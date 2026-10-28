"""Local media processing; no paid APIs, shell evaluation, or model uploads."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any

from .subtitles import validate_segments


class MediaError(RuntimeError):
    """An actionable failure of a local media tool or ASR backend."""


ALLOWED_MODELS = frozenset({"tiny", "base", "small", "medium", "large-v3", "turbo"})


def _run(command: list[str], *, cwd: Path | None = None, timeout: float | None = None) -> str:
    try:
        result = subprocess.run(
            command, cwd=cwd, stdin=subprocess.DEVNULL, capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=timeout, check=False,
        )
    except FileNotFoundError as exc:
        raise MediaError(f"找不到本地工具：{command[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise MediaError(f"本地工具执行超时：{Path(command[0]).name}") from exc
    except OSError as exc:
        raise MediaError(f"无法运行本地工具 {Path(command[0]).name}：{exc}") from exc
    if result.returncode:
        detail = result.stderr.strip()[-4000:] or f"退出状态 {result.returncode}"
        raise MediaError(f"{Path(command[0]).name} 执行失败：{detail}")
    return result.stdout


def _tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise MediaError(f"缺少 {name}，请先安装可用的 FFmpeg 工具")
    return path


def _input_file(path: Path, label: str) -> Path:
    try:
        resolved = Path(path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise MediaError(f"{label}不存在或不可访问：{path}") from exc
    if not resolved.is_file():
        raise MediaError(f"{label}必须是普通文件：{path}")
    return resolved


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def probe_media(path: Path) -> dict:
    path = _input_file(path, "输入媒体")
    raw = _run([
        _tool("ffprobe"), "-v", "error", "-show_entries",
        "format=duration,format_name,size:stream=index,codec_type,codec_name,width,height,duration",
        "-of", "json", str(path),
    ], timeout=30)
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise MediaError("ffprobe 返回了无法读取的媒体信息") from exc
    if not isinstance(data, dict):
        raise MediaError("ffprobe 返回的媒体信息不是有效对象")
    streams = data.get("streams", [])
    if not isinstance(streams, list):
        raise MediaError("ffprobe 返回的媒体流信息无效")
    streams = [stream for stream in streams if isinstance(stream, dict)]
    format_data = data.get("format", {})
    if not isinstance(format_data, dict):
        format_data = {}
    duration = _finite_float(format_data.get("duration"))
    if duration is None:
        durations = [_finite_float(stream.get("duration")) for stream in streams]
        duration = max((number for number in durations if number is not None), default=None)
    has_video = any(stream.get("codec_type") == "video" for stream in streams)
    has_audio = any(stream.get("codec_type") == "audio" for stream in streams)
    if not (has_video or has_audio):
        raise MediaError("输入文件没有可读取的视频或音频流")
    return {
        "path": str(path), "duration": duration, "size_bytes": path.stat().st_size,
        "format_name": format_data.get("format_name"), "streams": streams,
        "has_video": has_video, "has_audio": has_audio,
    }


def _ffmpeg_features(path: str) -> tuple[set[str], set[str]]:
    filters = _run([path, "-hide_banner", "-filters"], timeout=20)
    encoders = _run([path, "-hide_banner", "-encoders"], timeout=20)
    filter_names = {line.split()[1] for line in filters.splitlines() if re.match(r"^\s*[TSC.]{3}\s+\S+", line)}
    encoder_names = {line.split()[1] for line in encoders.splitlines() if re.match(r"^\s*[VAS.][A-Z.]{5}\s+\S+", line)}
    return filter_names, encoder_names


def media_capabilities() -> dict:
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    filters: set[str] = set()
    encoders: set[str] = set()
    errors: list[str] = []
    if ffmpeg:
        try:
            filters, encoders = _ffmpeg_features(ffmpeg)
        except MediaError as exc:
            errors.append(str(exc))
    try:
        asr_installed = importlib.util.find_spec("faster_whisper") is not None
        asr_version = importlib.metadata.version("faster-whisper") if asr_installed else None
    except (ImportError, ValueError, importlib.metadata.PackageNotFoundError):
        asr_installed, asr_version = False, None
    return {
        "ffmpeg": ffmpeg, "ffprobe": ffprobe,
        "asr_installed": asr_installed, "asr_backend": "faster-whisper", "asr_version": asr_version,
        "asr_readiness": "安装状态检查；实际可用性与模型缓存需通过转录确认",
        "soft_subtitles": {"libx264", "aac", "mov_text"}.issubset(encoders),
        "burn_in": "subtitles" in filters and {"libx264", "aac"}.issubset(encoders),
        "video_encoder": "libx264" if "libx264" in encoders else None,
        "audio_encoder": "aac" if "aac" in encoders else None,
        "errors": errors,
    }


def transcribe_media(
    path: Path, output_dir: Path, model: str = "small", language: str | None = None,
    allow_download: bool = False, model_dir: Path | None = None,
) -> dict:
    if not isinstance(model, str) or model not in ALLOWED_MODELS:
        raise MediaError(f"不支持的模型：{model}；可选 {', '.join(sorted(ALLOWED_MODELS))}")
    if not isinstance(allow_download, bool):
        raise MediaError("allow_download 必须是布尔值")
    if language is not None and (not isinstance(language, str) or not re.fullmatch(r"[a-z]{2,3}", language)):
        raise MediaError("language 必须是小写语言代码，例如 en、zh、ja，或留空自动识别")
    path = _input_file(path, "输入媒体")
    if not probe_media(path)["has_audio"]:
        raise MediaError("输入媒体没有音轨，无法识别语音")
    try:
        from faster_whisper import WhisperModel
    except (ImportError, OSError) as exc:
        raise MediaError("本项目的 faster-whisper 后端不可用；请安装项目的 ASR 依赖") from exc
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(model_dir).expanduser().resolve() if model_dir is not None else output_dir / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Names are allowlisted above: no arbitrary model repository or path is
        # accepted. local_files_only ensures default requests cannot download.
        recognizer = WhisperModel(
            model, device="cpu", compute_type="int8", download_root=str(cache_dir),
            local_files_only=not allow_download,
        )
        generated_segments, info = recognizer.transcribe(
            str(path), language=language, task="transcribe", beam_size=5,
            vad_filter=True, condition_on_previous_text=False,
        )
        segments = []
        for segment in generated_segments:
            text = segment.text.strip()
            if not text:
                continue
            segments.append({"id": len(segments) + 1, "start": float(segment.start), "end": float(segment.end), "text": text})
        if not segments:
            raise MediaError("未识别到有效语音；请检查音轨、语言或尝试更大的模型")
        segments = validate_segments(segments)
    except MediaError:
        raise
    except Exception as exc:
        hint = "" if allow_download else "；模型可能尚未缓存，可明确允许下载后重试"
        raise MediaError(f"本地语音识别失败{hint}：{exc}") from exc
    return {
        "language": info.language,
        "language_probability": float(info.language_probability),
        "segments": segments,
    }


def export_video(
    input_path: Path, subtitle_path: Path, output_path: Path,
    mode: str = "soft", language: str = "zho",
) -> dict:
    if mode not in {"soft", "burn_in"}:
        raise MediaError("mode 必须为 soft（内嵌可切换字幕）或 burn_in（烧录字幕）")
    if not isinstance(language, str) or not re.fullmatch(r"[a-z]{3}", language):
        raise MediaError("字幕语言必须为三字母小写代码，例如 zho")
    input_path = _input_file(input_path, "输入视频")
    subtitle_path = _input_file(subtitle_path, "字幕文件")
    if subtitle_path.suffix.lower() not in {".srt", ".vtt"}:
        raise MediaError("视频导出仅接受本项目导出的 SRT 或 VTT 字幕")
    output_path = Path(output_path).expanduser().absolute()
    if output_path.suffix.lower() != ".mp4":
        raise MediaError("视频输出文件必须使用 .mp4 扩展名")
    if os.path.lexists(output_path):
        raise MediaError(f"输出路径已存在，不会覆盖：{output_path}")
    if not probe_media(input_path)["has_video"]:
        raise MediaError("输入文件没有视频流")
    ffmpeg = _tool("ffmpeg")
    filters, encoders = _ffmpeg_features(ffmpeg)
    if not {"libx264", "aac"}.issubset(encoders):
        raise MediaError("当前 FFmpeg 缺少 H.264/AAC 编码器，无法导出兼容 MP4")
    if mode == "soft" and "mov_text" not in encoders:
        raise MediaError("当前 FFmpeg 缺少 mov_text 字幕编码器，无法内嵌中文字幕")
    if mode == "burn_in" and "subtitles" not in filters:
        raise MediaError("当前 FFmpeg 缺少 subtitles/libass 滤镜，无法烧录字幕；可使用 soft 导出可切换字幕视频")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # A fixed, generated subtitle name keeps original paths out of the FFmpeg
    # filter language. This also works with quotes, Unicode and colons in paths.
    with tempfile.TemporaryDirectory(prefix=".videolingo-export-", dir=output_path.parent) as temporary:
        work_dir = Path(temporary)
        local_subtitle = work_dir / f"captions{subtitle_path.suffix.lower()}"
        shutil.copyfile(subtitle_path, local_subtitle)
        temporary_output = work_dir / "output.mp4"
        command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n", "-i", str(input_path)]
        if mode == "soft":
            command += [
                "-i", str(local_subtitle), "-map", "0:v:0", "-map", "0:a?", "-map", "1:0",
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            ]
        else:
            command += [
                "-map", "0:v:0", "-map", "0:a?", "-vf",
                f"pad=ceil(iw/2)*2:ceil(ih/2)*2,subtitles=filename={local_subtitle.name}",
            ]
        command += [
            "-c:v", "libx264", "-crf", "20", "-preset", "medium", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
        ]
        if mode == "soft":
            command += ["-c:s", "mov_text", "-metadata:s:s:0", f"language={language}", "-disposition:s:0", "default"]
        command.append(str(temporary_output))
        _run(command, cwd=work_dir)
        metadata = probe_media(temporary_output)
        if mode == "soft" and not any(stream.get("codec_type") == "subtitle" for stream in metadata["streams"]):
            raise MediaError("导出视频没有字幕轨，已取消发布")
        # link() publishes atomically and fails if any file/directory/symlink
        # appeared at the target in the meantime. os.replace() would overwrite.
        try:
            os.link(temporary_output, output_path)
        except FileExistsError as exc:
            raise MediaError(f"输出路径已存在，不会覆盖：{output_path}") from exc
        except OSError as exc:
            raise MediaError(f"无法发布导出视频：{exc}") from exc
    metadata["path"] = str(output_path)
    metadata["subtitle_mode"] = mode
    metadata["subtitle_language"] = language
    return metadata

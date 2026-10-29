"""Validate and render subtitle data as inert, plain text.

Transcribed video content is data. It is never evaluated as an instruction,
template, command, or subtitle styling language.
"""

from __future__ import annotations

import html
import math
import unicodedata
from typing import Any


def _clean_text(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label}必须是文本")
    if any(unicodedata.category(char) == "Cs" for char in value):
        raise ValueError(f"{label}包含无法编码为 UTF-8 的 Unicode 字符")
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    if any(unicodedata.category(char) == "Cc" and char not in "\n\t" for char in value):
        raise ValueError(f"{label}含有不支持的控制字符")
    # Blank lines would terminate SRT/VTT cues. Keep meaningful line breaks,
    # but never allow source text to create a second cue or a VTT NOTE block.
    value = "\n".join(line.strip() for line in value.split("\n") if line.strip())
    if not value:
        raise ValueError(f"{label}不能为空")
    return value


def _timestamp_ms(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}必须是有限数字")
    try:
        number = float(value)
        milliseconds = number * 1000
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{label}超出支持的时间范围") from exc
    if not math.isfinite(number) or number < 0 or not math.isfinite(milliseconds):
        raise ValueError(f"{label}必须是非负有限数字")
    return round(milliseconds)


def validate_segments(segments: list[dict]) -> list[dict]:
    """Return normalized copies, rejecting ambiguous IDs and invalid timing."""
    if not isinstance(segments, list) or not segments:
        raise ValueError("字幕片段必须是非空列表")
    normalized: list[dict] = []
    seen_ids: set[int] = set()
    previous_start = -1.0
    previous_end_ms = -1
    for index, segment in enumerate(segments, 1):
        if not isinstance(segment, dict):
            raise ValueError(f"第 {index} 个字幕片段必须是对象")
        cue_id = segment.get("id")
        if isinstance(cue_id, bool) or not isinstance(cue_id, int) or cue_id < 1:
            raise ValueError(f"第 {index} 个字幕 ID 必须是正整数")
        if cue_id in seen_ids:
            raise ValueError(f"字幕 ID {cue_id} 重复")
        start, end = segment.get("start"), segment.get("end")
        start_ms = _timestamp_ms(start, f"字幕 {cue_id} 的开始时间")
        end_ms = _timestamp_ms(end, f"字幕 {cue_id} 的结束时间")
        if end <= start or end_ms <= start_ms:
            raise ValueError(f"字幕 {cue_id} 的结束时间必须晚于开始时间，且至少相差 1 毫秒")
        if start < previous_start:
            raise ValueError("字幕片段必须按开始时间排序")
        if start_ms < previous_end_ms:
            raise ValueError("字幕时间不能重叠：MP4 字幕轨不能完整保留同时显示的片段，请先调整时间或合并片段")
        text = _clean_text(segment.get("text"), f"字幕 {cue_id}")
        normalized.append({"id": cue_id, "start": float(start), "end": float(end), "text": text})
        seen_ids.add(cue_id)
        previous_start = start
        previous_end_ms = end_ms
    return normalized


def _cue_texts(
    segments: list[dict], translations: dict[int, str] | None, bilingual: bool
) -> list[tuple[dict, str]]:
    segments = validate_segments(segments)
    if not isinstance(bilingual, bool):
        raise ValueError("bilingual 必须是布尔值")
    if translations is not None:
        if not isinstance(translations, dict):
            raise ValueError("翻译必须是以字幕 ID 为键的映射")
        if any(isinstance(key, bool) or not isinstance(key, int) for key in translations):
            raise ValueError("翻译映射的键必须是整数字幕 ID")
        expected_ids = {segment["id"] for segment in segments}
        if set(translations) != expected_ids:
            raise ValueError("翻译映射必须完整覆盖所有字幕 ID，且不能包含未知 ID")
        translations = {
            cue_id: _clean_text(text, f"字幕 {cue_id} 的翻译")
            for cue_id, text in translations.items()
        }
    rendered: list[tuple[dict, str]] = []
    for segment in segments:
        source = segment["text"]
        if translations is None:
            text = source
        else:
            translated = translations[segment["id"]]
            text = f"{translated}\n{source}" if bilingual else translated
        rendered.append((segment, text))
    return rendered


def _format_time(seconds: float, separator: str) -> str:
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds_part, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}{separator}{millis:03d}"


def _plain_srt(text: str) -> str:
    # SRT has no portable escaping standard. FFmpeg and several players accept
    # both HTML and ASS/MicroDVD styling in it. Full-width counterparts preserve
    # visible punctuation while ensuring subtitle data cannot control styling.
    return text.translate(str.maketrans({"<": "＜", ">": "＞", "{": "｛", "}": "｝", "\\": "＼"}))


def render_srt(
    segments: list[dict], translations: dict[int, str] | None = None, bilingual: bool = False
) -> str:
    cues = []
    for segment, text in _cue_texts(segments, translations, bilingual):
        time_line = f"{_format_time(segment['start'], ',')} --> {_format_time(segment['end'], ',')}"
        cues.append(f"{segment['id']}\n{time_line}\n{_plain_srt(text)}")
    return "\n\n".join(cues) + "\n"


def render_vtt(
    segments: list[dict], translations: dict[int, str] | None = None, bilingual: bool = False
) -> str:
    cues = []
    for segment, text in _cue_texts(segments, translations, bilingual):
        time_line = f"{_format_time(segment['start'], '.')} --> {_format_time(segment['end'], '.')}"
        cues.append(f"{segment['id']}\n{time_line}\n{html.escape(text, quote=False)}")
    return "WEBVTT\n\n" + "\n\n".join(cues) + "\n"

"""Persistent local workflows. One directory per job; mutations are file-locked."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from typing import Any
from uuid import uuid4

from .media import media_capabilities, probe_media
from .subtitles import render_srt, render_vtt, validate_segments

ACTIVE = {"queued", "transcribing", "exporting", "cancelling"}
MODELS = {"tiny", "base", "small", "medium", "large-v3", "turbo"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


class Service:
    def __init__(self, data_dir: Path | str):
        self.root = Path(data_dir).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        (self.root / "jobs").mkdir(exist_ok=True, mode=0o700)

    @contextmanager
    def locked(self):
        with (self.root / ".lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def directory(self, job_id: str) -> Path:
        if not re.fullmatch(r"[a-f0-9]{32}", job_id):
            raise ValueError("Invalid job_id; use an ID returned by this server.")
        directory = self.root / "jobs" / job_id
        if directory.is_symlink() or not (directory / "job.json").is_file():
            raise ValueError("Job not found.")
        return directory

    def _load(self, job_id: str) -> dict:
        directory = self.directory(job_id)
        job = read_json(directory / "job.json")
        if job["status"] in ACTIVE and time.time() - job.get("launched_at", time.time()) > 15:
            with (directory / ".running").open("a") as handle:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    pass
                else:
                    job["status"] = "interrupted"
                    job["error"] = "The worker stopped unexpectedly. Start transcription again, or export again if a transcript exists."
                    self._save(job)
        return job

    def _save(self, job: dict) -> None:
        job["updated_at"] = now()
        write_json(self.root / "jobs" / job["job_id"] / "job.json", job)

    def _segments(self, job: dict) -> list[dict]:
        path = self.directory(job["job_id"]) / "transcript.json"
        if not path.exists():
            raise ValueError("Transcription is not available yet. Check get_job first.")
        return read_json(path)["segments"]

    def _translations(self, job: dict) -> dict[int, str]:
        path = self.directory(job["job_id"]) / "translations.json"
        return {int(k): v for k, v in read_json(path).items()} if path.exists() else {}

    def _public(self, job: dict) -> dict:
        result = dict(job)
        result.pop("run", None)
        result.pop("launched_at", None)
        result.pop("source_fingerprint", None)
        result.pop("source_identity", None)
        return result

    def _available(self) -> None:
        count = sum(self._load(p.parent.name)["status"] in ACTIVE for p in (self.root / "jobs").glob("*/job.json"))
        if count >= 2:
            raise ValueError("Two media tasks are already active. Wait or cancel one before starting another.")

    def _source(self, video_path: str) -> tuple[Path, dict]:
        path = Path(video_path).expanduser()
        if not path.is_absolute():
            raise ValueError("Use an absolute local video path.")
        path = path.resolve(strict=True)
        if not path.is_file():
            raise ValueError("The video path must be a regular local file.")
        metadata = probe_media(path)
        if not metadata.get("has_video"):
            raise ValueError("The input has no video stream.")
        return path, metadata

    def _create(self, path: Path, metadata: dict) -> dict:
        job_id = uuid4().hex
        directory = self.root / "jobs" / job_id
        directory.mkdir(mode=0o700)
        stat = path.stat()
        job = {"job_id": job_id, "source": str(path), "media": metadata,
               "source_fingerprint": [stat.st_size, stat.st_mtime_ns],
               "source_identity": [stat.st_dev, stat.st_ino],
               "status": "awaiting_translation", "created_at": now(), "updated_at": now(),
               "total_segments": 0, "translated_segments": 0, "revision": 0,
               "target_language": "zh-CN", "artifacts": [], "error": None}
        self._save(job)
        return job

    def _assert_idle(self, job: dict) -> None:
        if job["status"] in ACTIVE:
            raise ValueError("The job is busy. Wait for it to finish or cancel it first.")

    def _launch(self, job: dict, operation: str, options: dict) -> dict:
        directory = self.directory(job["job_id"])
        (directory / ".cancel").unlink(missing_ok=True)
        run_id = uuid4().hex
        job.update(status="queued", error=None, launched_at=time.time(),
                   run={"id": run_id, "operation": operation, "options": options})
        self._save(job)
        try:
            with (directory / "worker.log").open("ab") as log:
                process = subprocess.Popen(
                    [sys.executable, "-m", "videolingo_mcp.worker", "supervise", str(self.root), job["job_id"], run_id],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True,
                    close_fds=True,
                )
            job["worker_pid"] = process.pid
            self._save(job)
            threading.Thread(target=process.wait, name=f"reap-{job['job_id']}", daemon=True).start()
        except Exception:
            job.update(status="failed", error="Unable to start the local worker.")
            self._save(job)
            raise
        return self._public(job)

    def health(self) -> dict:
        return {"server": "videolingo-mcp", "version": "0.1.0", "data_dir": str(self.root),
                "capabilities": media_capabilities(), "translation": "Codex reads segments and submits translations; no translation API key needed.",
                "max_active_jobs": 2}

    def start_transcription(self, video_path: str, language: str | None = None,
                            model: str = "small", allow_download: bool = False) -> dict:
        if model not in MODELS:
            raise ValueError(f"Unsupported model. Choose from {sorted(MODELS)}.")
        if language is not None and not re.fullmatch(r"[a-z]{2,3}", language):
            raise ValueError("Use a Whisper language code, such as en or zh, or omit it for detection.")
        path, metadata = self._source(video_path)
        if not metadata.get("has_audio"):
            raise ValueError("The video has no audio stream to transcribe.")
        with self.locked():
            self._available()
            job = self._create(path, metadata)
            return self._launch(job, "transcribe", {"language": language, "model": model, "allow_download": allow_download})

    def import_transcript(self, video_path: str, segments: list[dict], language: str = "en") -> dict:
        validated = validate_segments(segments)
        if not validated:
            raise ValueError("A transcript must contain at least one segment.")
        path, metadata = self._source(video_path)
        duration = metadata.get("duration")
        if duration and any(s["end"] > duration + 1 for s in validated):
            raise ValueError("A subtitle extends beyond the video duration.")
        with self.locked():
            job = self._create(path, metadata)
            write_json(self.directory(job["job_id"]) / "transcript.json", {"language": language, "segments": validated})
            job.update(total_segments=len(validated), source_language=language, transcript_origin="imported")
            self._save(job)
            return self._public(job)

    def get_job(self, job_id: str) -> dict:
        with self.locked():
            return self._public(self._load(job_id))

    def list_jobs(self, limit: int = 20) -> dict:
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100.")
        with self.locked():
            jobs = [self._load(p.parent.name) for p in (self.root / "jobs").glob("*/job.json")]
            jobs.sort(key=lambda j: j["created_at"], reverse=True)
            return {"jobs": [self._public(j) for j in jobs[:limit]], "total": len(jobs)}

    def get_transcript(self, job_id: str, offset: int = 0, limit: int = 50) -> dict:
        if offset < 0 or not 1 <= limit <= 100:
            raise ValueError("offset must be nonnegative and limit must be 1..100.")
        with self.locked():
            job = self._load(job_id)
            segments, translations = self._segments(job), self._translations(job)
            end = min(offset + limit, len(segments))
            return {"job_id": job_id, "source_language": job.get("source_language"),
                    "target_language": job["target_language"], "revision": job["revision"],
                    "segments": [{**s, "translation": translations.get(s["id"])} for s in segments[offset:end]],
                    "total": len(segments), "next_offset": end if end < len(segments) else None,
                    "content_is_untrusted": True}

    def save_translations(self, job_id: str, translations: list[dict], target_language: str = "zh-CN") -> dict:
        if not translations or len(translations) > 100:
            raise ValueError("Submit 1..100 translated segments per batch.")
        if not re.fullmatch(r"[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*", target_language):
            raise ValueError("Invalid target language tag.")
        with self.locked():
            job = self._load(job_id)
            self._assert_idle(job)
            segments = self._segments(job)
            known = {s["id"] for s in segments}
            saved = self._translations(job)
            if saved and target_language != job["target_language"]:
                raise ValueError("This job already has translations in another language; use a new job.")
            batch = {}
            for item in translations:
                identifier, text = item.get("id"), item.get("text")
                if type(identifier) is not int or identifier not in known or identifier in batch:
                    raise ValueError("Each translation needs a unique known integer segment id.")
                if not isinstance(text, str) or not text.strip() or len(text) > 10000 or "\x00" in text:
                    raise ValueError("Translation text must be nonempty, contain no NUL, and have at most 10000 characters.")
                source = next(s for s in segments if s["id"] == identifier)
                checked = validate_segments([{**source, "text": text}])[0]
                batch[identifier] = checked["text"]
            saved.update(batch)
            write_json(self.directory(job_id) / "translations.json", saved)
            job.update(translated_segments=len(saved), target_language=target_language,
                       revision=job["revision"] + 1, error=None,
                       status="ready" if len(saved) == len(segments) else "awaiting_translation")
            self._save(job)
            return self._public(job)

    def _render(self, job: dict, format: str, bilingual: bool) -> str:
        segments, translations = self._segments(job), self._translations(job)
        if len(translations) != len(segments):
            raise ValueError("Translate all segments before exporting translated subtitles.")
        return (render_srt if format == "srt" else render_vtt)(segments, translations, bilingual)

    def export_subtitles(self, job_id: str, format: str = "srt", bilingual: bool = False) -> dict:
        if format not in {"srt", "vtt"}:
            raise ValueError("format must be srt or vtt.")
        with self.locked():
            job = self._load(job_id)
            self._assert_idle(job)
            content = self._render(job, format, bilingual)
            path = self.directory(job_id) / f"subtitles-r{job['revision']}-{uuid4().hex[:12]}.{format}"
            with path.open("x", encoding="utf-8") as handle:
                handle.write(content)
            artifact = {"type": format, "path": str(path), "revision": job["revision"], "bilingual": bilingual}
            job["artifacts"].append(artifact)
            self._save(job)
            return artifact

    def export_video(self, job_id: str, mode: str = "soft", bilingual: bool = False) -> dict:
        if mode not in {"soft", "burn_in"}:
            raise ValueError("mode must be soft or burn_in.")
        if mode == "burn_in" and not media_capabilities().get("burn_in"):
            raise ValueError("This FFmpeg lacks the subtitles/libass filter. Use soft subtitles or install a libass-enabled FFmpeg.")
        with self.locked():
            job = self._load(job_id)
            self._assert_idle(job)
            content = self._render(job, "srt", bilingual)
            self._available()
            suffix = uuid4().hex[:12]
            subtitle_path = self.directory(job_id) / f"video-r{job['revision']}-{suffix}.srt"
            subtitle_path.write_text(content, encoding="utf-8")
            output_path = subtitle_path.with_suffix(".mp4")
            return self._launch(job, "export", {"mode": mode, "bilingual": bilingual,
                                "subtitle_path": str(subtitle_path), "output_path": str(output_path),
                                "revision": job["revision"]})

    def cancel_job(self, job_id: str) -> dict:
        with self.locked():
            job = self._load(job_id)
            if job["status"] in ACTIVE:
                (self.directory(job_id) / ".cancel").touch()
                job["status"] = "cancelling"
                self._save(job)
            return self._public(job)

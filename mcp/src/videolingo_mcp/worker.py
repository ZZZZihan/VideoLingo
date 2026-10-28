"""Detached supervisor keeps jobs alive across client reconnects and owns cancellation."""

from __future__ import annotations

import errno
import fcntl
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

from .jobs import Service, read_json, write_json
from .media import export_video, transcribe_media


def verify_source(job: dict) -> None:
    stat = Path(job["source"]).stat()
    if ([stat.st_size, stat.st_mtime_ns] != job["source_fingerprint"]
            or ("source_identity" in job and [stat.st_dev, stat.st_ino] != job["source_identity"])):
        raise ValueError("The source video changed after this job was created. Start a new job.")


def execute(root: Path, job_id: str, run_id: str) -> None:
    service = Service(root)
    directory = service.directory(job_id)
    job = read_json(directory / "job.json")
    if job["run"]["id"] != run_id:
        raise RuntimeError("Worker generation is no longer current.")
    path = Path(job["source"])
    verify_source(job)
    run = job["run"]
    options = run["options"]
    if run["operation"] == "transcribe":
        result = transcribe_media(path, directory, model_dir=root / "models", **options)
        if not result.get("segments"):
            raise ValueError("No speech was detected. Try another language/model or import reviewed subtitles.")
    else:
        result = export_video(path, Path(options["subtitle_path"]), Path(options["output_path"]),
                              mode=options["mode"], language="zho" if job["target_language"].lower().startswith("zh") else "und")
    try:
        verify_source(job)
    except (ValueError, OSError):
        if run["operation"] == "export":
            Path(options["output_path"]).unlink(missing_ok=True)
        raise
    write_json(directory / f"result-{run_id}.json", result)


def _signal_group(child: subprocess.Popen, signum: int) -> bool:
    settle_until = time.monotonic() + 0.5
    while True:
        try:
            os.killpg(child.pid, signum)
            return True
        except ProcessLookupError:
            return False
        except PermissionError as error:
            # Darwin reports EPERM for zombies and during the exit transition.
            # Allow that transition to settle; never assume that a live leader
            # or descendant has exited merely because a signal was rejected.
            if error.errno != errno.EPERM or sys.platform != "darwin":
                raise
            if child.poll() is None:
                if time.monotonic() < settle_until:
                    time.sleep(0.02)
                    continue
                error.add_note("The process group leader has not yet exited.")
                raise
            try:
                result = subprocess.run(
                    ["/bin/ps", "-axo", "pgid=,stat="], check=True,
                    capture_output=True, text=True, timeout=5,
                )
                rows = [line.split() for line in result.stdout.splitlines() if line.strip()]
                if not rows or any(len(row) != 2 for row in rows):
                    raise ValueError("Unable to verify process group members")
                members = [(int(group), state) for group, state in rows]
            except (OSError, subprocess.SubprocessError, ValueError) as observation_error:
                error.add_note(f"Cannot verify process group members: {type(observation_error).__name__}")
                raise error
            if any(group == child.pid and not state.startswith("Z") for group, state in members):
                if time.monotonic() < settle_until:
                    time.sleep(0.02)
                    continue
                error.add_note("The process group still has live members.")
                raise error
            return False


def stop_child(child: subprocess.Popen) -> None:
    # The leader may exit on SIGTERM while a descendant ignores it. Keep the
    # process group identity until every member has been signalled, not just the
    # Popen leader; this function is only used for a group we launched ourselves.
    if not _signal_group(child, signal.SIGTERM):
        child.wait()
        return
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        child.poll()
        if not _signal_group(child, 0):
            child.wait()
            return
        time.sleep(0.05)
    _signal_group(child, signal.SIGKILL)
    child.wait()


def clean_temporary_exports(directory: Path) -> None:
    for path in directory.glob(".videolingo-export-*"):
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)


def supervise(root: Path, job_id: str, run_id: str) -> None:
    service = Service(root)
    directory = service.directory(job_id)
    cancelled = directory / ".cancel"
    stopping = False

    def on_signal(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    with (directory / ".running").open("a") as running:
        fcntl.flock(running, fcntl.LOCK_EX)
        with service.locked():
            job = read_json(directory / "job.json")
            if job["run"]["id"] != run_id:
                return
            if cancelled.exists() or stopping:
                job.update(status="cancelled" if cancelled.exists() else "interrupted", error=None)
                service._save(job)
                return
            job["status"] = "transcribing" if job["run"]["operation"] == "transcribe" else "exporting"
            service._save(job)
        child = None
        try:
            with (directory / f"task-{run_id}.log").open("wb") as log:
                child = subprocess.Popen([sys.executable, "-m", "videolingo_mcp.worker", "execute", str(root), job_id, run_id],
                                         stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                         start_new_session=True, close_fds=True)
                while child.poll() is None:
                    if cancelled.exists() or stopping:
                        stop_child(child)
                        break
                    time.sleep(0.2)
            with service.locked():
                job = read_json(directory / "job.json")
                if cancelled.exists() or stopping:
                    clean_temporary_exports(directory)
                    job.update(status="cancelled" if cancelled.exists() else "interrupted", error=None)
                    if job["run"]["operation"] == "export":
                        Path(job["run"]["options"]["output_path"]).unlink(missing_ok=True)
                elif child.returncode:
                    stop_child(child)
                    clean_temporary_exports(directory)
                    detail = (directory / f"task-{run_id}.log").read_text(encoding="utf-8", errors="replace")[-2000:]
                    job.update(status="failed", error=detail.strip() or f"Media process exited with code {child.returncode}.")
                else:
                    result = read_json(directory / f"result-{run_id}.json")
                    if job["run"]["operation"] == "transcribe":
                        write_json(directory / "transcript.json", result)
                        job.update(status="awaiting_translation", total_segments=len(result["segments"]),
                                   source_language=result.get("language"), transcript_origin="local_asr")
                    else:
                        options = job["run"]["options"]
                        job["artifacts"].append({"type": "video", "path": options["output_path"],
                                                 "revision": options["revision"], "mode": options["mode"],
                                                 "bilingual": options["bilingual"]})
                        job["status"] = "completed"
                service._save(job)
        except BaseException as error:
            cleanup_errors = []
            if child is not None:
                try:
                    stop_child(child)
                except Exception as cleanup_error:
                    cleanup_errors.append(f"Process cleanup: {cleanup_error}")
            try:
                clean_temporary_exports(directory)
            except Exception as cleanup_error:
                cleanup_errors.append(f"Temporary file cleanup: {cleanup_error}")
            with service.locked():
                job = read_json(directory / "job.json")
                detail = f"Worker failed: {type(error).__name__}: {error}"
                if cleanup_errors:
                    detail += "; " + "; ".join(cleanup_errors)
                job.update(status="failed", error=detail)
                service._save(job)
            raise


if __name__ == "__main__":
    action, directory, identifier, generation = sys.argv[1:]
    (supervise if action == "supervise" else execute)(Path(directory), identifier, generation)

"""Behavioral workflow checks without model downloads or a translation API."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from videolingo_mcp.jobs import Service, read_json, write_json


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is required")
class WorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="videolingo-workflow-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.video = self.root / "source clip.mp4"
        self.data_dir = self.root / "jobs"
        subprocess.run(
            [
                "ffmpeg", "-v", "error", "-nostdin", "-y",
                "-f", "lavfi", "-i", "color=c=black:s=160x90:r=10:d=2",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
                "-c:v", "mpeg4", "-c:a", "aac", "-shortest", str(self.video),
            ],
            check=True,
            capture_output=True,
            timeout=20,
        )
        self.service = Service(self.data_dir)
        self.segments = [
            {"id": 1, "start": 0.1, "end": 0.9, "text": "One."},
            {"id": 2, "start": 1.0, "end": 1.9, "text": "Two."},
        ]

    def import_job(self) -> dict:
        return self.service.import_transcript(str(self.video), self.segments, language="en")

    def ready_job(self) -> dict:
        job = self.import_job()
        return self.service.save_translations(job["job_id"], [
            {"id": 1, "text": "第一句。"}, {"id": 2, "text": "第二句。"},
        ])

    def wait_terminal(self, job_id: str, timeout: float = 20) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = self.service.get_job(job_id)
            if job["status"] not in {"queued", "transcribing", "exporting", "cancelling"}:
                return job
            time.sleep(0.05)
        self.fail(f"job did not finish: {self.service.get_job(job_id)}")

    def wait_for_file(self, path: Path, timeout: float = 10) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if path.exists():
                try:
                    return json.loads(path.read_text())
                except json.JSONDecodeError:
                    pass
            time.sleep(0.05)
        self.fail(f"background worker did not reach the processing fixture: {path}")

    @staticmethod
    def process_is_running(pid: int) -> bool:
        result = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, timeout=5,
        )
        state = result.stdout.strip()
        return bool(state) and not state.startswith("Z")

    def test_cancellation_stops_processing_process_and_its_descendant(self) -> None:
        # The wrapper runs real capability checks, but pauses the encoding call
        # with a real child process. This deterministically exercises process-
        # group cancellation without relying on a downloaded speech model or a
        # machine-dependent, expensive video encode.
        marker = self.root / "processing-pids.json"
        wrapper_dir = self.root / "slow-tools"
        wrapper_dir.mkdir()
        wrapper = wrapper_dir / "ffmpeg"
        real_ffmpeg = shutil.which("ffmpeg")
        wrapper.write_text(
            f"#!{sys.executable}\n"
            "import json, os, subprocess, sys, time\n"
            "from pathlib import Path\n"
            "if '-i' in sys.argv:\n"
            "    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
            f"    Path({str(marker)!r}).write_text(json.dumps({{'pid': os.getpid(), 'child': child.pid, 'group': os.getpgrp()}}))\n"
            "    time.sleep(30)\n"
            "    child.terminate()\n"
            "    child.wait()\n"
            f"os.execv({real_ffmpeg!r}, [{real_ffmpeg!r}, *sys.argv[1:]])\n",
            encoding="utf-8",
        )
        wrapper.chmod(0o700)
        job_id = self.ready_job()["job_id"]
        pids = None
        try:
            with patch.dict(os.environ, {"PATH": str(wrapper_dir) + os.pathsep + os.environ["PATH"]}):
                started = self.service.export_video(job_id, mode="soft")
            self.assertEqual(started["status"], "queued")
            pids = self.wait_for_file(marker)
            self.assertTrue(self.process_is_running(pids["pid"]))
            self.assertTrue(self.process_is_running(pids["child"]))
            self.assertEqual(self.service.get_job(job_id)["status"], "exporting")
            self.assertTrue(self.service.health()["server"])
            self.service.cancel_job(job_id)
            self.service.cancel_job(job_id)
            cancelled = self.wait_terminal(job_id)
            self.assertEqual(cancelled["status"], "cancelled", cancelled)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and any(self.process_is_running(pids[k]) for k in ["pid", "child"]):
                time.sleep(0.05)
            self.assertFalse(self.process_is_running(pids["pid"]))
            self.assertFalse(self.process_is_running(pids["child"]))
            self.assertFalse(any(Path(a["path"]).suffix == ".mp4" for a in cancelled["artifacts"]))
            self.assertEqual(self.service.cancel_job(job_id)["status"], "cancelled")
        finally:
            self.service.cancel_job(job_id)
            if pids:
                try:
                    os.killpg(pids["group"], signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def test_stopped_worker_is_reported_interrupted_without_losing_transcript(self) -> None:
        job_id = self.ready_job()["job_id"]
        # Crash recovery is driven by the persisted state plus the OS lock;
        # backdate a crash state to avoid a 15-second wall-clock wait.
        state_path = self.data_dir / "jobs" / job_id / "job.json"
        state = read_json(state_path)
        state.update(status="exporting", launched_at=time.time() - 60)
        write_json(state_path, state)
        recovered = Service(self.data_dir).get_job(job_id)
        self.assertEqual(recovered["status"], "interrupted")
        self.assertTrue(recovered["error"])
        self.assertEqual(recovered["translated_segments"], 2)
        self.assertEqual(len(self.service.get_transcript(job_id)["segments"]), 2)
        self.assertTrue(Path(self.service.export_subtitles(job_id)["path"]).is_file())

    def test_translation_batch_validation_is_atomic(self) -> None:
        job_id = self.import_job()["job_id"]
        for invalid in [
            [{"id": 1, "text": "不得部分写入"}, {"id": 999, "text": "未知编号"}],
            [{"id": 1, "text": "第一次"}, {"id": 1, "text": "重复编号"}],
            [{"id": 1, "text": "   "}],
            [{"id": 1, "text": "不可导出的\x1b控制字符"}],
        ]:
            with self.subTest(invalid=invalid):
                with self.assertRaises((ValueError, TypeError)):
                    self.service.save_translations(job_id, invalid)
                transcript = self.service.get_transcript(job_id)
                self.assertTrue(all(not segment.get("translation") for segment in transcript["segments"]))
                self.assertEqual(self.service.get_job(job_id)["translated_segments"], 0)

    def test_new_revision_preserves_old_exports_and_timestamps(self) -> None:
        job_id = self.ready_job()["job_id"]
        first = Path(self.service.export_subtitles(job_id, format="srt")["path"])
        first_content = first.read_text(encoding="utf-8")
        self.service.save_translations(job_id, [{"id": 1, "text": "修订后的第一句。"}])
        second = Path(self.service.export_subtitles(job_id, format="srt")["path"])
        third = Path(self.service.export_subtitles(job_id, format="vtt")["path"])
        self.assertNotEqual(first, second)
        self.assertEqual(first.read_text(encoding="utf-8"), first_content)
        self.assertIn("第一句。", first_content)
        self.assertNotIn("修订后", first_content)
        self.assertIn("修订后的第一句。", second.read_text(encoding="utf-8"))
        self.assertIn("00:00:00,100 --> 00:00:00,900", first_content)
        self.assertTrue(third.read_text(encoding="utf-8").startswith("WEBVTT"))
        artifacts = self.service.get_job(job_id)["artifacts"]
        paths = {str(artifact["path"]) for artifact in artifacts}
        self.assertTrue({str(first), str(second), str(third)} <= paths)
        self.assertTrue(all("revision" in artifact for artifact in artifacts))

    def test_soft_video_export_contains_chinese_subtitle_track(self) -> None:
        original = self.video.read_bytes()
        job_id = self.ready_job()["job_id"]
        result = self.service.export_video(job_id, mode="soft")
        exported = self.wait_terminal(result["job_id"])
        self.assertEqual(exported["status"], "completed", exported)
        videos = [Path(item["path"]) for item in exported["artifacts"] if Path(item["path"]).suffix == ".mp4"]
        self.assertTrue(videos, exported)
        output = videos[-1]
        self.assertTrue(output.is_relative_to(self.data_dir))
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(output)],
            check=True, capture_output=True, text=True, timeout=10,
        )
        streams = json.loads(probe.stdout)["streams"]
        self.assertTrue(any(stream["codec_type"] == "video" for stream in streams))
        self.assertTrue(any(stream["codec_type"] == "audio" for stream in streams))
        self.assertTrue(any(stream["codec_type"] == "subtitle" for stream in streams))
        extracted = subprocess.run(
            ["ffmpeg", "-v", "error", "-nostdin", "-i", str(output), "-map", "0:s:0", "-f", "srt", "-"],
            check=True, capture_output=True, text=True, timeout=10,
        )
        self.assertIn("第一句。", extracted.stdout)
        self.assertIn("第二句。", extracted.stdout)
        self.assertEqual(self.video.read_bytes(), original)

    def test_invalid_paths_pagination_and_segments_are_rejected(self) -> None:
        for segments in [
            [],
            [{"id": 1, "start": 0.1, "end": 0.1, "text": "empty interval"}],
            [{"id": 1, "start": -1, "end": 1, "text": "negative"}],
            [{"id": 1, "start": 0, "end": float("nan"), "text": "not finite"}],
        ]:
            with self.subTest(segments=segments):
                with self.assertRaises((ValueError, TypeError)):
                    self.service.import_transcript(str(self.video), segments, language="en")
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.service.import_transcript("relative.mp4", self.segments, language="en")
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.service.get_job("../outside")
        job_id = self.import_job()["job_id"]
        for kwargs in [{"offset": -1}, {"limit": 0}]:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    self.service.get_transcript(job_id, **kwargs)


if __name__ == "__main__":
    unittest.main()

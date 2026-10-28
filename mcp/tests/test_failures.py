"""Failure and lifecycle acceptance through real MCP stdio subprocesses.

Each run retains its media, task records and observations under .scratch.
Controlled FFmpeg wrappers inject failures at real process boundaries; they do
not replace MCP, task management or the operating-system cancellation path.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import unittest
from uuid import uuid4

from mcp import Client, StdioServerParameters, stdio_client


PROJECT = Path(__file__).resolve().parents[1]
EVIDENCE = PROJECT / ".scratch" / "qa-20261029" / "agent-review"
ACTIVE = {"queued", "transcribing", "exporting", "cancelling"}


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is required")
class FailureAcceptanceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.root = EVIDENCE / f"{self._testMethodName}-{uuid4().hex[:8]}"
        self.root.mkdir(parents=True)
        self.data = self.root / "data"
        self.video = self.root / "输入 视频.mp4"
        self.real_ffmpeg = shutil.which("ffmpeg")
        self.groups: set[int] = set()
        self.make_video(self.video)

    def tearDown(self) -> None:
        # Only groups recorded by this test's own wrapper are eligible.
        for group in self.groups:
            try:
                os.killpg(group, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def record(self, name: str, value) -> None:
        (self.root / f"{name}.json").write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8",
        )

    def make_video(self, path: Path, *, audio: bool = True, color: str = "blue") -> None:
        command = [self.real_ffmpeg, "-v", "error", "-nostdin", "-y", "-f", "lavfi",
                   "-i", f"color=c={color}:s=160x90:r=10:d=2"]
        if audio:
            command += ["-f", "lavfi", "-i", "sine=frequency=440:duration=2"]
        command += ["-c:v", "mpeg4"]
        if audio:
            command += ["-c:a", "aac", "-shortest"]
        command.append(str(path))
        subprocess.run(command, check=True, capture_output=True, timeout=20)

    @asynccontextmanager
    async def connect(self, env: dict[str, str] | None = None):
        parameters = StdioServerParameters(
            command=sys.executable, args=["-m", "videolingo_mcp"], cwd=PROJECT,
            env={"VIDEOLINGO_DATA_DIR": str(self.data), **(env or {})},
        )
        with (self.root / "mcp-stderr.log").open("a", encoding="utf-8") as log:
            async with Client(stdio_client(parameters, errlog=log)) as client:
                yield client

    async def call(self, client, name: str, **arguments) -> dict:
        result = await client.call_tool(name, arguments, read_timeout_seconds=15)
        self.assertFalse(result.is_error, result.content)
        self.assertIsInstance(result.structured_content, dict)
        return result.structured_content

    async def ready_job(self, client) -> str:
        job = await self.call(client, "import_transcript", video_path=str(self.video), language="en",
                              segments=[{"id": 1, "start": 0.1, "end": 1.9, "text": "A reviewed source line."}])
        await self.call(client, "save_translations", job_id=job["job_id"],
                        translations=[{"id": 1, "text": "经检查的原文。"}])
        return job["job_id"]

    async def wait_terminal(self, client, job_id: str, timeout: float = 20) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = await self.call(client, "get_job", job_id=job_id)
            if job["status"] not in ACTIVE:
                self.record("terminal", job)
                return job
            await asyncio.sleep(0.1)
        self.fail(f"job did not become terminal: {job}")

    async def wait_marker(self, path: Path) -> dict:
        for _ in range(100):
            if path.exists():
                try:
                    value = json.loads(path.read_text())
                except json.JSONDecodeError:
                    pass
                else:
                    self.groups.add(value["group"])
                    return value
            await asyncio.sleep(0.05)
        self.fail(f"processing fixture was not reached: {path}")

    def wrapper(self, body: str) -> tuple[dict[str, str], Path]:
        directory = self.root / "injected-tools"
        directory.mkdir()
        marker = self.root / "processing.json"
        program = directory / "ffmpeg"
        prefix = (
            f"#!{sys.executable}\n"
            "import json, os, shutil, signal, sys, time\n"
            "from pathlib import Path\n"
            "if '-i' in sys.argv:\n"
        )
        marked = (
            f"Path({str(marker)!r}).write_text(json.dumps({{'pid': os.getpid(), 'parent': os.getppid(), 'group': os.getpgrp()}}))\n"
            + body
        )
        text = prefix + "".join("    " + line + "\n" for line in marked.splitlines())
        text += f"os.execv({self.real_ffmpeg!r}, [{self.real_ffmpeg!r}, *sys.argv[1:]])\n"
        program.write_text(text, encoding="utf-8")
        program.chmod(0o700)
        return {"PATH": str(directory) + os.pathsep + os.environ["PATH"]}, marker

    @staticmethod
    def process_running(pid: int) -> bool:
        probe = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, timeout=5)
        status = probe.stdout.strip()
        return bool(status) and not status.startswith("Z")

    async def test_bad_media_is_rejected_before_creating_jobs(self) -> None:
        silent = self.root / "no-audio.mp4"
        self.make_video(silent, audio=False)
        broken = self.root / "corrupt.mp4"
        broken.write_bytes(b"not a video container")
        async with self.connect() as client:
            for path, explanation in [(silent, "audio"), (broken, "ffprobe")]:
                result = await client.call_tool("start_transcription", {"video_path": str(path)})
                self.assertTrue(result.is_error)
                message = " ".join(getattr(block, "text", "") for block in result.content)
                self.record(path.stem + "-error", message)
                self.assertIn(explanation, message)
            self.assertEqual((await self.call(client, "list_jobs"))["total"], 0)
            self.assertEqual((await self.call(client, "health"))["server"], "videolingo-mcp")

    async def test_empty_model_cache_fails_offline_with_actionable_error(self) -> None:
        async with self.connect({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}) as client:
            started = await self.call(client, "start_transcription", video_path=str(self.video),
                                      model="tiny", language="en", allow_download=False)
            failed = await self.wait_terminal(client, started["job_id"])
            self.assertEqual(failed["status"], "failed", failed)
            self.assertRegex(failed["error"], r"缓存|cached|local_files_only")
            self.assertFalse(failed["artifacts"])
            self.assertFalse(list((self.data / "models").glob("**/model.bin")))

    async def test_source_change_during_processing_does_not_publish_mismatched_video(self) -> None:
        replacement = self.root / "replacement.mp4"
        self.make_video(replacement, color="red")
        env, marker = self.wrapper(f"shutil.copyfile({str(replacement)!r}, {str(self.video)!r})")
        async with self.connect(env) as client:
            job_id = await self.ready_job(client)
            await self.call(client, "export_video", job_id=job_id)
            await self.wait_marker(marker)
            finished = await self.wait_terminal(client, job_id)
            self.assertEqual(finished["status"], "failed", finished)
            self.assertIn("changed", finished["error"].lower())
            self.assertFalse(any(artifact["type"] == "video" for artifact in finished["artifacts"]))

    async def test_cancellation_escalates_for_a_processing_descendant_ignoring_term(self) -> None:
        env, marker = self.wrapper("signal.signal(signal.SIGTERM, signal.SIG_IGN)\ntime.sleep(30)")
        async with self.connect(env) as client:
            job_id = await self.ready_job(client)
            await self.call(client, "export_video", job_id=job_id)
            process = await self.wait_marker(marker)
            await self.call(client, "cancel_job", job_id=job_id)
            cancelled = await self.wait_terminal(client, job_id, timeout=10)
            self.assertEqual(cancelled["status"], "cancelled", cancelled)
            for _ in range(50):
                if not self.process_running(process["pid"]):
                    break
                await asyncio.sleep(0.1)
            self.record("descendant-alive-after-cancel", self.process_running(process["pid"]))
            self.assertFalse(self.process_running(process["pid"]), "cancelled job left a running media descendant")

    async def test_same_size_and_mtime_source_replacement_is_detected(self) -> None:
        replacement = self.root / "replacement-with-preserved-time.mp4"
        self.make_video(replacement, color="red")
        # MP4 permits trailing data; align sizes without changing either clip's
        # decodability, then preserve mtime as a file-copy tool may do.
        length = max(self.video.stat().st_size, replacement.stat().st_size)
        for path in [self.video, replacement]:
            with path.open("ab") as handle:
                handle.write(b"\0" * (length - path.stat().st_size))
        original = self.video.stat()
        os.utime(replacement, ns=(original.st_atime_ns, original.st_mtime_ns))
        self.assertEqual(replacement.stat().st_size, original.st_size)
        self.assertEqual(replacement.stat().st_mtime_ns, original.st_mtime_ns)
        env, marker = self.wrapper(f"os.replace({str(replacement)!r}, {str(self.video)!r})")
        async with self.connect(env) as client:
            job_id = await self.ready_job(client)
            await self.call(client, "export_video", job_id=job_id)
            await self.wait_marker(marker)
            failed = await self.wait_terminal(client, job_id)
            self.assertEqual(failed["status"], "failed", failed)
            self.assertIn("changed", failed["error"].lower())
            self.assertFalse(any(artifact["type"] == "video" for artifact in failed["artifacts"]))

    async def test_execute_crash_cleans_unpublished_temporary_video(self) -> None:
        env, marker = self.wrapper(
            "Path(sys.argv[-1]).write_bytes(b'unpublished partial MP4')\n"
            "os.kill(os.getppid(), signal.SIGKILL)\n"
            "raise SystemExit(73)"
        )
        async with self.connect(env) as client:
            job_id = await self.ready_job(client)
            await self.call(client, "export_video", job_id=job_id)
            await self.wait_marker(marker)
            failed = await self.wait_terminal(client, job_id)
            self.assertEqual(failed["status"], "failed", failed)
            self.assertTrue(failed["error"])
            remaining = [str(path) for path in (self.data / "jobs" / job_id).glob(".videolingo-export-*")]
            self.record("remaining-temporary-directories", remaining)
            self.assertFalse(remaining, "failed processing left an unpublished temporary video")
            self.assertFalse(any(artifact["type"] == "video" for artifact in failed["artifacts"]))

    async def test_supervisor_term_survives_reconnect_and_allows_retry(self) -> None:
        env, marker = self.wrapper("time.sleep(30)")
        async with self.connect(env) as client:
            job_id = await self.ready_job(client)
            started = await self.call(client, "export_video", job_id=job_id)
            await self.wait_marker(marker)
            os.kill(started["worker_pid"], signal.SIGTERM)
        async with self.connect() as client:
            interrupted = await self.wait_terminal(client, job_id)
            self.record("interrupted-before-retry", interrupted)
            self.assertEqual(interrupted["status"], "interrupted", interrupted)
            self.assertEqual(interrupted["translated_segments"], 1)
            self.assertFalse(list((self.data / "jobs" / job_id).glob(".videolingo-export-*")))
            await self.call(client, "export_video", job_id=job_id)
            finished = await self.wait_terminal(client, job_id)
            self.assertEqual(finished["status"], "completed", finished)
            self.assertEqual(sum(artifact["type"] == "video" for artifact in finished["artifacts"]), 1)


if __name__ == "__main__":
    unittest.main()

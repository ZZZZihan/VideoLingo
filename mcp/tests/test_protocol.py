"""Exercise the public MCP contract through a real stdio subprocess."""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import asynccontextmanager
from pathlib import Path

from mcp import Client, StdioServerParameters


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_TOOLS = {
    "health",
    "start_transcription",
    "import_transcript",
    "get_job",
    "list_jobs",
    "get_transcript",
    "save_translations",
    "export_subtitles",
    "export_video",
    "cancel_job",
}


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is required")
class StdioProtocolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="videolingo-protocol-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.data_dir = self.root / "任务 数据"
        self.video = self.root / "原始 视频.mp4"
        subprocess.run(
            [
                "ffmpeg", "-v", "error", "-nostdin", "-y",
                "-f", "lavfi", "-i", "color=c=blue:s=160x90:r=10:d=2",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
                "-c:v", "mpeg4", "-c:a", "aac", "-shortest", str(self.video),
            ],
            check=True,
            capture_output=True,
            timeout=20,
        )

    @asynccontextmanager
    async def connect(self, extra_env: dict[str, str] | None = None):
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "videolingo_mcp"],
            cwd=PROJECT_ROOT,
            env={"VIDEOLINGO_DATA_DIR": str(self.data_dir), **(extra_env or {})},
        )
        async with Client(params) as client:
            yield client

    async def call(self, client, name: str, arguments: dict | None = None) -> dict:
        result = await client.call_tool(name, arguments or {}, read_timeout_seconds=15)
        self.assertFalse(result.is_error, result.content)
        self.assertIsInstance(result.structured_content, dict)
        return result.structured_content

    async def test_registered_tools_have_schemas_and_structured_results(self) -> None:
        async with self.connect() as client:
            listing = await client.list_tools()
            tools = {tool.name: tool for tool in listing.tools}
            self.assertTrue(REQUIRED_TOOLS <= tools.keys())
            for name in REQUIRED_TOOLS:
                with self.subTest(tool=name):
                    self.assertEqual(tools[name].input_schema["type"], "object")
                    self.assertIsNotNone(tools[name].output_schema)
            health = await self.call(client, "health")
            self.assertTrue(health)

    async def test_translation_roundtrip_survives_a_new_connection(self) -> None:
        segments = [
            {"id": 1, "start": 0.1, "end": 0.9, "text": "Hello world."},
            {"id": 2, "start": 1.0, "end": 1.9, "text": "Welcome back."},
        ]
        original_bytes = self.video.read_bytes()
        async with self.connect() as client:
            job = await self.call(client, "import_transcript", {
                "video_path": str(self.video), "segments": segments, "language": "en",
            })
            job_id = job["job_id"]
            self.assertEqual(job["status"], "awaiting_translation")
            first = await self.call(client, "get_transcript", {
                "job_id": job_id, "offset": 0, "limit": 1,
            })
            self.assertEqual(first["total"], 2)
            self.assertEqual(first["next_offset"], 1)
            self.assertEqual(first["segments"][0]["text"], "Hello world.")
            partial = await self.call(client, "save_translations", {
                "job_id": job_id, "translations": [{"id": 1, "text": "你好，世界。"}],
            })
            self.assertEqual(partial["status"], "awaiting_translation")
            result = await client.call_tool("export_subtitles", {"job_id": job_id})
            self.assertTrue(result.is_error, result)
            self.assertIn("Translate all segments", " ".join(getattr(item, "text", "") for item in result.content))
            complete = await self.call(client, "save_translations", {
                "job_id": job_id, "translations": [{"id": 2, "text": "欢迎回来。"}],
            })
            self.assertEqual(complete["status"], "ready")
            exported = await self.call(client, "export_subtitles", {
                "job_id": job_id, "format": "srt", "bilingual": True,
            })
            subtitle_path = Path(exported["path"])
            self.assertTrue(subtitle_path.is_relative_to(self.data_dir))
            text = subtitle_path.read_text(encoding="utf-8")
            for expected in ["你好，世界。", "Hello world.", "欢迎回来。", "Welcome back."]:
                self.assertIn(expected, text)
            self.assertIn("00:00:00,100 --> 00:00:00,900", text)

        async with self.connect() as client:
            restored = await self.call(client, "get_job", {"job_id": job_id})
            self.assertEqual(restored["job_id"], job_id)
            self.assertEqual(restored["translated_segments"], 2)
            last = await self.call(client, "get_transcript", {
                "job_id": job_id, "offset": 1, "limit": 1,
            })
            self.assertIsNone(last["next_offset"])
            self.assertEqual(last["segments"][0]["translation"], "欢迎回来。")
        self.assertEqual(self.video.read_bytes(), original_bytes)

    async def test_bad_arguments_are_tool_errors_and_server_remains_usable(self) -> None:
        async with self.connect() as client:
            for arguments in [{"job_id": "missing"}, {"job_id": "../outside"}]:
                with self.subTest(arguments=arguments):
                    result = await client.call_tool("get_job", arguments)
                    self.assertTrue(result.is_error)
                    self.assertTrue(result.content)
                    self.assertIn("Invalid job_id", " ".join(getattr(item, "text", "") for item in result.content))
            result = await client.call_tool("import_transcript", {
                "video_path": str(self.video), "segments": [], "language": "en",
            })
            self.assertTrue(result.is_error)
            health = await self.call(client, "health")
            self.assertTrue(health)

    async def test_detached_export_finishes_after_stdio_disconnect(self) -> None:
        real_ffmpeg = shutil.which("ffmpeg")
        wrapper_dir = self.root / "delayed-tools"
        wrapper_dir.mkdir()
        marker = self.root / "encoding-started"
        wrapper = wrapper_dir / "ffmpeg"
        wrapper.write_text(
            f"#!{sys.executable}\n"
            "import os, sys, time\n"
            "from pathlib import Path\n"
            "if '-i' in sys.argv:\n"
            f"    Path({str(marker)!r}).touch()\n"
            "    time.sleep(2)\n"
            f"os.execv({real_ffmpeg!r}, [{real_ffmpeg!r}, *sys.argv[1:]])\n",
            encoding="utf-8",
        )
        wrapper.chmod(0o700)
        env = {"PATH": str(wrapper_dir) + os.pathsep + os.environ["PATH"]}
        async with self.connect(env) as client:
            imported = await self.call(client, "import_transcript", {
                "video_path": str(self.video),
                "segments": [{"id": 1, "start": 0.1, "end": 1.9, "text": "Test clip."}],
            })
            job_id = imported["job_id"]
            await self.call(client, "save_translations", {
                "job_id": job_id, "translations": [{"id": 1, "text": "测试视频。"}],
            })
            await self.call(client, "export_video", {"job_id": job_id})
            for _ in range(100):
                if marker.exists():
                    break
                await asyncio.sleep(0.05)
            self.assertTrue(marker.exists())
            running = await self.call(client, "get_job", {"job_id": job_id})
            self.assertEqual(running["status"], "exporting")

        async with self.connect() as client:
            for _ in range(100):
                finished = await self.call(client, "get_job", {"job_id": job_id})
                if finished["status"] not in {"queued", "exporting", "cancelling"}:
                    break
                await asyncio.sleep(0.1)
            self.assertEqual(finished["status"], "completed", finished)
            self.assertTrue(any(Path(a["path"]).is_file() and a["type"] == "video" for a in finished["artifacts"]))


if __name__ == "__main__":
    unittest.main()

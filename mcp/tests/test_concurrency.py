"""Check independent MCP clients sharing the same persistent workflow directory."""

import asyncio
from contextlib import asynccontextmanager
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from mcp import Client, StdioServerParameters


PROJECT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is required")
class ConcurrentClientsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="videolingo-concurrent-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.video = self.root / "source.mp4"
        self.data = self.root / "data"
        subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-f", "lavfi", "-i",
                        "color=c=blue:s=160x90:r=10:d=2", "-c:v", "mpeg4", str(self.video)],
                       check=True, capture_output=True, timeout=15)

    @asynccontextmanager
    async def connect(self, extra_env=None):
        async with Client(StdioServerParameters(command=sys.executable, args=["-m", "videolingo_mcp"],
                         cwd=PROJECT, env={"VIDEOLINGO_DATA_DIR": str(self.data), **(extra_env or {})})) as client:
            yield client

    async def call(self, client, name, args):
        result = await client.call_tool(name, args)
        self.assertFalse(result.is_error, result.content)
        return result.structured_content

    async def import_job(self, client):
        return await self.call(client, "import_transcript", {"video_path": str(self.video), "segments": [
            {"id": i + 1, "start": i * .4, "end": (i + 1) * .4, "text": f"Line {i + 1}"}
            for i in range(4)]})

    async def test_simultaneous_translation_batches_do_not_lose_updates(self):
        async with self.connect() as first, self.connect() as second:
            jid = (await self.import_job(first))["job_id"]
            await asyncio.gather(
                self.call(first, "save_translations", {"job_id": jid, "translations": [
                    {"id": 1, "text": "第一句"}, {"id": 3, "text": "第三句"}]}),
                self.call(second, "save_translations", {"job_id": jid, "translations": [
                    {"id": 2, "text": "第二句"}, {"id": 4, "text": "第四句"}]}),
            )
            job = await self.call(first, "get_job", {"job_id": jid})
            self.assertEqual((job["status"], job["translated_segments"], job["revision"]), ("ready", 4, 2))
            exported = await self.call(second, "export_subtitles", {"job_id": jid})
            text = Path(exported["path"]).read_text()
            for expected in ["第一句", "第二句", "第三句", "第四句"]:
                self.assertIn(expected, text)

    async def test_media_limit_is_shared_across_server_processes(self):
        wrappers = self.root / "tools"
        wrappers.mkdir()
        wrapper = wrappers / "ffmpeg"
        real_ffmpeg = shutil.which("ffmpeg")
        wrapper.write_text(f"#!{sys.executable}\nimport os,sys,time\n"
                           "if '-i' in sys.argv: time.sleep(30)\n"
                           f"os.execv({real_ffmpeg!r}, [{real_ffmpeg!r}, *sys.argv[1:]])\n")
        wrapper.chmod(0o700)
        env = {"PATH": str(wrappers) + os.pathsep + os.environ["PATH"]}
        async with self.connect(env) as first, self.connect(env) as second:
            ids = []
            try:
                for _ in range(3):
                    jid = (await self.import_job(first))["job_id"]
                    ids.append(jid)
                    await self.call(first, "save_translations", {"job_id": jid, "translations": [
                        {"id": i + 1, "text": f"译文 {i + 1}"} for i in range(4)]})
                await asyncio.gather(
                    self.call(first, "export_video", {"job_id": ids[0]}),
                    self.call(second, "export_video", {"job_id": ids[1]}),
                )
                result = await second.call_tool("export_video", {"job_id": ids[2]})
                self.assertTrue(result.is_error)
                self.assertIn("Two media tasks", " ".join(getattr(c, "text", "") for c in result.content))
                third = await self.call(first, "get_job", {"job_id": ids[2]})
                self.assertEqual(third["status"], "ready")
                self.assertFalse(third["artifacts"])
                edits = await second.call_tool("save_translations", {"job_id": ids[0], "translations": [
                    {"id": 1, "text": "不能在导出期间更改"}]})
                self.assertTrue(edits.is_error)
            finally:
                for jid in ids:
                    await self.call(first, "cancel_job", {"job_id": jid})
                for _ in range(100):
                    jobs = (await self.call(first, "list_jobs", {}))["jobs"]
                    if all(j["status"] not in {"queued", "exporting", "cancelling"} for j in jobs):
                        break
                    await asyncio.sleep(.1)
                self.assertTrue(all(j["status"] in {"cancelled", "ready"} for j in jobs), jobs)

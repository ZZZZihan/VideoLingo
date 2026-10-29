"""Media regression checks using real FFmpeg round trips and retained fixtures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import unittest
from uuid import uuid4

from videolingo_mcp.jobs import Service
from videolingo_mcp.media import MediaError, export_video, media_capabilities
from videolingo_mcp.subtitles import render_srt, render_vtt, validate_segments


PROJECT = Path(__file__).resolve().parents[1]
FIXTURES = PROJECT / ".scratch" / "qa-20261029" / "agent-media"


def run_tool(arguments: list[str]) -> str:
    return subprocess.run(
        arguments, check=True, capture_output=True, text=True,
        encoding="utf-8", errors="strict", timeout=30,
    ).stdout


def probe(path: Path) -> dict:
    return json.loads(run_tool([
        "ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path),
    ]))


def seconds(timestamp: str) -> float:
    hours, minutes, remainder = timestamp.replace(",", ".").split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(remainder)


def extract_cues(video: Path) -> tuple[str, list[dict]]:
    content = run_tool([
        "ffmpeg", "-v", "error", "-nostdin", "-i", str(video),
        "-map", "0:s:0", "-f", "srt", "-",
    ])
    cues = []
    for block in re.split(r"\n\s*\n", content.strip()):
        lines = block.splitlines()
        if len(lines) < 3:
            continue
        start, end = lines[1].split(" --> ")
        cues.append({"start": seconds(start), "end": seconds(end), "text": "\n".join(lines[2:])})
    return content, cues


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is required")
class MediaEdgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = FIXTURES / f"suite-{uuid4().hex[:12]}"
        cls.root.mkdir(parents=True)
        cls.multi_audio = cls.root / "two-audio-tracks.mp4"
        cls.silent_odd = cls.root / "silent-odd-dimensions.mp4"
        run_tool([
            "ffmpeg", "-v", "error", "-nostdin", "-f", "lavfi", "-i",
            "color=c=blue:s=160x90:r=10:d=5",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=5",
            "-f", "lavfi", "-i", "sine=frequency=880:duration=5",
            "-map", "0:v", "-map", "1:a", "-map", "2:a",
            "-c:v", "mpeg4", "-c:a", "aac",
            "-metadata:s:a:0", "language=eng", "-metadata:s:a:1", "language=zho",
            str(cls.multi_audio),
        ])
        run_tool([
            "ffmpeg", "-v", "error", "-nostdin", "-f", "lavfi", "-i",
            "testsrc=size=161x91:rate=10:duration=5",
            "-c:v", "libx264", "-pix_fmt", "yuv444p", str(cls.silent_odd),
        ])

    def setUp(self) -> None:
        self.directory = self.root / self.id().rsplit(".", 1)[-1]
        self.directory.mkdir()

    def export(self, segments: list[dict], translations: dict | None = None,
               *, bilingual: bool = False, format: str = "srt",
               name: str = "result", source: Path | None = None) -> Path:
        subtitle = self.directory / f"{name} 中文 : '符号.{format}"
        renderer = render_srt if format == "srt" else render_vtt
        subtitle.write_text(renderer(segments, translations, bilingual), encoding="utf-8")
        output = self.directory / f"{name}.mp4"
        export_video(source or self.multi_audio, subtitle, output)
        return output

    def test_bilingual_special_symbols_and_millisecond_timing_round_trip(self) -> None:
        segments = [
            {"id": 1, "start": 0.123, "end": 1.234,
             "text": "Use <b>literal</b> & C:\\clip {\\an8} 🙂"},
            {"id": 2, "start": 2.345, "end": 3.456, "text": "Second line\nLine three"},
        ]
        translations = {1: "中文 ＜不是标记＞ & \"引号\" 🙂", 2: "第二句\n第三行"}
        output = self.export(segments, translations, bilingual=True)
        content, cues = extract_cues(output)
        (self.directory / "extracted.srt").write_text(content, encoding="utf-8")
        self.assertEqual(len(cues), 2)
        self.assertEqual(cues[0]["text"],
                         '中文 ＜不是标记＞ & "引号" 🙂\nUse ＜b＞literal＜/b＞ & C:＼clip ｛＼an8｝ 🙂')
        self.assertEqual(cues[1]["text"], "第二句\n第三行\nSecond line\nLine three")
        for expected, actual in zip(segments, cues):
            self.assertAlmostEqual(actual["start"], expected["start"], delta=0.001)
            self.assertAlmostEqual(actual["end"], expected["end"], delta=0.001)
        subtitle_stream = next(s for s in probe(output)["streams"] if s["codec_type"] == "subtitle")
        self.assertEqual(subtitle_stream["tags"]["language"], "zho")
        self.assertEqual(subtitle_stream["disposition"]["default"], 1)

    def test_vtt_markup_is_literal_after_video_round_trip(self) -> None:
        text = '<b>literal</b> & &amp; {\\an8}\\N {\\b1}visible{\\b0} C:\\clip 🙂'
        expected = '<b>literal</b> & &amp; ｛＼an8｝＼N ｛＼b1｝visible｛＼b0｝ C:＼clip 🙂'
        subtitle = self.directory / "literal 中文 : 'captions.vtt"
        subtitle.write_text(render_vtt([{"id": 1, "start": 0.25, "end": 2.75, "text": text}]), encoding="utf-8")
        original = subtitle.read_bytes()
        # The standalone VTT retains the exact literal ASS punctuation and
        # standard HTML entities. Only a temporary video-input copy may apply
        # the documented full-width display punctuation normalization.
        self.assertIn(b"&lt;b&gt;literal&lt;/b&gt; &amp; &amp;amp;", original)
        self.assertIn(b"{\\an8}\\N {\\b1}visible{\\b0}", original)
        self.assertNotIn("｛".encode("utf-8"), original)
        output = self.directory / "literal.mp4"
        export_video(self.multi_audio, subtitle, output)
        self.assertEqual(subtitle.read_bytes(), original, "Video export modified the original standalone VTT")
        content, cues = extract_cues(output)
        (self.directory / "extracted.srt").write_text(content, encoding="utf-8")
        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0]["text"], expected)
        self.assertEqual((cues[0]["start"], cues[0]["end"]), (0.25, 2.75))
        ass = run_tool(["ffmpeg", "-v", "error", "-i", str(output), "-map", "0:s:0", "-f", "ass", "-"])
        self.assertIn(expected, ass)
        self.assertNotIn("{\\b1}", ass)
        self.assertNotIn("{\\b0}", ass)
        self.assertNotIn("{\\an8}", ass)
        self.assertNotIn("\\N", ass, "A literal backslash-N became an ASS line break")

    def test_vtt_burn_in_matches_literal_display_and_keeps_original_file(self) -> None:
        if not media_capabilities()["burn_in"]:
            self.skipTest("This host lacks the real subtitles/libass filter; literal VTT burn-in must run on capable CI")
        text = '<b>QA</b> {\\an8}\\N {\\b1}LITERAL{\\b0}'
        expected = '<b>QA</b> ｛＼an8｝＼N ｛＼b1｝LITERAL｛＼b0｝'
        videos = []
        for name, payload in [("original", text), ("reference", expected)]:
            subtitle = self.directory / f"{name} 中文 : 'captions.vtt"
            subtitle.write_text(render_vtt([
                {"id": 1, "start": 0.5, "end": 2.5, "text": payload},
            ]), encoding="utf-8")
            original = subtitle.read_bytes()
            output = self.directory / f"{name}.mp4"
            export_video(self.multi_audio, subtitle, output, mode="burn_in")
            self.assertEqual(subtitle.read_bytes(), original, "Burn-in modified its source VTT")
            self.assertFalse(any(stream["codec_type"] == "subtitle" for stream in probe(output)["streams"]))
            videos.append(output)

        def rgb_frame(path: Path, at: float) -> bytes:
            frame = subprocess.run([
                "ffmpeg", "-v", "error", "-nostdin", "-ss", str(at), "-i", str(path),
                "-map", "0:v:0", "-frames:v", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "-",
            ], check=True, capture_output=True, timeout=15).stdout
            self.assertEqual(len(frame), 160 * 90 * 3)
            return frame

        # A real rendered reference catches deleted brace blocks, accidental
        # top alignment, bold markup, and a backslash-N treated as a line break.
        # Merely finding any changed pixels would miss those content losses.
        for at in [0.1, 1.0, 3.0]:
            with self.subTest(at=at):
                self.assertEqual(rgb_frame(videos[0], at), rgb_frame(videos[1], at))
        source_frame = rgb_frame(self.multi_audio, 1.0)
        rendered = rgb_frame(videos[0], 1.0)
        changed_pixels = sum(
            max(abs(source_frame[index + channel] - rendered[index + channel]) for channel in range(3)) >= 40
            for index in range(0, len(source_frame), 3)
        )
        self.assertGreater(changed_pixels, 20, "The literal reference comparison passed without any visible subtitle")

    def test_long_bilingual_payload_keeps_every_character(self) -> None:
        source, translated = "原" * 8000, "译" * 8000
        output = self.export([{"id": 1, "start": 0.1, "end": 3.9, "text": source}],
                             {1: translated}, bilingual=True)
        _, cues = extract_cues(output)
        self.assertEqual(len(cues), 1)
        # FFmpeg's SRT encoder wraps very long lines. Those extra line breaks
        # are presentation changes; every subtitle character must survive.
        self.assertEqual(cues[0]["text"].replace("\n", ""), translated + source)
        self.assertIn("\n", cues[0]["text"])

    def test_oversized_payload_fails_without_publishing_corrupted_video(self) -> None:
        subtitle = self.directory / "oversized.srt"
        subtitle.write_text(render_srt([{"id": 1, "start": 0.1, "end": 3.9, "text": "中" * 30000}]),
                            encoding="utf-8")
        output = self.directory / "must-not-exist.mp4"
        with self.assertRaises((MediaError, ValueError)):
            export_video(self.multi_audio, subtitle, output)
        self.assertFalse(output.exists())
        self.assertFalse(list(self.directory.glob(".videolingo-export-*")))

    def test_all_audio_tracks_and_full_video_duration_are_preserved(self) -> None:
        output = self.export([{"id": 1, "start": 0.1, "end": 0.2, "text": "短字幕"}])
        before, after = probe(self.multi_audio), probe(output)
        audio_before = [s for s in before["streams"] if s["codec_type"] == "audio"]
        audio_after = [s for s in after["streams"] if s["codec_type"] == "audio"]
        self.assertEqual(len(audio_after), len(audio_before), "Export silently dropped an original audio track")
        self.assertEqual([s["tags"].get("language") for s in audio_after], ["eng", "zho"])
        self.assertAlmostEqual(float(after["format"]["duration"]), float(before["format"]["duration"]), delta=0.1)

    def test_overlapping_cues_are_rejected_before_creating_a_job(self) -> None:
        cases = [
            [{"id": 1, "start": 0.1, "end": 2.0, "text": "First"},
             {"id": 2, "start": 1.0, "end": 3.0, "text": "Overlap"}],
            [{"id": 1, "start": 0.1, "end": 2.0, "text": "Same start one"},
             {"id": 2, "start": 0.1, "end": 3.0, "text": "Same start two"}],
        ]
        service = Service(self.directory / "job-data")
        for index, segments in enumerate(cases):
            with self.subTest(case=index):
                for operation in [validate_segments, render_srt, render_vtt]:
                    with self.assertRaisesRegex(ValueError, r"overlap|重叠"):
                        operation(segments)
                with self.assertRaisesRegex(ValueError, r"overlap|重叠"):
                    service.import_transcript(str(self.multi_audio), segments, language="en")
                self.assertEqual(service.list_jobs()["total"], 0)
        adjacent = [{"id": 1, "start": 0.1, "end": 2.0, "text": "First"},
                    {"id": 2, "start": 2.0, "end": 3.0, "text": "Adjacent"}]
        output = self.export(adjacent, name="adjacent")
        _, actual = extract_cues(output)
        self.assertEqual([cue["text"] for cue in actual], ["First", "Adjacent"])
        self.assertEqual((actual[0]["end"], actual[1]["start"]), (2.0, 2.0))

    def test_silent_odd_dimensions_and_short_cue_do_not_truncate_video(self) -> None:
        output = self.export([{"id": 1, "start": 0.1, "end": 0.2, "text": "无音轨也能导出"}],
                             source=self.silent_odd)
        info = probe(output)
        self.assertFalse(any(s["codec_type"] == "audio" for s in info["streams"]))
        video = next(s for s in info["streams"] if s["codec_type"] == "video")
        self.assertEqual((video["width"], video["height"]), (162, 92))
        self.assertAlmostEqual(float(info["format"]["duration"]), 5.0, delta=0.1)
        _, cues = extract_cues(output)
        self.assertEqual(cues[0]["text"], "无音轨也能导出")

    def test_existing_file_directory_and_symlink_cannot_be_overwritten(self) -> None:
        subtitle = self.directory / "captions.srt"
        subtitle.write_text(render_srt([{"id": 1, "start": 0, "end": 1, "text": "你好"}]), encoding="utf-8")
        source_hash = hashlib.sha256(self.multi_audio.read_bytes()).hexdigest()
        marker = self.directory / "original.mp4"
        marker.write_bytes(b"existing user content")
        folder = self.directory / "folder.mp4"
        folder.mkdir()
        link = self.directory / "link.mp4"
        link.symlink_to(marker)
        for target in [marker, folder, link]:
            with self.subTest(target=target.name), self.assertRaisesRegex(MediaError, "已存在|覆盖"):
                export_video(self.multi_audio, subtitle, target)
        self.assertEqual(marker.read_bytes(), b"existing user content")
        self.assertTrue(folder.is_dir())
        self.assertTrue(link.is_symlink())
        self.assertEqual(hashlib.sha256(self.multi_audio.read_bytes()).hexdigest(), source_hash)

    def test_missing_burn_in_support_is_actionable_and_creates_no_output(self) -> None:
        if media_capabilities()["burn_in"]:
            self.skipTest("This host has libass; the unavailable-capability branch does not apply")
        subtitle = self.directory / "captions.srt"
        subtitle.write_text(render_srt([{"id": 1, "start": 0, "end": 1, "text": "你好"}]), encoding="utf-8")
        output = self.directory / "burn.mp4"
        with self.assertRaisesRegex(MediaError, r"libass.*soft|soft.*libass"):
            export_video(self.multi_audio, subtitle, output, mode="burn_in")
        self.assertFalse(output.exists())

    def test_burn_in_changes_picture_during_cue_and_preserves_full_video(self) -> None:
        if not media_capabilities()["burn_in"]:
            self.skipTest("This host lacks the real subtitles/libass filter; successful burn-in requires CI or another capable host")
        subtitle = self.directory / "burn-in 中文 : 'captions.srt"
        subtitle.write_text(render_srt([
            {"id": 1, "start": 0.5, "end": 2.5, "text": "Subtitle QA 中文字幕"},
        ]), encoding="utf-8")
        output = self.directory / "burn-in.mp4"
        export_video(self.multi_audio, subtitle, output, mode="burn_in")
        info = probe(output)
        self.assertFalse(any(stream["codec_type"] == "subtitle" for stream in info["streams"]))
        self.assertEqual(sum(stream["codec_type"] == "video" for stream in info["streams"]), 1)
        self.assertAlmostEqual(float(info["format"]["duration"]), 5.0, delta=0.1)
        # Decode the complete result, rather than inferring readability from
        # its stream metadata or from a single frame.
        run_tool(["ffmpeg", "-v", "error", "-nostdin", "-i", str(output), "-f", "null", "-"])

        def frame(path: Path, at: float) -> bytes:
            result = subprocess.run([
                "ffmpeg", "-v", "error", "-nostdin", "-ss", str(at), "-i", str(path),
                "-map", "0:v:0", "-frames:v", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "-",
            ], check=True, capture_output=True, timeout=15)
            self.assertEqual(len(result.stdout), 160 * 90 * 3)
            return result.stdout

        def visibly_changed_pixels(original: bytes, rendered: bytes) -> int:
            # Ignore small lossy-encoding changes. Real white subtitle glyphs
            # differ strongly from the generated clip's uniform blue picture.
            return sum(
                max(abs(original[index + channel] - rendered[index + channel]) for channel in range(3)) >= 40
                for index in range(0, len(original), 3)
            )

        during = visibly_changed_pixels(frame(self.multi_audio, 1.0), frame(output, 1.0))
        before = visibly_changed_pixels(frame(self.multi_audio, 0.1), frame(output, 0.1))
        after = visibly_changed_pixels(frame(self.multi_audio, 3.0), frame(output, 3.0))
        self.assertGreater(during, 20, "No visible subtitle glyphs were burned into a frame inside the cue")
        self.assertLessEqual(before, 5, "The picture changed before the subtitle should be visible")
        self.assertLessEqual(after, 5, "Subtitle glyphs remain after the cue should have ended")

    def test_unencodable_unicode_is_rejected_before_saving(self) -> None:
        with self.assertRaises(ValueError):
            validate_segments([{"id": 1, "start": 0, "end": 1, "text": "bad\ud800"}])


if __name__ == "__main__":
    unittest.main()

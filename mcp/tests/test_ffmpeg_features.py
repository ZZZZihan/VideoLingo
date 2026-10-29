"""FFmpeg filter-list format regression without local tool/model dependencies."""

from __future__ import annotations

import unittest
from unittest.mock import call, patch

from videolingo_mcp import media


# Verbatim selected rows from /opt/homebrew/opt/ffmpeg-full/bin/ffmpeg 8.1,
# captured on macOS on 2026-10-29. Its legend still shows T.. and .S., although
# actual filter rows now contain two flags.
FFMPEG_8_FILTERS = """Filters:
  T.. = Timeline support
  .S. = Slice threading
  A = Audio input/output
  V = Video input/output
  N = Dynamic number and/or type of input/output
  | = Source or sink filter
  ------
 .. anull             A->A       Pass the source unchanged to the output.
 T. volume            A->A       Change input volume.
 .. ass               V->V       Render ASS subtitles onto input video using the libass library.
 T. drawtext          V->V       Draw text on top of video frames using libfreetype library.
 .. null              V->V       Pass the source unchanged to the output.
 .. scale             V->V       Scale the input video size and/or convert the image format.
 .. subtitles         V->V       Render text subtitles onto input video using the libass library.
 .. trim              V->V       Pick one continuous section from the input, drop the rest.
 .. showvolume        A->V       Convert input audio volume to video output.
"""

# Minimal compatibility fixture for older FFmpeg's three-flag listing format.
# This is a format fixture, not a claim that an older binary ran on this host.
LEGACY_FILTERS = """Filters:
  T.. = Timeline support
  .S. = Slice threading
  ..C = Command support
  A = Audio input/output
  V = Video input/output
  N = Dynamic number and/or type of input/output
  | = Source or sink filter
  ------
 ... ass               V->V       Render ASS subtitles onto input video using the libass library.
 T.C drawtext          V->V       Draw text on top of video frames using libfreetype library.
 ..C scale             V->V       Scale the input video size and/or convert the image format.
 ... subtitles         V->V       Render text subtitles onto input video using the libass library.
 T.C volume            A->A       Change input volume.
 ... anullsrc          |->A       Null audio source, return empty audio frames.
 ... nullsink          V->|       Do absolutely nothing with the input video.
 ... concat            N->N       Concatenate audio and video streams.
"""

# Encoder rows were captured from the same FFmpeg 8.1 binary above.
ENCODERS = """Encoders:
 V..... = Video
 A..... = Audio
 S..... = Subtitle
 .F.... = Frame-level multithreading
 ..S... = Slice-level multithreading
 ...X.. = Codec is experimental
 ....B. = Supports draw_horiz_band
 .....D = Supports direct rendering method 1
 ------
 V....D libx264              libx264 H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10 (codec h264)
 A....D aac                  AAC (Advanced Audio Coding)
 S..... mov_text             3GPP Timed Text subtitle
"""

NON_FILTER_TEXT = """Filters: subtitles ass drawtext
  T.. = Timeline support
  .S. = Slice threading
  ..C = Command support
  ------
 ... subtitles unavailable in this build; the documentation says V->V.
 .. subtitles Render text subtitles onto video; this description mentions V->V.
 ... ass = Render ASS subtitles onto input video.
 ... drawtext invalid->V Draw text on top of video frames.
 . subtitles V->V A single flag is not a supported listing format.
 .... subtitles V->V Four flags are not a supported listing format.
  Description: subtitles V->V Render text subtitles.
"""


class FFmpegFeatureTests(unittest.TestCase):
    def features(self, filters: str, encoders: str = ENCODERS) -> tuple[set[str], set[str]]:
        with patch.object(media, "_run", side_effect=[filters, encoders]) as run:
            result = media._ffmpeg_features("/fixture/ffmpeg")
        self.assertEqual(run.call_args_list, [
            call(["/fixture/ffmpeg", "-hide_banner", "-filters"], timeout=20),
            call(["/fixture/ffmpeg", "-hide_banner", "-encoders"], timeout=20),
        ])
        return result

    def capabilities(self, filters: str, encoders: str = ENCODERS) -> dict:
        with patch.object(media.shutil, "which", side_effect=lambda name: f"/fixture/{name}"), \
                patch.object(media, "_run", side_effect=[filters, encoders]), \
                patch.object(media.importlib.util, "find_spec", return_value=None):
            return media.media_capabilities()

    def test_real_ffmpeg_8_two_flag_rows_include_subtitles_ass_and_timeline_filters(self) -> None:
        filters, encoders = self.features(FFMPEG_8_FILTERS)
        self.assertEqual(filters, {"anull", "volume", "ass", "drawtext", "null", "scale",
                                   "subtitles", "trim", "showvolume"})
        self.assertTrue({"libx264", "aac", "mov_text"}.issubset(encoders))

    def test_legacy_three_flag_rows_preserve_timeline_and_source_sink_signatures(self) -> None:
        filters, _ = self.features(LEGACY_FILTERS)
        self.assertEqual(filters, {"ass", "drawtext", "scale", "subtitles", "volume",
                                   "anullsrc", "nullsink", "concat"})

    def test_headers_legends_and_filter_names_in_descriptions_are_not_filter_rows(self) -> None:
        filters, _ = self.features(NON_FILTER_TEXT)
        self.assertEqual(filters, set())

    def test_health_reports_burn_in_and_soft_subtitles_for_both_listing_formats(self) -> None:
        for listing in [LEGACY_FILTERS, FFMPEG_8_FILTERS]:
            with self.subTest(listing="legacy" if listing is LEGACY_FILTERS else "ffmpeg8"):
                health = self.capabilities(listing)
                self.assertTrue(health["burn_in"])
                self.assertTrue(health["soft_subtitles"])
                self.assertEqual(health["errors"], [])

    def test_health_does_not_infer_subtitles_from_ass_or_descriptive_text(self) -> None:
        for listing in [NON_FILTER_TEXT, " .. ass V->V Render ASS subtitles.\n"]:
            with self.subTest(listing=listing):
                health = self.capabilities(listing)
                self.assertFalse(health["burn_in"])
                self.assertTrue(health["soft_subtitles"])
                self.assertEqual(health["errors"], [])

if __name__ == "__main__":
    unittest.main()

from pathlib import Path
import subprocess
import hashlib
import os
import shutil
import sys
import types
import pytest
from core.workflow.media import prepare_source, render_deliverables, normalize_public_url, subtitle_layout_issues, _font_path, _validate_segments, _wrap

def _binary(name):
    directory = os.environ.get('VIDEOLINGO_FFMPEG_DIR')
    if directory:
        path = Path(directory) / name
        return str(path) if path.is_file() and os.access(path, os.X_OK) else None
    return shutil.which(name)

FFMPEG = _binary('ffmpeg')
FFPROBE = _binary('ffprobe')


def test_source_sidecar_and_invalid_ids(tmp_path, monkeypatch):
    monkeypatch.setattr('core.workflow.media._probe', lambda *_: {'format': {'duration': '0.8'}})
    video = tmp_path / 'source.mp4'
    video.write_bytes(b'placeholder')
    sub = tmp_path / 'source.srt'
    sub.write_text('1\n00:00:00,100 --> 00:00:01,000\nHello {world}\\test\n\n', encoding='utf-8')
    result = prepare_source(str(video), tmp_path / 'job')
    assert result['segments'][0]['text'] == 'Hello {world}\\test'
    assert result['subtitle_source'] == 'sidecar'
    assert result['segments'][0]['end'] == .8
    assert result['metadata']['subtitle_timing_adjustments'][0]['original_end'] == 1
    sub.write_text('1\n00:00:00,100 --> 00:00:01,000\na\n\n1\n00:00:01,000 --> 00:00:02,000\nb\n')
    with pytest.raises(ValueError, match='Duplicate'):
        prepare_source(str(video), tmp_path / 'job')


def test_missing_and_transcriber(tmp_path, monkeypatch):
    monkeypatch.setattr('core.workflow.media._probe', lambda *_: {'format': {'duration': '2'}})
    video = tmp_path / 'source.mp4'
    video.write_bytes(b'x')
    with pytest.raises(ValueError, match='No source subtitles'):
        prepare_source(str(video), tmp_path / 'job')
    result = prepare_source(str(video), tmp_path / 'job', transcriber=lambda *_: [{'id': 1, 'start': 0., 'end': 1., 'text': 'hello'}])
    assert result['subtitle_source'] == 'transcribed'


@pytest.mark.skipif(not FFMPEG or not FFPROBE, reason='FFmpeg and ffprobe required; configure VIDEOLINGO_FFMPEG_DIR or PATH')
def test_real_render_and_audio(tmp_path):
    video = tmp_path / 'source.mp4'
    subprocess.run([FFMPEG, '-v', 'error', '-y', '-f', 'lavfi', '-i', 'testsrc=size=321x241:rate=5:duration=1.5', '-f', 'lavfi', '-i', 'sine=frequency=440:duration=1.5', '-f', 'lavfi', '-i', 'sine=frequency=660:duration=1.5', '-map', '0:v', '-map', '1:a', '-map', '2:a', '-c:v', 'libx264', '-pix_fmt', 'yuv444p', '-c:a', 'aac', str(video)], check=True)
    before = hashlib.sha256(video.read_bytes()).digest()
    segments = [{'id': 1, 'start': .1, 'end': 1.1, 'text': 'Hello world'}]
    result = render_deliverables(video, segments, {1: '你好，世界'}, tmp_path / 'deliverables', ffmpeg=FFMPEG, ffprobe=FFPROBE)
    assert result['validation_data']['full_decode'] == 'passed'
    assert result['validation_data']['output_audio_streams'] == 2
    assert hashlib.sha256(video.read_bytes()).digest() == before
    assert '你好，世界' in Path(result['bilingual_srt']).read_text()
    assert Path(result['previews'][0]).is_file()
    with pytest.raises(FileExistsError, match='already contains'):
        render_deliverables(video, segments, {1: '你好，世界'}, tmp_path / 'deliverables', ffmpeg=FFMPEG, ffprobe=FFPROBE)
    with pytest.raises(ValueError, match='segment IDs: \\[1\\]'):
        render_deliverables(video, segments, {1: '超' * 1000}, tmp_path / 'overflow', ffmpeg=FFMPEG, ffprobe=FFPROBE)


@pytest.mark.parametrize('segments', [
    [{'id': 1, 'start': True, 'end': 1, 'text': 'a'}],
    [{'id': 1, 'start': 0, 'end': float('nan'), 'text': 'a'}],
    [{'id': 1, 'start': 0, 'end': 0.001, 'text': 'a'}],
    [{'id': 1, 'start': 0, 'end': 1, 'text': 'a\n\n2'}],
    [{'id': 1, 'start': 0, 'end': 1, 'text': '\ud800'}],
    [{'id': 1, 'start': 0, 'end': 1, 'text': 'a'}, {'id': 2, 'start': .5, 'end': 2, 'text': 'b'}],
])
def test_invalid_segments(segments):
    with pytest.raises(ValueError):
        _validate_segments(segments)


def test_word_wrapping():
    class Font:
        def getlength(self, text):
            return len(text)
    lines = _wrap('hello world 中文', Font(), 7, 4)
    assert ''.join(lines) == 'hello world 中文'
    assert all('hello' not in line or 'hello' == line.strip() for line in lines)
    assert _wrap('unbreakable', Font(), 4, 4) is None


def test_layout_issues_identify_language_without_ffmpeg(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError('Layout checking must not invoke FFmpeg')
    monkeypatch.setattr('core.workflow.media._run', fail)
    segments = [{'id': 1, 'start': 0, 'end': 1, 'text': 'hello'},
                {'id': 2, 'start': 1, 'end': 2, 'text': 'word' * 300}]
    assert subtitle_layout_issues(segments, {1: '超' * 1000, 2: '正常'}) == [
        {'id': 1, 'language': 'translation'}, {'id': 2, 'language': 'source'}]


def test_url_selection_and_subtitle_cache_recovery(tmp_path, monkeypatch):
    calls = []
    class FakeYoutubeDL:
        def __init__(self, opts): self.opts = opts
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def extract_info(self, source, download):
            calls.append((self.opts, download))
            if download:
                directory = Path(self.opts['outtmpl']).parent
                if not self.opts.get('skip_download'):
                    (directory / 'video.mp4').write_bytes(b'video')
                (directory / 'video.en-US.vtt').write_text('WEBVTT\n\n00:00:00.100 --> 00:00:01.000\nHi\n')
            return {'subtitles': {'en-US': [{}]}, 'automatic_captions': {'en': [{}]}, 'title': 'Example'}
    monkeypatch.setitem(sys.modules, 'yt_dlp', types.SimpleNamespace(YoutubeDL=FakeYoutubeDL))
    monkeypatch.setattr('core.workflow.media._probe', lambda *_: {'format': {'duration': '2'}})
    source = 'https://www.youtube.com/watch?v=O8t50OxsfB8&si=tracking'
    result = prepare_source(source, tmp_path)
    assert result['subtitle_source'] == 'downloaded_manual'
    assert calls[-1][0]['subtitleslangs'] == ['^en\\-US$']
    assert result['metadata']['source_url'] == 'https://www.youtube.com/watch?v=O8t50OxsfB8'
    dest = Path(result['video_path']).parent
    (dest / 'video.en-US.vtt').unlink()
    prepare_source(source, tmp_path)
    assert calls[-1][0]['skip_download'] is True


@pytest.mark.parametrize('url', [
    'https://user:pass@example.org/video',
    'https://example.org/video?token=secret',
    'https://example.org/video?X-Amz-Signature=secret',
    'https://www.youtube.com/watch?v=O8t50OxsfB8&token=secret',
    'https://example.org/token/secret',
])
def test_reject_access_urls(url):
    with pytest.raises(ValueError):
        normalize_public_url(url)


def test_public_url_identity():
    assert normalize_public_url('https://youtube.com/watch?v=O8t50OxsfB8&t=9') == 'https://youtube.com/watch?v=O8t50OxsfB8'
    assert normalize_public_url('https://youtu.be/O8t50OxsfB8?si=abc') == 'https://youtu.be/O8t50OxsfB8'


def test_configured_font_path(tmp_path, monkeypatch):
    font = tmp_path / 'font.ttc'
    font.touch()
    monkeypatch.setenv('VIDEOLINGO_FONT_PATH', str(font))
    assert _font_path() == str(font)
    font.unlink()
    with pytest.raises(RuntimeError, match='VIDEOLINGO_FONT_PATH'):
        _font_path()

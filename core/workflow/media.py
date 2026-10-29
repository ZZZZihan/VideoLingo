"""Deterministic source preparation and subtitle rendering for the workflow."""
from __future__ import annotations

import hashlib
import html
import json
import math
import os
import re
import shutil
import subprocess
import unicodedata
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


def _run(args, **kwargs):
    return subprocess.run([str(x) for x in args], check=True, capture_output=True, text=True, **kwargs)


def _time(value):
    parts = value.replace(',', '.').split(':')
    if len(parts) not in (2, 3):
        raise ValueError(f'Invalid subtitle timestamp: {value}')
    try:
        seconds = float(parts[-1]) + 60 * int(parts[-2]) + (3600 * int(parts[-3]) if len(parts) == 3 else 0)
    except ValueError as exc:
        raise ValueError(f'Invalid subtitle timestamp: {value}') from exc
    if not math.isfinite(seconds) or seconds < 0 or float(parts[-1]) >= 60 or int(parts[-2]) >= 60:
        raise ValueError(f'Invalid subtitle timestamp: {value}')
    return seconds


def _validate_segments(segments):
    seen = set()
    previous_end = 0
    if not segments:
        raise ValueError('No usable subtitle segments found')
    for s in segments:
        ident = s.get('id')
        if not isinstance(ident, int) or isinstance(ident, bool) or ident <= 0 or ident in seen:
            raise ValueError(f'Invalid or duplicate subtitle ID: {ident}')
        seen.add(ident)
        start, end = s.get('start'), s.get('end')
        if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start or round(end * 100) <= round(start * 100):
            raise ValueError(f'Invalid subtitle timing for ID {ident}')
        if start < previous_end:
            raise ValueError(f'Out-of-order or overlapping subtitle ID {ident}')
        previous_end = end
        _validate_text(s.get('text'), ident)
    return segments


def _validate_text(value, ident):
    if not isinstance(value, str) or not value.strip() or any(unicodedata.category(c) in ('Cc', 'Cs') for c in value):
        raise ValueError(f'Invalid subtitle text for ID {ident}')
    try:
        value.encode('utf-8')
    except UnicodeEncodeError as exc:
        raise ValueError(f'Invalid subtitle text for ID {ident}') from exc


def _parse_subtitles(path):
    raw = path.read_text(encoding='utf-8-sig').replace('\r\n', '\n').replace('\r', '\n')
    segments = []
    used = set()
    for block in re.split(r'\n\s*\n', raw):
        lines = block.strip().splitlines()
        if not lines or lines[0].startswith(('WEBVTT', 'NOTE', 'STYLE', 'REGION')):
            continue
        idx = next ( (i for i, line in enumerate(lines) if '-->' in line), None)
        if idx is None:
            continue
        match = re.match(r'^\s*(\S+)\s*-->\s*(\S+)', lines[idx])
        if not match:
            raise ValueError('Invalid subtitle timing line')
        ident = int(lines[idx-1].strip()) if idx and lines[idx-1].strip().isdigit() else len(segments) + 1
        if ident in used:
            raise ValueError(f'Duplicate subtitle ID: {ident}')
        used.add(ident)
        content = html.unescape(re.sub(r'<[^>]*>', '', ' '.join(lines[idx+1:]))).strip()
        segments.append({'id': ident, 'start': _time(match.group(1)), 'end': _time(match.group(2)), 'text': content})
    return _validate_segments(segments)


def normalize_public_url(url):
    """Return a shareable public URL, rejecting unknown query credentials.

    YouTube watch URLs retain their video ID. Other sites must use a query-free
    public URL; never include signed media URLs or embedded credentials.
    """
    p = urlsplit(url)
    if p.scheme not in ('http', 'https') or not p.hostname or p.username or p.password or p.fragment:
        raise ValueError('Use a public URL without embedded credentials or fragments')
    host = p.hostname.lower()
    try:
        port = p.port
    except ValueError as exc:
        raise ValueError('Invalid public URL') from exc
    if port not in (None, 80, 443):
        raise ValueError('Use a public URL with a standard port')
    if re.search(r'(?i)(?:^|[/_.-])(token|signature|signed|credential|secret|apikey|auth)(?:$|[/_.-])', p.path):
        raise ValueError('Access-bearing URL paths are not supported')
    pairs = parse_qsl(p.query, keep_blank_values=True)
    youtube = host in {'youtube.com', 'www.youtube.com', 'm.youtube.com', 'youtu.be', 'www.youtu.be'}
    if youtube:
        allowed = {'v', 't', 'start', 'end', 'si', 'feature', 'pp'}
        if any(k not in allowed for k, _ in pairs):
            raise ValueError('Signed or unsupported URL parameters are not allowed')
        ids = [v for k, v in pairs if k == 'v']
        if p.path == '/watch':
            if len(ids) != 1 or not re.fullmatch(r'[A-Za-z0-9_-]{11}', ids[0]):
                raise ValueError('A YouTube watch URL requires one valid video ID')
            query = urlencode({'v': ids[0]})
        else:
            if ids:
                raise ValueError('Unexpected YouTube video ID parameter')
            query = ''
    else:
        if pairs:
            raise ValueError('URL query parameters are not supported for this site')
        query = ''
    return urlunsplit((p.scheme, host, p.path, query, ''))


def prepare_source(source: str, job_dir: Path, *, source_language='en', subtitle_path=None, max_height=1080, transcriber=None):
    job_dir = Path(job_dir).resolve()
    job_dir.mkdir(parents=True, exist_ok=True)
    url = urlsplit(source)
    metadata = {}
    if url.scheme in ('http', 'https'):
        source = normalize_public_url(source)
        try:
            import yt_dlp
        except ImportError as exc:
            raise RuntimeError('yt-dlp is required for URL sources') from exc
        key = hashlib.sha256(source.encode()).hexdigest()[:16]
        dest = job_dir / ('source_' + key)
        dest.mkdir(exist_ok=True)
        opts = {'outtmpl': str(dest / 'video.%(ext)s'), 'format': f'bestvideo[height<={int(max_height)}]+bestaudio/best[height<={int(max_height)}]/best', 'merge_output_format': 'mp4', 'subtitlesformat': 'vtt/srt/best', 'quiet': True, 'no_warnings': True, 'noplaylist': True, 'overwrites': False}
        existing = [p for p in dest.glob('video.*') if p.suffix.lower() in ('.mp4', '.mkv', '.webm', '.mov')]
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(source, download=False)
        def choose_language(languages):
            matches = [x for x in languages if x == source_language or x.startswith(source_language + '-') or x.startswith(source_language + '_')]
            return min(matches, key=lambda x: (x != source_language, x)) if matches else None
        language = choose_language(info.get('subtitles') or {})
        manual = bool(language)
        if not language:
            language = choose_language(info.get('automatic_captions') or {})
        candidates = [dest / f'video.{language}.{ext}' for ext in ('vtt', 'srt')] if language else []
        marker = dest / 'subtitle-cache.json'
        cache_key = {'language': language, 'kind': 'manual' if manual else 'automatic'}
        try:
            cache_valid = json.loads(marker.read_text(encoding='utf-8')) == cache_key
        except (OSError, ValueError):
            cache_valid = False
        cached_subtitle = next((p for p in candidates if p.is_file()), None) if cache_valid else None
        if not existing or (language and not cached_subtitle and not subtitle_path):
            download_opts = dict(opts, skip_download=bool(existing), overwrites=not cache_valid, writesubtitles=manual and bool(language), writeautomaticsub=not manual and bool(language), subtitleslangs=['^' + re.escape(language) + '$'] if language and not subtitle_path else [])
            if subtitle_path:
                download_opts.update(writesubtitles=False, writeautomaticsub=False)
            with yt_dlp.YoutubeDL(download_opts) as ydl:
                ydl.extract_info(source, download=True)
            if language and not subtitle_path and any(p.is_file() for p in candidates):
                marker.write_text(json.dumps(cache_key), encoding='utf-8')
                cache_valid = True
        videos = [p for p in dest.glob('video.*') if p.suffix.lower() in ('.mp4', '.mkv', '.webm', '.mov')]
        if len(videos) != 1:
            raise RuntimeError('Download did not produce exactly one video')
        video = videos[0]
        metadata = {k: info[k] for k in ('title', 'uploader', 'duration', 'id') if info.get(k) is not None}
        metadata['source_url'] = source
        subtitle = Path(subtitle_path).expanduser().resolve() if subtitle_path else next((p for p in candidates if p.is_file() and cache_valid), None)
        kind = 'provided' if subtitle_path else ('downloaded_manual' if manual else 'downloaded_automatic')
    else:
        video = Path(source).expanduser().resolve()
        if not video.is_file():
            raise FileNotFoundError(f'Video not found: {video}')
        metadata = {'title': video.stem}
        subtitle = Path(subtitle_path).expanduser().resolve() if subtitle_path else next((p for p in (video.with_suffix('.srt'), video.with_suffix('.vtt')) if p.is_file()), None)
        kind = 'provided' if subtitle_path else 'sidecar'
    if subtitle is not None:
        if not subtitle.is_file():
            raise FileNotFoundError(f'Subtitle not found: {subtitle}')
        segments = _parse_subtitles(subtitle)
    elif transcriber:
        segments = _validate_segments(transcriber(str(video.resolve()), job_dir, source_language))
        kind = 'transcribed'
    else:
        raise ValueError('No source subtitles found; provide subtitles or a transcriber')
    ffprobe = shutil.which('ffprobe')
    if not ffprobe:
        raise RuntimeError('ffprobe is required to validate source duration')
    duration = float(_probe(ffprobe, video)['format']['duration'])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('Invalid video duration')
    adjustments = []
    for segment in segments:
        if segment['start'] >= duration:
            raise ValueError(f'Subtitle ID {segment["id"]} starts at or after video end')
        if segment['end'] > duration:
            adjustments.append({'id': segment['id'], 'original_end': segment['end'], 'adjusted_end': duration})
            segment['end'] = duration
    _validate_segments(segments)
    metadata['duration'] = duration
    metadata['subtitle_timing_adjustments'] = adjustments
    return {'video_path': str(video.resolve()), 'segments': segments, 'source_language': source_language, 'metadata': metadata, 'subtitle_source': kind}


def _stamp(t, ass=False):
    unit = 100 if ass else 1000
    n = round(t * unit)
    if ass:
        return f'{n//(3600*unit)}:{n//(60*unit)%60:02}:{n//unit%60:02}.{n%unit:02}'
    return f'{n//(3600*unit):02}:{n//(60*unit)%60:02}:{n//unit%60:02},{n%unit:03}'


def _font_path():
    configured = os.environ.get('VIDEOLINGO_FONT_PATH')
    if configured:
        if not Path(configured).is_file():
            raise RuntimeError('VIDEOLINGO_FONT_PATH must name an existing Chinese-capable font')
        return configured
    for path in ('/System/Library/Fonts/PingFang.ttc', '/System/Library/Fonts/STHeiti Medium.ttc', '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc', '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'):
        if Path(path).is_file():
            return path
    if shutil.which('fc-match'):
        result = _run(['fc-match', '-f', '%{file}', ':lang=zh']).stdout
        if Path(result).is_file():
            return result
    raise RuntimeError('A Chinese-capable font is required for subtitle layout')


def _wrap(text, font, width, max_lines):
    lines, line = [], ''
    # A Latin word is indivisible; CJK and punctuation are break opportunities.
    tokens = re.findall(r'[^\W_]+(?:[\'’\-][^\W_]+)*|\s+|.', text, flags=re.UNICODE)
    tokens = [piece for token in tokens for piece in (list(token) if any('\u2e80' <= ch <= '\u9fff' or '\uf900' <= ch <= '\ufaff' for ch in token) else [token])]
    for token in tokens:
        if line and font.getlength(line + token) > width:
            lines.append(line)
            line = token
        else:
            line += token
    if line:
        lines.append(line)
    if len(lines) > max_lines or any(font.getlength(x) > width for x in lines):
        return None
    return lines


def _probe(ffprobe, path):
    return json.loads(_run([ffprobe, '-v', 'error', '-show_streams', '-show_format', '-of', 'json', path]).stdout)


def _subtitle_layout(segments, translations):
    from PIL import ImageFont
    _validate_segments(segments)
    font_path = _font_path()
    zh_font, en_font = ImageFont.truetype(font_path, 48), ImageFont.truetype(font_path, 34)
    wrapped, issues = {}, []
    for segment in segments:
        ident = segment['id']
        translated = translations.get(ident)
        _validate_text(translated, ident)
        z = _wrap(translated, zh_font, 1760, 2)
        e = _wrap(segment['text'], en_font, 1760, 2)
        if z is None:
            issues.append({'id': ident, 'language': 'translation'})
        if e is None:
            issues.append({'id': ident, 'language': 'source'})
        wrapped[ident] = (z, e)
    return wrapped, issues, zh_font.getname()[0]


def subtitle_layout_issues(segments, translations):
    """Identify segments that cannot fit without changing text or timestamps."""
    return _subtitle_layout(segments, translations)[1]


def render_deliverables(video_path, segments, translations: dict[int, str], output_dir, *, layout='subtitle_band', ffmpeg=None, ffprobe=None):
    if layout != 'subtitle_band':
        raise ValueError(f'Unsupported layout: {layout}')
    wrapped, issues, family = _subtitle_layout(segments, translations)
    if issues:
        raise ValueError(f'Subtitle text does not fit; revise segment IDs: {sorted({issue["id"] for issue in issues})}')
    video = Path(video_path).resolve()
    out = Path(output_dir).resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    out.mkdir(parents=True, exist_ok=True)
    if video.parent == out:
        raise ValueError('Output directory must differ from source directory')
    reserved = ('bilingual.ass', 'bilingual.srt', 'chinese.srt', 'english.srt', 'final.mp4', 'validation.json', 'previews')
    if any((out / name).exists() for name in reserved):
        raise FileExistsError('Output directory already contains workflow deliverables; choose a new directory')
    ffmpeg = ffmpeg or shutil.which('ffmpeg')
    ffprobe = ffprobe or shutil.which('ffprobe')
    if not ffmpeg or not ffprobe:
        raise RuntimeError('ffmpeg and ffprobe are required')
    if not re.search(r'^\s*\S+\s+ass\s', _run([ffmpeg, '-hide_banner', '-filters']).stdout, re.M):
        raise RuntimeError('FFmpeg needs the ass filter (libass)')
    source_info = _probe(ffprobe, video)
    duration = float(source_info['format']['duration'])
    if any(s['end'] > duration + 0.05 for s in segments):
        raise ValueError('Subtitle timing exceeds video duration')
    header = '''[Script Info]\nScriptType: v4.00+\nPlayResX: 1920\nPlayResY: 1080\nWrapStyle: 2\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Chinese,SubFont,48,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,8,80,80,0,1\nStyle: English,SubFont,34,&H00DDD3CB,&H00DDD3CB,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,8,80,80,0,1\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n'''
    # libass uses the font's internal family name, not the filename.
    header = header.replace('SubFont', family.replace(',', ''))
    ass, bi, zh, en = [header], [], [], []
    def escape(line):
        # The word joiner prevents ASS escape sequences while preserving the
        # visible backslash; escaped braces remain visible in libass.
        return line.replace('\\', '\\\u2060').replace('{', r'\{').replace('}', r'\}')
    def srt_escape(line):
        return html.escape(line).replace('{', '&#123;').replace('}', '&#125;')
    for s in segments:
        i, a, b = s['id'], s['start'], s['end']
        z, e = wrapped[i]
        for style, lines, y in (('Chinese', z, 902), ('English', e, 1004)):
            ass.append(f'Dialogue: 0,{_stamp(a, True)},{_stamp(b, True)},{style},,0,0,0,,{{\\pos(960,{y})}}' + r'\N'.join(map(escape, lines)) + '\n')
        prefix = f'{i}\n{_stamp(a)} --> {_stamp(b)}\n'
        bi.append(prefix + srt_escape(translations[i]) + '\n' + srt_escape(s['text']) + '\n\n')
        zh.append(prefix + srt_escape(translations[i]) + '\n\n')
        en.append(prefix + srt_escape(s['text']) + '\n\n')
    paths = {'ass': out / 'bilingual.ass', 'bilingual_srt': out / 'bilingual.srt', 'chinese_srt': out / 'chinese.srt', 'english_srt': out / 'english.srt', 'video': out / 'final.mp4', 'validation': out / 'validation.json'}
    for key, content in (('ass', ass), ('bilingual_srt', bi), ('chinese_srt', zh), ('english_srt', en)):
        paths[key].write_text(''.join(content), encoding='utf-8')
    # Relative ASS path avoids escaping arbitrary user paths in filter expressions.
    filt = 'scale=1920:880:force_original_aspect_ratio=decrease:flags=lanczos,pad=1920:1080:(ow-iw)/2:0:color=0x09131C,setsar=1,ass=bilingual.ass'
    cmd = [ffmpeg, '-hide_banner', '-nostdin', '-y', '-i', video, '-map', '0:v:0', '-map', '0:a?', '-vf', filt, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'copy', '-movflags', '+faststart', paths['video']]
    try:
        _run(cmd, cwd=out)
        audio_mode = 'copy'
    except subprocess.CalledProcessError as exc:
        raise RuntimeError('Rendering failed; input audio may be incompatible with MP4: ' + exc.stderr[-1200:]) from exc
    result = _probe(ffprobe, paths['video'])
    source_audio = len([x for x in source_info['streams'] if x['codec_type'] == 'audio'])
    output_audio = len([x for x in result['streams'] if x['codec_type'] == 'audio'])
    output_duration = float(result['format']['duration'])
    if source_audio != output_audio or abs(duration - output_duration) > max(0.15, duration * 0.001):
        raise RuntimeError('Output stream count or duration differs from source')
    _run([ffmpeg, '-v', 'error', '-xerror', '-i', paths['video'], '-map', '0:v:0', '-map', '0:a?', '-f', 'null', '-'])
    preview_dir = out / 'previews'
    preview_dir.mkdir(exist_ok=True)
    times = sorted(set(round(min(duration - 0.05, max(0, (s['start'] + s['end']) / 2)), 3) for s in [segments[round(i * (len(segments)-1) / min(9, len(segments)-1))] for i in range(min(10, len(segments)))])) if len(segments) > 1 else [round(min(duration - 0.05, (segments[0]['start'] + segments[0]['end']) / 2), 3)]
    previews = []
    for i, t in enumerate(times):
        target = preview_dir / f'{i+1:02}.jpg'
        _run([ffmpeg, '-v', 'error', '-ss', str(max(0, t)), '-i', paths['video'], '-frames:v', '1', '-y', target])
        previews.append(str(target))
    validation = {'source_duration': duration, 'output_duration': output_duration, 'source_audio_streams': source_audio, 'output_audio_streams': output_audio, 'audio_mode': audio_mode, 'subtitle_count': len(segments), 'ass_events': len(segments)*2, 'full_decode': 'passed', 'preview_times': times, 'font': family}
    paths['validation'].write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding='utf-8')
    return {**{key: str(value) for key, value in paths.items()}, 'previews': previews, 'validation_data': validation}

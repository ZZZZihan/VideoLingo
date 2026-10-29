"""Run local VideoLingo ASR and preserve its raw segment text."""
from pathlib import Path
import json
import socket


def _disable_network():
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    def connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise OSError('Network access is disabled for local ASR')
        return original_connect(sock, address)
    def connect_ex(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise OSError('Network access is disabled for local ASR')
        return original_connect_ex(sock, address)
    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex


def segments_from_result(result, joiner=' '):
    from core.workflow.translation import validate_segments
    if not isinstance(result, dict) or not isinstance(result.get('segments'), list):
        raise ValueError('Local ASR returned invalid segments')
    segments = []
    for raw in result['segments']:
        if not isinstance(raw, dict):
            raise ValueError('Local ASR returned an invalid segment')
        text = raw.get('text')
        if not isinstance(text, str) or not text.strip():
            words = raw.get('words')
            if not isinstance(words, list) or any(not isinstance(w, dict) or not isinstance(w.get('word'), str) for w in words):
                raise ValueError('Local ASR returned a segment without readable text')
            text = joiner.join(w['word'].strip() for w in words if w['word'].strip())
        if not text.strip():
            raise ValueError('Local ASR returned an empty segment')
        segments.append({'id': len(segments) + 1, 'start': raw.get('start'),
                         'end': raw.get('end'), 'text': text})
    return validate_segments(segments)


def main():
    _disable_network()
    from core import _2_asr as asr
    from core.utils.config_utils import get_joiner, get_source_language

    # Capture backend output before the standard spreadsheet conversion, which
    # discards words longer than 30 characters. This also covers cache hits.
    class RawResult:
        def __init__(self, value):
            self.value = value
            self.empty = not bool(value.get('segments'))
    asr.process_transcription = RawResult
    def save_raw(result):
        segments = segments_from_result(result.value, get_joiner(get_source_language()))
        Path('segments.json').write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding='utf-8')
    asr.save_results = save_raw
    asr.transcribe()
    if not Path('segments.json').is_file():
        raise RuntimeError('Local ASR did not produce validated segments')


if __name__ == '__main__':
    main()

"""Actual CLI, FFmpeg, subtitles and checkpoints; no network/model/browser calls."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from core.workflow.state import file_digest, read_json, write_json
from core.workflow.translation import (api_settings, call_llm, parse_translations, seal_request,
                                       translation_schema, validate_request)


ROOT = Path(__file__).resolve().parents[1]


def cli(*args):
    result = subprocess.run([sys.executable, str(ROOT / 'workflow.py'), *map(str, args)],
                            cwd=ROOT, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'), reason='FFmpeg required')
def test_cli_real_media_and_external_language_round_trip(tmp_path):
    source = tmp_path / 'source.mp4'
    subprocess.run([shutil.which('ffmpeg'), '-v', 'error', '-f', 'lavfi', '-i',
                    'testsrc2=size=320x240:rate=10:duration=1.8', '-f', 'lavfi', '-i',
                    'sine=frequency=440:duration=1.8', '-c:v', 'libx264', '-c:a', 'aac', str(source)],
                   check=True, capture_output=True)
    source_hash = file_digest(source)
    captions = tmp_path / 'source.vtt'
    captions.write_text('WEBVTT\n\n00:00:00.100 --> 00:00:00.800\nThe top ten together hold 2%.\n\n'
                        '00:00:00.900 --> 00:00:02.200\nApple and Alphabet.\n', encoding='utf-8')
    directory = tmp_path / 'job'
    data = cli('run', source, '--subtitles', captions, '--provider', 'external', '--job-dir', directory)
    assert data['status'] == 'awaiting_translation'
    assert data['source']['segments'][-1]['end'] < 2.2
    assert data['source']['metadata']['subtitle_timing_adjustments'][-1]['id'] == 2

    def answer(result):
        request = read_json(Path(data['pending_requests'][0]))['request']
        response = tmp_path / 'response.json'
        write_json(response, {'request_id': request['request_id'], 'source_digest': request['source_digest'], 'result': result})
        return cli('import-response', '--job-dir', directory, '--file', response)

    data = answer({'translations': [{'id': 1, 'text': '前十名各持有2%。'}, {'id': 2, 'text': '苹果与谷歌母公司。'}]})
    assert data['status'] == 'awaiting_review'
    data = answer({'issues': [{'id': 1, 'reason': '原文是合计持有', 'replacement': '前十名合计持有2%。'}], 'uncertainties': []})
    assert data['status'] == 'awaiting_review'
    data = answer({'issues': [], 'uncertainties': ['专名按术语表核对。']})
    assert data['status'] == 'completed'
    artifacts = data['artifacts']
    assert artifacts['validation_data']['full_decode'] == 'passed'
    assert artifacts['validation_data']['output_audio_streams'] == 1
    assert len(artifacts['previews']) >= 2
    assert all(Path(p).is_file() for p in artifacts['previews'])
    assert '合计持有2%' in Path(artifacts['chinese_srt']).read_text()
    assert 'The top ten together' in Path(artifacts['english_srt']).read_text()
    assert file_digest(source) == source_hash
    assert cli('resume', '--job-dir', directory)['artifacts']['video'] == artifacts['video']
    assert cli('status', '--job-dir', directory)['publication']['stage'] == 'prepared'
    assert not (directory / 'douyin-browser-profile').exists()


def test_codex_provider_uses_stdin_schema_and_requested_configuration(tmp_path, monkeypatch):
    request = seal_request({'operation': 'translate', 'source_digest': 'source',
                            'segments': [{'id': 1, 'text': 'Hello'}]})
    seen = []
    monkeypatch.setattr('core.workflow.translation.shutil.which', lambda _name: '/local/codex')
    def execute(command, **kwargs):
        seen.append((command, kwargs))
        result_path = Path(command[command.index('--output-last-message') + 1])
        write_json(result_path, {'translations': [{'id': 1, 'text': '你好'}]})
        return type('Completed', (), {'returncode': 0, 'stderr': 'private provider diagnostics'})()
    monkeypatch.setattr('core.workflow.translation.subprocess.run', execute)
    assert call_llm(request, translation_schema(), provider='codex', work_dir=tmp_path)['translations'][0]['text'] == '你好'
    command, args = seen[0]
    assert command[command.index('--model') + 1] == 'gpt-6.1-sol'
    assert 'model_reasoning_effort="medium"' in command
    assert command[command.index('--sandbox') + 1] == 'read-only'
    assert command[-1] == '-'
    assert request['instructions'] in args['input']
    assert list(tmp_path.iterdir()) == []


def test_instruction_changes_break_external_request_binding():
    request = seal_request({'operation': 'review', 'source_digest': 'source', 'segments': []})
    request['instructions'] = 'Different rules'
    with pytest.raises(ValueError, match='inconsistent'):
        validate_request(request)


def test_malformed_api_yaml_does_not_echo_credential(tmp_path):
    config = tmp_path / 'config.yaml'
    config.write_text('api:\n  key: private-credential: invalid-yaml\n')
    with pytest.raises(ValueError) as error:
        api_settings(config)
    assert 'private-credential' not in str(error.value)


@pytest.mark.parametrize('entries', [
    [{'id': 1, 'text': 'a'}, {'id': 1, 'text': 'b'}],
    [{'id': True, 'text': 'a'}],
    [{'id': 2, 'text': 'a'}],
    [{'id': 1, 'text': 'a', 'start': 0}],
    [{'id': 1, 'text': '\u0001'}],
])
def test_response_cannot_duplicate_change_or_inject_segment_structure(entries):
    with pytest.raises(ValueError):
        parse_translations({'translations': entries}, {1})

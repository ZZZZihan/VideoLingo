import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
from ruamel.yaml import YAML

from core.workflow import asr, asr_worker


def test_isolated_config_never_copies_other_providers(tmp_path):
    config = {'api': {'key': 'secret-api'}, 'whisper': {
        'backend': 'qwen', 'qwen_model': '1.7b', 'qwen_engine': 'auto', 'model': 'large-v3',
        'runtime': 'mai', 'mai_api_key': 'secret-mai', 'elevenlabs_api_key': 'secret-eleven'},
        'openai_tts': {'api_key': 'secret-tts'}, 'model_dir': './models',
        'audio': {'raw_sample_rate': 32000, 'raw_bitrate': '128k'}}
    result = asr.isolated_config(config, 'en', root=tmp_path)
    assert result['model_dir'] == str(tmp_path / 'models')
    assert result['whisper']['runtime'] == 'local'
    assert 'secret' not in json.dumps(result)
    assert set(result) == {'whisper', 'model_dir', 'demucs', 'audio', 'language_split_with_space', 'language_split_without_space'}


def test_actual_raw_output_keeps_long_and_unaligned_text():
    original = 'Pneumonoultramicroscopicsilicovolcanoconiosis «hello»'
    result = {'segments': [{'start': 1, 'end': 4, 'text': original,
                            'words': [{'word': 'Pneumonoultramicroscopicsilicovolcanoconiosis'}]}]}
    assert asr_worker.segments_from_result(result)[0]['text'] == original
    with pytest.raises(ValueError, match='timing'):
        asr_worker.segments_from_result({'segments': [{'start': 2, 'end': 1, 'text': original}]})


def test_worker_intercepts_raw_backend_result(monkeypatch, tmp_path):
    import core
    from core.utils import config_utils
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(asr_worker, '_disable_network', lambda: None)
    monkeypatch.setattr(config_utils, 'get_source_language', lambda: 'en')
    monkeypatch.setattr(config_utils, 'get_joiner', lambda language: ' ')
    fake = SimpleNamespace()
    def transcribe():
        value = fake.process_transcription({'segments': [{'text': 'x' * 40, 'start': 0, 'end': 1}]})
        assert not value.empty
        fake.save_results(value)
    fake.transcribe = transcribe
    monkeypatch.setattr(core, '_2_asr', fake, raising=False)
    asr_worker.main()
    assert json.loads((tmp_path / 'segments.json').read_text())[0]['text'] == 'x' * 40


def test_worker_blocks_outbound_connections():
    script = '''
import socket
from core.workflow.asr_worker import _disable_network
_disable_network()
for method in ("connect", "connect_ex"):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        getattr(sock, method)(("127.0.0.1", 9))
    except OSError as exc:
        assert "Network access is disabled" in str(exc)
    else:
        raise AssertionError("network was allowed")
    finally:
        sock.close()
'''
    subprocess.run([sys.executable, '-c', script], check=True, timeout=10)


def test_adapter_uses_fresh_isolated_attempts_and_offline_settings(monkeypatch, tmp_path):
    root = tmp_path / 'repo'
    root.mkdir()
    (root / 'config.yaml').write_text('whisper:\n  backend: qwen\n  mai_api_key: secret\nmodel_dir: ./models\n')
    video = tmp_path / 'video.mp4'
    video.write_bytes(b'video')
    monkeypatch.setattr(asr, 'ROOT', root)
    attempts = []
    def run(args, *, cwd, env, **kwargs):
        attempts.append(cwd)
        config = YAML(typ='safe').load((cwd / 'config.yaml').read_text())
        assert config['model_dir'] == str(root / 'models')
        assert 'secret' not in (cwd / 'config.yaml').read_text()
        assert env['HF_HUB_OFFLINE'] == '1' and env['TRANSFORMERS_OFFLINE'] == '1'
        (cwd / 'segments.json').write_text('[{"id":1,"start":0,"end":1,"text":"hello"}]')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(asr.subprocess, 'run', run)
    job = tmp_path / 'job'
    assert asr.transcribe(video, job, 'en')[0]['text'] == 'hello'
    assert asr.transcribe(video, job, 'en')[0]['text'] == 'hello'
    assert attempts[0] != attempts[1]

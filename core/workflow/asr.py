"""Isolated, offline adapter to the existing local ASR backend."""
import os
from pathlib import Path
import subprocess
import sys
import uuid

from .state import read_json, write_json
from .translation import validate_segments

ROOT = Path(__file__).resolve().parents[2]


def isolated_config(config, language, root=ROOT):
    """Copy only local ASR settings; no provider credentials enter the job."""
    whisper = config.get('whisper', {})
    if not isinstance(whisper, dict):
        raise ValueError('Invalid whisper configuration')
    selected = {k: whisper[k] for k in ('backend', 'qwen_model', 'qwen_engine', 'model') if k in whisper}
    if any(not isinstance(v, str) for v in selected.values()):
        raise ValueError('Local ASR model settings must be strings')
    selected.update(runtime='local', language=language, detected_language=language, cache=False)
    model_dir = Path(config.get('model_dir', './_model_cache')).expanduser()
    if not model_dir.is_absolute():
        model_dir = Path(root) / model_dir
    audio = config.get('audio', {})
    if not isinstance(audio, dict):
        raise ValueError('Invalid audio configuration')
    if 'raw_sample_rate' in audio and type(audio['raw_sample_rate']) is not int:
        raise ValueError('audio.raw_sample_rate must be an integer')
    if 'raw_bitrate' in audio and not isinstance(audio['raw_bitrate'], str):
        raise ValueError('audio.raw_bitrate must be a string')
    for key in ('language_split_with_space', 'language_split_without_space'):
        value = config.get(key, [])
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError(f'{key} must be a list of language codes')
    return {'whisper': selected, 'demucs': False, 'model_dir': str(model_dir.resolve()),
            'audio': {k: audio[k] for k in ('raw_sample_rate', 'raw_bitrate') if k in audio},
            'language_split_with_space': config.get('language_split_with_space', []),
            'language_split_without_space': config.get('language_split_without_space', [])}


def transcribe(video_path, job_dir, language):
    from ruamel.yaml import YAML
    yaml = YAML(typ='safe')
    try:
        config = yaml.load((ROOT / 'config.yaml').read_text(encoding='utf-8'))
    except Exception:
        raise ValueError('Cannot read repository config.yaml; check its YAML syntax') from None
    safe_config = isolated_config(config, language, root=ROOT)
    directory = Path(job_dir).resolve() / 'asr' / uuid.uuid4().hex
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / 'config.yaml').open('w', encoding='utf-8') as stream:
        YAML().dump(safe_config, stream)
    output = directory / 'output'
    output.mkdir()
    write_json(output / 'input_manifest.json', {'path': str(Path(video_path).resolve(strict=True)), 'type': 'video'})
    environment = dict(os.environ)
    environment['PYTHONPATH'] = str(ROOT) + os.pathsep + environment.get('PYTHONPATH', '')
    environment.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    result = subprocess.run([sys.executable, '-m', 'core.workflow.asr_worker'], cwd=directory,
                            env=environment, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError('Local ASR failed. Check that its backend and models are installed locally, '
                           'or provide an SRT/VTT file. Network access was disabled.')
    return validate_segments(read_json(directory / 'segments.json'))

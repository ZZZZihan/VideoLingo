"""Recovery tests exercise saved jobs across independent resume/import calls."""
import copy
from pathlib import Path
import subprocess
import sys

import pytest

from core.workflow import runner
from core.workflow.state import Job, digest, file_digest, job_lock, read_json, write_json
from core.workflow.translation import provider_identity


SEGMENTS = [
    {'id': 1, 'start': .1, 'end': .8, 'text': 'The top ten together hold 2%.'},
    {'id': 2, 'start': .9, 'end': 1.5, 'text': 'Apple and Alphabet are different companies.'},
]


def make_job(tmp_path, *, provider='external', batch_size=40, config=None):
    source = tmp_path / 'source.mp4'
    source.write_bytes(b'local source for orchestration tests')
    directory = tmp_path / 'job'
    config = config or tmp_path / 'config.yaml'
    options = {'source': str(source), 'subtitles': None, 'subtitle_sha256': None,
               'source_language': 'en', 'target_language': 'zh-CN', 'provider': provider,
               'model': 'gpt-6.1-sol', 'reasoning': 'medium', 'batch_size': batch_size,
               'glossary': [], 'config_path': str(config), 'asr': False, 'max_height': 1080,
               'provider_identity': provider_identity(provider, 'gpt-6.1-sol', 'medium', config)}
    with job_lock(directory):
        Job.create(directory, options)
    return directory


def prepare(source, *_args, **_kwargs):
    return {'video_path': source, 'segments': copy.deepcopy(SEGMENTS), 'source_language': 'en',
            'metadata': {'title': 'Example', 'source_url': 'https://www.youtube.com/watch?v=example'},
            'subtitle_source': 'provided'}


def render(_video, _segments, values, directory):
    directory.mkdir(parents=True)
    video = directory / 'final.mp4'
    video.write_bytes(str(sorted(values.items())).encode())
    report = directory / 'validation.json'
    write_json(report, {'full_decode': 'mocked_in_orchestration_test'})
    return {'video': str(video), 'validation': str(report), 'previews': []}


def provider(request, *_args, **_kwargs):
    if request['operation'] == 'translate':
        return {'translations': [{'id': s['id'], 'text': f'中文译文 {s["id"]}'} for s in request['segments']]}
    return {'issues': [], 'uncertainties': []}


@pytest.fixture
def adapters(monkeypatch):
    monkeypatch.setattr('core.workflow.media.prepare_source', prepare)
    monkeypatch.setattr('core.workflow.media.render_deliverables', render)
    # Real font/layout checks are covered by the media and end-to-end tests.
    monkeypatch.setattr('core.workflow.layout.subtitle_layout_issues', lambda *_args: [])
    return monkeypatch


def respond(directory, request_file, result):
    request = read_json(Path(request_file))['request']
    response = {'request_id': request['request_id'], 'source_digest': request['source_digest'], 'result': result}
    path = directory.parent / 'response.json'
    write_json(path, response)
    return runner.import_response(directory, path)


def test_external_batches_review_and_import_binding(tmp_path, adapters):
    directory = make_job(tmp_path, batch_size=1)
    data = runner.resume_job(directory)
    assert data['status'] == 'awaiting_translation'
    first, second = data['pending_requests']
    request = read_json(Path(first))
    bad = directory.parent / 'wrong-source.json'
    write_json(bad, {'request_id': request['request']['request_id'], 'source_digest': '0' * 64,
                     'result': {'translations': [{'id': 1, 'text': '译文'}]}})
    with pytest.raises(ValueError, match='source digest'):
        runner.import_response(directory, bad)
    with pytest.raises(ValueError, match='exactly the requested'):
        respond(directory, first, {'translations': []})
    # Responses may arrive out of order; the first imported batch is not lost.
    data = respond(directory, second, {'translations': [{'id': 2, 'text': '苹果和谷歌母公司是两家公司。'}]})
    assert data['translated_segments'] == 1
    data = respond(directory, first, {'translations': [{'id': 1, 'text': '前十名合计持有2%。'}]})
    assert data['status'] == 'awaiting_review'
    pending = list(data['pending_requests'])
    for request_file in pending:
        contract = read_json(Path(request_file))
        assert 'collective versus per-item' in contract['instructions']
        assert contract['instructions'] == contract['request']['instructions']
        data = respond(directory, request_file, {'issues': [], 'uncertainties': []})
    assert data['status'] == 'completed'
    assert len(read_json(directory / 'translations.json')['translations']) == 2
    assert read_json(directory / 'semantic-review.json')['human_acceptance'] is False
    assert read_json(directory / 'publication.json')['stage'] == 'prepared'


def test_semantic_correction_is_reviewed_without_changing_source(tmp_path, adapters):
    directory = make_job(tmp_path, provider='codex')
    calls = []
    def language(request, *args, **kwargs):
        calls.append(request['operation'])
        if request['operation'] == 'review' and len(calls) == 2:
            return {'issues': [{'id': 1, 'reason': 'collective, not each',
                                'replacement': '前十名合计持有2%。'}], 'uncertainties': []}
        return provider(request, *args, **kwargs)
    result = runner.resume_job(directory, llm=language)
    assert result['status'] == 'completed'
    assert calls == ['translate', 'review', 'review']
    assert result['source']['segments'] == SEGMENTS
    assert read_json(directory / 'translations.json')['translations'][0]['text'] == '前十名合计持有2%。'


def test_review_limit_persists_across_external_resumes(tmp_path, adapters):
    directory = make_job(tmp_path)
    result = runner.resume_job(directory)
    result = respond(directory, result['pending_requests'][0], {
        'translations': [{'id': 1, 'text': '初稿'}, {'id': 2, 'text': '第二段'}]})
    for attempt in range(3):
        assert result['status'] == 'awaiting_review'
        result = respond(directory, result['pending_requests'][0], {
            'issues': [{'id': 1, 'reason': '仍需纠正', 'replacement': f'第{attempt}轮修订'}],
            'uncertainties': []})
    assert result['status'] == 'needs_revision'
    requests = list((directory / 'requests').glob('*.json'))
    result = runner.resume_job(directory)
    assert result['status'] == 'needs_revision'
    assert list((directory / 'requests').glob('*.json')) == requests
    assert len(next(iter(read_json(directory / 'review-progress.json')['batches'].values()))) == 3


def test_cached_review_cycle_stops(tmp_path, adapters):
    directory = make_job(tmp_path, provider='codex')
    calls = []
    def cycle(request, *args, **kwargs):
        if request['operation'] == 'translate':
            return {'translations': [{'id': 1, 'text': '译文A'}, {'id': 2, 'text': '第二段'}]}
        current = request['segments'][0]['translation']
        calls.append(current)
        return {'issues': [{'id': 1, 'reason': '修订', 'replacement': '译文B' if current == '译文A' else '译文A'}],
                'uncertainties': []}
    assert runner.resume_job(directory, llm=cycle)['status'] == 'needs_revision'
    assert calls == ['译文A', '译文B']
    assert runner.resume_job(directory, llm=lambda *_args, **_kwargs: pytest.fail('cached cycle made another request'))['status'] == 'needs_revision'


def test_overflow_only_sends_affected_cue_then_reviews_all(tmp_path, adapters):
    directory = make_job(tmp_path, provider='codex')
    adapters.setattr('core.workflow.layout.subtitle_layout_issues', lambda _s, values:
                     [{'id': 1, 'language': 'translation'}] if len(values[1]) > 10 else [])
    calls = []
    def language(request, *args, **kwargs):
        calls.append((request['operation'], [s['id'] for s in request['segments']]))
        if request['operation'] == 'translate':
            return {'translations': [{'id': 1, 'text': '非常冗长的中文表达' * 10}, {'id': 2, 'text': '第二段'}]}
        if request['operation'] == 'fit_translation':
            return {'translations': [{'id': 1, 'text': '前十合计2%。'}]}
        return provider(request, *args, **kwargs)
    assert runner.resume_job(directory, llm=language)['status'] == 'completed'
    assert calls == [('translate', [1, 2]), ('fit_translation', [1]), ('review', [1, 2])]


def test_external_layout_round_limit_persists(tmp_path, adapters):
    directory = make_job(tmp_path)
    adapters.setattr('core.workflow.layout.subtitle_layout_issues', lambda *_args: [{'id': 1, 'language': 'translation'}])
    result = runner.resume_job(directory)
    result = respond(directory, result['pending_requests'][0], {
        'translations': [{'id': 1, 'text': '长译文'}, {'id': 2, 'text': '第二段'}]})
    for attempt in range(2):
        assert result['status'] == 'awaiting_layout'
        result = respond(directory, result['pending_requests'][0], {'translations': [{'id': 1, 'text': f'修订{attempt}'}]})
    assert result['status'] == 'needs_revision'
    assert runner.resume_job(directory)['status'] == 'needs_revision'


def test_source_overflow_stops_without_rewriting_original(tmp_path, adapters):
    directory = make_job(tmp_path, provider='codex')
    adapters.setattr('core.workflow.layout.subtitle_layout_issues', lambda *_args: [{'id': 1, 'language': 'source'}])
    calls = []
    def language(request, *args, **kwargs):
        calls.append(request['operation'])
        return provider(request, *args, **kwargs)
    result = runner.resume_job(directory, llm=language)
    assert result['status'] == 'needs_revision'
    assert calls == ['translate']
    assert result['source']['segments'] == SEGMENTS


def test_interrupted_translation_only_calls_missing_batch(tmp_path, adapters):
    directory = make_job(tmp_path, provider='codex', batch_size=1)
    def interrupted(request, *args, **kwargs):
        if request['operation'] == 'translate' and request['segments'][0]['id'] == 2:
            raise RuntimeError('interrupted provider')
        return provider(request, *args, **kwargs)
    with pytest.raises(RuntimeError, match='interrupted'):
        runner.resume_job(directory, llm=interrupted)
    assert len(read_json(directory / 'translations.json')['translations']) == 1
    calls = []
    def resumed(request, *args, **kwargs):
        calls.append((request['operation'], [s['id'] for s in request['segments']]))
        return provider(request, *args, **kwargs)
    assert runner.resume_job(directory, llm=resumed)['status'] == 'completed'
    assert ('translate', [1]) not in calls
    assert ('translate', [2]) in calls


def test_package_crash_reuses_verified_video(tmp_path, adapters):
    import core.workflow.publishing as publication
    directory = make_job(tmp_path, provider='codex')
    original = publication.prepare_publication
    paths = []
    def crash(job, video, metadata):
        paths.append(video)
        original(job, video, metadata)
        raise RuntimeError('crash after package write')
    adapters.setattr(publication, 'prepare_publication', crash)
    with pytest.raises(RuntimeError, match='crash after'):
        runner.resume_job(directory, llm=provider)
    checkpoint = Job(directory).data['render_checkpoint']
    assert checkpoint['artifacts']['video'] == str(paths[0])
    adapters.setattr(publication, 'prepare_publication', original)
    def unexpected(*_args, **_kwargs):
        pytest.fail('verified stages must not run again')
    result = runner.resume_job(directory, llm=unexpected, render=unexpected)
    assert result['status'] == 'completed'
    assert result['artifacts']['video'] == str(paths[0])
    assert len(list((directory / 'renders').iterdir())) == 1


def test_failed_render_preserved_and_retry_uses_new_directory(tmp_path, adapters):
    directory = make_job(tmp_path, provider='codex')
    attempts = []
    def fail(*args):
        attempts.append(args[-1])
        args[-1].mkdir(parents=True)
        (args[-1] / 'partial').write_bytes(b'keep')
        raise RuntimeError('render failure')
    with pytest.raises(RuntimeError, match='render failure'):
        runner.resume_job(directory, llm=provider, render=fail)
    result = runner.resume_job(directory, llm=lambda *_args, **_kwargs: pytest.fail('language cache lost'))
    assert result['status'] == 'completed'
    assert Path(result['artifacts']['video']).parent != attempts[0]
    assert (attempts[0] / 'partial').read_bytes() == b'keep'


@pytest.mark.parametrize('changed', ['source', 'translation', 'video', 'review'])
def test_completed_job_rejects_changed_verified_inputs(tmp_path, adapters, changed):
    directory = make_job(tmp_path, provider='codex')
    result = runner.resume_job(directory, llm=provider)
    path = {'source': Path(result['source']['video_path']),
            'translation': directory / 'translations.json',
            'video': Path(result['artifacts']['video']),
            'review': directory / 'semantic-review.json'}[changed]
    if changed == 'translation':
        data = read_json(path)
        data['translations'][0]['text'] = '未审校的修改'
        write_json(path, data)
    elif changed == 'review':
        data = read_json(path)
        data['human_acceptance'] = True
        write_json(path, data)
    else:
        path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='changed'):
        runner.resume_job(directory, llm=provider)


@pytest.mark.parametrize('changed', ['video', 'metadata'])
def test_publication_package_is_bound_to_verified_video(tmp_path, adapters, changed):
    directory = make_job(tmp_path, provider='codex')
    runner.resume_job(directory, llm=provider)
    package = read_json(directory / 'publication.json')
    if changed == 'video':
        other = tmp_path / 'other.mp4'
        other.write_bytes(b'wrong video')
        package.update(video_path=str(other), video_sha256=file_digest(other))
    else:
        package['metadata']['title'] = '被替换的发布内容'
    write_json(directory / 'publication.json', package)
    adapters.setattr('core.workflow.publishing.publish_douyin', lambda *_args, **_kwargs: pytest.fail('changed publication was uploaded'))
    with pytest.raises(ValueError, match='Publication video/metadata changed'):
        runner.publish_job(directory)


def test_api_key_rotation_allowed_model_drift_rejected(tmp_path, adapters):
    config = tmp_path / 'api.yaml'
    config.write_text('api:\n  model: local-model\n  base_url: http://localhost:1234/v1\n  key: first-key\n')
    directory = make_job(tmp_path, provider='api', config=config)
    def language(request, *args, **kwargs):
        if request['operation'] == 'translate':
            config.write_text(config.read_text().replace('first-key', 'rotated-key'))
        return provider(request, *args, **kwargs)
    assert runner.resume_job(directory, llm=language)['status'] == 'completed'
    assert 'rotated-key' not in (directory / 'workflow.json').read_text()
    assert 'first-key' not in (directory / 'workflow.json').read_text()
    # A fresh unfinished job is bound to its original API model and endpoint.
    other = tmp_path / 'other'
    other.mkdir()
    second = make_job(other, provider='api', config=config)
    config.write_text(config.read_text().replace('local-model', 'changed-model'))
    with pytest.raises(ValueError, match='model/endpoint changed'):
        runner.resume_job(second, llm=provider)


def test_job_lock_excludes_second_process(tmp_path):
    directory = tmp_path / 'locked'
    code = ('from pathlib import Path\nfrom core.workflow.state import job_lock\n'
            'import sys\nwith job_lock(Path(sys.argv[1])):\n    pass\n')
    with job_lock(directory):
        result = subprocess.run([sys.executable, '-c', code, str(directory)], capture_output=True, text=True)
        assert result.returncode != 0
        assert 'already running' in result.stderr
    with job_lock(directory):
        pass


def test_run_settings_and_unrelated_folder_are_protected(tmp_path, adapters):
    source = tmp_path / 'source.mp4'
    source.write_bytes(b'source')
    directory = tmp_path / 'existing'
    directory.mkdir()
    (directory / 'user.txt').write_text('keep')
    with pytest.raises(ValueError, match='must be empty'):
        runner.start_job(str(source), directory, provider='external')
    assert (directory / 'user.txt').read_text() == 'keep'
    empty = tmp_path / 'new'
    assert runner.start_job(str(source), empty, provider='external')['status'] == 'awaiting_translation'
    with pytest.raises(ValueError, match='settings differ'):
        runner.start_job(str(source), empty, provider='external', batch_size=1)

"""A resumable video pipeline: deterministic stages around language tasks."""
from __future__ import annotations

from pathlib import Path
import re
from urllib.parse import urlsplit
import uuid

from .asr import transcribe as _asr
from .state import Job, check_files, digest, file_digest, job_lock, read_json, write_json
from .translation import (batches, call_llm, clean_text, parse_translations,
                          provider_identity, review_schema, seal_request, structural_report,
                          translation_request, translation_schema, validate_request, validate_segments)

ROOT = Path(__file__).resolve().parents[2]


def _source_argument(source):
    url = urlsplit(source)
    if url.scheme in ('http', 'https'):
        from .media import normalize_public_url
        return normalize_public_url(source)
    return str(Path(source).expanduser().resolve(strict=True))


def start_job(source, job_dir, *, subtitles=None, source_language='en', target_language='zh-CN',
              provider='codex', model='gpt-6.1-sol', reasoning='medium', batch_size=40,
              glossary=None, config_path=None, asr=False, max_height=1080):
    if provider not in ('codex', 'api', 'external'):
        raise ValueError('provider must be codex, api or external')
    if not 1 <= batch_size <= 100 or not 144 <= max_height <= 4320:
        raise ValueError('batch_size must be 1..100 and max_height must be 144..4320')
    if not re.fullmatch(r'[a-z]{2,3}(?:-[A-Za-z0-9]+)*', source_language):
        raise ValueError('Choose an explicit source language, such as en')
    if not re.fullmatch(r'[A-Za-z]{2,3}(?:-[A-Za-z0-9]+)*', target_language):
        raise ValueError('Invalid target language')
    if not target_language.lower().startswith('zh'):
        raise ValueError('This bilingual video layout currently supports Chinese translations')
    if reasoning not in ('low', 'medium', 'high', 'xhigh', 'ultra'):
        raise ValueError('Invalid reasoning effort')
    source = _source_argument(source)
    caption_path = str(Path(subtitles).expanduser().resolve(strict=True)) if subtitles else None
    glossary = glossary or []
    if not isinstance(glossary, list) or any(not isinstance(t, dict) or
            not isinstance(t.get('src'), str) or not isinstance(t.get('tgt'), str) for t in glossary):
        raise ValueError('Glossary must be an array of {src, tgt, note?} objects')
    config_path = str(Path(config_path or ROOT / 'config.yaml').expanduser().resolve())
    options = {'source': source, 'subtitles': caption_path,
               'subtitle_sha256': file_digest(Path(caption_path)) if caption_path else None,
               'source_language': source_language, 'target_language': target_language,
               'provider': provider, 'model': model, 'reasoning': reasoning,
               'batch_size': batch_size, 'glossary': glossary,
               'config_path': config_path,
               'provider_identity': provider_identity(provider, model, reasoning, config_path),
               'asr': bool(asr), 'max_height': max_height}
    with job_lock(Path(job_dir)):
        Job.create(Path(job_dir), options)
    return resume_job(job_dir)


def _translations(job):
    path = job.directory / 'translations.json'
    if not path.exists():
        return {}
    saved = read_json(path)
    if saved.get('source_digest') != job.data['source_digest']:
        raise ValueError('Translation checkpoint belongs to different source data')
    return parse_translations({'translations': saved['translations']},
                              {item['id'] for item in saved['translations']})


def _save_translations(job, values):
    write_json(job.directory / 'translations.json', {
        'source_digest': job.data['source_digest'],
        'target_language': job.data['options']['target_language'],
        'translations': [{'id': i, 'text': text} for i, text in sorted(values.items())]})


def _request(job, payload, schema, llm):
    """Cache only validated responses; external responses have the same contract."""
    options = job.data['options']
    if provider_identity(options['provider'], options['model'], options['reasoning'], options['config_path']) != options['provider_identity']:
        raise ValueError('API model/endpoint changed; use a new job for different language settings')
    # The provider's non-secret identity is part of the cache key.
    payload.update(seal_request({**payload, 'provider_identity': options['provider_identity']}))
    validate_request(payload)
    request_id = payload['request_id']
    request_path = job.directory / 'requests' / f'{request_id}.json'
    response_path = job.directory / 'responses' / f'{request_id}.json'
    write_json(request_path, {'instructions': payload['instructions'],
        'request': payload, 'response_schema': schema,
        'response_envelope': {'request_id': request_id, 'source_digest': job.data['source_digest'], 'result': '<JSON result>'}})
    if response_path.exists():
        saved = read_json(response_path)
        if saved.get('source_digest') != job.data['source_digest'] or saved.get('request_id') != request_id:
            raise ValueError('Cached language response has a different source/request')
        return saved['result'], response_path
    if options['provider'] == 'external':
        return None, request_path
    result = llm(payload, schema, provider=options['provider'], work_dir=job.directory,
                 model=options['model'], reasoning=options['reasoning'], config_path=Path(options['config_path']))
    # Validation happens before publication of this cache.
    return result, response_path


def _cache(job, request_id, response, path):
    write_json(path, {'request_id': request_id, 'source_digest': job.data['source_digest'], 'result': response})


def _validate_review(result, batch):
    ids = {s['id'] for s in batch}
    if not isinstance(result, dict) or set(result) != {'issues', 'uncertainties'}:
        raise ValueError('Review requires exactly issues and uncertainties arrays')
    if not isinstance(result['issues'], list) or not isinstance(result['uncertainties'], list):
        raise ValueError('Review issues and uncertainties must be arrays')
    result['uncertainties'] = [clean_text(v, 'review uncertainty') for v in result['uncertainties']]
    seen = set()
    for issue in result['issues']:
        if not isinstance(issue, dict) or set(issue) != {'id', 'reason', 'replacement'}:
            raise ValueError('Each review issue requires id, reason, replacement')
        ident = issue['id']
        if type(ident) is not int or ident not in ids or ident in seen:
            raise ValueError('Review has an unknown or duplicate segment ID')
        seen.add(ident)
        issue['replacement'] = clean_text(issue['replacement'], f'review replacement {ident}')
        issue['reason'] = clean_text(issue['reason'], f'review reason {ident}')
    return result


def _review_progress(job):
    path = job.directory / 'review-progress.json'
    data = read_json(path) if path.exists() else {'source_digest': job.data['source_digest'], 'batches': {}}
    if data.get('source_digest') != job.data['source_digest']:
        raise ValueError('Review checkpoint belongs to different source data')
    return path, data


def _language_stages(job, llm):
    from .layout import fit_translations
    segments, options = job.data['source']['segments'], job.data['options']
    groups = batches(segments, options['batch_size'])
    values = _translations(job)
    known = {s['id'] for s in segments}
    if not set(values).issubset(known):
        raise ValueError('Translation checkpoint contains unknown IDs')
    pending = []
    for group in groups:
        missing = [s for s in group if s['id'] not in values]
        if not missing:
            continue
        payload = translation_request(segments, missing, job.data['source_digest'],
                                      options['target_language'], options['glossary'])
        response, path = _request(job, payload, translation_schema(), llm)
        if response is None:
            pending.append(str(path))
            continue
        parsed = parse_translations(response, {s['id'] for s in missing})
        _cache(job, payload['request_id'], response, path)
        values.update(parsed)
        _save_translations(job, values)
    if pending:
        return job.save('awaiting_translation', pending_requests=pending,
                        translated_segments=len(values), total_segments=len(segments))
    parse_translations({'translations': [{'id': i, 'text': t} for i, t in values.items()]}, known)
    if not fit_translations(job, values, llm, _request, _cache, _save_translations):
        return job.data
    write_json(job.directory / 'structure-check.json', structural_report(segments, values))
    # Review every pair. Numeric checks alone cannot catch 'each' versus 'in total'.
    pending, completed, uncertainties = [], [], []
    progress_path, progress = _review_progress(job)
    for group in groups:
        key = digest([s['id'] for s in group])
        history = progress['batches'].setdefault(key, [])
        visited = set()
        while True:
            payload = {'operation': 'review', 'source_digest': job.data['source_digest'],
                       'target_language': options['target_language'], 'glossary': options['glossary'],
                       'provider_identity': options['provider_identity'],
                       'segments': [{**s, 'translation': values[s['id']]} for s in group]}
            lo, hi = segments.index(group[0]), segments.index(group[-1]) + 1
            payload.update(previous_context=segments[max(0, lo-2):lo], following_context=segments[hi:hi+2])
            payload = seal_request(payload)
            if payload['request_id'] in visited:
                return job.save('needs_revision', pending_requests=[],
                                unresolved_issues=history[-1]['issues'],
                                revision_reason='Semantic corrections cycle between previous translations.')
            visited.add(payload['request_id'])
            recorded = any(record['request_id'] == payload['request_id'] for record in history)
            if len(history) >= 3 and not recorded:
                return job.save('needs_revision', pending_requests=[],
                                unresolved_issues=history[-1]['issues'],
                                revision_reason='Semantic review reached three rounds for this batch; use a new job with corrected input/glossary.')
            response, path = _request(job, payload, review_schema(), llm)
            if response is None:
                pending.append(str(path))
                break
            result = _validate_review(response, group)
            _cache(job, payload['request_id'], result, path)
            if not recorded:
                history.append({'request_id': payload['request_id'], 'issues': result['issues']})
                # Save the consumed review round BEFORE applying its corrections.
                # After a crash the same response can be replayed without consuming
                # another round, and resumes cannot reset the total limit.
                write_json(progress_path, progress)
            if not result['issues']:
                completed.append(payload['request_id'])
                uncertainties.extend(result['uncertainties'])
                break
            if len(history) >= 3:
                return job.save('needs_revision', pending_requests=[],
                                unresolved_issues=result['issues'],
                                revision_reason='Semantic review still reports errors after three rounds.')
            # Corrections do not change IDs/timestamps and are reviewed again.
            before = digest(values)
            for issue in result['issues']:
                values[issue['id']] = issue['replacement']
            if digest(values) == before:
                return job.save('needs_revision', pending_requests=[],
                                unresolved_issues=result['issues'],
                                revision_reason='Reviewer reported an error without a changed translation.')
            _save_translations(job, values)
            if not fit_translations(job, values, llm, _request, _cache, _save_translations):
                return job.data
    if pending:
        return job.save('awaiting_review', pending_requests=pending, translated_segments=len(values),
                        total_segments=len(segments))
    revision = digest(values)
    write_json(job.directory / 'semantic-review.json', {
        'source_digest': job.data['source_digest'],
        'translation_digest': revision, 'request_ids': completed,
        'review_type': 'language_model', 'provider': options['provider'],
        'provider_identity': options['provider_identity'],
        'requested_model': options['provider_identity'].get('model', options['provider_identity'].get('requested_model')),
        'uncertainties': uncertainties, 'human_acceptance': False})
    write_json(job.directory / 'structure-check.json', structural_report(segments, values))
    return job.save('ready_to_render', pending_requests=[], translated_segments=len(values),
                    total_segments=len(segments), translation_digest=revision, uncertainties=uncertainties)


def _check_language(job):
    values = _translations(job)
    if digest(values) != job.data.get('translation_digest'):
        raise ValueError('Reviewed translations changed; use a new job to revise them')
    review = read_json(job.directory / 'semantic-review.json')
    if review.get('source_digest') != job.data['source_digest'] or review.get('translation_digest') != digest(values):
        raise ValueError('Semantic review does not match the current translations')
    return values


def _artifact_hashes(artifacts):
    paths = [Path(v) for v in artifacts.values() if isinstance(v, str) and Path(v).is_file()]
    paths += [Path(p) for p in artifacts.get('previews', [])]
    if not artifacts.get('video') or not Path(artifacts['video']).is_file():
        raise ValueError('Renderer did not produce a verified video')
    return [{'path': str(p), 'sha256': file_digest(p)} for p in dict.fromkeys(paths)]


def _publication_binding(package):
    return digest({k: package[k] for k in ('video_path', 'video_sha256', 'metadata')})


def _check_publication(job):
    package = read_json(job.directory / 'publication.json')
    if _publication_binding(package) != job.data.get('publication_binding'):
        raise ValueError('Publication video/metadata changed after preparation; use a new job')
    if Path(package['video_path']) != Path(job.data['artifacts']['video']):
        raise ValueError('Publication package does not use the verified video')


def resume_job(job_dir, *, llm=None, prepare=None, render=None):
    from .media import prepare_source, render_deliverables
    from .publishing import prepare_publication
    llm, prepare, render = llm or call_llm, prepare or prepare_source, render or render_deliverables
    with job_lock(Path(job_dir)):
        job = Job(job_dir)
        try:
            options = job.data['options']
            if not job.data.get('source'):
                job.save('preparing', error=None)
                result = prepare(options['source'], job.directory, source_language=options['source_language'],
                                 subtitle_path=options['subtitles'], max_height=options['max_height'],
                                 transcriber=_asr if options['asr'] else None)
                result['segments'] = validate_segments(result['segments'])
                result['video_sha256'] = file_digest(Path(result['video_path']))
                signature = digest({'video_sha256': result['video_sha256'], 'segments': result['segments'],
                                    'source_language': options['source_language']})
                job.save('translating', source=result, source_digest=signature)
                job.event('source_prepared', subtitle_source=result['subtitle_source'],
                          segment_count=len(result['segments']))
            job.assert_source()
            if options['subtitles'] and file_digest(Path(options['subtitles'])) != options['subtitle_sha256']:
                raise ValueError('Input subtitles changed after job creation; use a new job')
            if job.data.get('status') == 'completed':
                _check_language(job)
                job.check_artifacts()
                _check_publication(job)
                return job.data
            checkpoint = job.data.get('render_checkpoint')
            if checkpoint:
                _check_language(job)
                if checkpoint['source_digest'] != job.data['source_digest'] or checkpoint['translation_digest'] != job.data['translation_digest']:
                    raise ValueError('Verified render belongs to different source/translations')
                check_files(checkpoint['hashes'])
                artifacts, hashes = checkpoint['artifacts'], checkpoint['hashes']
            else:
                job.save('translating', error=None)
                _language_stages(job, llm)
                if job.data['status'] != 'ready_to_render':
                    return job.data
                values = _check_language(job)
                # Each failed attempt has a fresh directory, preserving its files.
                attempt = job.directory / 'renders' / uuid.uuid4().hex[:12]
                job.save('rendering', rendering_directory=str(attempt))
                artifacts = render(job.data['source']['video_path'], job.data['source']['segments'], values, attempt)
                job.assert_source()
                hashes = _artifact_hashes(artifacts)
                hashes += [{'path': str(job.directory / name), 'sha256': file_digest(job.directory / name)}
                           for name in ('translations.json', 'semantic-review.json', 'structure-check.json', 'review-progress.json', 'layout-progress.json')
                           if (job.directory / name).is_file()]
                checkpoint = {'source_digest': job.data['source_digest'],
                              'translation_digest': job.data['translation_digest'],
                              'artifacts': artifacts, 'hashes': hashes}
                # This checkpoint precedes package creation. A crash after writing
                # publication.json reuses this exact video and never renders a new path.
                job.save('render_verified', render_checkpoint=checkpoint)
                job.event('render_verified', video=artifacts['video'])
            metadata = job.data['source'].get('metadata', {})
            title = clean_text(str(metadata.get('title') or '双语视频'))[:26] + '｜中英字幕'
            author = metadata.get('uploader', '')
            description = '\n'.join(filter(None, [f'原作者：{author}' if author else '',
                f'来源：{metadata.get("source_url", "")}' if metadata.get('source_url') else '',
                '中英双语字幕版。原声和原片内容保留。']))
            package = prepare_publication(job.directory, Path(artifacts['video']), {
                'title': title, 'description': description, 'author': author,
                'source_url': metadata.get('source_url', ''), 'chapters': []})
            return job.save('completed', artifacts=artifacts, artifact_hashes=hashes,
                            publication_stage=package['stage'], publication_binding=_publication_binding(package), error=None)
        except Exception as exc:
            # Provider errors are already sanitized at the boundary. Never retain
            # subprocess commands, environment variables, or raw network responses.
            job.save('failed', error={'type': type(exc).__name__, 'message': str(exc)[:1000]})
            raise


def import_response(job_dir, response_file):
    envelope = read_json(Path(response_file))
    with job_lock(Path(job_dir)):
        job = Job(job_dir)
        job.assert_source()
        if not isinstance(envelope, dict) or set(envelope) != {'request_id', 'source_digest', 'result'}:
            raise ValueError('Response file requires exactly request_id, source_digest and result')
        ident = envelope['request_id']
        if not isinstance(ident, str) or not re.fullmatch(r'[0-9a-f]{64}', ident):
            raise ValueError('Invalid request_id')
        if envelope['source_digest'] != job.data.get('source_digest'):
            raise ValueError('Response source digest does not match this job')
        request_path = job.directory / 'requests' / f'{ident}.json'
        if not request_path.is_file():
            raise ValueError('This job did not request that response')
        payload = read_json(request_path)['request']
        validate_request(payload)
        if payload['request_id'] != ident or payload['source_digest'] != job.data['source_digest']:
            raise ValueError('Request checkpoint is inconsistent')
        if payload['operation'] in ('translate', 'fit_translation'):
            parse_translations(envelope['result'], {s['id'] for s in payload['segments']})
        else:
            _validate_review(envelope['result'], payload['segments'])
        destination = job.directory / 'responses' / f'{ident}.json'
        if destination.exists() and read_json(destination) != envelope:
            raise ValueError('A different response is already saved for this request')
        write_json(destination, envelope)
    return resume_job(job_dir)


def publish_job(job_dir, *, submit=False, headless=False, profile_dir=None):
    from .publishing import publish_douyin
    with job_lock(Path(job_dir)):
        job = Job(job_dir)
        if job.data['status'] != 'completed':
            raise ValueError('Complete and verify the video before uploading')
        job.assert_source()
        _check_language(job)
        job.check_artifacts()
        _check_publication(job)
        result = publish_douyin(Path(job_dir), publish=submit, headless=headless, profile_dir=profile_dir)
        job.save(publication_stage=result['stage'])
        return result

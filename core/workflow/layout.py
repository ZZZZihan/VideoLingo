"""Code detects overflow; language calls are limited to the affected Chinese cues."""
from .media import subtitle_layout_issues
from .state import digest, read_json, write_json
from .translation import batches, parse_translations, seal_request, translation_schema


def fit_translations(job, values, llm, request, cache, save):
    segments, options = job.data['source']['segments'], job.data['options']
    path = job.directory / 'layout-progress.json'
    progress = read_json(path) if path.exists() else {'source_digest': job.data['source_digest'], 'attempts': {}}
    if progress['source_digest'] != job.data['source_digest']:
        raise ValueError('Layout checkpoint belongs to different source data')
    visited = set()
    while True:
        issues = subtitle_layout_issues(segments, values)
        if not issues:
            return True
        source_ids = [s['id'] for s in issues if s['language'] == 'source']
        if source_ids:
            job.save('needs_revision', pending_requests=[], unresolved_layout=issues,
                     revision_reason=f'Original captions are too long for this layout: {source_ids}. Provide more granular source captions in a new job; original speech was not shortened.')
            return False
        signature = digest(values)
        if signature in visited:
            job.save('needs_revision', pending_requests=[], unresolved_layout=issues,
                     revision_reason='Layout revisions cycle without fitting; no content was truncated.')
            return False
        visited.add(signature)
        ids = {s['id'] for s in issues}
        pending = []
        for group in batches([s for s in segments if s['id'] in ids], options['batch_size']):
            lo, hi = segments.index(group[0]), segments.index(group[-1]) + 1
            payload = seal_request({'operation': 'fit_translation', 'source_digest': job.data['source_digest'],
                'target_language': options['target_language'], 'glossary': options['glossary'],
                'provider_identity': options['provider_identity'],
                'previous_context': segments[max(0, lo - 2):lo], 'following_context': segments[hi:hi + 2],
                'segments': [{**s, 'translation': values[s['id']]} for s in group],
                'layout': {'font_size': 48, 'max_lines': 2, 'width_pixels': 1760}})
            histories = [progress['attempts'].setdefault(str(s['id']), []) for s in group]
            if any(len(history) >= 2 and payload['request_id'] not in history for history in histories):
                job.save('needs_revision', pending_requests=[], unresolved_layout=issues,
                         revision_reason='Chinese subtitles still exceed layout after two revisions; no text was dropped.')
                return False
            response, response_path = request(job, payload, translation_schema(), llm)
            if response is None:
                pending.append(str(response_path))
                continue
            changes = parse_translations(response, {s['id'] for s in group})
            cache(job, payload['request_id'], response, response_path)
            for history in histories:
                if payload['request_id'] not in history:
                    history.append(payload['request_id'])
            write_json(path, progress)
            before = digest(values)
            values.update(changes)
            save(job, values)
            if digest(values) == before:
                job.save('needs_revision', pending_requests=[], unresolved_layout=issues,
                         revision_reason='No faithful shorter translation was returned; no content was truncated.')
                return False
        if pending:
            job.save('awaiting_layout', pending_requests=pending, unresolved_layout=issues)
            return False

"""Fixed video workflow; run ``python workflow.py --help`` from the repository."""
import argparse
import json
from pathlib import Path
import sys

from core.workflow.runner import import_response, publish_job, resume_job, start_job
from core.workflow.state import Job, read_json


def parser():
    result = argparse.ArgumentParser(description='VideoLingo: code-controlled bilingual video workflow')
    commands = result.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run', help='Prepare, translate, review, render and verify a video')
    run.add_argument('source', help='Public video URL or existing local video')
    run.add_argument('--job-dir', required=True, type=Path)
    run.add_argument('--subtitles', type=Path, help='Existing original-language SRT or VTT')
    run.add_argument('--source-language', default='en')
    run.add_argument('--target-language', default='zh-CN')
    run.add_argument('--provider', choices=['codex', 'api', 'external'], default='codex')
    run.add_argument('--model', default='gpt-6.1-sol', help='Requested model for Codex CLI')
    run.add_argument('--reasoning', default='medium', choices=['low', 'medium', 'high', 'xhigh', 'ultra'])
    run.add_argument('--batch-size', default=40, type=int)
    run.add_argument('--max-height', default=1080, type=int)
    run.add_argument('--glossary', type=Path, help='JSON array of {src,tgt,note?}')
    run.add_argument('--config', type=Path, help='Existing VideoLingo config.yaml for api provider')
    run.add_argument('--asr', action='store_true', help='If captions are absent, use installed/cached local ASR from repository config.yaml')
    for name in ('resume', 'status'):
        sub = commands.add_parser(name)
        sub.add_argument('--job-dir', required=True, type=Path)
    external = commands.add_parser('import-response', help='Validate an external translation/review response and continue')
    external.add_argument('--job-dir', required=True, type=Path)
    external.add_argument('--file', required=True, type=Path)
    publish = commands.add_parser('publish', help='Upload to Douyin in a separate visible browser; explicit --publish submits once')
    publish.add_argument('--job-dir', required=True, type=Path)
    publish.add_argument('--publish', action='store_true', help='Explicitly submit publication; otherwise upload/fill only')
    publish.add_argument('--profile-dir', type=Path)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == 'run':
            data = start_job(str(args.source), args.job_dir, subtitles=args.subtitles,
                source_language=args.source_language, target_language=args.target_language,
                provider=args.provider, model=args.model, reasoning=args.reasoning,
                batch_size=args.batch_size, glossary=read_json(args.glossary) if args.glossary else None,
                config_path=args.config, asr=args.asr, max_height=args.max_height)
        elif args.command == 'resume':
            data = resume_job(args.job_dir)
        elif args.command == 'status':
            data = Job(args.job_dir).data
            publication = args.job_dir / 'publication.json'
            if publication.is_file():
                data['publication'] = read_json(publication)
        elif args.command == 'import-response':
            data = import_response(args.job_dir, args.file)
        else:
            data = publish_job(args.job_dir, submit=args.publish, profile_dir=args.profile_dir)
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        # Detailed tracebacks may expose provider configuration. The job keeps a
        # bounded, readable stage error instead of dumping SDK/network internals.
        print(json.dumps({'error': type(exc).__name__, 'message': str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

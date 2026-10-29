"""Atomic, content-bound checkpoints for one local video job."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def file_digest(path: Path) -> str:
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path: Path, value):
    """A failed write never replaces a valid checkpoint with partial JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@contextmanager
def job_lock(job_dir: Path):
    """Lock the job, including imports and publication, across CLI processes.

    OS locks release on process exit, so a crash never leaves a stale PID lock.
    """
    job_dir = Path(job_dir).expanduser().resolve()
    job_dir.mkdir(parents=True, exist_ok=True)
    stream = (job_dir / '.lock').open('a+b')
    try:
        if os.name == 'nt':
            import msvcrt
            stream.write(b'\0')
            stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError('This job is already running in another process') from exc
        else:
            import fcntl
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError('This job is already running in another process') from exc
        yield
    finally:
        stream.close()


class Job:
    def __init__(self, directory: Path):
        self.directory = Path(directory).expanduser().resolve()
        self.path = self.directory / 'workflow.json'
        self.data = read_json(self.path)
        if self.data.get('schema_version') != 1:
            raise ValueError('Unsupported workflow checkpoint version')

    @classmethod
    def create(cls, directory: Path, options: dict):
        directory = Path(directory).expanduser().resolve()
        if (directory / 'workflow.json').exists():
            job = cls(directory)
            if job.data['options'] != options:
                raise ValueError('Job inputs/settings differ; use a new job directory')
            return job
        # Never turn an unrelated folder of user assets into a managed job.
        if any(p.name != '.lock' for p in directory.iterdir()):
            raise ValueError('New job directory must be empty')
        write_json(directory / 'workflow.json', {
            'schema_version': 1, 'options': options, 'status': 'preparing',
            'events': [], 'artifacts': {},
        })
        return cls(directory)

    def save(self, status: str | None = None, **updates):
        if status is not None:
            self.data['status'] = status
        self.data.update(updates)
        self.data['updated_at'] = datetime.now(timezone.utc).isoformat()
        write_json(self.path, self.data)
        return self.data

    def event(self, stage: str, **details):
        self.data['events'].append({'stage': stage, 'at': datetime.now(timezone.utc).isoformat(), **details})
        self.save()

    def assert_source(self):
        source = self.data.get('source')
        if source and file_digest(Path(source['video_path'])) != source['video_sha256']:
            raise ValueError('Source video changed after preparation; use a new job')
        if source and digest({'video_sha256': source['video_sha256'], 'segments': source['segments'],
                              'source_language': self.data['options']['source_language']}) != self.data['source_digest']:
            raise ValueError('Prepared source checkpoint changed; use a new job')

    def check_artifacts(self):
        """Completed means the exact verified files are still present."""
        records = self.data.get('artifact_hashes', [])
        if not records:
            raise ValueError('Completed job has no verified artifacts')
        check_files(records)


def check_files(records):
    for record in records:
        path = Path(record['path'])
        if not path.is_file() or file_digest(path) != record['sha256']:
            raise ValueError(f'Verified artifact is missing or changed: {path.name}')

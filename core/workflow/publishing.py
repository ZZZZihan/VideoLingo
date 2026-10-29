"""Conservative, resumable Douyin publication via an optional Playwright browser.

Browser profiles are local credentials: keep them out of jobs and version control.
No network or browser access occurs until ``publish_douyin`` is called.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import subprocess
import secrets
import tempfile
import time
from pathlib import Path
from typing import Any

_PACKAGE = 'publication.json'
_ALLOWED_METADATA = ('title', 'description', 'author', 'source_url', 'chapters', 'cover')
_TERMINAL = {'published', 'publication_unknown', 'upload_unknown', 'publishing'}


def _save(job_dir: Path, data: dict) -> dict:
    job_dir.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.publication-', suffix='.tmp', dir=job_dir)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, job_dir / _PACKAGE)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    return data


def prepare_publication(job_dir: Path, video_path: Path, metadata: dict) -> dict:
    """Create a non-secret publication package without opening a browser.

    Existing packages are preserved, including their recovery state. A different
    video or metadata requires a separate job to avoid losing submission evidence.
    """
    job_dir, video_path = Path(job_dir), Path(video_path).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    clean = {key: metadata[key] for key in _ALLOWED_METADATA if key in metadata}
    if not isinstance(clean.get('title'), str) or not clean['title'].strip():
        raise ValueError('publication title is required')
    with video_path.open('rb') as stream:
        video_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    package_path = job_dir / _PACKAGE
    if package_path.exists():
        existing = json.loads(package_path.read_text(encoding='utf-8'))
        if existing.get('video_path') != str(video_path) or existing.get('metadata') != clean or existing.get('video_sha256') != video_hash:
            raise ValueError('existing publication package differs; use a separate job')
        return existing
    return _save(job_dir, {'version': 1, 'platform': 'douyin', 'stage': 'prepared',
                           'video_path': str(video_path), 'video_sha256': video_hash, 'metadata': clean,
                           'limitations': ['cover and chapters are retained in the package and require manual entry']})


class PlaywrightDouyinAdapter:
    """Best-effort semantic UI adapter; site changes may require manual action."""
    URL = 'https://creator.douyin.com/creator-micro/content/upload'

    def __init__(self, profile_dir: Path, headless: bool, session: dict | None = None):
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError('Install optional playwright and its Chromium browser to use Douyin publishing') from exc
        self.headless = headless
        self._playwright = sync_playwright().start()
        self._process = None
        try:
            # This is a dedicated browser process and profile, never the user's
            # existing browser. It survives the Playwright client disconnecting.
            profile_dir = Path(profile_dir).resolve()
            port_file = profile_dir / 'DevToolsActivePort'
            session_file = profile_dir / '.videolingo-browser-session.json'
            if session is not None:
                saved = json.loads(session_file.read_text(encoding='utf-8'))
                if saved.get('id') != session.get('id') or saved.get('target_id') != session.get('target_id'):
                    raise RuntimeError('Browser session binding differs')
                os.kill(int(saved['pid']), 0)
                port = port_file.read_text().splitlines()[0]
                if not port.isdigit() or not 0 < int(port) < 65536 or port != saved['port']:
                    raise RuntimeError('Browser endpoint no longer matches')
                self.browser = self._playwright.chromium.connect_over_cdp('http://127.0.0.1:' + port)
                self.context = self.browser.contexts[0]
                self.page = None
                for candidate in self.context.pages:
                    client = self.context.new_cdp_session(candidate)
                    try:
                        target = client.send('Target.getTargetInfo')['targetInfo']['targetId']
                    finally:
                        client.detach()
                    if target == session['target_id']:
                        self.page = candidate
                        break
                if self.page is None:
                    raise RuntimeError('The original browser page is no longer available')
                self.binding = session
                return
            pid_file = profile_dir / '.videolingo-browser-pid'
            if port_file.exists():
                try:
                    pid = int(pid_file.read_text())
                    os.kill(pid, 0)
                except ProcessLookupError:
                    port_file.unlink()
                    pid_file.unlink(missing_ok=True)
                except (OSError, ValueError):
                    raise RuntimeError('Browser profile state is unclear; choose another profile')
                else:
                    raise RuntimeError('A browser is already using this profile; complete the open page and close it before retrying')
            executable = self._playwright.chromium.executable_path
            command = [executable, '--user-data-dir=' + str(profile_dir),
                       '--remote-debugging-address=127.0.0.1', '--remote-debugging-port=0',
                       '--no-first-run', '--no-default-browser-check', 'about:blank']
            if headless:
                command.insert(-1, '--headless=new')
            self._process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            fd = os.open(pid_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, 'w') as handle:
                handle.write(str(self._process.pid))
            deadline = time.monotonic() + 15
            while not port_file.is_file() and self._process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.1)
            if not port_file.is_file():
                raise RuntimeError('Unable to start the dedicated browser; check the profile and Chromium installation')
            port = port_file.read_text().splitlines()[0]
            if not port.isdigit() or not 0 < int(port) < 65536:
                raise RuntimeError('Invalid local browser endpoint')
            self.browser = self._playwright.chromium.connect_over_cdp('http://127.0.0.1:' + port)
            self.context = self.browser.contexts[0]
            self.page = self.context.pages[0] if self.context.pages else self.context.new_page()
            self.page.goto(self.URL, wait_until='domcontentloaded', timeout=60000)
            client = self.context.new_cdp_session(self.page)
            try:
                target_id = client.send('Target.getTargetInfo')['targetInfo']['targetId']
            finally:
                client.detach()
            self.binding = {'id': secrets.token_hex(24), 'target_id': target_id}
            fd = os.open(session_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, 'w') as handle:
                json.dump({**self.binding, 'pid': self._process.pid, 'port': port}, handle)
        except BaseException:
            if self._process and self._process.poll() is None:
                self._process.terminate()
            self._playwright.stop()
            raise

    def session(self):
        return self.binding

    def resume_ready(self, metadata, video_path, expected_sha256):
        # A same-tab marker is lost on navigation; verify current form too.
        if self.page.evaluate('() => window.__videolingoUploadSession') != self.binding['id']:
            return False
        title = self.page.get_by_placeholder(re.compile('标题|作品标题')).first
        if title.input_value() != metadata['title']:
            return False
        description = metadata.get('description', '')
        if description and self.page.get_by_placeholder(re.compile('介绍|描述|添加作品简介')).first.input_value() != description:
            return False
        # Read the selected browser File in bounded chunks. Name and size alone
        # cannot distinguish a replacement with different content.
        try:
            field = self.page.locator('input[type=file]').first
            files = field.evaluate('(input) => Array.from(input.files || []).map(f => ({name:f.name, size:f.size}))')
            size = Path(video_path).stat().st_size
            if size <= 0 or files != [{'name': Path(video_path).name, 'size': size}]:
                return False
            field.evaluate('(input) => { window.__videolingoCheckedFile = input.files[0]; }')
            digest = hashlib.sha256()
            chunk_size = 256 * 1024
            for offset in range(0, size, chunk_size):
                encoded = field.evaluate("""async (input, range) => {
                    if (!input.files || input.files.length !== 1 || input.files[0] !== window.__videolingoCheckedFile) throw Error('file unavailable or changed');
                    const bytes = new Uint8Array(await input.files[0].slice(range.start, range.end).arrayBuffer());
                    let binary = '';
                    for (let i = 0; i < bytes.length; i += 8192)
                        binary += String.fromCharCode(...bytes.subarray(i, i + 8192));
                    return btoa(binary);
                }""", {'start': offset, 'end': min(offset + chunk_size, size)})
                chunk = base64.b64decode(encoded, validate=True)
                if len(chunk) != min(chunk_size, size - offset):
                    return False
                digest.update(chunk)
            if digest.hexdigest() != expected_sha256:
                return False
            # Ensure the input was not cleared or changed while the hash was read.
            if not field.evaluate('(input) => input.files && input.files.length === 1 && input.files[0] === window.__videolingoCheckedFile'):
                return False
        except Exception:
            return False
        return self.page.get_by_role('button', name=re.compile('^发布$')).first.is_enabled()

    def close(self):
        # Disconnect only. The separate browser remains available for manual work.
        self._playwright.stop()

    def status(self):
        page = self.page
        if page.get_by_text(re.compile('短信验证|安全验证|验证码')).first.is_visible():
            return 'awaiting_verification'
        if re.search(r'/login', page.url) or page.get_by_text(re.compile('扫码登录|登录抖音|登录后')).first.is_visible():
            return 'needs_login'
        return 'ready'

    def wait_for_ready(self, timeout_seconds=120):
        """Give a person time to log in or complete pre-upload verification."""
        status = self.status()
        if self.headless:
            return status
        for _ in range(timeout_seconds):
            if status == 'ready':
                break
            self.page.wait_for_timeout(1000)
            status = self.status()
        return status

    def wait_for_outcome(self, timeout_seconds=120):
        """Wait for manual verification; final publication remains unconfirmed."""
        result = self.outcome()
        if self.headless or result['stage'] != 'awaiting_verification':
            return result
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            self.page.wait_for_timeout(1000)
            result = self.outcome()
            if result['stage'] != 'awaiting_verification':
                break
        return result

    def upload(self, video_path: str, metadata: dict):
        page = self.page
        page.locator('input[type=file]').first.set_input_files(video_path, timeout=30000)
        # Upload completion and form readiness are explicitly awaited before saving state.
        title = page.get_by_placeholder(re.compile('标题|作品标题')).first
        title.wait_for(state='visible', timeout=600000)
        title.fill(metadata['title'])
        description = metadata.get('description', '')
        if description:
            field = page.get_by_placeholder(re.compile('介绍|描述|添加作品简介')).first
            field.fill(description, timeout=30000)
        button = page.get_by_role('button', name=re.compile('^发布$')).first
        button.wait_for(state='visible', timeout=600000)
        # A visible button alone is insufficient: ensure it is enabled.
        for _ in range(600):
            if button.is_enabled():
                page.evaluate('(id) => { window.__videolingoUploadSession = id; }', self.binding['id'])
                return
            page.wait_for_timeout(1000)
        raise TimeoutError('upload did not become ready')

    def submit(self):
        self.page.get_by_role('button', name=re.compile('^发布$')).first.click(timeout=30000)
        self.page.wait_for_timeout(3000)

    def outcome(self):
        status = self.status()
        if status == 'awaiting_verification':
            return {'stage': status}
        # No verified site receipt ties a displayed URL to this submission.
        return {'stage': 'publication_unknown'}


def publish_douyin(job_dir: Path, *, publish: bool = False, headless: bool = False,
                   profile_dir: Path | None = None, adapter: Any = None) -> dict:
    """Upload and submit at most once; resume only the bound browser page."""
    job_dir = Path(job_dir)
    data = json.loads((job_dir / _PACKAGE).read_text(encoding='utf-8'))
    stage = data['stage']
    if stage == 'published':
        # Older packages may have treated a URL as proof. No verified receipt
        # parser is currently available, so require a manual check.
        data.update(stage='publication_unknown', manual_action_required=True,
                    manual_action='Confirm this work in the creator account; the saved URL is not linked to a verified submission receipt.')
        return _save(job_dir, data)
    if stage == 'uploading':
        data['stage'] = stage = 'upload_unknown'
    if stage == 'publishing':
        data['stage'] = stage = 'publication_unknown'
    observing = stage in {'publication_unknown', 'awaiting_verification'} and data.get('verification_phase') != 'preflight'
    resuming = stage == 'uploaded' or observing
    if stage == 'upload_unknown':
        data.update(manual_action_required=True, manual_action='Inspect the existing browser page; upload will not be repeated automatically.')
        return _save(job_dir, data)
    if stage not in {'prepared', 'needs_login', 'awaiting_verification', 'uploaded', 'publication_unknown'}:
        raise ValueError(f'unsupported publication stage: {stage}')
    if resuming and not data.get('browser_session'):
        data.update(manual_action_required=True, manual_action='The original browser session cannot be verified; inspect the page and creator account manually.')
        return _save(job_dir, data)
    owned = adapter is None
    if owned and headless:
        raise ValueError('Use a visible browser so login, verification and manual continuation remain accessible')
    if owned:
        profile = Path(profile_dir).expanduser() if profile_dir else Path.home() / '.videolingo' / 'douyin-browser-profile'
        profile.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            adapter = PlaywrightDouyinAdapter(profile, headless, data['browser_session'] if resuming else None)
        except Exception:
            if not resuming:
                raise
            data.update(manual_action_required=True, manual_action='The bound browser page is unavailable; inspect the creator account manually before any further action.')
            return _save(job_dir, data)
    try:
        if observing:
            try:
                observed = adapter.outcome()
                data['stage'] = ('awaiting_verification' if observed.get('stage') == 'awaiting_verification'
                                 else 'publication_unknown')
                data.update(manual_action_required=True, manual_action='Check the creator account to confirm publication. A page URL alone cannot establish which work was submitted; no submission was repeated.')
            except Exception:
                data.update(manual_action_required=True, manual_action='The original page cannot be checked; inspect the creator account manually.')
            return _save(job_dir, data)
        if stage == 'uploaded':
            if not publish:
                data.update(manual_action_required=True, manual_action='Review the original browser page; rerun with --publish to submit if the page still matches.')
                return _save(job_dir, data)
            try:
                with Path(data['video_path']).open('rb') as stream:
                    intact = hashlib.file_digest(stream, 'sha256').hexdigest() == data['video_sha256']
                ready = intact and adapter.resume_ready(data['metadata'], data['video_path'], data['video_sha256'])
            except Exception:
                ready = False
            if not ready:
                data.update(manual_action_required=True, manual_action='The original form or video changed; inspect the browser manually. Nothing was submitted.')
                return _save(job_dir, data)
        else:
            status = adapter.wait_for_ready() if hasattr(adapter, 'wait_for_ready') else adapter.status()
            if status in {'needs_login', 'awaiting_verification'}:
                data.update(stage=status, manual_action_required=True,
                            manual_action='Complete login or verification in the open browser, then close that browser and retry with this profile.')
                if status == 'awaiting_verification':
                    data['verification_phase'] = 'preflight'
                return _save(job_dir, data)
            if status != 'ready':
                raise ValueError(f'unexpected browser status: {status}')
            data.pop('manual_action_required', None)
            data.pop('manual_action', None)
            data.pop('verification_phase', None)
            with Path(data['video_path']).open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != data['video_sha256']:
                    raise ValueError('publication video changed; use a new job')
            if hasattr(adapter, 'session'):
                data['browser_session'] = adapter.session()
            data['stage'] = 'uploading'
            _save(job_dir, data)
            try:
                adapter.upload(data['video_path'], data['metadata'])
            except Exception:
                data.update(stage='upload_unknown', manual_action_required=True,
                            manual_action='Inspect the open browser; upload will not be repeated automatically.')
                _save(job_dir, data)
                raise
            data['stage'] = 'uploaded'
            _save(job_dir, data)
            if not publish:
                data.update(manual_action_required=True, manual_action='Review the original browser page; rerun with --publish to submit if the page still matches.')
                return _save(job_dir, data)
        data.pop('manual_action_required', None)
        data.pop('manual_action', None)
        data['stage'] = 'publishing'
        _save(job_dir, data)
        try:
            adapter.submit()
            result = adapter.wait_for_outcome() if hasattr(adapter, 'wait_for_outcome') else adapter.outcome()
            if result.get('stage') == 'awaiting_verification':
                data.update(stage='awaiting_verification', verification_phase='submission', manual_action_required=True,
                            manual_action='Complete verification in the original browser page, then check the creator account.')
            else:
                data.update(stage='publication_unknown', manual_action_required=True,
                            manual_action='Check the creator account to confirm publication; no verified submission receipt is available. Do not submit again until the result is confirmed.')
        except Exception:
            data.update(stage='publication_unknown', manual_action_required=True,
                        manual_action='Inspect the original browser and creator account; do not submit again until the result is confirmed.')
            _save(job_dir, data)
            raise
        return _save(job_dir, data)
    finally:
        if owned:
            adapter.close()

"""Offline CDP lifecycle test against a disposable local page and browser."""
import contextlib
import hashlib
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

import pytest

REQUIRE_BROWSER = os.environ.get('VIDEOLINGO_TEST_REQUIRE_BROWSER') == '1'
if REQUIRE_BROWSER:
    from playwright import sync_api as playwright
else:
    playwright = pytest.importorskip('playwright.sync_api')
from core.workflow.publishing import PlaywrightDouyinAdapter


@pytest.fixture
def local_browser(tmp_path, monkeypatch):
    configured = os.environ.get('VIDEOLINGO_TEST_CHROMIUM')
    if configured:
        executable = Path(configured)
    else:
        with playwright.sync_playwright() as p:
            executable = Path(p.chromium.executable_path)
    if not executable.is_file() or not os.access(executable, os.X_OK):
        if REQUIRE_BROWSER:
            pytest.fail('Required Chromium executable is unavailable: ' + str(executable))
        pytest.skip('Chromium unavailable; set VIDEOLINGO_TEST_CHROMIUM to an installed executable')
    monkeypatch.setattr(playwright.BrowserType, 'executable_path', property(lambda self: str(executable)))
    startup_logs = []
    startup_processes = []
    popen = subprocess.Popen
    def capture_startup(command, **kwargs):
        is_test_browser = '--remote-debugging-port=0' in command
        if is_test_browser:
            log = tempfile.TemporaryFile()
            startup_logs.append(log)
            kwargs['stderr'] = log
        process = popen(command, **kwargs)
        if is_test_browser:
            startup_processes.append(process)
        return process
    monkeypatch.setattr(subprocess, 'Popen', capture_startup)
    original_goto = playwright.Page.goto
    def bounded_goto(page, url, **kwargs):
        kwargs['timeout'] = min(kwargs.get('timeout', 10000), 10000)
        return original_goto(page, url, **kwargs)
    monkeypatch.setattr(playwright.Page, 'goto', bounded_goto)
    monkeypatch.setattr(PlaywrightDouyinAdapter, 'URL', 'about:blank')
    profile = tmp_path / 'profile'
    profile.mkdir()
    adapters = []
    used_ci_wrapper = False
    def create(session=None):
        nonlocal used_ci_wrapper
        try:
            adapter = PlaywrightDouyinAdapter(profile, True, session)
        except RuntimeError as error:
            diagnostic = ''
            if startup_logs:
                startup_logs[-1].seek(0)
                diagnostic = startup_logs[-1].read().decode('utf-8', errors='replace')[-4000:]
            # Ubuntu CI can forbid Chromium user namespaces. Only this trusted
            # about:blank test may retry without the browser sandbox, and only
            # after the actual browser reports that precise startup failure.
            sandbox_unavailable = ('no usable sandbox' in diagnostic.lower()
                                   or 'running as root without --no-sandbox' in diagnostic.lower())
            if (session is None and not used_ci_wrapper and sys.platform == 'linux'
                    and os.environ.get('VIDEOLINGO_TEST_ALLOW_UNSANDBOXED_CHROMIUM') == '1'
                    and sandbox_unavailable):
                print('Local-only CI browser startup diagnostic: ' + diagnostic)
                launcher = tmp_path / 'chromium-local-test'
                launcher.write_text('#!/bin/sh\nexec ' + shlex.quote(str(executable)) + ' --no-sandbox "$@"\n')
                launcher.chmod(0o700)
                monkeypatch.setattr(playwright.BrowserType, 'executable_path', property(lambda self: str(launcher)))
                used_ci_wrapper = True
                adapter = PlaywrightDouyinAdapter(profile, True, session)
            else:
                if diagnostic and session is None:
                    raise RuntimeError(str(error) + '\nBrowser startup diagnostic: ' + diagnostic) from error
                raise
        adapters.append(adapter)
        return adapter
    try:
        yield create
    finally:
        for adapter in adapters:
            with contextlib.suppress(Exception):
                adapter.close()
        for process in startup_processes:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        for log in startup_logs:
            log.close()


def test_browser_reconnects_to_exact_tab_and_detects_changed_form(local_browser, tmp_path):
    video = tmp_path / 'video.mp4'
    video.write_bytes(b'local test media')
    expected_hash = hashlib.sha256(video.read_bytes()).hexdigest()
    metadata = {'title': 'Local test', 'description': 'Local description'}
    first = local_browser()
    first.page.set_content('''<!doctype html><html><body>
        <input type="file"><input placeholder="作品标题"><input placeholder="作品描述">
        <button>发布</button></body></html>''')
    first.upload(str(video), metadata)
    binding = first.session().copy()
    first.page.evaluate("() => { window.testMarker = 'retained'; }")
    first.context.new_page().goto('about:blank')
    first.close()
    resumed = local_browser(binding)
    assert resumed.session() == binding
    assert resumed.page.evaluate('() => window.testMarker') == 'retained'
    assert resumed.resume_ready(metadata, str(video), expected_hash)
    resumed.page.get_by_placeholder('作品标题').fill('Changed title')
    assert not resumed.resume_ready(metadata, str(video), expected_hash)
    resumed.page.get_by_placeholder('作品标题').fill(metadata['title'])
    assert resumed.resume_ready(metadata, str(video), expected_hash)
    different = tmp_path / 'other' / video.name
    different.parent.mkdir()
    different.write_bytes(b'X' * video.stat().st_size)
    resumed.page.locator('input[type=file]').set_input_files(str(different))
    assert not resumed.resume_ready(metadata, str(video), expected_hash)
    resumed.page.locator('input[type=file]').set_input_files([])
    assert not resumed.resume_ready(metadata, str(video), expected_hash)
    resumed.page.close()
    resumed.close()
    with pytest.raises(RuntimeError, match='original browser page'):
        local_browser(binding)

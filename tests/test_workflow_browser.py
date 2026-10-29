"""Offline CDP lifecycle test against a disposable local page and browser."""
import contextlib
import hashlib
import os
from pathlib import Path
import subprocess

import pytest

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
        pytest.skip('Chromium unavailable; set VIDEOLINGO_TEST_CHROMIUM to an installed executable')
    monkeypatch.setattr(playwright.BrowserType, 'executable_path', property(lambda self: str(executable)))
    original_goto = playwright.Page.goto
    def bounded_goto(page, url, **kwargs):
        kwargs['timeout'] = min(kwargs.get('timeout', 10000), 10000)
        return original_goto(page, url, **kwargs)
    monkeypatch.setattr(playwright.Page, 'goto', bounded_goto)
    monkeypatch.setattr(PlaywrightDouyinAdapter, 'URL', 'about:blank')
    profile = tmp_path / 'profile'
    profile.mkdir()
    adapters = []
    process = None
    def create(session=None):
        nonlocal process
        adapter = PlaywrightDouyinAdapter(profile, True, session)
        adapters.append(adapter)
        if adapter._process is not None:
            process = adapter._process
        return adapter
    try:
        yield create
    finally:
        for adapter in adapters:
            with contextlib.suppress(Exception):
                adapter.close()
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


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

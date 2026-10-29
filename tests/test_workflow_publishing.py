import json
from pathlib import Path

import pytest

from core.workflow.publishing import prepare_publication, publish_douyin


class FakeBrowser:
    def __init__(self, status='ready', outcome=None, fail=None):
        self.initial_status = status
        self.result = outcome or {'stage': 'publication_unknown'}
        self.fail = fail
        self.uploads = 0
        self.submissions = 0

    def status(self):
        return self.initial_status

    def upload(self, path, metadata):
        self.uploads += 1
        if self.fail == 'upload':
            raise RuntimeError('upload interrupted')

    def submit(self):
        self.submissions += 1
        if self.fail == 'submit':
            raise RuntimeError('submission interrupted')

    def outcome(self):
        return self.result


@pytest.fixture
def job(tmp_path):
    video = tmp_path / 'video.mp4'
    video.write_bytes(b'video')
    job = tmp_path / 'job'
    prepare_publication(job, video, {'title': 'Title', 'description': 'Text', 'chapters': [{'start': 0}], 'password': 'not saved'})
    return job


def test_prepare_keeps_structured_metadata_and_rejects_changed_package(job, tmp_path):
    data = json.loads((job / 'publication.json').read_text())
    assert data['metadata']['chapters'] == [{'start': 0}]
    assert 'password' not in data['metadata']
    with pytest.raises(ValueError):
        prepare_publication(job, tmp_path / 'video.mp4', {'title': 'Changed'})


def test_default_upload_never_publishes_or_reuploads(job):
    browser = FakeBrowser()
    assert publish_douyin(job, adapter=browser)['stage'] == 'uploaded'
    assert publish_douyin(job, publish=True, adapter=browser)['stage'] == 'uploaded'
    assert (browser.uploads, browser.submissions) == (1, 0)


def test_login_can_be_retried_without_upload(job):
    browser = FakeBrowser(status='needs_login')
    assert publish_douyin(job, adapter=browser)['stage'] == 'needs_login'
    browser.initial_status = 'ready'
    assert publish_douyin(job, adapter=browser)['stage'] == 'uploaded'
    assert browser.uploads == 1


def test_publish_requires_proof_and_never_retries_uncertain_submission(job):
    browser = FakeBrowser(outcome={'stage': 'published'})
    assert publish_douyin(job, publish=True, adapter=browser)['stage'] == 'publication_unknown'
    assert publish_douyin(job, publish=True, adapter=browser)['stage'] == 'publication_unknown'
    assert (browser.uploads, browser.submissions) == (1, 1)


def test_publication_url_alone_is_not_proof(job):
    browser = FakeBrowser(outcome={'stage': 'published', 'publication_url': 'https://www.douyin.com/video/123', 'publication_id': '123'})
    result = publish_douyin(job, publish=True, adapter=browser)
    assert result['stage'] == 'publication_unknown'
    assert result['manual_action_required']


@pytest.mark.parametrize('failure, expected', [('upload', 'upload_unknown'), ('submit', 'publication_unknown')])
def test_interrupted_actions_cannot_repeat(job, failure, expected):
    browser = FakeBrowser(fail=failure)
    with pytest.raises(RuntimeError):
        publish_douyin(job, publish=True, adapter=browser)
    assert publish_douyin(job, publish=True, adapter=browser)['stage'] == expected
    assert browser.uploads == 1
    assert browser.submissions == (1 if failure == 'submit' else 0)


def test_verification_after_submission_never_submits_again(job):
    browser = FakeBrowser(outcome={'stage': 'awaiting_verification'})
    assert publish_douyin(job, publish=True, adapter=browser)['stage'] == 'awaiting_verification'
    assert publish_douyin(job, publish=True, adapter=browser)['stage'] == 'awaiting_verification'
    assert browser.submissions == 1


def test_crash_after_publish_intent_is_unknown(job):
    path = job / 'publication.json'
    data = json.loads(path.read_text())
    data['stage'] = 'publishing'
    path.write_text(json.dumps(data))
    browser = FakeBrowser()
    assert publish_douyin(job, publish=True, adapter=browser)['stage'] == 'publication_unknown'
    assert browser.submissions == 0


def test_verification_remains_open_for_manual_completion(job):
    class WaitingBrowser(FakeBrowser):
        def wait_for_outcome(self):
            # Simulates a user completing verification in the same live page.
            return {'stage': 'published', 'publication_url': 'https://www.douyin.com/video/123', 'publication_id': '123'}
    browser = WaitingBrowser()
    result = publish_douyin(job, publish=True, adapter=browser)
    assert result['stage'] == 'publication_unknown'
    assert (browser.uploads, browser.submissions) == (1, 1)


def test_preflight_verification_can_resume(job):
    browser = FakeBrowser(status='awaiting_verification')
    assert publish_douyin(job, adapter=browser)['verification_phase'] == 'preflight'
    browser.initial_status = 'ready'
    assert publish_douyin(job, adapter=browser)['stage'] == 'uploaded'
    assert browser.uploads == 1


def test_changed_video_is_never_uploaded(job, tmp_path):
    (tmp_path / 'video.mp4').write_bytes(b'different')
    browser = FakeBrowser()
    with pytest.raises(ValueError, match='video changed'):
        publish_douyin(job, adapter=browser)
    assert browser.uploads == 0


def test_owned_browser_is_left_for_manual_completion(job, tmp_path, monkeypatch):
    import core.workflow.publishing as publishing
    instances = []
    class DetachedFake(FakeBrowser):
        def __init__(self, profile, headless, session=None):
            super().__init__()
            self.disconnected = False
            self.page_is_open = True
            instances.append(self)
        def close(self):
            self.disconnected = True  # disconnect does not close the detached browser
    monkeypatch.setattr(publishing, 'PlaywrightDouyinAdapter', DetachedFake)
    result = publish_douyin(job, profile_dir=tmp_path / 'profile')
    assert result['stage'] == 'uploaded' and result['manual_action_required']
    assert instances[0].disconnected and instances[0].page_is_open
    resumed = publish_douyin(job, publish=True, profile_dir=tmp_path / 'profile')
    assert resumed['manual_action_required']
    assert len(instances) == 1 and instances[0].submissions == 0


class ResumableBrowser(FakeBrowser):
    def __init__(self, ready=True, outcome=None):
        super().__init__(outcome=outcome)
        self.ready = ready
        self.observations = 0
    def session(self):
        return {'id': 'session', 'target_id': 'page'}
    def resume_ready(self, metadata, video_path, expected_sha256):
        return self.ready
    def outcome(self):
        self.observations += 1
        return self.result


def test_bound_upload_resumes_with_explicit_submit_once(job):
    browser = ResumableBrowser(outcome={'stage': 'published', 'publication_url': 'https://www.douyin.com/video/123', 'publication_id': '123'})
    assert publish_douyin(job, adapter=browser)['stage'] == 'uploaded'
    assert publish_douyin(job, publish=True, adapter=browser)['stage'] == 'publication_unknown'
    assert publish_douyin(job, publish=True, adapter=browser)['stage'] == 'publication_unknown'
    assert (browser.uploads, browser.submissions) == (1, 1)


def test_changed_form_blocks_resume(job):
    browser = ResumableBrowser()
    publish_douyin(job, adapter=browser)
    browser.ready = False
    result = publish_douyin(job, publish=True, adapter=browser)
    assert result['stage'] == 'uploaded' and result['manual_action_required']
    assert browser.submissions == 0


def test_lost_bound_page_blocks_resume(job, tmp_path, monkeypatch):
    import core.workflow.publishing as publishing
    browser = ResumableBrowser()
    publish_douyin(job, adapter=browser)
    def lost(*args):
        raise RuntimeError('page missing')
    monkeypatch.setattr(publishing, 'PlaywrightDouyinAdapter', lost)
    result = publish_douyin(job, publish=True, profile_dir=tmp_path / 'profile')
    assert result['stage'] == 'uploaded' and result['manual_action_required']
    assert browser.submissions == 0


def test_verification_resume_only_observes_bound_page(job):
    browser = ResumableBrowser(outcome={'stage': 'awaiting_verification'})
    publish_douyin(job, publish=True, adapter=browser)
    assert publish_douyin(job, publish=True, adapter=browser)['stage'] == 'awaiting_verification'
    browser.result = {'stage': 'published', 'publication_url': 'https://www.douyin.com/video/123', 'publication_id': '123'}
    result = publish_douyin(job, publish=True, adapter=browser)
    assert result['stage'] == 'publication_unknown'
    assert result['manual_action_required']
    assert (browser.uploads, browser.submissions) == (1, 1)


def test_crashed_submission_only_observes(job):
    browser = ResumableBrowser()
    publish_douyin(job, adapter=browser)
    path = job / 'publication.json'
    data = json.loads(path.read_text())
    data['stage'] = 'publishing'
    path.write_text(json.dumps(data))
    assert publish_douyin(job, publish=True, adapter=browser)['stage'] == 'publication_unknown'
    assert browser.submissions == 0 and browser.observations == 1


def test_legacy_url_only_published_state_is_not_trusted(job):
    path = job / 'publication.json'
    data = json.loads(path.read_text())
    data.update(stage='published', publication_url='https://www.douyin.com/video/123', publication_id='123')
    path.write_text(json.dumps(data))
    result = publish_douyin(job, adapter=FakeBrowser())
    assert result['stage'] == 'publication_unknown'
    assert result['manual_action_required']

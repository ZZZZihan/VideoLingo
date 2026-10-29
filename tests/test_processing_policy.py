import json
from types import SimpleNamespace

import pandas as pd
import pytest


def test_policy_defaults_and_invalid_values(monkeypatch):
    import core.processing_policy as policy
    settings = {}
    monkeypatch.setattr(policy, 'load_key_or', lambda key, default: settings.get(key, default))
    assert policy.mode() == 'legacy'
    assert policy.enabled('polish', True) is True
    settings['processing_policy.mode'] = 'code_first'
    assert policy.enabled('polish', True) is False
    settings['processing_policy.polish'] = True
    assert policy.enabled('polish', False) is True
    settings['processing_policy.polish'] = 'false'
    with pytest.raises(ValueError, match='boolean'):
        policy.enabled('polish', True)
    settings['processing_policy.mode'] = 'typo'
    with pytest.raises(ValueError, match='Unknown'):
        policy.mode()


def test_code_first_split_uses_rules_and_falls_back_for_impossible_word(monkeypatch):
    import core._3_2_split_meaning as module
    monkeypatch.setattr(module, 'processing_mode', lambda: 'code_first')
    monkeypatch.setattr(module, 'get_source_language', lambda: 'en')
    monkeypatch.setattr(module, 'get_joiner', lambda language: ' ')
    calls = []
    monkeypatch.setattr(module, 'split_sentence', lambda sentence, *args, **kwargs: calls.append(sentence) or sentence)
    # Simulate a tokenizer that treats a long word as multiple pieces.
    def nlp(text):
        words = text.split()
        return [SimpleNamespace(text=x) for word in words for x in (list(word) if word == 'unbreakable' else [word])]

    sentence = 'one two three four five six'
    result = module.parallel_split_sentences([sentence], 3, 1, nlp)
    assert ' '.join(result) == sentence
    assert all(len(nlp(line)) <= 3 for line in result)
    with pytest.raises(ValueError, match='Cannot safely split'):
        module.parallel_split_sentences(['unbreakable'], 3, 1, nlp)
    assert calls == ['unbreakable']


def test_code_first_writes_custom_terms_without_summary(monkeypatch, tmp_path):
    import core._4_1_summarize as module
    import core.processing_policy as policy
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'output' / 'log').mkdir(parents=True)
    pd.DataFrame([['agent', '智能体', '软件']], columns=['src', 'tgt', 'note']).to_excel('custom_terms.xlsx', index=False)
    monkeypatch.setattr(policy, 'load_key_or', lambda key, default: 'code_first' if key.endswith('.mode') else default)
    monkeypatch.setattr(module, 'ask_gpt', lambda *args, **kwargs: pytest.fail('unexpected summary call'))
    module.get_summary()
    result = json.loads((tmp_path / 'output' / 'log' / 'terminology.json').read_text())
    assert result['terms'] == [{'src': 'agent', 'tgt': '智能体', 'note': '软件'}]
    assert module.search_things_to_note_in_prompt('An agent works') is not None


def test_code_first_translation_calls_only_faithful(monkeypatch):
    import core.translate_lines as module
    import core.processing_policy as policy
    monkeypatch.setattr(policy, 'load_key_or', lambda key, default: 'code_first' if key.endswith('.mode') else default)
    monkeypatch.setattr(module, 'load_key', lambda key: True)
    monkeypatch.setattr(module, 'check_cancel', lambda: None)
    monkeypatch.setattr(module, 'generate_shared_prompt', lambda *args: '')
    monkeypatch.setattr(module, 'get_prompt_faithfulness', lambda *args: '')
    calls = []
    def ask(*args, **kwargs):
        calls.append(kwargs['log_title'])
        return {'1': {'origin': 'Hello', 'direct': '你好'}}
    monkeypatch.setattr(module, 'ask_gpt', ask)
    assert module.translate_lines('Hello', None, None, None, None)[0] == '你好'
    assert calls == ['translate_faithfulness']

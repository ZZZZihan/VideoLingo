"""Control which pipeline stages require semantic model calls.

Older configuration files retain their original behaviour. This opt-in policy
controls the existing pipeline; the subtitle CLI has its own fixed stages.
"""
from core.utils import load_key_or


def mode():
    value = load_key_or('processing_policy.mode', 'legacy')
    if value not in ('legacy', 'code_first'):
        raise ValueError(f'Unknown processing policy: {value!r}')
    return value


def enabled(name, legacy_value):
    if mode() == 'legacy':
        return legacy_value
    value = load_key_or(f'processing_policy.{name}', False)
    if not isinstance(value, bool):
        raise ValueError(f'processing_policy.{name} must be a boolean')
    return value

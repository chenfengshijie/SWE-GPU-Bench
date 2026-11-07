"""Token accounting from provider-reported counters; missing is never zero."""
from __future__ import annotations

import json
from pathlib import Path


def response_usage(completion, elapsed_seconds: float) -> dict:
    def get(value, key):
        return value.get(key) if isinstance(value, dict) else getattr(value, key, None)
    def count(value):
        return value if isinstance(value, int) and not isinstance(value, bool) else None
    usage = get(completion, 'usage')
    inp = count(get(usage, 'input_tokens'))
    out = count(get(usage, 'output_tokens'))
    return {'source': 'responses_api', 'response_id': get(completion, 'id') if isinstance(get(completion, 'id'), str) else None,
            'elapsed_seconds': elapsed_seconds, 'input_tokens': inp, 'output_tokens': out,
            'cached_input_tokens': count(get(get(usage, 'input_tokens_details'), 'cached_tokens')),
            'reasoning_tokens': count(get(get(usage, 'output_tokens_details'), 'reasoning_tokens')),
            'total_tokens': count(get(usage, 'total_tokens')) if count(get(usage, 'total_tokens')) is not None else
                (inp + out if inp is not None and out is not None else None)}


def summarize_usage(events: list[dict]) -> dict:
    fields = ['input_tokens', 'output_tokens', 'cached_input_tokens', 'reasoning_tokens', 'total_tokens']
    reported = [e for e in events if e.get('input_tokens') is not None and e.get('output_tokens') is not None]
    result = {'source': 'responses_api', 'status': 'reported' if reported and len(reported) == len(events) else
              ('partial' if reported else 'missing'), 'attempted_calls': len(events), 'reported_calls': len(reported)}
    for field in fields:
        values = [e[field] for e in reported if isinstance(e.get(field), int)]
        result[field] = sum(values) if values else None
    result['api_elapsed_seconds'] = round(sum(e.get('elapsed_seconds', 0) for e in events), 3) if events else None
    return result


def read_usage(path: Path) -> dict:
    events = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return summarize_usage(events)

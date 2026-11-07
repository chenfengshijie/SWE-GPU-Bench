from types import SimpleNamespace
from unittest.mock import Mock, patch

from swe_gpu_bench.shared.usage import response_usage, summarize_usage
from swe_gpu_bench.shared.models import ModelProfile
from swe_gpu_bench.adapters.llm import generate_text_from_profile


def test_provider_usage_includes_cache_and_reasoning_without_double_counting():
    completion = SimpleNamespace(id='r1', usage=SimpleNamespace(input_tokens=100, output_tokens=40,
        total_tokens=140, input_tokens_details=SimpleNamespace(cached_tokens=60),
        output_tokens_details=SimpleNamespace(reasoning_tokens=25)))
    event = response_usage(completion, 1.5)
    result = summarize_usage([event])
    assert result['total_tokens'] == 140
    assert result['cached_input_tokens'] == 60
    assert result['reasoning_tokens'] == 25
    assert result['status'] == 'reported'
    assert summarize_usage([])['input_tokens'] is None
    assert summarize_usage([event, {'error_type': 'TimeoutError'}])['status'] == 'partial'


def test_callback_records_response_and_does_not_retry_accounting_failure():
    client = Mock()
    client.responses.create.return_value = SimpleNamespace(output_text='patch', id='r1',
        usage=SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15))
    profile = ModelProfile(name='x', kind='text', provider='openai_responses', model='x',
                           api_key_env='FIXTURE_KEY', extra_env={'FIXTURE_KEY': 'fixture'})
    events = []
    with patch('openai.OpenAI', return_value=client):
        assert generate_text_from_profile('prompt', profile=profile, usage_callback=events.append) == 'patch'
    assert events[0]['total_tokens'] == 15
    client.responses.create.reset_mock()
    def fail(_):
        raise OSError('disk full')
    import pytest
    with patch('openai.OpenAI', return_value=client), pytest.raises(OSError):
        generate_text_from_profile('prompt', profile=profile, usage_callback=fail)
    assert client.responses.create.call_count == 1

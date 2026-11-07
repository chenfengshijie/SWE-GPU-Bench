from pathlib import Path
from unittest import mock
import pytest
from swe_gpu_bench.shared.config import resolve_profile
from swe_gpu_bench.shared.models import ModelProfile
from swe_gpu_bench.adapters.llm import generate_text_from_profile


def test_high_config_reaches_actual_responses_request(tmp_path):
    config=tmp_path/'profiles.toml'
    config.write_text('[profiles.text.review]\nprovider="openai_responses"\nmodel="gpt-5.6-sol-algorithm"\nreasoning_effort="high"\n')
    profile=resolve_profile(profile_name='review',profile_kind='text',config_path=config)
    client=mock.Mock();client.responses.create.return_value.output_text='ok'
    with mock.patch('openai.OpenAI',return_value=client):
        assert generate_text_from_profile('fixture',profile=profile)=='ok'
    assert client.responses.create.call_args.kwargs['reasoning']=={'effort':'high'}
    assert client.responses.create.call_args.kwargs['model']=='gpt-5.6-sol-algorithm'
    assert resolve_profile(profile_name='review',profile_kind='text',config_path=config,reasoning_effort='medium').reasoning_effort=='medium'


def test_omitted_setting_preserves_default():
    client=mock.Mock();client.responses.create.return_value.output_text='ok'
    with mock.patch('openai.OpenAI',return_value=client):
        generate_text_from_profile('fixture',profile=ModelProfile(name='x',kind='text',provider='openai_responses'))
    assert 'reasoning' not in client.responses.create.call_args.kwargs


def test_explicit_effort_never_silently_disappears_on_retry():
    client=mock.Mock();client.responses.create.side_effect=RuntimeError('unsupported setting')
    profile=ModelProfile(name='x',kind='text',provider='openai_responses',reasoning_effort='high')
    with mock.patch('openai.OpenAI',return_value=client),mock.patch('time.sleep'):
        with pytest.raises(RuntimeError):generate_text_from_profile('fixture',profile=profile)
    assert client.responses.create.call_count==3
    assert all(c.kwargs['reasoning']=={'effort':'high'} for c in client.responses.create.call_args_list)


def test_unknown_provider_does_not_ignore_setting():
    with pytest.raises(ValueError):
        generate_text_from_profile('fixture',profile=ModelProfile(name='x',kind='text',provider='openai_compatible',reasoning_effort='high'))


def test_invalid_setting_rejected():
    with pytest.raises(ValueError):ModelProfile(name='x',kind='text',reasoning_effort='hihg')

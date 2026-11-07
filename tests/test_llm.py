import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from swe_gpu_bench.core import llm
from swe_gpu_bench.core.config import ModelProfile


def make_profile(**overrides) -> ModelProfile:
    values = dict(
        name="oracle_default",
        kind="text",
        provider="openai_compatible",
        model="gpt-5.2",
        base_url="https://oracle.example/v1",
        api_key_env="ORACLE_API_KEY",
        extra_env={"ORACLE_API_KEY": "oracle-secret"},
    )
    values.update(overrides)
    return ModelProfile(**values)


def fake_completion(content: str) -> mock.Mock:
    return mock.Mock(choices=[mock.Mock(message=mock.Mock(content=content))])


class CompleteTests(unittest.TestCase):
    def test_complete_uses_openai_compatible_client_from_profile(self):
        fake_client = mock.Mock()
        fake_client.chat.completions.create.return_value = fake_completion("diff --git a/x b/x\n")

        with mock.patch("openai.OpenAI", return_value=fake_client) as openai_ctor:
            result = llm.complete("patch prompt", profile=make_profile(), max_tokens=2048, env_file=None)

        self.assertEqual(result, "diff --git a/x b/x\n")
        openai_ctor.assert_called_once_with(api_key="oracle-secret", base_url="https://oracle.example/v1")
        call_kwargs = fake_client.chat.completions.create.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "gpt-5.2")
        self.assertEqual(call_kwargs["max_tokens"], 2048)

    def test_complete_reads_api_key_from_env_file(self):
        fake_client = mock.Mock()
        fake_client.chat.completions.create.return_value = fake_completion("ok")
        profile = make_profile(extra_env={})

        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text('ORACLE_API_KEY="file-secret"\n', encoding="utf-8")
            with mock.patch.dict(os.environ, {}, clear=True), \
                 mock.patch("openai.OpenAI", return_value=fake_client) as openai_ctor:
                llm.complete("prompt", profile=profile, env_file=env_file)

        openai_ctor.assert_called_once_with(api_key="file-secret", base_url="https://oracle.example/v1")

    def test_complete_without_profile_uses_openai_environment(self):
        fake_client = mock.Mock()
        fake_client.chat.completions.create.return_value = fake_completion("ok")
        env = {"OPENAI_API_KEY": "env-secret", "OPENAI_BASE_URL": "https://env.example/v1", "OPENAI_MODEL": "gpt-x"}

        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch("openai.OpenAI", return_value=fake_client) as openai_ctor:
            llm.complete("prompt", env_file=None)

        openai_ctor.assert_called_once_with(api_key="env-secret", base_url="https://env.example/v1")
        self.assertEqual(fake_client.chat.completions.create.call_args.kwargs["model"], "gpt-x")

    def test_complete_missing_api_key_raises_without_retry(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch("openai.OpenAI") as openai_ctor:
            with self.assertRaisesRegex(ValueError, "ORACLE_API_KEY"):
                llm.complete("prompt", profile=make_profile(extra_env={}), env_file=None)
        openai_ctor.assert_not_called()

    def test_complete_raises_last_exception_after_retry_exhaustion(self):
        fake_client = mock.Mock()
        fake_client.chat.completions.create.side_effect = RuntimeError("gateway down")

        with mock.patch("openai.OpenAI", return_value=fake_client), mock.patch("time.sleep"):
            with self.assertRaisesRegex(RuntimeError, "gateway down"):
                llm.complete("prompt", profile=make_profile(), env_file=None)

        self.assertEqual(fake_client.chat.completions.create.call_count, 3)

    def test_complete_retries_without_temperature_on_wrapped_error(self):
        fake_client = mock.Mock()
        fake_client.chat.completions.create.side_effect = [
            RuntimeError(
                "litellm.BadRequestError: AzureException BadRequestError - "
                "{\"error\":{\"message\":\"Unsupported parameter: \\'temperature\\' is not supported\"}}"
            ),
            fake_completion('{"edits":[{"path":"x","old":"a","new":"b"}]}'),
        ]

        with mock.patch("openai.OpenAI", return_value=fake_client):
            result = llm.complete("prompt", profile=make_profile(), env_file=None)

        self.assertIn('"edits"', result)
        first_call, second_call = fake_client.chat.completions.create.call_args_list
        self.assertEqual(first_call.kwargs["temperature"], 0)
        self.assertNotIn("temperature", second_call.kwargs)

    def test_complete_maps_context_length_errors(self):
        fake_client = mock.Mock()
        fake_client.chat.completions.create.side_effect = RuntimeError("context_length_exceeded")

        with mock.patch("openai.OpenAI", return_value=fake_client):
            with self.assertRaises(llm.ContextLengthExceededError):
                llm.complete("prompt", profile=make_profile(), env_file=None)


class ResponseHelperTests(unittest.TestCase):
    def test_profile_label_prefers_profile_name_then_model(self):
        self.assertEqual(llm.profile_label(make_profile()), "oracle_default")
        self.assertEqual(llm.profile_label(make_profile(name="inline", model="org/model")), "org_model")

    def test_parse_json_response_extracts_embedded_object(self):
        self.assertEqual(llm.parse_json_response('prefix {"a": 1} suffix'), {"a": 1})
        self.assertEqual(llm.parse_json_response("")["parse_error"], "empty_response")
        self.assertEqual(llm.parse_json_response("nope")["parse_error"], "invalid_json")

    def test_trim_text_marks_truncation(self):
        self.assertEqual(llm.trim_text("abc", 5), "abc")
        self.assertTrue(llm.trim_text("abcdef", 3).startswith("abc\n\n[TRUNCATED]"))


if __name__ == "__main__":
    unittest.main()

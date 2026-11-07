import tempfile
import unittest
from pathlib import Path
from unittest import mock

from swe_gpu_bench.shared.models import ModelProfile


class LlmTargetTests(unittest.TestCase):
    def test_responses_profile_uses_requested_model_and_response_text(self):
        from swe_gpu_bench.adapters import llm
        client = mock.Mock()
        client.responses.create.return_value.output_text = '{"setup_commands":[],"test_commands":["test"]}'
        profile = ModelProfile(name="provider_review", kind="text", provider="openai_responses",
                               model="gpt-5.6-sol-algorithm", base_url="https://gateway.example",
                               api_key_env="FIXTURE_KEY", extra_env={"FIXTURE_KEY": "fixture-key"})
        with mock.patch("openai.OpenAI", return_value=client) as ctor:
            result = llm.generate_text_from_profile("recipe prompt", profile=profile, max_tokens=10000, request_timeout=120)
        self.assertEqual(result, client.responses.create.return_value.output_text)
        ctor.assert_called_once_with(api_key="fixture-key", base_url="https://gateway.example", timeout=120, max_retries=0)
        client.responses.create.assert_called_once_with(model="gpt-5.6-sol-algorithm", input="recipe prompt",
                                                        max_output_tokens=10000, store=False)
        client.chat.completions.create.assert_not_called()

    def test_resolve_generation_target_uses_eval_env_for_kimi(self):
        from swe_gpu_bench.adapters import llm

        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / "eval.env"
            env_file.write_text(
                'MOONSHOT_BASE_URL="https://moonshot.example/v1"\n'
                'MOONSHOT_API_KEY="moonshot-secret"\n',
                encoding="utf-8",
            )

            target = llm.resolve_generation_target("kimi", env_file=env_file)

        self.assertEqual(target.provider, "openai-compatible")
        self.assertEqual(target.model_name, "moonshot/kimi-k2.5")
        self.assertEqual(target.base_url, "https://moonshot.example/v1")
        self.assertEqual(target.api_key, "moonshot-secret")

    def test_generate_text_from_profile_uses_openai_compatible_client(self):
        from swe_gpu_bench.adapters import llm

        fake_client = mock.Mock()
        fake_client.chat.completions.create.return_value.choices = [
            mock.Mock(message=mock.Mock(content="diff --git a/x b/x\n"))
        ]
        profile = ModelProfile(
            name="oracle_default",
            kind="text",
            provider="openai_compatible",
            model="gpt-5.2",
            base_url="https://oracle.example/v1",
            api_key_env="ORACLE_API_KEY",
            extra_env={"ORACLE_API_KEY": "oracle-secret"},
        )

        with mock.patch("openai.OpenAI", return_value=fake_client) as openai_ctor:
            result = llm.generate_text_from_profile(
                "patch prompt",
                profile=profile,
                max_tokens=2048,
            )

        self.assertEqual(result, "diff --git a/x b/x\n")
        openai_ctor.assert_called_once_with(api_key="oracle-secret", base_url="https://oracle.example/v1")
        call_kwargs = fake_client.chat.completions.create.call_args
        self.assertEqual(call_kwargs.kwargs["model"], "gpt-5.2")

    def test_generate_text_from_profile_loads_configured_key_from_env_file(self):
        from swe_gpu_bench.adapters import llm

        fake_client = mock.Mock()
        fake_client.chat.completions.create.return_value.choices = [
            mock.Mock(message=mock.Mock(content='{"edits": []}'))
        ]
        profile = ModelProfile(
            name="gpt5mini_20250807",
            kind="agent",
            provider="litellm",
            model="gpt-5-mini-2025-08-07",
            base_url="https://provider.example/v1",
            api_key_env="PROVIDER_API_KEY",
        )

        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text('PROVIDER_API_KEY="fixture-secret"\n', encoding="utf-8")

            with mock.patch("openai.OpenAI", return_value=fake_client) as openai_ctor:
                llm.generate_text_from_profile(
                    "patch prompt",
                    profile=profile,
                    env_file=env_file,
                )

        openai_ctor.assert_called_once_with(api_key="fixture-secret", base_url="https://provider.example/v1")

    def test_generate_text_routes_glm_models(self):
        from swe_gpu_bench.adapters import llm

        fake_client = mock.Mock()
        fake_client.chat.completions.create.return_value.choices = [
            mock.Mock(message=mock.Mock(content="diff --git a/x b/x\n"))
        ]

        with mock.patch("openai.OpenAI", return_value=fake_client) as openai_ctor:
            result = llm.generate_text(
                "patch prompt",
                model_spec="glm",
                env_overrides={
                    "GLM_BASE_URL": "https://glm.example/v4",
                    "GLM_API_KEY": "glm-secret",
                    "GLM_MODEL": "glm-5",
                },
            )

        self.assertEqual(result, "diff --git a/x b/x\n")
        openai_ctor.assert_called_once_with(api_key="glm-secret", base_url="https://glm.example/v4")
        call_kwargs = fake_client.chat.completions.create.call_args
        self.assertEqual(call_kwargs.kwargs["model"], "glm-5")

    def test_generate_text_raises_last_exception_after_retry_exhaustion(self):
        from swe_gpu_bench.adapters import llm

        fake_client = mock.Mock()
        fake_client.chat.completions.create.side_effect = RuntimeError("gateway down")

        with mock.patch("openai.OpenAI", return_value=fake_client), \
             mock.patch("time.sleep"):
            with self.assertRaisesRegex(RuntimeError, "gateway down"):
                llm.generate_text(
                    "patch prompt",
                    model_spec="glm",
                    env_overrides={
                        "GLM_BASE_URL": "https://glm.example/v4",
                        "GLM_API_KEY": "glm-secret",
                        "GLM_MODEL": "glm-5",
                    },
                )

        self.assertEqual(fake_client.chat.completions.create.call_count, 3)

    def test_generate_text_from_profile_raises_last_exception_after_retry_exhaustion(self):
        from swe_gpu_bench.adapters import llm

        fake_client = mock.Mock()
        fake_client.chat.completions.create.side_effect = RuntimeError("profile gateway down")
        profile = ModelProfile(
            name="oracle_default",
            kind="text",
            provider="openai_compatible",
            model="gpt-5.2",
            base_url="https://oracle.example/v1",
            api_key_env="ORACLE_API_KEY",
            extra_env={"ORACLE_API_KEY": "oracle-secret"},
        )

        with mock.patch("openai.OpenAI", return_value=fake_client), \
             mock.patch("time.sleep"):
            with self.assertRaisesRegex(RuntimeError, "profile gateway down"):
                llm.generate_text_from_profile(
                    "patch prompt",
                    profile=profile,
                    max_tokens=2048,
                )

        self.assertEqual(fake_client.chat.completions.create.call_count, 3)

    def test_profile_timeout_is_bounded_and_error_diagnostics_omit_credentials(self):
        from swe_gpu_bench.adapters import llm
        import contextlib
        import io
        fake_client = mock.Mock()
        fake_client.chat.completions.create.side_effect = RuntimeError("provider echoed fixture-private-key")
        profile = ModelProfile(name="fixture", kind="text", model="fixture",
                               api_key_env="FIXTURE_KEY", extra_env={"FIXTURE_KEY": "fixture-private-key"})
        stderr = io.StringIO()
        with mock.patch("openai.OpenAI", return_value=fake_client) as ctor, \
             mock.patch("time.sleep"), contextlib.redirect_stderr(stderr):
            with self.assertRaises(RuntimeError):
                llm.generate_text_from_profile("prompt", profile=profile, request_timeout=120)
        self.assertEqual(fake_client.chat.completions.create.call_count, 3)
        self.assertEqual(ctor.call_args.kwargs["timeout"], 120)
        self.assertEqual(ctor.call_args.kwargs["max_retries"], 0)
        self.assertNotIn("fixture-private-key", stderr.getvalue())
        self.assertIn("RuntimeError", stderr.getvalue())

    def test_generate_text_from_profile_retries_without_temperature_on_wrapped_error(self):
        from swe_gpu_bench.adapters import llm

        fake_client = mock.Mock()
        fake_client.chat.completions.create.side_effect = [
            RuntimeError(
                "litellm.BadRequestError: AzureException BadRequestError - "
                "{\"error\":{\"message\":\"Unsupported parameter: \\'temperature\\' is not supported\"}}"
            ),
            mock.Mock(choices=[mock.Mock(message=mock.Mock(content='{"edits":[{"path":"x","old":"a","new":"b"}]}'))]),
        ]
        profile = ModelProfile(
            name="gpt51_codex_20251113",
            kind="agent",
            provider="litellm",
            model="gpt-5.1-codex-2025-11-13",
            base_url="https://provider.example/v1",
            api_key_env="PROVIDER_API_KEY",
            extra_env={"PROVIDER_API_KEY": "fixture-secret"},
        )

        with mock.patch("openai.OpenAI", return_value=fake_client):
            result = llm.generate_text_from_profile("patch prompt", profile=profile)

        self.assertIn('"edits"', result)
        first_call = fake_client.chat.completions.create.call_args_list[0]
        second_call = fake_client.chat.completions.create.call_args_list[1]
        self.assertEqual(first_call.kwargs["temperature"], 0)
        self.assertNotIn("temperature", second_call.kwargs)

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from swe_gpu_bench.core.agent import (
    CodexAgentRunner,
    OpenCodeAgentRunner,
    build_agent_runner,
    parse_codex_error,
    parse_opencode_error,
)
from swe_gpu_bench.evaluate.agent_eval import EvalCasePaths
from swe_gpu_bench.core.config import ModelProfile


def make_case_paths(root: Path) -> EvalCasePaths:
    case_dir = root / "case"
    repo_dir = case_dir / "repo"
    inputs_dir = case_dir / "inputs"
    artifacts_dir = case_dir / "artifacts"
    command_logs_dir = case_dir / "command-logs"
    for path in (repo_dir, inputs_dir, artifacts_dir, command_logs_dir):
        path.mkdir(parents=True, exist_ok=True)
    prompt_path = inputs_dir / "prompt.md"
    prompt_path.write_text("prompt\n", encoding="utf-8")
    instance_path = inputs_dir / "instance.json"
    instance_path.write_text("{}\n", encoding="utf-8")
    return EvalCasePaths(
        case_dir=case_dir,
        repo_dir=repo_dir,
        inputs_dir=inputs_dir,
        artifacts_dir=artifacts_dir,
        command_logs_dir=command_logs_dir,
        instance_path=instance_path,
        prompt_path=prompt_path,
        agent_stdout_log=command_logs_dir / "agent.stdout.jsonl",
        agent_stderr_log=command_logs_dir / "agent.stderr.log",
        result_path=case_dir / "result.json",
    )


class EvalAgentsTests(unittest.TestCase):
    def test_build_agent_runner_passes_codex_model_flag(self) -> None:
        profile = ModelProfile(
            name="codex_provider",
            kind="agent",
            framework="codex",
            model="gpt-5.1-codex-2025-11-13",
        )

        runner = build_agent_runner(profile=profile)

        self.assertIsInstance(runner, CodexAgentRunner)

        with tempfile.TemporaryDirectory() as tmp:
            case_paths = make_case_paths(Path(tmp))
            with mock.patch("swe_gpu_bench.core.agent.subprocess.run") as mocked_run:
                mocked_run.return_value = mock.Mock(returncode=0)

                result = runner.invoke(case_paths)

        command = mocked_run.call_args.args[0]
        self.assertIn("--model", command)
        self.assertIn("gpt-5.1-codex-2025-11-13", command)
        self.assertEqual(result.command, command)

    def test_parse_codex_error_returns_nested_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "agent.stdout.jsonl"
            log_path.write_text(
                "\n".join(
                    [
                        json.dumps({"type": "thread.started"}),
                        json.dumps(
                            {
                                "type": "error",
                                "message": json.dumps(
                                    {
                                        "type": "error",
                                        "status": 400,
                                        "error": {
                                            "type": "invalid_request_error",
                                            "message": "The model is not supported for ChatGPT auth.",
                                        },
                                    }
                                ),
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            self.assertEqual(
                parse_codex_error(log_path),
                "The model is not supported for ChatGPT auth.",
            )

    def test_parse_opencode_error_returns_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "agent.stdout.jsonl"
            log_path.write_text(
                json.dumps(
                    {
                        "type": "error",
                        "error": {
                            "name": "UnknownError",
                            "data": {"message": "\"/chat/completions\" cannot be parsed as a URL."},
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            self.assertEqual(
                parse_opencode_error(log_path),
                "\"/chat/completions\" cannot be parsed as a URL.",
            )

    def test_build_agent_runner_uses_profile_driven_opencode_config(self) -> None:
        profile = ModelProfile(
            name="opencode_kimi",
            kind="agent",
            framework="opencode",
            provider="moonshot",
            model="kimi-k2.5",
            base_url="https://moonshot.example/v1",
            api_key_env="MOONSHOT_API_KEY",
        )

        with mock.patch.dict("os.environ", {"MOONSHOT_API_KEY": "test-key"}):
            runner = build_agent_runner(profile=profile)

        self.assertIsInstance(runner, OpenCodeAgentRunner)
        config = json.loads(runner._config_path.read_text(encoding="utf-8"))
        runner._config_path.unlink()
        self.assertEqual(config["permission"]["webfetch"], "deny")
        self.assertEqual(config["permission"]["websearch"], "deny")

    def test_opencode_runner_rejects_invalid_base_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case_paths = make_case_paths(Path(tmp))

            runner = OpenCodeAgentRunner(model="moonshot/kimi-k2.5")
            result = runner.invoke(case_paths, extra_env={"MOONSHOT_BASE_URL": '"/chat/completions"'})

            self.assertEqual(result.exit_code, 2)
            self.assertIn("invalid OpenCode provider configuration", result.error_message or "")

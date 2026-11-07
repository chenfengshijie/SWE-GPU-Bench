import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from swe_gpu_bench.adapters.agent import (
    CodexAgentRunner,
    OpenCodeAgentRunner,
    build_agent_runner,
    parse_codex_error,
    parse_opencode_error,
    parse_opencode_session_stats,
)
from swe_gpu_bench.evaluate.use_cases.eval import EvalCasePaths
from swe_gpu_bench.shared.models import ModelProfile


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
    def test_opencode_isolates_working_directory_and_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = make_case_paths(Path(tmp))
            with mock.patch("swe_gpu_bench.adapters.agent.subprocess.run") as run:
                run.return_value = mock.Mock(returncode=0)
                OpenCodeAgentRunner(model="provider/model").invoke(paths)
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("--dir") + 1], str(paths.case_dir.resolve()))
            env = run.call_args.kwargs["env"]
            self.assertEqual(env["PWD"], str(paths.case_dir.resolve()))
            self.assertTrue(env["XDG_DATA_HOME"].startswith(str(paths.artifacts_dir)))

    def test_opencode_token_accounting_keeps_cached_and_reasoning_subsets(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            step = {"type": "step_finish", "part": {"id": "one", "tokens": {
                "input": 100, "output": 10, "reasoning": 20,
                "cache": {"read": 50, "write": 5}, "total": 185}}}
            path.write_text(json.dumps(step) + "\n" + json.dumps(step) + "\n")
            stats = parse_opencode_session_stats(path)
            self.assertEqual(stats["input_tokens"], 155)
            self.assertEqual(stats["output_tokens"], 30)
            self.assertEqual(stats["total_tokens"], 185)
            self.assertEqual(stats["turns"], 1)
            task = {'type': 'tool_use', 'part': {'tool': 'task', 'state': {'metadata': {'sessionId': 'child'}}}}
            path.write_text(json.dumps(step) + '\n' + json.dumps(task) + '\n')
            nested = parse_opencode_session_stats(path)
            self.assertEqual(nested['usage_status'], 'partial')
            self.assertEqual(nested['subagent_sessions'], 1)
            self.assertEqual(nested['total_tokens'], 185)
            path.write_text(json.dumps({"type": "step_finish", "part": {}}) + "\n")
            self.assertEqual(parse_opencode_session_stats(path)["usage_status"], "missing")
            self.assertIsNone(parse_opencode_session_stats(path)["total_tokens"])

    def test_missing_tool_host_nested_error_is_infrastructure_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'agent.jsonl'
            path.write_text(json.dumps({'type': 'item.completed', 'item': {
                'type': 'error', 'message': 'failed to spawn codex-code-mode-host: host executable was not found'}}) + '\n')
            self.assertIn('codex-code-mode-host', parse_codex_error(path))

    def test_build_agent_runner_passes_codex_model_flag(self) -> None:
        profile = ModelProfile(
            name="codex_provider",
            kind="agent",
            framework="codex",
            model="gpt-5.1-codex-2025-11-13",
            reasoning_effort="high",
        )

        runner = build_agent_runner(profile=profile)

        self.assertIsInstance(runner, CodexAgentRunner)

        with tempfile.TemporaryDirectory() as tmp:
            case_paths = make_case_paths(Path(tmp))
            with mock.patch("swe_gpu_bench.adapters.agent.subprocess.run") as mocked_run:
                mocked_run.return_value = mock.Mock(returncode=0)

                result = runner.invoke(case_paths)

        command = mocked_run.call_args.args[0]
        self.assertIn("--model", command)
        self.assertIn("gpt-5.1-codex-2025-11-13", command)
        self.assertIn('model_reasoning_effort="high"', command)
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

        runner = build_agent_runner(profile=profile)

        self.assertIsInstance(runner, OpenCodeAgentRunner)

    def test_opencode_runner_rejects_invalid_base_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case_paths = make_case_paths(Path(tmp))

            runner = OpenCodeAgentRunner(model="moonshot/kimi-k2.5")
            result = runner.invoke(case_paths, extra_env={"MOONSHOT_BASE_URL": '"/chat/completions"'})

            self.assertEqual(result.exit_code, 2)
            self.assertIn("invalid OpenCode provider configuration", result.error_message or "")


def test_codex_completed_turn_recovers_stream_warning(tmp_path):
    import json
    from swe_gpu_bench.adapters.agent import parse_codex_error
    p = tmp_path / 'agent.jsonl'
    warning = {'type': 'error', 'message': 'Reconnecting... stream disconnected before completion'}
    completed = {'type': 'turn.completed', 'usage': {'input_tokens': 12}}
    for events, expected in [
        ([warning, completed], None),
        ([warning], warning['message']),
        ([completed, warning], warning['message']),
        ([{'type': 'error', 'message': 'unknown fatal error'}, completed], 'unknown fatal error'),
        ([{'type': 'turn.failed', 'error': {'message': 'failed'}}, completed], 'failed'),
    ]:
        p.write_text(''.join(json.dumps(e) + '\n' for e in events))
        assert parse_codex_error(p) == expected


def test_codex_completed_turn_recovers_explicit_tpm_reconnect_only(tmp_path):
    import json
    from swe_gpu_bench.adapters.agent import parse_codex_error
    p = tmp_path / 'agent.jsonl'
    message = 'Reconnecting... 2/5 (rate limit exceeded: Your requests have exceeded token rate limit.)'
    warning = {'type': 'error', 'message': message}
    completed = {'type': 'turn.completed'}
    failed = {'type': 'turn.failed', 'error': {'message': 'quota exhausted'}}
    host = {'type': 'item.completed', 'item': {'type': 'error', 'message': 'codex-code-mode-host failed'}}
    for events, expected in [
        ([warning, completed], None),
        ([warning], message),
        ([completed, warning], message),
        ([warning, failed, completed], 'quota exhausted'),
        ([warning, host, completed], 'codex-code-mode-host failed'),
        ([{'type': 'error', 'message': 'rate limit exceeded'}, completed], 'rate limit exceeded'),
    ]:
        p.write_text(''.join(json.dumps(e) + '\n' for e in events))
        assert parse_codex_error(p) == expected

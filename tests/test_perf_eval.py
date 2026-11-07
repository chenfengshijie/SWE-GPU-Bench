import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from swe_gpu_bench.core.agent import AgentInvocationResult


def make_perf_record() -> dict:
    return {
        "instance_id": "deepspeedai__DeepSpeed__7886",
        "repo": "deepspeedai/DeepSpeed",
        "pr_number": 7886,
        "base_commit": "base123",
        "head_commit": "head456",
        "problem_statement": "Improve the performance regression.",
        "setup_commands": ["setup"],
        "test_commands": ["test"],
        "perf_setup_commands": ["perf-setup"],
        "perf_commands": ["perf"],
        "test_patch": "",
        "perf_context": {"perf_kind": "latency"},
        "verification_status": "ready",
    }


class PerfEvalStatsTests(unittest.TestCase):
    def test_build_perf_evaluation_stats_includes_perf_phases(self):
        from swe_gpu_bench.evaluate.perf_eval import build_perf_evaluation_stats

        stats = build_perf_evaluation_stats(
            setup_records=[{"phase": "setup", "exit_code": 0}],
            test_records=[{"phase": "test", "exit_code": 0}],
            perf_setup_records=[{"phase": "perf_setup", "exit_code": 0}],
            perf_records=[{"phase": "perf", "exit_code": 1}],
            elapsed_seconds=5.0,
        )

        self.assertEqual(stats["total_commands"], 4)
        self.assertEqual(stats["test_commands"], 1)
        self.assertEqual(stats["perf_setup_commands"], 1)
        self.assertEqual(stats["perf_commands"], 1)
        self.assertEqual(stats["failed_commands"], 1)


class PerfEvalExecutionTests(unittest.TestCase):
    def test_render_perf_eval_prompt_uses_template_without_test_file_restriction(self):
        from swe_gpu_bench.evaluate import perf_eval

        repo_root = Path(__file__).resolve().parents[1]
        template = (repo_root / "prompts" / "agent_perf_eval.md").read_text(encoding="utf-8")
        prompt = perf_eval.render_perf_eval_prompt(template, make_perf_record())

        self.assertNotIn("Do not modify existing test files", prompt)

    def test_execute_perf_eval_only_generates_patch(self):
        from swe_gpu_bench.evaluate import perf_eval

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path = root / "perf_dataset.jsonl"
            dataset_path.write_text(json.dumps(make_perf_record()) + "\n", encoding="utf-8")
            prompt_template = root / "agent_perf_eval.md"
            prompt_template.write_text("Template\n", encoding="utf-8")

            class FakeRunner:
                name = "codex"

                def invoke(self, case_paths, extra_env=None):
                    case_paths.agent_stdout_log.write_text("", encoding="utf-8")
                    case_paths.agent_stderr_log.write_text("", encoding="utf-8")
                    return AgentInvocationResult(
                        command=["codex", "exec"],
                        exit_code=0,
                        stdout_path=case_paths.agent_stdout_log,
                        stderr_path=case_paths.agent_stderr_log,
                        elapsed_seconds=1.0,
                        agent_stats={"elapsed_seconds": 1.0},
                    )

            def fake_clone(case_paths, repo_slug, head_sha, pr_number=None, cache_root=None):
                (case_paths.repo_dir / "README.md").write_text("ok\n", encoding="utf-8")

            phases = []

            def fake_run_commands(commands, phase, case_paths):
                phases.append(phase)
                return [
                    {
                        "phase": phase,
                        "command": command,
                        "exit_code": 0,
                        "elapsed_seconds": 0.1,
                        "stdout_log": str(case_paths.command_logs_dir / f"{phase}-01.stdout.log"),
                        "stderr_log": str(case_paths.command_logs_dir / f"{phase}-01.stderr.log"),
                    }
                    for command in commands
                ]

            with mock.patch("swe_gpu_bench.evaluate.perf_eval.clone_or_update_repo", side_effect=fake_clone), \
                 mock.patch("swe_gpu_bench.evaluate.perf_eval.capture_diff", return_value="diff --git a/x b/x\n"), \
                 mock.patch("swe_gpu_bench.evaluate.perf_eval.run_commands", side_effect=fake_run_commands):
                result = perf_eval.execute_perf_eval(
                    instance_id=make_perf_record()["instance_id"],
                    dataset_path=dataset_path,
                    eval_root=root / "PerfEvalResults",
                    work_root=root / "PerfEvalWork",
                    prompt_template_path=prompt_template,
                    agent_runner=FakeRunner(),
                )

            self.assertEqual(result["status"], "generated")
            self.assertEqual(result["patch_capture_status"], "captured")
            self.assertEqual(result["perf_verdict"], "not_judged")
            self.assertEqual(result["evaluation_stats"]["perf_commands"], 1)
            self.assertEqual(result["evaluation_stats"]["perf_setup_commands"], 1)
            self.assertEqual(result["candidate_perf_results"], [])
            self.assertEqual(phases, ["setup", "test", "perf_setup", "perf"])
            case_root = root / "PerfEvalResults" / make_perf_record()["instance_id"]
            self.assertTrue((case_root / "artifacts" / "agent.diff").exists())
            manifest_path = case_root / "artifacts" / "environment_manifest.json"
            self.assertTrue(manifest_path.exists())
            self.assertTrue((case_root / "command-logs" / "agent.stdout.jsonl").exists())
            self.assertFalse((case_root / "repo").exists())
            self.assertFalse((case_root / "artifacts" / ".venv").exists())
            self.assertEqual(result["case_dir"], str(case_root))
            self.assertEqual(result["work_dir"], str(root / "PerfEvalWork" / make_perf_record()["instance_id"]))
            self.assertEqual(result["agent_trajectory_path"], str(case_root / "command-logs" / "agent.stdout.jsonl"))
            self.assertEqual(result["environment_restore_mode"], "full_base_replay")
            self.assertEqual(result["environment_restore_results"]["succeeded"], True)
            self.assertEqual(result["environment_manifest_path"], str(manifest_path))
            prompt_text = (case_root / "inputs" / "prompt.md").read_text(encoding="utf-8")
            self.assertNotIn("The workflow has already executed the following setup commands", prompt_text)
            self.assertIn("environment_manifest.json", prompt_text)
            instance_payload = json.loads((case_root / "inputs" / "instance.json").read_text(encoding="utf-8"))
            self.assertNotIn("test_commands", instance_payload)
            self.assertNotIn("perf_commands", instance_payload)

    def test_execute_perf_eval_marks_empty_patch_when_diff_empty(self):
        from swe_gpu_bench.evaluate import perf_eval

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = make_perf_record()
            dataset_path = root / "perf_dataset.jsonl"
            dataset_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            prompt_template = root / "agent_perf_eval.md"
            prompt_template.write_text("Template\n", encoding="utf-8")

            class FakeRunner:
                name = "codex"

                def invoke(self, case_paths, extra_env=None):
                    return AgentInvocationResult(
                        command=["codex", "exec"],
                        exit_code=0,
                        stdout_path=case_paths.agent_stdout_log,
                        stderr_path=case_paths.agent_stderr_log,
                        elapsed_seconds=1.0,
                        agent_stats={},
                    )

            def fake_clone(case_paths, repo_slug, head_sha, pr_number=None, cache_root=None):
                (case_paths.repo_dir / "README.md").write_text("ok\n", encoding="utf-8")

            def fake_run_commands(commands, phase, case_paths):
                return [
                    {
                        "phase": phase,
                        "command": command,
                        "exit_code": 0,
                        "elapsed_seconds": 0.1,
                        "stdout_log": str(case_paths.command_logs_dir / f"{phase}-01.stdout.log"),
                        "stderr_log": str(case_paths.command_logs_dir / f"{phase}-01.stderr.log"),
                    }
                    for command in commands
                ]

            with mock.patch("swe_gpu_bench.evaluate.perf_eval.clone_or_update_repo", side_effect=fake_clone), \
                 mock.patch("swe_gpu_bench.evaluate.perf_eval.capture_diff", return_value=""), \
                 mock.patch("swe_gpu_bench.evaluate.perf_eval.run_commands", side_effect=fake_run_commands):
                result = perf_eval.execute_perf_eval(
                    instance_id=record["instance_id"],
                    dataset_path=dataset_path,
                    eval_root=root / "PerfEvalResults",
                    work_root=root / "PerfEvalWork",
                    prompt_template_path=prompt_template,
                    agent_runner=FakeRunner(),
                )

            self.assertEqual(result["status"], "generated")
            self.assertEqual(result["patch_capture_status"], "empty")
            self.assertEqual(result["perf_capture_status"], "not_run")
            self.assertEqual(result["candidate_perf_results"], [])

    def test_execute_perf_eval_uses_blocked_restore_mode_without_test_or_perf(self):
        from swe_gpu_bench.evaluate import perf_eval

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = make_perf_record()
            record["verification_status"] = "blocked"
            dataset_path = root / "perf_dataset.jsonl"
            dataset_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            prompt_template = root / "agent_perf_eval.md"
            prompt_template.write_text("Template\n", encoding="utf-8")

            class FakeRunner:
                name = "codex"

                def invoke(self, case_paths, extra_env=None):
                    return AgentInvocationResult(
                        command=["codex", "exec"],
                        exit_code=0,
                        stdout_path=case_paths.agent_stdout_log,
                        stderr_path=case_paths.agent_stderr_log,
                        elapsed_seconds=1.0,
                        agent_stats={},
                    )

            phases = []

            def fake_clone(case_paths, repo_slug, head_sha, pr_number=None, cache_root=None):
                (case_paths.repo_dir / "README.md").write_text("ok\n", encoding="utf-8")

            def fake_run_commands(commands, phase, case_paths):
                phases.append(phase)
                return [
                    {
                        "phase": phase,
                        "command": command,
                        "exit_code": 0,
                        "elapsed_seconds": 0.1,
                        "stdout_log": str(case_paths.command_logs_dir / f"{phase}-01.stdout.log"),
                        "stderr_log": str(case_paths.command_logs_dir / f"{phase}-01.stderr.log"),
                    }
                    for command in commands
                ]

            with mock.patch("swe_gpu_bench.evaluate.perf_eval.clone_or_update_repo", side_effect=fake_clone), \
                 mock.patch("swe_gpu_bench.evaluate.perf_eval.capture_diff", return_value="diff --git a/x b/x\n"), \
                 mock.patch("swe_gpu_bench.evaluate.perf_eval.run_commands", side_effect=fake_run_commands):
                result = perf_eval.execute_perf_eval(
                    instance_id=record["instance_id"],
                    dataset_path=dataset_path,
                    eval_root=root / "PerfEvalResults",
                    work_root=root / "PerfEvalWork",
                    prompt_template_path=prompt_template,
                    agent_runner=FakeRunner(),
                )

            self.assertEqual(phases, ["setup", "perf_setup"])
            self.assertEqual(result["environment_restore_mode"], "setup_and_perf_setup_only")
            self.assertEqual(result["evaluation_stats"]["test_commands"], 0)
            self.assertEqual(result["evaluation_stats"]["perf_commands"], 0)


class PerfEvalFailureTests(unittest.TestCase):
    def test_execute_perf_eval_writes_error_result_when_clone_fails(self):
        from swe_gpu_bench.evaluate import perf_eval

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = make_perf_record()
            dataset_path = root / "perf_dataset.jsonl"
            dataset_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            prompt_template = root / "agent_perf_eval.md"
            prompt_template.write_text("Template\n", encoding="utf-8")

            with mock.patch(
                "swe_gpu_bench.evaluate.perf_eval.clone_or_update_repo",
                side_effect=subprocess.CalledProcessError(128, ["git", "clone"]),
            ):
                result = perf_eval.execute_perf_eval(
                    instance_id=record["instance_id"],
                    dataset_path=dataset_path,
                    eval_root=root / "PerfEvalResults",
                    work_root=root / "PerfEvalWork",
                    prompt_template_path=prompt_template,
                )

            self.assertEqual(result["status"], "error")
            self.assertEqual(result["failure_category"], "REPOSITORY_PREPARATION_FAILURE")

    def test_execute_perf_eval_allows_base_test_failure_and_still_runs_agent(self):
        from swe_gpu_bench.evaluate import perf_eval

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = make_perf_record()
            dataset_path = root / "perf_dataset.jsonl"
            dataset_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            prompt_template = root / "agent_perf_eval.md"
            prompt_template.write_text("Template\n", encoding="utf-8")

            class FakeRunner:
                name = "codex"

                def invoke(self, case_paths, extra_env=None):
                    return AgentInvocationResult(
                        command=["codex", "exec"],
                        exit_code=0,
                        stdout_path=case_paths.agent_stdout_log,
                        stderr_path=case_paths.agent_stderr_log,
                        elapsed_seconds=1.0,
                        agent_stats={},
                    )

            def fake_clone(case_paths, repo_slug, head_sha, pr_number=None, cache_root=None):
                (case_paths.repo_dir / "README.md").write_text("ok\n", encoding="utf-8")

            def fake_run_commands(commands, phase, case_paths):
                exit_code = 1 if phase == "test" else 0
                return [
                    {
                        "phase": phase,
                        "command": command,
                        "exit_code": exit_code,
                        "elapsed_seconds": 0.1,
                        "stdout_log": str(case_paths.command_logs_dir / f"{phase}-01.stdout.log"),
                        "stderr_log": str(case_paths.command_logs_dir / f"{phase}-01.stderr.log"),
                    }
                    for command in commands
                ]

            with mock.patch("swe_gpu_bench.evaluate.perf_eval.clone_or_update_repo", side_effect=fake_clone), \
                 mock.patch("swe_gpu_bench.evaluate.perf_eval.capture_diff", return_value="diff --git a/x b/x\n"), \
                 mock.patch("swe_gpu_bench.evaluate.perf_eval.run_commands", side_effect=fake_run_commands):
                result = perf_eval.execute_perf_eval(
                    instance_id=record["instance_id"],
                    dataset_path=dataset_path,
                    eval_root=root / "PerfEvalResults",
                    work_root=root / "PerfEvalWork",
                    prompt_template_path=prompt_template,
                    agent_runner=FakeRunner(),
                )

            self.assertEqual(result["status"], "generated")
            self.assertEqual(result["patch_capture_status"], "captured")
            self.assertEqual(result["environment_restore_results"]["succeeded"], True)
            self.assertEqual(result["environment_restore_results"]["failed_phase"], None)
            self.assertEqual(result["environment_restore_results"]["soft_failed_phases"], ["test"])

    def test_execute_perf_eval_marks_setup_failure_as_error(self):
        from swe_gpu_bench.evaluate import perf_eval

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = make_perf_record()
            dataset_path = root / "perf_dataset.jsonl"
            dataset_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            prompt_template = root / "agent_perf_eval.md"
            prompt_template.write_text("Template\n", encoding="utf-8")

            class FakeRunner:
                name = "codex"

                def invoke(self, case_paths, extra_env=None):
                    raise AssertionError("agent should not run when setup restore fails")

            def fake_clone(case_paths, repo_slug, head_sha, pr_number=None, cache_root=None):
                (case_paths.repo_dir / "README.md").write_text("ok\n", encoding="utf-8")

            def fake_run_commands(commands, phase, case_paths):
                exit_code = 1 if phase == "setup" else 0
                return [
                    {
                        "phase": phase,
                        "command": command,
                        "exit_code": exit_code,
                        "elapsed_seconds": 0.1,
                        "stdout_log": str(case_paths.command_logs_dir / f"{phase}-01.stdout.log"),
                        "stderr_log": str(case_paths.command_logs_dir / f"{phase}-01.stderr.log"),
                    }
                    for command in commands
                ]

            with mock.patch("swe_gpu_bench.evaluate.perf_eval.clone_or_update_repo", side_effect=fake_clone), \
                 mock.patch("swe_gpu_bench.evaluate.perf_eval.run_commands", side_effect=fake_run_commands):
                result = perf_eval.execute_perf_eval(
                    instance_id=record["instance_id"],
                    dataset_path=dataset_path,
                    eval_root=root / "PerfEvalResults",
                    work_root=root / "PerfEvalWork",
                    prompt_template_path=prompt_template,
                    agent_runner=FakeRunner(),
                )

            self.assertEqual(result["status"], "error")
            self.assertEqual(result["failure_category"], "ENVIRONMENT_RESTORE_FAILURE")
            self.assertEqual(result["patch_capture_status"], "not_run")
            self.assertEqual(result["environment_restore_results"]["failed_phase"], "setup")

    def test_execute_perf_eval_uses_shared_repo_cache(self):
        from swe_gpu_bench.evaluate import perf_eval

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = make_perf_record()
            dataset_path = root / "perf_dataset.jsonl"
            dataset_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            prompt_template = root / "agent_perf_eval.md"
            prompt_template.write_text("Template\n", encoding="utf-8")
            eval_root = root / "PerfEvalResults" / "codex"
            work_root = root / "PerfEvalWork" / "codex"
            captured = {}

            class FakeRunner:
                name = "codex"

                def invoke(self, case_paths, extra_env=None):
                    return AgentInvocationResult(
                        command=["codex", "exec"],
                        exit_code=0,
                        stdout_path=case_paths.agent_stdout_log,
                        stderr_path=case_paths.agent_stderr_log,
                        elapsed_seconds=1.0,
                        agent_stats={},
                    )

            def fake_clone(case_paths, repo_slug, head_sha, pr_number=None, cache_root=None):
                captured["cache_root"] = cache_root
                (case_paths.repo_dir / "README.md").write_text("ok\n", encoding="utf-8")

            with mock.patch("swe_gpu_bench.evaluate.perf_eval.clone_or_update_repo", side_effect=fake_clone), \
                 mock.patch("swe_gpu_bench.evaluate.perf_eval.capture_diff", return_value=""), \
                 mock.patch("swe_gpu_bench.evaluate.perf_eval.run_commands", return_value=[]):
                perf_eval.execute_perf_eval(
                    instance_id=record["instance_id"],
                    dataset_path=dataset_path,
                    eval_root=eval_root,
                    work_root=work_root,
                    prompt_template_path=prompt_template,
                    agent_runner=FakeRunner(),
                )

            self.assertEqual(captured["cache_root"], work_root / "_repo_cache")

    def test_execute_perf_eval_preserves_lightweight_artifacts_only(self):
        from swe_gpu_bench.evaluate import perf_eval

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = make_perf_record()
            dataset_path = root / "perf_dataset.jsonl"
            dataset_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            prompt_template = root / "agent_perf_eval.md"
            prompt_template.write_text("Template\n", encoding="utf-8")

            class FakeRunner:
                name = "codex"

                def invoke(self, case_paths, extra_env=None):
                    (case_paths.artifacts_dir / "notes.txt").write_text("hello\n", encoding="utf-8")
                    (case_paths.artifacts_dir / ".cache").mkdir(parents=True, exist_ok=True)
                    (case_paths.artifacts_dir / ".cache" / "skip.txt").write_text("skip\n", encoding="utf-8")
                    large_file = case_paths.artifacts_dir / "large.bin"
                    large_file.write_bytes(b"x" * (11 * 1024 * 1024))
                    return AgentInvocationResult(
                        command=["codex", "exec"],
                        exit_code=0,
                        stdout_path=case_paths.agent_stdout_log,
                        stderr_path=case_paths.agent_stderr_log,
                        elapsed_seconds=1.0,
                        agent_stats={},
                    )

            def fake_clone(case_paths, repo_slug, head_sha, pr_number=None, cache_root=None):
                (case_paths.repo_dir / "README.md").write_text("ok\n", encoding="utf-8")

            with mock.patch("swe_gpu_bench.evaluate.perf_eval.clone_or_update_repo", side_effect=fake_clone), \
                 mock.patch("swe_gpu_bench.evaluate.perf_eval.capture_diff", return_value="diff --git a/x b/x\n"), \
                 mock.patch("swe_gpu_bench.evaluate.perf_eval.run_commands", return_value=[]):
                result = perf_eval.execute_perf_eval(
                    instance_id=record["instance_id"],
                    dataset_path=dataset_path,
                    eval_root=root / "PerfEvalResults",
                    work_root=root / "PerfEvalWork",
                    prompt_template_path=prompt_template,
                    agent_runner=FakeRunner(),
                )

            saved_artifacts = root / "PerfEvalResults" / record["instance_id"] / "artifacts"
            self.assertTrue((saved_artifacts / "notes.txt").exists())
            self.assertFalse((saved_artifacts / ".cache").exists())
            self.assertFalse((saved_artifacts / "large.bin").exists())
            self.assertIn("notes.txt", result["preserved_artifacts"])


if __name__ == "__main__":
    unittest.main()

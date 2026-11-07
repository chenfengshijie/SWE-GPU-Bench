import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock


def make_record() -> dict:
    return {
        "instance_id": "org__repo__101",
        "repo": "org/repo",
        "pr_number": 101,
        "base_commit": "base123",
        "head_commit": "head456",
        "problem_statement": "Make it faster.",
        "setup_commands": ["setup"],
        "test_commands": ["test"],
        "perf_setup_commands": ["perf-setup"],
        "perf_commands": ["perf"],
        "test_patch": "",
        "perf_kind": "latency",
        "measurement_hint": "ns_per_call",
    }


def write_dataset(path: Path, record: dict) -> None:
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")


def init_git_repo(repo: Path, files: dict[str, str]) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
    for rel_path, content in files.items():
        path = repo / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "init"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def build_patch_from_head(repo: Path, rel_path: str, new_content: str) -> str:
    path = repo / rel_path
    path.write_text(new_content, encoding="utf-8")
    diff = subprocess.run(
        ["git", "diff", "--", rel_path],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    subprocess.run(["git", "checkout", "HEAD", "--", rel_path], cwd=repo, check=True, capture_output=True, text=True)
    return diff


class PerfValidateExecutionTests(unittest.TestCase):
    def test_execute_perf_validate_runs_base_then_patch_then_patched_perf(self):
        from swe_gpu_bench.evaluate import perf_validate
        from swe_gpu_bench.core.patch import PatchCheckResult

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = make_record()
            dataset_path = root / "dataset.jsonl"
            write_dataset(dataset_path, record)
            pred_root = root / "oracle-patches"
            patch_dir = pred_root / record["instance_id"]
            patch_dir.mkdir(parents=True)
            (patch_dir / "pred.diff").write_text("diff --git a/x b/x\n", encoding="utf-8")

            phases = []

            def fake_run(commands, phase, case_paths):
                phases.append(phase)
                return [
                    {
                        "phase": phase,
                        "command": command,
                        "exit_code": 0,
                        "elapsed_seconds": 0.1,
                        "stdout_log": str(case_paths.command_logs_dir / f"{phase}.stdout.log"),
                        "stderr_log": str(case_paths.command_logs_dir / f"{phase}.stderr.log"),
                    }
                    for command in commands
                ]

            with mock.patch("swe_gpu_bench.evaluate.perf_validate.clone_or_update_repo"), \
                 mock.patch("swe_gpu_bench.evaluate.perf_validate.check_patch_applies", return_value=PatchCheckResult("valid")), \
                 mock.patch("swe_gpu_bench.evaluate.perf_validate.apply_pred_patch", return_value=PatchCheckResult("applied")), \
                 mock.patch("swe_gpu_bench.evaluate.agent_eval.run_commands", side_effect=fake_run), \
                 mock.patch("swe_gpu_bench.evaluate.perf_validate.call_perf_judge", return_value={"verdict": "improved", "confidence": "high"}):
                result = perf_validate.execute_perf_validate(
                    instance_id=record["instance_id"],
                    dataset_path=dataset_path,
                    eval_root=root / "eval",
                    patch_root=pred_root,
                    work_root=root / "work",
                    force=True,
                )

            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["perf_status"], "improved")
            self.assertEqual(
                phases,
                ["setup", "base_perf_setup", "base_perf", "test", "patched_perf_setup", "patched_perf"],
            )
            self.assertEqual(result["evaluation_stats"]["perf_commands"], 2)
            self.assertEqual(result["base_perf_results"][0]["command"], "perf")
            self.assertEqual(result["patched_perf_results"][0]["command"], "perf")

    def test_execute_perf_validate_missing_patch_exits_before_commands(self):
        from swe_gpu_bench.evaluate import perf_validate

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = make_record()
            dataset_path = root / "dataset.jsonl"
            write_dataset(dataset_path, record)

            with mock.patch("swe_gpu_bench.evaluate.perf_validate.clone_or_update_repo"), \
                 mock.patch("swe_gpu_bench.evaluate.agent_eval.run_commands") as mocked_run:
                result = perf_validate.execute_perf_validate(
                    instance_id=record["instance_id"],
                    dataset_path=dataset_path,
                    eval_root=root / "eval",
                    patch_root=root / "missing-patches",
                    work_root=root / "work",
                    force=True,
                )

            self.assertEqual(result["status"], "error")
            self.assertEqual(result["failure_category"], "PRED_PATCH_MISSING")
            mocked_run.assert_not_called()

    def test_execute_perf_validate_reads_agent_diff_from_perf_agent_eval_root(self):
        from swe_gpu_bench.evaluate import perf_validate
        from swe_gpu_bench.core.patch import PatchCheckResult

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = make_record()
            dataset_path = root / "dataset.jsonl"
            write_dataset(dataset_path, record)
            agent_eval_root = root / "PerfEvalResults" / "codex"
            agent_case_dir = agent_eval_root / record["instance_id"] / "artifacts"
            agent_case_dir.mkdir(parents=True)
            (agent_case_dir / "agent.diff").write_text("diff --git a/x b/x\n", encoding="utf-8")

            phases = []

            def fake_run(commands, phase, case_paths):
                phases.append(phase)
                return [
                    {
                        "phase": phase,
                        "command": command,
                        "exit_code": 0,
                        "elapsed_seconds": 0.1,
                        "stdout_log": str(case_paths.command_logs_dir / f"{phase}.stdout.log"),
                        "stderr_log": str(case_paths.command_logs_dir / f"{phase}.stderr.log"),
                    }
                    for command in commands
                ]

            with mock.patch("swe_gpu_bench.evaluate.perf_validate.clone_or_update_repo"), \
                 mock.patch("swe_gpu_bench.evaluate.perf_validate.check_patch_applies", return_value=PatchCheckResult("valid")), \
                 mock.patch("swe_gpu_bench.evaluate.perf_validate.apply_pred_patch", return_value=PatchCheckResult("applied")), \
                 mock.patch("swe_gpu_bench.evaluate.agent_eval.run_commands", side_effect=fake_run), \
                 mock.patch("swe_gpu_bench.evaluate.perf_validate.call_perf_judge", return_value={"verdict": "improved", "confidence": "high"}):
                result = perf_validate.execute_perf_validate(
                    instance_id=record["instance_id"],
                    dataset_path=dataset_path,
                    eval_root=root / "validate",
                    patch_root=agent_eval_root,
                    patch_source="agent",
                    work_root=root / "work",
                    force=True,
                )

            self.assertEqual(result["patch_source"], "agent")
            self.assertEqual(result["agent_patch_path"], str(agent_case_dir / "agent.diff"))
            self.assertEqual(result["perf_status"], "improved")
            self.assertEqual(
                phases,
                ["setup", "base_perf_setup", "base_perf", "test", "patched_perf_setup", "patched_perf"],
            )

    def test_execute_perf_validate_skips_patched_perf_when_correctness_fails(self):
        from swe_gpu_bench.evaluate import perf_validate
        from swe_gpu_bench.core.patch import PatchCheckResult

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = make_record()
            dataset_path = root / "dataset.jsonl"
            write_dataset(dataset_path, record)
            pred_root = root / "oracle-patches"
            patch_dir = pred_root / record["instance_id"]
            patch_dir.mkdir(parents=True)
            (patch_dir / "pred.diff").write_text("diff --git a/x b/x\n", encoding="utf-8")

            def fake_run(commands, phase, case_paths):
                exit_code = 1 if phase == "test" else 0
                return [{"phase": phase, "command": command, "exit_code": exit_code, "elapsed_seconds": 0.1} for command in commands]

            with mock.patch("swe_gpu_bench.evaluate.perf_validate.clone_or_update_repo"), \
                 mock.patch("swe_gpu_bench.evaluate.perf_validate.check_patch_applies", return_value=PatchCheckResult("valid")), \
                 mock.patch("swe_gpu_bench.evaluate.perf_validate.apply_pred_patch", return_value=PatchCheckResult("applied")), \
                 mock.patch("swe_gpu_bench.evaluate.agent_eval.run_commands", side_effect=fake_run), \
                 mock.patch("swe_gpu_bench.evaluate.perf_validate.call_perf_judge") as mocked_judge:
                result = perf_validate.execute_perf_validate(
                    instance_id=record["instance_id"],
                    dataset_path=dataset_path,
                    eval_root=root / "eval",
                    patch_root=pred_root,
                    work_root=root / "work",
                    force=True,
                )

            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["perf_status"], "test_failed")
            self.assertEqual(result["patched_perf_results"], [])
            mocked_judge.assert_not_called()

    def test_execute_perf_validate_does_not_judge_failed_perf_command(self):
        from swe_gpu_bench.evaluate import perf_validate
        from swe_gpu_bench.core.patch import PatchCheckResult

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = make_record()
            dataset_path = root / "dataset.jsonl"
            write_dataset(dataset_path, record)
            pred_root = root / "oracle-patches"
            patch_dir = pred_root / record["instance_id"]
            patch_dir.mkdir(parents=True)
            (patch_dir / "pred.diff").write_text("diff --git a/x b/x\n", encoding="utf-8")

            def fake_run(commands, phase, case_paths):
                exit_code = 1 if phase == "patched_perf" else 0
                return [{"phase": phase, "command": command, "exit_code": exit_code, "elapsed_seconds": 0.1} for command in commands]

            with mock.patch("swe_gpu_bench.evaluate.perf_validate.clone_or_update_repo"), \
                 mock.patch("swe_gpu_bench.evaluate.perf_validate.check_patch_applies", return_value=PatchCheckResult("valid")), \
                 mock.patch("swe_gpu_bench.evaluate.perf_validate.apply_pred_patch", return_value=PatchCheckResult("applied")), \
                 mock.patch("swe_gpu_bench.evaluate.agent_eval.run_commands", side_effect=fake_run), \
                 mock.patch("swe_gpu_bench.evaluate.perf_validate.call_perf_judge") as mocked_judge:
                result = perf_validate.execute_perf_validate(
                    instance_id=record["instance_id"],
                    dataset_path=dataset_path,
                    eval_root=root / "eval",
                    patch_root=pred_root,
                    work_root=root / "work",
                    force=True,
                )

            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["perf_status"], "unclear")
            mocked_judge.assert_not_called()

    def test_execute_perf_validate_restores_test_patch_targets_before_apply(self):
        from swe_gpu_bench.evaluate import perf_validate
        from swe_gpu_bench.core.patch import PatchCheckResult

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = make_record()
            seed_repo = root / "repo-seed"
            init_git_repo(
                seed_repo,
                {
                    "src/main.py": "x = 1\n",
                    "tests/test_perf.py": "assert 1\n",
                },
            )
            record["test_patch"] = build_patch_from_head(seed_repo, "tests/test_perf.py", "assert 2\n")
            dataset_path = root / "dataset.jsonl"
            write_dataset(dataset_path, record)
            pred_root = root / "oracle-patches"
            patch_dir = pred_root / record["instance_id"]
            patch_dir.mkdir(parents=True)
            (patch_dir / "pred.diff").write_text("diff --git a/src/main.py b/src/main.py\n", encoding="utf-8")

            def fake_clone(case_paths, repo_slug, head_sha, pr_number=None, cache_root=None):
                init_git_repo(
                    case_paths.repo_dir,
                    {
                        "src/main.py": "x = 1\n",
                        "tests/test_perf.py": "assert 1\n",
                    },
                )

            def fake_apply_pred_patch(repo_dir, patch_text):
                (repo_dir / "tests" / "test_perf.py").write_text("assert 999\n", encoding="utf-8")
                return PatchCheckResult("applied")

            def fake_run(commands, phase, case_paths):
                return [
                    {
                        "phase": phase,
                        "command": command,
                        "exit_code": 0,
                        "elapsed_seconds": 0.1,
                        "stdout_log": str(case_paths.command_logs_dir / f"{phase}.stdout.log"),
                        "stderr_log": str(case_paths.command_logs_dir / f"{phase}.stderr.log"),
                    }
                    for command in commands
                ]

            with mock.patch("swe_gpu_bench.evaluate.perf_validate.clone_or_update_repo", side_effect=fake_clone), \
                 mock.patch("swe_gpu_bench.evaluate.perf_validate.check_patch_applies", return_value=PatchCheckResult("valid")), \
                 mock.patch("swe_gpu_bench.evaluate.perf_validate.apply_pred_patch", side_effect=fake_apply_pred_patch), \
                 mock.patch("swe_gpu_bench.evaluate.agent_eval.run_commands", side_effect=fake_run), \
                 mock.patch("swe_gpu_bench.evaluate.perf_validate.call_perf_judge", return_value={"verdict": "improved", "confidence": "high"}):
                result = perf_validate.execute_perf_validate(
                    instance_id=record["instance_id"],
                    dataset_path=dataset_path,
                    eval_root=root / "eval",
                    patch_root=pred_root,
                    work_root=root / "work",
                    force=True,
                )

            repo_dir = Path(result["work_dir"]) / "repo"
            self.assertEqual(result["status"], "passed")
            self.assertNotIn(result["test_patch_status"], {"restore_failed", "apply_failed"})
            self.assertEqual((repo_dir / "tests" / "test_perf.py").read_text(encoding="utf-8"), "assert 2\n")


class PerfValidateFailureTests(unittest.TestCase):
    def test_execute_perf_validate_reports_repository_preparation_failure(self):
        from swe_gpu_bench.evaluate import perf_validate

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = make_record()
            dataset_path = root / "dataset.jsonl"
            write_dataset(dataset_path, record)
            patch_dir = root / "oracle-patches" / record["instance_id"]
            patch_dir.mkdir(parents=True)
            (patch_dir / "pred.diff").write_text("diff --git a/x b/x\n", encoding="utf-8")

            with mock.patch(
                "swe_gpu_bench.evaluate.perf_validate.clone_or_update_repo",
                side_effect=subprocess.CalledProcessError(128, ["git", "clone"]),
            ), mock.patch("swe_gpu_bench.evaluate.agent_eval.run_commands") as mocked_run:
                result = perf_validate.execute_perf_validate(
                    instance_id=record["instance_id"],
                    dataset_path=dataset_path,
                    eval_root=root / "eval",
                    patch_root=root / "oracle-patches",
                    work_root=root / "work",
                    force=True,
                )

            self.assertEqual(result["status"], "error")
            self.assertEqual(result["failure_category"], "REPOSITORY_PREPARATION_FAILURE")
            mocked_run.assert_not_called()
            self.assertTrue((root / "eval" / record["instance_id"] / "result.json").exists())


if __name__ == "__main__":
    unittest.main()

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from swe_gpu_bench.core.patch import PatchCheckResult
from swe_gpu_bench.generate.verify import (
    build_verification_stats,
    build_verification_result,
    execute_verification,
    execute_dataset_verification,
    load_case_inputs,
    prepare_verification_case_directory,
    run_commit_aware_commands,
    run_dataset_verification_batch,
    run_recorded_commands,
)
from swe_gpu_bench.cli import main
from swe_gpu_bench.core.selection import select_records


def make_source_case(root: Path, name: str = "xformers-pr104") -> Path:
    case_dir = root / "TestCases" / name
    (case_dir / "inputs").mkdir(parents=True)
    (case_dir / "artifacts").mkdir(parents=True)
    sample = {
        "repo": "facebookresearch/xformers",
        "number": 104,
        "head_sha": "70c70dc3436cc8b1e34f15746e42b1bafa073bfe",
    }
    original_result = {
        "status": "reproduced",
        "setup_commands": [
            "cd repo && uv venv .venv",
            "cd repo && source .venv/bin/activate && uv pip install pkg",
        ],
        "test_commands": [
            "cd repo && source .venv/bin/activate && pytest tests/test_attentions.py::test_causal",
            "cd repo && source .venv/bin/activate && pytest tests/test_favor.py::test_favor_approximation_accuracy",
        ],
    }
    (case_dir / "inputs" / "sample.json").write_text(
        json.dumps(sample), encoding="utf-8"
    )
    (case_dir / "artifacts" / "result.json").write_text(
        json.dumps(original_result), encoding="utf-8"
    )
    return case_dir


def make_dataset_record(
    instance_id: str = "acme__project__7",
    repo: str = "acme/project",
    pr_number: int = 7,
    base_commit: str = "base123",
    head_commit: str = "head123",
    setup_commands=None,
    test_commands=None,
):
    if setup_commands is None:
        setup_commands = ["cd repo", "pwd"]
    if test_commands is None:
        test_commands = ["cd repo", "pwd"]
    return {
        "instance_id": instance_id,
        "dataset": "AgentExploreCmd",
        "repo": repo,
        "pr_number": pr_number,
        "base_commit": base_commit,
        "head_commit": head_commit,
        "setup_commands": list(setup_commands),
        "test_commands": list(test_commands),
        "verification_status": "reproduced",
    }


def write_dataset(path: Path, records) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return path


class LoadCaseInputsTests(unittest.TestCase):
    def test_load_case_inputs_reads_sample_and_original_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_case = make_source_case(Path(tmp))

            sample, original_result = load_case_inputs(source_case)

            self.assertEqual(sample["repo"], "facebookresearch/xformers")
            self.assertEqual(original_result["status"], "reproduced")
            self.assertEqual(len(original_result["setup_commands"]), 2)

    def test_load_case_inputs_rejects_missing_required_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_case = make_source_case(Path(tmp))
            (source_case / "artifacts" / "result.json").write_text(
                json.dumps({"status": "reproduced", "setup_commands": []}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "test_commands"):
                load_case_inputs(source_case)

    def test_load_case_inputs_normalizes_legacy_absolute_case_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_case = make_source_case(Path(tmp))
            (source_case / "artifacts" / "result.json").write_text(
                json.dumps(
                    {
                        "status": "reproduced",
                        "setup_commands": [
                            "uv venv /data/workspace/TestCases/xformers-pr104/artifacts/.venv"
                        ],
                        "test_commands": [
                            "cd /data/workspace/TestCases/xformers-pr104/repo && pytest -q"
                        ],
                    }
                ),
                encoding="utf-8",
            )

            _, original_result = load_case_inputs(source_case)

            self.assertEqual(
                original_result["setup_commands"],
                ["uv venv $PWD/artifacts/.venv"],
            )
            self.assertEqual(
                original_result["test_commands"],
                ["cd $PWD/repo && pytest -q"],
            )


class PrepareVerificationCaseTests(unittest.TestCase):
    def test_prepare_verification_case_directory_copies_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_case = make_source_case(root)
            sample, original_result = load_case_inputs(source_case)

            case_paths = prepare_verification_case_directory(
                benchmark_root=root / "BenchMarkTest",
                source_case_dir=source_case,
                sample=sample,
                original_result=original_result,
                force=False,
            )

            self.assertEqual(case_paths.case_dir.name, "xformers-pr104")
            self.assertTrue((case_paths.case_dir / "repo").exists())
            self.assertEqual(
                json.loads((case_paths.inputs_dir / "sample.json").read_text()),
                sample,
            )
            self.assertEqual(
                json.loads((case_paths.inputs_dir / "original_result.json").read_text()),
                original_result,
            )


class CommandExecutionTests(unittest.TestCase):
    def test_run_recorded_commands_continues_after_setup_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_case = make_source_case(root)
            sample, original_result = load_case_inputs(source_case)
            case_paths = prepare_verification_case_directory(
                benchmark_root=root / "BenchMarkTest",
                source_case_dir=source_case,
                sample=sample,
                original_result=original_result,
                force=False,
            )

            calls = []

            def fake_run(cmd, cwd, env, stdout, stderr, text, check):
                calls.append(cmd)
                stdout.write("stdout\n")
                stderr.write("stderr\n")
                return subprocess.CompletedProcess(cmd, 1 if len(calls) == 2 else 0)

            with mock.patch(
                "swe_gpu_bench.core.shell.subprocess.run",
                side_effect=fake_run,
            ):
                records = run_recorded_commands(case_paths, original_result)

            self.assertEqual(
                [record["phase"] for record in records],
                ["setup", "setup", "test", "test"],
            )
            self.assertEqual(records[1]["exit_code"], 1)
            self.assertEqual(records[-1]["exit_code"], 0)
            self.assertIn("elapsed_seconds", records[0])
            self.assertTrue(Path(records[0]["stdout_log"]).exists())

    def test_run_recorded_commands_runs_all_tests_after_successful_setup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_case = make_source_case(root)
            sample, original_result = load_case_inputs(source_case)
            case_paths = prepare_verification_case_directory(
                benchmark_root=root / "BenchMarkTest",
                source_case_dir=source_case,
                sample=sample,
                original_result=original_result,
                force=False,
            )

            def fake_run(cmd, cwd, env, stdout, stderr, text, check):
                stdout.write("ok\n")
                stderr.write("")
                return subprocess.CompletedProcess(cmd, 0)

            with mock.patch(
                "swe_gpu_bench.core.shell.subprocess.run",
                side_effect=fake_run,
            ):
                records = run_recorded_commands(case_paths, original_result)

            self.assertEqual(
                [record["phase"] for record in records],
                ["setup", "setup", "test", "test"],
            )


class BuildVerificationResultTests(unittest.TestCase):
    def test_build_verification_stats_summarizes_commands(self):
        stats = build_verification_stats(
            [
                {"phase": "setup", "exit_code": 0},
                {"phase": "setup", "exit_code": 1},
                {"phase": "test", "exit_code": 0},
            ],
            elapsed_seconds=3.21,
        )

        self.assertEqual(
            stats,
            {
                "elapsed_seconds": 3.21,
                "total_commands": 3,
                "setup_commands": 2,
                "test_commands": 1,
                "failed_commands": 1,
                "last_exit_code": 0,
            },
        )

    def test_build_verification_result_marks_verified_and_matches_original(self):
        sample = {
            "repo": "facebookresearch/xformers",
            "number": 104,
            "head_sha": "abc",
        }
        original_result = {
            "status": "reproduced",
            "setup_commands": ["setup"],
            "test_commands": ["test"],
        }
        command_records = [
            {"phase": "setup", "command": "setup", "exit_code": 0},
            {"phase": "test", "command": "test", "exit_code": 0},
        ]

        result = build_verification_result(
            sample=sample,
            source_case_dir=Path("/tmp/TestCases/xformers-pr104"),
            benchmark_case_dir=Path("/tmp/BenchMarkTest/xformers-pr104"),
            original_result=original_result,
            command_records=command_records,
            elapsed_seconds=1.5,
        )

        self.assertEqual(result["verification_status"], "verified")
        self.assertTrue(result["comparison"]["matches_original_verdict"])
        self.assertEqual(
            result["verification_stats"],
            {
                "elapsed_seconds": 1.5,
                "total_commands": 2,
                "setup_commands": 1,
                "test_commands": 1,
                "failed_commands": 0,
                "last_exit_code": 0,
            },
        )

    def test_build_verification_result_fails_when_an_earlier_test_fails(self):
        sample = {
            "repo": "facebookresearch/xformers",
            "number": 104,
            "head_sha": "abc",
        }
        original_result = {
            "status": "failed",
            "setup_commands": ["setup"],
            "test_commands": ["test-1", "test-2"],
        }
        command_records = [
            {"phase": "setup", "command": "setup", "exit_code": 0},
            {"phase": "test", "command": "test-1", "exit_code": 2},
            {"phase": "test", "command": "test-2", "exit_code": 0},
        ]

        result = build_verification_result(
            sample=sample,
            source_case_dir=Path("/tmp/TestCases/xformers-pr104"),
            benchmark_case_dir=Path("/tmp/BenchMarkTest/xformers-pr104"),
            original_result=original_result,
            command_records=command_records,
        )

        self.assertEqual(result["verification_status"], "test_failed")
        self.assertTrue(result["comparison"]["matches_original_verdict"])

    def test_build_verification_result_reports_setup_failure_even_when_tests_pass(self):
        sample = {
            "repo": "facebookresearch/xformers",
            "number": 104,
            "head_sha": "abc",
        }
        original_result = {
            "status": "reproduced",
            "setup_commands": ["setup-1", "setup-2"],
            "test_commands": ["test"],
        }
        command_records = [
            {"phase": "setup", "command": "setup-1", "exit_code": 1},
            {"phase": "setup", "command": "setup-2", "exit_code": 0},
            {"phase": "test", "command": "test", "exit_code": 0},
        ]

        result = build_verification_result(
            sample=sample,
            source_case_dir=Path("/tmp/TestCases/xformers-pr104"),
            benchmark_case_dir=Path("/tmp/BenchMarkTest/xformers-pr104"),
            original_result=original_result,
            command_records=command_records,
        )

        self.assertEqual(result["verification_status"], "setup_failed")


class ExecuteVerificationTests(unittest.TestCase):
    def test_execute_verification_writes_result_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_case = make_source_case(root)

            def fake_clone(case_paths, repo_slug, head_sha, pr_number, cache_root=None):
                (case_paths.repo_dir / "README.md").write_text(
                    "%s@%s\n" % (repo_slug, head_sha), encoding="utf-8"
                )

            with mock.patch(
                "swe_gpu_bench.generate.verify.run_recorded_commands",
                return_value=[
                    {"phase": "setup", "command": "setup", "exit_code": 0},
                    {"phase": "test", "command": "test", "exit_code": 0},
                ],
            ):
                result = execute_verification(
                    source_case_dir=source_case,
                    benchmark_root=root / "BenchMarkTest",
                    force=False,
                    clone_repo_fn=fake_clone,
                )

            result_path = root / "BenchMarkTest" / "xformers-pr104" / "artifacts" / "result.json"
            self.assertTrue(result_path.exists())
            self.assertEqual(result["verification_status"], "verified")
            self.assertIn("verification_stats", result)
            self.assertEqual(result["verification_stats"]["total_commands"], 2)

    def test_execute_verification_returns_repository_preparation_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_case = make_source_case(root)

            def failing_clone(case_paths, repo_slug, head_sha, pr_number):
                raise subprocess.CalledProcessError(1, ["git", "clone"])

            result = execute_verification(
                source_case_dir=source_case,
                benchmark_root=root / "BenchMarkTest",
                force=False,
                clone_repo_fn=failing_clone,
            )

            self.assertEqual(result["verification_status"], "repository_preparation_failed")
            self.assertIn("verification_stats", result)
            self.assertEqual(result["verification_stats"]["total_commands"], 0)


class DatasetVerificationTests(unittest.TestCase):
    def test_select_records_selects_requested_instances_and_skips_existing_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path = write_dataset(
                root / "data" / "dataset" / "agentexplore_reproduced_full.jsonl",
                [
                    make_dataset_record("acme__project__7", pr_number=7),
                    make_dataset_record("acme__project__8", pr_number=8),
                    make_dataset_record("acme__project__9", pr_number=9),
                ],
            )
            eval_root = root / "AgentExploreEval"
            (eval_root / "acme__project__8").mkdir(parents=True)
            (eval_root / "acme__project__8" / "result.json").write_text("{}", encoding="utf-8")
            instance_id_file = root / "instance_ids.txt"
            instance_id_file.write_text("acme__project__9\n", encoding="utf-8")

            def done(record):
                return (eval_root / record["instance_id"] / "result.json").exists()

            records, skipped = select_records(
                dataset_path,
                instance_ids=["acme__project__8"],
                instance_id_file=instance_id_file,
                done=done,
            )

            self.assertEqual([record["instance_id"] for record in records], ["acme__project__9"])
            self.assertEqual(skipped, 1)

            forced_records, _ = select_records(dataset_path, all_instances=True)

            self.assertEqual(
                [record["instance_id"] for record in forced_records],
                ["acme__project__7", "acme__project__8", "acme__project__9"],
            )

    def make_verification_case(self, root: Path):
        source_case = make_source_case(root)
        sample, original_result = load_case_inputs(source_case)
        return prepare_verification_case_directory(
            benchmark_root=root / "BenchMarkTest",
            source_case_dir=source_case,
            sample=sample,
            original_result=original_result,
            force=False,
        )

    def test_run_commit_aware_commands_applies_test_patch_then_gold_patch_on_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_paths = self.make_verification_case(Path(tmp))
            record = {
                **make_dataset_record(setup_commands=["cd repo", "pwd"], test_commands=["cd repo", "pwd"]),
                "patch": "gold",
                "test_patch": "tests",
            }
            checkouts = []
            subprocess_cwds = []
            events = []

            def fake_checkout(repo_dir, commit):
                checkouts.append((repo_dir, commit))

            def fake_apply_test_patch(test_patch, repo_dir):
                events.append(("test_patch", test_patch))
                return "applied_clean"

            def fake_apply_patch(repo_dir, patch):
                events.append(("patch", patch))
                return PatchCheckResult(status="applied")

            def fake_run(cmd, cwd, env, stdout, stderr, text, check):
                subprocess_cwds.append(Path(cwd))
                phase = Path(stdout.name).name.rsplit("-", 1)[0]
                events.append(("run", phase))
                return subprocess.CompletedProcess(cmd, 1 if phase == "base_test" else 0)

            with mock.patch("swe_gpu_bench.core.shell.subprocess.run", side_effect=fake_run):
                phase_results = run_commit_aware_commands(
                    case_paths=case_paths,
                    record=record,
                    checkout_commit_fn=fake_checkout,
                    apply_test_patch_fn=fake_apply_test_patch,
                    apply_patch_fn=fake_apply_patch,
                )

            self.assertEqual(checkouts, [(case_paths.repo_dir, "base123")])
            self.assertEqual(
                events,
                [("run", "setup"), ("test_patch", "tests"), ("run", "base_test"), ("patch", "gold"), ("run", "gold_test")],
            )
            self.assertEqual(phase_results["test_patch_status"], "applied_clean")
            self.assertEqual(phase_results["patch_status"], "applied")
            self.assertEqual(phase_results["post_patch_setup_results"], [])
            self.assertEqual(subprocess_cwds, [case_paths.repo_dir.resolve()] * 3)

    def test_run_commit_aware_commands_rebuilds_before_gold_tests_when_setup_builds_repo(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_paths = self.make_verification_case(Path(tmp))
            record = {
                **make_dataset_record(setup_commands=["uv pip install -e repo"], test_commands=["pytest"]),
                "patch": "gold",
                "test_patch": "",
            }
            phases = []

            def fake_run(cmd, cwd, env, stdout, stderr, text, check):
                phase = Path(stdout.name).name.rsplit("-", 1)[0]
                phases.append(phase)
                return subprocess.CompletedProcess(cmd, 1 if phase == "base_test" else 0)

            with mock.patch("swe_gpu_bench.core.shell.subprocess.run", side_effect=fake_run):
                phase_results = run_commit_aware_commands(
                    case_paths=case_paths,
                    record=record,
                    checkout_commit_fn=lambda repo_dir, commit: None,
                    apply_patch_fn=lambda repo_dir, patch: PatchCheckResult(status="applied"),
                )

            self.assertEqual(phases, ["setup", "base_test", "post_patch_setup", "gold_test"])
            self.assertEqual(phase_results["test_patch_status"], "no_test_patch")

    def test_run_commit_aware_commands_stops_after_setup_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_paths = self.make_verification_case(Path(tmp))
            record = {
                **make_dataset_record(setup_commands=["echo setup-1", "echo setup-2"], test_commands=["echo test-1"]),
                "patch": "gold",
                "test_patch": "tests",
            }
            exit_codes = iter([1, 0])
            apply_test_patch_fn = mock.Mock()

            def fake_run(cmd, cwd, env, stdout, stderr, text, check):
                return subprocess.CompletedProcess(cmd, next(exit_codes))

            with mock.patch("swe_gpu_bench.core.shell.subprocess.run", side_effect=fake_run):
                phase_results = run_commit_aware_commands(
                    case_paths=case_paths,
                    record=record,
                    checkout_commit_fn=lambda repo_dir, commit: None,
                    apply_test_patch_fn=apply_test_patch_fn,
                )

            self.assertEqual([item["exit_code"] for item in phase_results["setup_results"]], [1, 0])
            self.assertEqual(phase_results["base_test_results"], [])
            self.assertEqual(phase_results["test_patch_status"], "not_run")
            apply_test_patch_fn.assert_not_called()

    def test_run_commit_aware_commands_skips_gold_patch_when_base_tests_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_paths = self.make_verification_case(Path(tmp))
            record = {**make_dataset_record(test_commands=["echo test-1"]), "patch": "gold", "test_patch": "tests"}
            apply_patch_fn = mock.Mock()

            def fake_run(cmd, cwd, env, stdout, stderr, text, check):
                return subprocess.CompletedProcess(cmd, 0)

            with mock.patch("swe_gpu_bench.core.shell.subprocess.run", side_effect=fake_run):
                phase_results = run_commit_aware_commands(
                    case_paths=case_paths,
                    record=record,
                    checkout_commit_fn=lambda repo_dir, commit: None,
                    apply_test_patch_fn=lambda test_patch, repo_dir: "applied_clean",
                    apply_patch_fn=apply_patch_fn,
                )

            self.assertEqual([item["exit_code"] for item in phase_results["base_test_results"]], [0])
            self.assertEqual(phase_results["patch_status"], "not_run")
            apply_patch_fn.assert_not_called()

    def test_execute_dataset_verification_writes_invalid_input_result_and_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_root = root / "AgentExploreEval"
            work_root = root / "AgentExploreWork"
            result = execute_dataset_verification(
                record={
                    "instance_id": "acme__project__7",
                    "base_commit": "base123",
                    "head_commit": "head123",
                },
                eval_root=eval_root,
                work_root=work_root,
                force=False,
            )

            result_path = eval_root / "acme__project__7" / "result.json"
            self.assertTrue(result_path.exists())
            self.assertEqual(result["verification_status"], "invalid_input")
            self.assertEqual(result["cleanup"]["repo_removed"], True)
            self.assertEqual(result["cleanup"]["artifacts_removed"], True)
            self.assertFalse((work_root / "acme__project__7" / "repo").exists())
            self.assertFalse((work_root / "acme__project__7" / "artifacts").exists())

    def test_execute_dataset_verification_reports_gold_patch_that_does_not_apply(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_root = root / "AgentExploreEval"
            work_root = root / "AgentExploreWork"
            record = {**make_dataset_record(test_commands=["echo test"]), "patch": "gold", "test_patch": "tests"}

            def fake_clone(case_paths, repo_slug, head_sha, pr_number=None, cache_root=None):
                case_paths.repo_dir.mkdir(parents=True, exist_ok=True)
                case_paths.artifacts_dir.mkdir(parents=True, exist_ok=True)

            def fake_run(cmd, cwd, env, stdout, stderr, text, check):
                return subprocess.CompletedProcess(cmd, 1 if "base_test" in stdout.name else 0)

            with mock.patch("swe_gpu_bench.core.shell.subprocess.run", side_effect=fake_run), \
                 mock.patch("swe_gpu_bench.generate.verify.apply_test_patch", return_value="applied_clean"), \
                 mock.patch(
                     "swe_gpu_bench.generate.verify.apply_pred_patch",
                     return_value=PatchCheckResult(status="apply_failed"),
                 ):
                result = execute_dataset_verification(
                    record=record,
                    eval_root=eval_root,
                    work_root=work_root,
                    force=False,
                    clone_repo_fn=fake_clone,
                    checkout_commit_fn=lambda repo_dir, commit: None,
                )

            self.assertEqual(result["verification_status"], "patch_apply_failed")
            self.assertEqual(result["test_patch_status"], "applied_clean")
            self.assertEqual([item["phase"] for item in result["commands"]], ["setup", "setup", "base_test"])
            self.assertTrue((eval_root / "acme__project__7" / "result.json").exists())
            self.assertEqual(result["cleanup"]["repo_removed"], True)
            self.assertEqual(result["cleanup"]["artifacts_removed"], True)

    def test_execute_dataset_verification_verifies_fail_to_pass_with_real_patches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed = root / "seed"
            seed.mkdir()

            def git(repo, *args):
                return subprocess.run(
                    ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", *args],
                    cwd=repo, check=True, capture_output=True, text=True,
                ).stdout

            git(seed, "init")
            (seed / "value.txt").write_text("1\n", encoding="utf-8")
            git(seed, "add", ".")
            git(seed, "commit", "-m", "base")
            (seed / "value.txt").write_text("2\n", encoding="utf-8")
            gold_patch = git(seed, "diff")
            git(seed, "checkout", "--", "value.txt")
            (seed / "tests").mkdir()
            (seed / "tests" / "check.sh").write_text('test "$(cat value.txt)" = 2\n', encoding="utf-8")
            git(seed, "add", "-N", "tests/check.sh")
            test_patch = git(seed, "diff")
            git(seed, "reset", "--", "tests/check.sh")
            record = {
                **make_dataset_record(setup_commands=["true"], test_commands=["cd repo && bash tests/check.sh"]),
                "patch": gold_patch,
                "test_patch": test_patch,
            }

            def fake_clone(case_paths, repo_slug, head_sha, pr_number=None, cache_root=None):
                shutil.rmtree(case_paths.repo_dir)
                git(root, "clone", "--quiet", str(seed), str(case_paths.repo_dir))

            result = execute_dataset_verification(
                record=record,
                eval_root=root / "VerifyEval",
                work_root=root / "VerifyWork",
                force=False,
                clone_repo_fn=fake_clone,
                checkout_commit_fn=lambda repo_dir, commit: None,
            )

            self.assertEqual(result["summary"], "Verification status: verified")
            self.assertEqual([item["exit_code"] for item in result["base_test_results"]], [1])
            self.assertEqual([item["exit_code"] for item in result["gold_test_results"]], [0])

    def test_run_dataset_verification_batch_skips_existing_results_without_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path = write_dataset(
                root / "data" / "dataset" / "agentexplore_reproduced_full.jsonl",
                [
                    make_dataset_record("acme__project__7", pr_number=7),
                    make_dataset_record("acme__project__8", pr_number=8),
                ],
            )
            eval_root = root / "AgentExploreEval"
            work_root = root / "AgentExploreWork"
            (eval_root / "acme__project__7").mkdir(parents=True)
            (eval_root / "acme__project__7" / "result.json").write_text("{}", encoding="utf-8")
            processed = []

            def fake_execute(record, eval_root, work_root, force, clone_repo_fn=None, checkout_commit_fn=None):
                processed.append((record["instance_id"], force))
                return {
                    "instance_id": record["instance_id"],
                    "verification_status": "verified",
                }

            with mock.patch(
                "swe_gpu_bench.generate.verify.execute_dataset_verification",
                side_effect=fake_execute,
            ):
                summary = run_dataset_verification_batch(
                    dataset_path=dataset_path,
                    eval_root=eval_root,
                    work_root=work_root,
                    all_instances=True,
                    concurrency=1,
                    force=False,
                )

            self.assertEqual(processed, [("acme__project__8", False)])
            self.assertEqual(summary["counts"]["skipped"], 1)
            self.assertEqual(summary["counts"]["processed"], 1)


class CliTests(unittest.TestCase):
    def test_cli_forwards_case_dir_arguments_to_execute_verification(self):
        with mock.patch("swe_gpu_bench.generate.verify.execute_verification") as execute:
            execute.return_value = {"verification_status": "verified"}

            exit_code = main(
                [
                    "generate",
                    "verify",
                    "--case-dir", "/tmp/TestCases/xformers-pr104",
                    "--benchmark-root", "/tmp/BenchMarkTest",
                    "--force",
                ]
            )

        self.assertEqual(exit_code, 0)
        kwargs = execute.call_args.kwargs
        self.assertEqual(kwargs["source_case_dir"], Path("/tmp/TestCases/xformers-pr104"))
        self.assertEqual(kwargs["benchmark_root"], Path("/tmp/BenchMarkTest"))
        self.assertTrue(kwargs["force"])

    def test_cli_forwards_dataset_batch_arguments(self):
        with mock.patch("swe_gpu_bench.generate.verify.run_dataset_verification_batch") as batch_execute:
            batch_execute.return_value = {"counts": {"processed": 1}}

            exit_code = main(
                [
                    "generate",
                    "verify",
                    "--dataset", "/tmp/data/dataset/swe_gpu_bench.jsonl",
                    "--eval-root", "/tmp/VerifyEval",
                    "--work-root", "/tmp/VerifyWork",
                    "--instance-id", "acme__project__7,acme__project__8",
                    "--instance-id", "acme__project__9",
                    "--concurrency", "8",
                    "--force",
                ]
            )

        self.assertEqual(exit_code, 0)
        kwargs = batch_execute.call_args.kwargs
        self.assertEqual(kwargs["dataset_path"], Path("/tmp/data/dataset/swe_gpu_bench.jsonl"))
        self.assertEqual(kwargs["eval_root"], Path("/tmp/VerifyEval"))
        self.assertEqual(kwargs["work_root"], Path("/tmp/VerifyWork"))
        self.assertEqual(kwargs["instance_ids"], ["acme__project__7,acme__project__8", "acme__project__9"])
        self.assertEqual(kwargs["concurrency"], 8)
        self.assertTrue(kwargs["force"])

    def test_cli_requires_selection_for_dataset_mode(self):
        with self.assertRaises(SystemExit):
            main(["generate", "verify", "--dataset", "/tmp/data.jsonl"])

    def test_cli_rejects_selection_with_case_dir(self):
        with self.assertRaises(SystemExit):
            main(["generate", "verify", "--case-dir", "/tmp/case", "--all"])


if __name__ == "__main__":
    unittest.main()

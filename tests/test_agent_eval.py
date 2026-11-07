import json
import subprocess
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]

from swe_gpu_bench.core.agent import AgentInvocationResult, OpenCodeAgentRunner
from swe_gpu_bench.core.patch import apply_test_patch, extract_test_patch_paths, restore_test_patch_targets
from swe_gpu_bench.core.shell import all_commands_succeeded, did_commands_succeed, setup_builds_repo
from swe_gpu_bench.evaluate.agent_eval import (
    build_eval_result,
    build_evaluation_stats,
    execute_eval,
    prepare_eval_case_dir,
    render_eval_prompt,
    run_commands,
)


def make_eval_record() -> dict:
    return {
        "instance_id": "facebookresearch__xformers-104",
        "repo": "facebookresearch/xformers",
        "base_commit": "base123",
        "head_commit": "head456",
        "problem_statement": "Fix the bug.",
        "setup_commands": ["cd repo && echo setup"],
        "test_commands": ["cd repo && echo test"],
        "test_patch": "",
    }


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


def build_new_file_patch_from_head(repo: Path, rel_path: str, new_content: str) -> str:
    path = repo / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(new_content, encoding="utf-8")
    subprocess.run(["git", "add", "-N", rel_path], cwd=repo, check=True, capture_output=True, text=True)
    diff = subprocess.run(
        ["git", "diff", "--", rel_path],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    subprocess.run(["git", "reset", "--", rel_path], cwd=repo, check=True, capture_output=True, text=True)
    path.unlink()
    return diff


class EvalStatsTests(unittest.TestCase):
    def test_build_evaluation_stats_summarizes_setup_and_test_records(self):
        stats = build_evaluation_stats(
            setup_records=[
                {"phase": "setup", "exit_code": 0},
                {"phase": "setup", "exit_code": 1},
            ],
            post_patch_setup_records=[
                {"phase": "post_patch_setup", "exit_code": 0},
            ],
            test_records=[
                {"phase": "test", "exit_code": 0},
            ],
            elapsed_seconds=4.56,
        )

        self.assertEqual(
            stats,
            {
                "elapsed_seconds": 4.56,
                "total_commands": 4,
                "setup_commands": 2,
                "post_patch_setup_commands": 1,
                "test_commands": 1,
                "failed_commands": 1,
                "last_exit_code": 0,
            },
        )

    def test_did_commands_succeed_only_checks_last_command(self):
        self.assertTrue(did_commands_succeed([]))
        self.assertTrue(did_commands_succeed([{"exit_code": 1}, {"exit_code": 0}]))
        self.assertFalse(did_commands_succeed([{"exit_code": 0}, {"exit_code": 1}]))


class EvalCommandTests(unittest.TestCase):
    def test_run_commands_records_elapsed_seconds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = root / "prompt.md"
            prompt.write_text("prompt", encoding="utf-8")
            case_paths = prepare_eval_case_dir(
                record=make_eval_record(),
                eval_root=root / "EvalResults",
                prompt_template_path=prompt,
                force=False,
            )

            def fake_run(cmd, cwd, env, stdout, stderr, text, check):
                stdout.write("ok\n")
                return subprocess.CompletedProcess(cmd, 0)

            with mock.patch("swe_gpu_bench.core.shell.subprocess.run", side_effect=fake_run):
                records = run_commands(["cd repo && echo setup"], "setup", case_paths)

            self.assertEqual(records[0]["exit_code"], 0)
            self.assertIn("elapsed_seconds", records[0])
            self.assertTrue(Path(records[0]["stdout_log"]).exists())

    def test_setup_builds_repo_detects_repo_install_or_build_steps(self):
        self.assertTrue(setup_builds_repo(["uv pip install --python artifacts/.venv/bin/python -e repo"]))
        self.assertTrue(setup_builds_repo(["cd repo && python setup.py install"]))
        self.assertTrue(setup_builds_repo(["cmake --build artifacts/build"]))
        self.assertFalse(setup_builds_repo(["uv venv artifacts/.venv", "uv pip install torch pytest"]))

    def test_all_commands_succeeded_requires_every_command(self):
        self.assertFalse(all_commands_succeeded([]))
        self.assertFalse(all_commands_succeeded([{"exit_code": 1}, {"exit_code": 0}]))
        self.assertTrue(all_commands_succeeded([{"exit_code": 0}, {"exit_code": 0}]))

    def test_prepare_eval_case_dir_writes_only_agent_visible_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = {
                **make_eval_record(),
                "patch": "gold",
                "test_patch": "hidden tests",
                "head_commit": "head456",
                "pr_number": 104,
                "merged_at": "2024-01-03T00:00:00Z",
            }
            case_paths = prepare_eval_case_dir(record=record, eval_root=root / "EvalResults", force=False)

            payload = json.loads(case_paths.instance_path.read_text(encoding="utf-8"))

        self.assertEqual(
            set(payload),
            {"instance_id", "repo", "base_commit", "problem_statement", "setup_commands"},
        )

    def test_render_eval_prompt_says_environment_is_prepared(self):
        prompt = render_eval_prompt("Template\n", make_eval_record())
        self.assertIn("already executed the following setup commands", prompt)
        self.assertIn("Reuse the prepared environment under `artifacts/`", prompt)

    def test_render_eval_prompt_uses_template_without_test_file_restriction(self):
        template = (ROOT / "prompts" / "agent_eval.md").read_text(encoding="utf-8")
        prompt = render_eval_prompt(template, make_eval_record())

        self.assertNotIn("Do not modify existing test files", prompt)


class TestPatchRestoreTests(unittest.TestCase):
    def test_extract_test_patch_paths_deduplicates_and_handles_dev_null_headers(self):
        test_patch = (
            "diff --git a/tests/test_main.py b/tests/test_main.py\n"
            "index 1111111..2222222 100644\n"
            "--- a/tests/test_main.py\n"
            "+++ b/tests/test_main.py\n"
            "@@ -1 +1 @@\n"
            "-assert 1\n"
            "+assert 2\n"
            "diff --git a//dev/null b/tests/test_added.py\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/tests/test_added.py\n"
            "@@ -0,0 +1 @@\n"
            "+assert True\n"
            "diff --git a/tests/test_old.py b//dev/null\n"
            "deleted file mode 100644\n"
            "--- a/tests/test_old.py\n"
            "+++ /dev/null\n"
            "@@ -1 +0,0 @@\n"
            "-assert False\n"
            "diff --git a/tests/test_main.py b/tests/test_main.py\n"
            "index 1111111..3333333 100644\n"
            "--- a/tests/test_main.py\n"
            "+++ b/tests/test_main.py\n"
            "@@ -1 +1 @@\n"
            "-assert 2\n"
            "+assert 3\n"
        )

        self.assertEqual(
            extract_test_patch_paths(test_patch),
            ["tests/test_main.py", "tests/test_added.py", "tests/test_old.py"],
        )

    def test_restore_test_patch_targets_resets_tracked_files_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            init_git_repo(
                repo,
                {
                    "src/main.py": "x = 1\n",
                    "tests/test_main.py": "assert 1\n",
                },
            )
            test_patch = build_patch_from_head(repo, "tests/test_main.py", "assert 2\n")

            (repo / "src" / "main.py").write_text("x = 2\n", encoding="utf-8")
            (repo / "tests" / "test_main.py").write_text("assert 999\n", encoding="utf-8")

            result = restore_test_patch_targets(test_patch, repo)

            self.assertEqual(result.status, "restored")
            self.assertEqual((repo / "tests" / "test_main.py").read_text(encoding="utf-8"), "assert 1\n")
            self.assertEqual((repo / "src" / "main.py").read_text(encoding="utf-8"), "x = 2\n")

    def test_apply_test_patch_removes_untracked_new_file_before_apply(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            init_git_repo(repo, {"src/main.py": "x = 1\n"})
            test_patch = build_new_file_patch_from_head(
                repo,
                "tests/test_added.py",
                "def test_added():\n    assert True\n",
            )

            added = repo / "tests" / "test_added.py"
            added.parent.mkdir(parents=True, exist_ok=True)
            added.write_text("def test_added():\n    assert False\n", encoding="utf-8")

            status = apply_test_patch(test_patch, repo)

            self.assertNotIn(status, {"restore_failed", "apply_failed"})
            self.assertEqual(
                added.read_text(encoding="utf-8"),
                "def test_added():\n    assert True\n",
            )


class EvalResultTests(unittest.TestCase):
    def test_build_eval_result_includes_evaluation_stats(self):
        record = make_eval_record()
        case_paths = mock.Mock()
        case_paths.case_dir = Path("/tmp/EvalResults/facebookresearch__xformers-104")
        case_paths.agent_stdout_log = case_paths.case_dir / "artifacts/command-logs/agent.stdout.jsonl"
        case_paths.agent_stderr_log = case_paths.case_dir / "artifacts/command-logs/agent.stderr.log"

        result = build_eval_result(
            record,
            case_paths,
            "codex",
            datetime(2026, 4, 9, 0, 0, 0, tzinfo=timezone.utc),
            agent_result=AgentInvocationResult(
                command=["codex", "exec"],
                exit_code=0,
                stdout_path=Path("stdout.log"),
                stderr_path=Path("stderr.log"),
                elapsed_seconds=1.23,
                agent_stats={"elapsed_seconds": 1.23, "turns": 2},
            ),
            agent_diff="diff --git a/x b/x",
            setup_records=[{"phase": "setup", "command": "setup", "exit_code": 0, "elapsed_seconds": 0.1}],
            post_patch_setup_records=[
                {"phase": "post_patch_setup", "command": "rebuild", "exit_code": 0, "elapsed_seconds": 0.15}
            ],
            test_patch_status="no_test_patch",
            test_records=[{"phase": "test", "command": "test", "exit_code": 0, "elapsed_seconds": 0.2}],
        )

        self.assertEqual(result["status"], "passed")
        stats = dict(result["evaluation_stats"])
        self.assertGreater(stats.pop("elapsed_seconds"), 0)
        self.assertEqual(
            stats,
            {
                "total_commands": 3,
                "setup_commands": 1,
                "post_patch_setup_commands": 1,
                "test_commands": 1,
                "failed_commands": 0,
                "last_exit_code": 0,
            },
        )
        self.assertEqual(result["setup_commands"], record["setup_commands"])
        self.assertEqual(len(result["post_patch_setup_results"]), 1)


class OpenCodeAgentRunnerTests(unittest.TestCase):
    def test_invoke_builds_opencode_command_and_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = root / "prompt.md"
            prompt.write_text("fix it", encoding="utf-8")
            case_paths = prepare_eval_case_dir(
                record=make_eval_record(),
                eval_root=root / "EvalResults",
                prompt_template_path=prompt,
                force=False,
            )

            captured = {}

            def fake_run(command, cwd, text, stdout, stderr, env, check):
                captured["command"] = command
                captured["cwd"] = cwd
                captured["env"] = env
                stdout.write('{"type":"session.completed"}\n')
                return subprocess.CompletedProcess(command, 0)

            with mock.patch("swe_gpu_bench.core.agent.subprocess.run", side_effect=fake_run):
                result = OpenCodeAgentRunner(
                    model="deepseek/deepseek-reasoner", config_path=Path("opencode.json").resolve()
                ).invoke(case_paths)

            self.assertEqual(
                captured["command"][:5],
                ["opencode", "run", "--format", "json", "--model"],
            )
            self.assertEqual(captured["command"][5], "deepseek/deepseek-reasoner")
            self.assertEqual(captured["command"][-1], case_paths.prompt_path.read_text(encoding="utf-8"))
            self.assertEqual(captured["cwd"], str(case_paths.case_dir))
            self.assertEqual(
                captured["env"]["OPENCODE_CONFIG"],
                str((Path.cwd() / "opencode.json").resolve()),
            )
            self.assertEqual(captured["env"]["UV_CACHE_DIR"], str(case_paths.artifacts_dir / ".uv-cache"))
            self.assertEqual(captured["env"]["XDG_CACHE_HOME"], str(case_paths.artifacts_dir / ".cache"))
            self.assertEqual(result.exit_code, 0)
            self.assertIn("elapsed_seconds", result.agent_stats)

    def test_invoke_returns_configuration_error_when_opencode_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = root / "prompt.md"
            prompt.write_text("Template\n", encoding="utf-8")
            case_paths = prepare_eval_case_dir(
                record=make_eval_record(),
                eval_root=root / "EvalResults",
                prompt_template_path=prompt,
                force=False,
            )

            with mock.patch(
                "swe_gpu_bench.core.agent.subprocess.run",
                side_effect=FileNotFoundError,
            ):
                result = OpenCodeAgentRunner(model="moonshot/kimi-k2.5").invoke(case_paths)

            self.assertEqual(result.exit_code, 127)
            self.assertEqual(result.error_message, "opencode executable not found")


class ExecuteEvalTests(unittest.TestCase):
    def test_execute_eval_writes_evaluation_stats_to_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path = root / "dataset.jsonl"
            dataset_path.write_text(json.dumps(make_eval_record()) + "\n", encoding="utf-8")
            prompt_template = root / "agent_eval.md"
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
                        agent_stats={"elapsed_seconds": 1.0, "turns": 1},
                    )

            def fake_clone(case_paths, repo_slug, head_sha, pr_number=None, cache_root=None):
                (case_paths.repo_dir / "README.md").write_text(
                    f"{repo_slug}@{head_sha}\n", encoding="utf-8"
                )

            def fake_run_commands(commands, phase, case_paths):
                return [
                    {
                        "phase": phase,
                        "command": command,
                        "exit_code": 0,
                        "elapsed_seconds": 0.5,
                        "stdout_log": str(case_paths.command_logs_dir / f"{phase}-01.stdout.log"),
                        "stderr_log": str(case_paths.command_logs_dir / f"{phase}-01.stderr.log"),
                    }
                    for command in commands
                ]

            with mock.patch("swe_gpu_bench.evaluate.agent_eval.clone_or_update_repo", side_effect=fake_clone), \
                 mock.patch("swe_gpu_bench.evaluate.agent_eval.capture_diff", return_value=""), \
                 mock.patch("swe_gpu_bench.evaluate.agent_eval.run_commands", side_effect=fake_run_commands):
                result = execute_eval(
                    instance_id="facebookresearch__xformers-104",
                    dataset_path=dataset_path,
                    eval_root=root / "EvalResults",
                    prompt_template_path=prompt_template,
                    agent_runner=FakeRunner(),
                    force=False,
                )

            self.assertIn("evaluation_stats", result)
            self.assertEqual(result["evaluation_stats"]["total_commands"], 2)
            self.assertEqual(result["evaluation_stats"]["failed_commands"], 0)
            written = json.loads((root / "EvalResults" / "facebookresearch__xformers-104" / "result.json").read_text())
            self.assertEqual(written["evaluation_stats"]["setup_commands"], 1)
            self.assertEqual(written["evaluation_stats"]["post_patch_setup_commands"], 0)
            self.assertEqual(written["setup_commands"], make_eval_record()["setup_commands"])
            self.assertEqual(written["post_patch_setup_results"], [])
            self.assertEqual(written["test_patch_status"], "no_test_patch")

    def test_execute_eval_restores_test_patch_targets_before_applying_patch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = make_eval_record()
            dataset_path = root / "dataset.jsonl"
            prompt_template = root / "agent_eval.md"
            prompt_template.write_text("Template\n", encoding="utf-8")

            repo_seed = root / "repo-seed"
            init_git_repo(
                repo_seed,
                {
                    "src/main.py": "x = 1\n",
                    "tests/test_main.py": "assert 1\n",
                },
            )
            record["test_patch"] = build_patch_from_head(repo_seed, "tests/test_main.py", "assert 2\n")
            dataset_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            class FakeRunner:
                name = "codex"

                def invoke(self, case_paths, extra_env=None):
                    (case_paths.repo_dir / "src" / "main.py").write_text("x = 2\n", encoding="utf-8")
                    (case_paths.repo_dir / "tests" / "test_main.py").write_text("assert 999\n", encoding="utf-8")
                    case_paths.agent_stdout_log.write_text("", encoding="utf-8")
                    case_paths.agent_stderr_log.write_text("", encoding="utf-8")
                    return AgentInvocationResult(
                        command=["codex", "exec"],
                        exit_code=0,
                        stdout_path=case_paths.agent_stdout_log,
                        stderr_path=case_paths.agent_stderr_log,
                        elapsed_seconds=1.0,
                        agent_stats={},
                    )

            def fake_clone(case_paths, repo_slug, head_sha, pr_number=None, cache_root=None):
                init_git_repo(
                    case_paths.repo_dir,
                    {
                        "src/main.py": "x = 1\n",
                        "tests/test_main.py": "assert 1\n",
                    },
                )

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

            with mock.patch("swe_gpu_bench.evaluate.agent_eval.clone_or_update_repo", side_effect=fake_clone), \
                 mock.patch("swe_gpu_bench.evaluate.agent_eval.run_commands", side_effect=fake_run_commands):
                result = execute_eval(
                    instance_id=record["instance_id"],
                    dataset_path=dataset_path,
                    eval_root=root / "EvalResults",
                    prompt_template_path=prompt_template,
                    agent_runner=FakeRunner(),
                    force=False,
                )

            repo_dir = Path(result["case_dir"]) / "repo"
            self.assertEqual(result["status"], "passed")
            self.assertNotIn(result["test_patch_status"], {"restore_failed", "apply_failed"})
            self.assertEqual((repo_dir / "tests" / "test_main.py").read_text(encoding="utf-8"), "assert 2\n")
            self.assertEqual((repo_dir / "src" / "main.py").read_text(encoding="utf-8"), "x = 2\n")

    def test_execute_eval_hides_commits_after_base_from_the_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = make_eval_record()
            dataset_path = root / "dataset.jsonl"
            dataset_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            prompt_template = root / "agent_eval.md"
            prompt_template.write_text("Template\n", encoding="utf-8")
            seen = {}

            def git_log_all(repo_dir):
                return subprocess.run(
                    ["git", "log", "--all", "--format=%s"], cwd=repo_dir, check=True, capture_output=True, text=True
                ).stdout.split()

            class FakeRunner:
                name = "codex"

                def invoke(self, case_paths, extra_env=None):
                    seen["visible"] = git_log_all(case_paths.repo_dir)
                    case_paths.agent_stdout_log.write_text("", encoding="utf-8")
                    case_paths.agent_stderr_log.write_text("", encoding="utf-8")
                    return AgentInvocationResult(
                        command=["codex", "exec"],
                        exit_code=0,
                        stdout_path=case_paths.agent_stdout_log,
                        stderr_path=case_paths.agent_stderr_log,
                        elapsed_seconds=1.0,
                        agent_stats={},
                    )

            def fake_clone(case_paths, repo_slug, head_sha, pr_number=None, cache_root=None):
                repo = case_paths.repo_dir
                init_git_repo(repo, {"src/main.py": "x = 1\n"})
                (repo / "src" / "main.py").write_text("x = 2\n", encoding="utf-8")
                subprocess.run(
                    ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-am", "fix"],
                    cwd=repo, check=True, capture_output=True, text=True,
                )
                subprocess.run(["git", "checkout", "--quiet", "HEAD~1"], cwd=repo, check=True, capture_output=True)

            def fake_run_commands(commands, phase, case_paths):
                return [{"phase": phase, "command": command, "exit_code": 0} for command in commands]

            with mock.patch("swe_gpu_bench.evaluate.agent_eval.clone_or_update_repo", side_effect=fake_clone), \
                 mock.patch("swe_gpu_bench.evaluate.agent_eval.run_commands", side_effect=fake_run_commands):
                execute_eval(
                    instance_id=record["instance_id"],
                    dataset_path=dataset_path,
                    eval_root=root / "EvalResults",
                    prompt_template_path=prompt_template,
                    agent_runner=FakeRunner(),
                    force=False,
                )

            self.assertEqual(seen["visible"], ["init"])

    def test_execute_eval_reruns_setup_after_agent_when_repo_install_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = make_eval_record()
            record["setup_commands"] = ["uv pip install --python artifacts/.venv/bin/python -e repo"]
            dataset_path = root / "dataset.jsonl"
            dataset_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            prompt_template = root / "agent_eval.md"
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

            with mock.patch("swe_gpu_bench.evaluate.agent_eval.clone_or_update_repo", side_effect=fake_clone), \
                 mock.patch("swe_gpu_bench.evaluate.agent_eval.capture_diff", return_value=""), \
                 mock.patch("swe_gpu_bench.evaluate.agent_eval.run_commands", side_effect=fake_run_commands):
                result = execute_eval(
                    instance_id=record["instance_id"],
                    dataset_path=dataset_path,
                    eval_root=root / "EvalResults",
                    prompt_template_path=prompt_template,
                    agent_runner=FakeRunner(),
                    force=False,
                )

            self.assertEqual(phases, ["setup", "post_patch_setup", "test"])
            self.assertEqual(result["evaluation_stats"]["post_patch_setup_commands"], 1)
            self.assertEqual(len(result["post_patch_setup_results"]), 1)

    def test_execute_eval_continues_when_setup_recovers_by_final_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = make_eval_record()
            record["setup_commands"] = ["echo fail", "echo recover"]
            dataset_path = root / "dataset.jsonl"
            dataset_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            prompt_template = root / "agent_eval.md"
            prompt_template.write_text("Template\n", encoding="utf-8")

            fake_runner = mock.Mock()
            fake_runner.name = "codex"
            fake_runner.invoke.return_value = AgentInvocationResult(
                command=["codex", "exec"],
                exit_code=0,
                stdout_path=root / "stdout.log",
                stderr_path=root / "stderr.log",
                elapsed_seconds=1.0,
                agent_stats={},
            )

            def fake_clone(case_paths, repo_slug, head_sha, pr_number=None, cache_root=None):
                (case_paths.repo_dir / "README.md").write_text("ok\n", encoding="utf-8")

            def fake_run_commands(commands, phase, case_paths):
                if phase == "setup":
                    exit_codes = [1, 0]
                else:
                    exit_codes = [0 for _ in commands]
                return [
                    {
                        "phase": phase,
                        "command": command,
                        "exit_code": exit_codes[index],
                        "elapsed_seconds": 0.1,
                        "stdout_log": str(case_paths.command_logs_dir / f"{phase}-{index + 1:02d}.stdout.log"),
                        "stderr_log": str(case_paths.command_logs_dir / f"{phase}-{index + 1:02d}.stderr.log"),
                    }
                    for index, command in enumerate(commands)
                ]

            with mock.patch("swe_gpu_bench.evaluate.agent_eval.clone_or_update_repo", side_effect=fake_clone), \
                 mock.patch("swe_gpu_bench.evaluate.agent_eval.capture_diff", return_value=""), \
                 mock.patch("swe_gpu_bench.evaluate.agent_eval.run_commands", side_effect=fake_run_commands):
                result = execute_eval(
                    instance_id=record["instance_id"],
                    dataset_path=dataset_path,
                    eval_root=root / "EvalResults",
                    prompt_template_path=prompt_template,
                    agent_runner=fake_runner,
                    force=False,
                )

            fake_runner.invoke.assert_called_once()
            self.assertEqual(result["status"], "passed")

    def test_execute_eval_stops_when_setup_final_command_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = make_eval_record()
            record["setup_commands"] = ["echo still broken"]
            dataset_path = root / "dataset.jsonl"
            dataset_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            prompt_template = root / "agent_eval.md"
            prompt_template.write_text("Template\n", encoding="utf-8")

            fake_runner = mock.Mock()
            fake_runner.name = "codex"

            def fake_clone(case_paths, repo_slug, head_sha, pr_number=None, cache_root=None):
                (case_paths.repo_dir / "README.md").write_text("ok\n", encoding="utf-8")

            def fake_run_commands(commands, phase, case_paths):
                return [
                    {
                        "phase": phase,
                        "command": command,
                        "exit_code": 1,
                        "elapsed_seconds": 0.1,
                        "stdout_log": str(case_paths.command_logs_dir / f"{phase}-01.stdout.log"),
                        "stderr_log": str(case_paths.command_logs_dir / f"{phase}-01.stderr.log"),
                    }
                    for command in commands
                ]

            with mock.patch("swe_gpu_bench.evaluate.agent_eval.clone_or_update_repo", side_effect=fake_clone), \
                 mock.patch("swe_gpu_bench.evaluate.agent_eval.run_commands", side_effect=fake_run_commands):
                result = execute_eval(
                    instance_id=record["instance_id"],
                    dataset_path=dataset_path,
                    eval_root=root / "EvalResults",
                    prompt_template_path=prompt_template,
                    agent_runner=fake_runner,
                    force=False,
                )

            fake_runner.invoke.assert_not_called()
            self.assertEqual(result["status"], "error")
            self.assertEqual(result["failure_category"], "SETUP_FAILURE")
            self.assertEqual(result["test_patch_status"], "not_run")
            self.assertEqual(result["test_results"], [])

    def test_execute_eval_repository_preparation_failure_includes_evaluation_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path = root / "dataset.jsonl"
            dataset_path.write_text(json.dumps(make_eval_record()) + "\n", encoding="utf-8")
            prompt_template = root / "agent_eval.md"
            prompt_template.write_text("Template\n", encoding="utf-8")

            with mock.patch(
                "swe_gpu_bench.evaluate.agent_eval.clone_or_update_repo",
                side_effect=subprocess.CalledProcessError(1, ["git", "clone"]),
            ):
                failed = execute_eval(
                    instance_id="facebookresearch__xformers-104",
                    dataset_path=dataset_path,
                    eval_root=root / "EvalResults",
                    prompt_template_path=prompt_template,
                    force=True,
                    progress_callback=None,
                    agent_runner=type("Runner", (), {"name": "codex"})(),
                )

            self.assertEqual(failed["status"], "error")
            self.assertIn("evaluation_stats", failed)
            self.assertEqual(failed["evaluation_stats"]["total_commands"], 0)
            self.assertEqual(failed["agent_stats"], {})


if __name__ == "__main__":
    unittest.main()

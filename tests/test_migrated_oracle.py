import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock


def make_oracle_record() -> dict:
    return {
        "instance_id": "org__repo__101",
        "repo": "org/repo",
        "pr_number": 101,
        "base_commit": "base123",
        "head_commit": "head456",
        "problem_statement": "Fix the kernel launch bug.",
        "patch": (
            "diff --git a/src/main.py b/src/main.py\n"
            "index 1111111..2222222 100644\n"
            "--- a/src/main.py\n"
            "+++ b/src/main.py\n"
            "@@ -1 +1 @@\n"
            "-x = 1\n"
            "+x = 2\n"
            "diff --git a/tests/test_main.py b/tests/test_main.py\n"
            "index 3333333..4444444 100644\n"
            "--- a/tests/test_main.py\n"
            "+++ b/tests/test_main.py\n"
            "@@ -1 +1 @@\n"
            "-assert 1\n"
            "+assert 2\n"
        ),
        "test_patch": "",
        "setup_commands": ["cd repo && echo setup"],
        "test_commands": ["cd repo && echo test"],
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


class OraclePromptTests(unittest.TestCase):
    def test_build_oracle_context_collects_readme_and_non_test_files(self):
        from swe_gpu_bench.evaluate.use_cases import oracle

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo_dir = root / "repo"
            (repo_dir / "src").mkdir(parents=True)
            (repo_dir / "tests").mkdir(parents=True)
            (repo_dir / "README.md").write_text("Project README\n", encoding="utf-8")
            (repo_dir / "src" / "main.py").write_text("print('hello')\n", encoding="utf-8")
            (repo_dir / "tests" / "test_main.py").write_text("def test_ok(): pass\n", encoding="utf-8")

            context = oracle.build_oracle_context(make_oracle_record(), repo_dir)

        self.assertEqual(context.oracle_files, ["src/main.py"])
        self.assertIn("[start of README.md]", context.code_block)
        self.assertIn("[start of src/main.py]", context.code_block)
        self.assertIn("[line numbers are reference only; do not copy them into patches]", context.code_block)
        self.assertIn("0001| Project README", context.code_block)
        self.assertIn("0001| print('hello')", context.code_block)
        self.assertNotIn("tests/test_main.py", context.code_block)

    def test_render_oracle_prompt_uses_task_code_and_patch_example(self):
        from swe_gpu_bench.evaluate.use_cases import oracle

        prompt = oracle.render_oracle_prompt(
            "Template header.\n",
            problem_statement="Fix it",
            code_block=(
                "[start of README.md]\n"
                "[line numbers are reference only; do not copy them into patches]\n"
                "0001| hi\n"
                "[end of README.md]\n"
            ),
        )

        self.assertIn("<task>", prompt)
        self.assertIn("Fix it", prompt)
        self.assertIn("<code>", prompt)
        self.assertIn('"edits"', prompt)
        self.assertIn('"old": "exact existing text block to replace"', prompt)
        self.assertIn("The `0001|` prefixes are not part of the files", prompt)
        self.assertIn("Do not return a unified diff", prompt)
        self.assertIn("The system will apply your JSON edits locally and generate the final unified diff", prompt)


class OraclePatchTests(unittest.TestCase):
    def test_extract_patch_from_response_prefers_diff_block(self):
        from swe_gpu_bench.evaluate.use_cases import oracle

        raw = """Here is the patch:

```diff
diff --git a/src/main.py b/src/main.py
index 1111111..2222222 100644
--- a/src/main.py
+++ b/src/main.py
@@ -1 +1 @@
-x = 1
+x = 2
```
"""
        patch = oracle.extract_patch_from_response(raw)

        self.assertTrue(patch.startswith("diff --git a/src/main.py b/src/main.py"))
        self.assertIn("@@ -1 +1 @@", patch)

    def test_check_patch_applies_reports_valid_patch(self):
        from swe_gpu_bench.evaluate.use_cases import oracle

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
            (repo / "src").mkdir()
            (repo / "src" / "main.py").write_text("x = 1\n", encoding="utf-8")
            subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True, text=True)
            subprocess.run(
                ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "init"],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
            )
            patch_text = (
                "diff --git a/src/main.py b/src/main.py\n"
                "index bafc5d9..2489a8b 100644\n"
                "--- a/src/main.py\n"
                "+++ b/src/main.py\n"
                "@@ -1 +1 @@\n"
                "-x = 1\n"
                "+x = 2\n"
            )

            result = oracle.check_patch_applies(repo, patch_text)

        self.assertEqual(result.status, "valid")

    def test_build_patch_from_structured_response_generates_git_diff(self):
        from swe_gpu_bench.evaluate.use_cases import oracle

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
            (repo / "src").mkdir()
            (repo / "src" / "main.py").write_text("x = 1\n", encoding="utf-8")
            subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True, text=True)
            subprocess.run(
                ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "init"],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
            )
            response = json.dumps({"edits": [{"path": "src/main.py", "old": "x = 1\n", "new": "x = 2\n"}]})

            result = oracle.build_patch_from_structured_response(repo, response)

            self.assertEqual(result.status, "ok")
            self.assertIn("diff --git a/src/main.py b/src/main.py", result.patch_text)
            self.assertIn("-x = 1", result.patch_text)
            self.assertIn("+x = 2", result.patch_text)
            self.assertEqual((repo / "src" / "main.py").read_text(encoding="utf-8"), "x = 1\n")
            self.assertEqual(oracle.check_patch_applies(repo, result.patch_text).status, "valid")

    def test_build_patch_from_structured_response_includes_new_files(self):
        from swe_gpu_bench.evaluate.use_cases import oracle

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
            (repo / "src").mkdir()
            (repo / "src" / "main.py").write_text("x = 1\n", encoding="utf-8")
            subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True, text=True)
            subprocess.run(
                ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "init"],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
            )
            response = json.dumps(
                {
                    "edits": [
                        {"path": "src/main.py", "old": "x = 1\n", "new": "x = 2\n"},
                        {"path": "src/new.py", "old": "", "new": "y = 3\n"},
                    ]
                }
            )

            result = oracle.build_patch_from_structured_response(repo, response)

            self.assertEqual(result.status, "ok")
            self.assertIn("diff --git a/src/main.py b/src/main.py", result.patch_text)
            self.assertIn("diff --git a/src/new.py b/src/new.py", result.patch_text)
            self.assertIn("new file mode", result.patch_text)
            self.assertIn("+y = 3", result.patch_text)
            self.assertFalse((repo / "src" / "new.py").exists())
            self.assertEqual(oracle.check_patch_applies(repo, result.patch_text).status, "valid")


class OracleValidateTests(unittest.TestCase):
    def test_failed_setup_or_rebuild_cannot_be_hidden_by_last_success(self):
        from swe_gpu_bench.evaluate.use_cases import oracle

        for failed_phase in ("setup", "post_patch_setup", "portable_setup"):
            with self.subTest(phase=failed_phase), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                row = make_oracle_record()
                row["setup_commands"] = ["pip install -e repo", "echo done"]
                if failed_phase == "portable_setup":
                    row["command_contract"] = "portable_unified_v1"
                dataset = root / "dataset.jsonl"
                dataset.write_text(json.dumps(row) + "\n")
                template = root / "prompt.md"
                template.write_text("Fix the task.")
                pred = root / "pred" / row["instance_id"]
                pred.mkdir(parents=True)
                (pred / "pred.diff").write_text("example patch")
                phases = []

                def run(commands, phase, paths):
                    phases.append(phase)
                    return [{"phase": phase, "command": c,
                             "exit_code": int(phase == failed_phase and i == 0),
                             "elapsed_seconds": 0, "stdout_log": "", "stderr_log": ""}
                            for i, c in enumerate(commands)]

                with mock.patch.object(oracle, "clone_or_update_repo"), \
                     mock.patch.object(oracle, "check_patch_applies", return_value=oracle.PatchCheckResult(status="valid")), \
                     mock.patch.object(oracle, "apply_pred_patch", return_value=oracle.PatchCheckResult(status="applied")), \
                     mock.patch.object(oracle, "run_commands", side_effect=run):
                    result = oracle.execute_oracle_validate(row["instance_id"], dataset, root / "eval", root / "pred", template)
                if failed_phase == "portable_setup":
                    self.assertEqual(result["status"], "passed")
                    self.assertNotIn("post_patch_setup", phases)
                    self.assertIn("test", phases)
                else:
                    self.assertEqual(result["status"], "error")
                    self.assertEqual(result["failure_category"], failed_phase.upper() + "_FAILURE")
                    self.assertNotIn("test", phases)

    def test_oracle_preserves_reasoning_and_records_effective_model(self):
        from swe_gpu_bench.evaluate.use_cases import oracle
        from swe_gpu_bench.shared.models import ModelProfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = make_oracle_record()
            dataset = root / "dataset.jsonl"
            dataset.write_text(json.dumps(row) + "\n")
            template = root / "prompt.md"
            template.write_text("Fix the task.")
            profile = ModelProfile(name="smoke", kind="text", model="gpt-6.1-sol", provider="openai_responses", reasoning_effort="high")

            def prepare(repo, cache, record):
                init_git_repo(repo, {"src/main.py": "x = 1\n"})

            response = json.dumps({"edits": [{"path": "src/main.py", "old": "x = 1", "new": "x = 2"}]})
            with mock.patch.object(oracle, "_prepare_oracle_repo", side_effect=prepare), \
                 mock.patch.object(oracle, "generate_text_from_profile", return_value=response) as generate:
                result = oracle.generate_oracle_patch(row["instance_id"], dataset, root / "output", template, profile=profile)
            self.assertEqual(generate.call_args.kwargs["profile"].reasoning_effort, "high")
            self.assertEqual(result["model"], "gpt-6.1-sol")
            self.assertEqual(result["reasoning_effort"], "high")
            self.assertEqual(result["pred_patch_apply_check_status"], "valid")

    def test_execute_oracle_validate_applies_saved_patch_and_runs_tests(self):
        from swe_gpu_bench.evaluate.use_cases import oracle

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = make_oracle_record()
            dataset_path = root / "dataset.jsonl"
            dataset_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            prompt_template = root / "oracle_prompt.md"
            prompt_template.write_text("Template header.\n", encoding="utf-8")
            pred_patch_dir = root / "oracle-patches"
            patch_dir = pred_patch_dir / record["instance_id"]
            patch_dir.mkdir(parents=True)
            patch_path = patch_dir / "pred.diff"
            patch_path.write_text(
                "diff --git a/src/main.py b/src/main.py\n"
                "index 1111111..2222222 100644\n"
                "--- a/src/main.py\n"
                "+++ b/src/main.py\n"
                "@@ -1 +1 @@\n"
                "-x = 1\n"
                "+x = 2\n",
                encoding="utf-8",
            )

            def fake_clone(case_paths, repo_slug, head_sha, pr_number=None, cache_root=None, strict_commit=False):
                (case_paths.repo_dir / "src").mkdir(parents=True)
                (case_paths.repo_dir / "README.md").write_text("ok\n", encoding="utf-8")
                (case_paths.repo_dir / "src" / "main.py").write_text("x = 1\n", encoding="utf-8")

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

            with mock.patch("swe_gpu_bench.evaluate.use_cases.oracle.clone_or_update_repo", side_effect=fake_clone), \
                 mock.patch("swe_gpu_bench.evaluate.use_cases.oracle.run_commands", side_effect=fake_run_commands):
                result = oracle.execute_oracle_validate(
                    instance_id=record["instance_id"],
                    dataset_path=dataset_path,
                    eval_root=root / "OracleEvalResults",
                    pred_patch_root=pred_patch_dir,
                    prompt_template_path=prompt_template,
                )

        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["pred_patch_apply_check_status"], "valid")
        self.assertEqual(result["pred_patch_apply_status"], "applied")

    def test_execute_oracle_validate_restores_test_patch_targets_before_apply(self):
        from swe_gpu_bench.evaluate.use_cases import oracle

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = make_oracle_record()
            seed_repo = root / "repo-seed"
            init_git_repo(
                seed_repo,
                {
                    "src/main.py": "x = 1\n",
                    "tests/test_main.py": "assert 1\n",
                },
            )
            record["test_patch"] = build_patch_from_head(seed_repo, "tests/test_main.py", "assert 2\n")
            dataset_path = root / "dataset.jsonl"
            dataset_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            prompt_template = root / "oracle_prompt.md"
            prompt_template.write_text("Template header.\n", encoding="utf-8")
            pred_patch_dir = root / "oracle-patches"
            patch_dir = pred_patch_dir / record["instance_id"]
            patch_dir.mkdir(parents=True)
            patch_path = patch_dir / "pred.diff"
            patch_path.write_text("diff --git a/src/main.py b/src/main.py\n", encoding="utf-8")

            def fake_clone(case_paths, repo_slug, head_sha, pr_number=None, cache_root=None, strict_commit=False):
                init_git_repo(
                    case_paths.repo_dir,
                    {
                        "src/main.py": "x = 1\n",
                        "tests/test_main.py": "assert 1\n",
                    },
                )

            def fake_apply_pred_patch(repo_dir, patch_text):
                (repo_dir / "tests" / "test_main.py").write_text("assert 999\n", encoding="utf-8")
                return oracle.PatchCheckResult(status="applied")

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

            with mock.patch("swe_gpu_bench.evaluate.use_cases.oracle.clone_or_update_repo", side_effect=fake_clone), \
                 mock.patch("swe_gpu_bench.evaluate.use_cases.oracle.check_patch_applies", return_value=oracle.PatchCheckResult(status="valid")), \
                 mock.patch("swe_gpu_bench.evaluate.use_cases.oracle.apply_pred_patch", side_effect=fake_apply_pred_patch), \
                 mock.patch("swe_gpu_bench.evaluate.use_cases.oracle.run_commands", side_effect=fake_run_commands):
                result = oracle.execute_oracle_validate(
                    instance_id=record["instance_id"],
                    dataset_path=dataset_path,
                    eval_root=root / "OracleEvalResults",
                    pred_patch_root=pred_patch_dir,
                    prompt_template_path=prompt_template,
                )
            repo_dir = Path(result["case_dir"]) / "repo"
            self.assertEqual(result["status"], "passed")
            self.assertNotIn(result["test_patch_status"], {"restore_failed", "apply_failed"})
            self.assertEqual((repo_dir / "tests" / "test_main.py").read_text(encoding="utf-8"), "assert 2\n")


class OracleBatchTests(unittest.TestCase):
    def test_generate_oracle_patch_marks_context_length_cases_as_skipped(self):
        from swe_gpu_bench.evaluate.use_cases import oracle

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = make_oracle_record()
            dataset_path = root / "dataset.jsonl"
            dataset_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            prompt_template = root / "oracle_prompt.md"
            prompt_template.write_text("Template header.\n", encoding="utf-8")

            def fake_clone(case_paths, repo_slug, head_sha, pr_number=None, cache_root=None, strict_commit=False):
                (case_paths.repo_dir / "src").mkdir(parents=True)
                (case_paths.repo_dir / "README.md").write_text("ok\n", encoding="utf-8")
                (case_paths.repo_dir / "src" / "main.py").write_text("x = 1\n", encoding="utf-8")

            with mock.patch("swe_gpu_bench.evaluate.use_cases.oracle.clone_or_update_repo", side_effect=fake_clone), \
                 mock.patch(
                     "swe_gpu_bench.evaluate.use_cases.oracle.generate_text",
                     side_effect=oracle.ContextLengthExceededError("context_length_exceeded"),
                 ):
                metadata = oracle.generate_oracle_patch(
                    instance_id=record["instance_id"],
                    dataset_path=dataset_path,
                    output_root=root / "OraclePatches",
                    prompt_template_path=prompt_template,
                    model_spec="gpt-default",
                )

            self.assertEqual(metadata["structured_edit_status"], "skipped")
            self.assertEqual(metadata["pred_patch_apply_check_status"], "skipped")
            self.assertEqual(metadata["skip_reason"], "context_length_exceeded")
            self.assertFalse(Path(metadata["pred_patch_path"]).exists())

    def test_generate_oracle_patch_retries_repo_prepare_after_git_failure(self):
        from swe_gpu_bench.evaluate.use_cases import oracle

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = make_oracle_record()
            dataset_path = root / "dataset.jsonl"
            dataset_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            prompt_template = root / "oracle_prompt.md"
            prompt_template.write_text("Template header.\n", encoding="utf-8")
            calls: list[str] = []

            def fake_clone(case_paths, repo_slug, head_sha, pr_number=None, cache_root=None, strict_commit=False):
                calls.append("clone")
                if len(calls) == 1:
                    raise subprocess.CalledProcessError(128, ["git", "fetch", "--all", "--tags"], stderr="Broken pipe")
                (case_paths.repo_dir / "src").mkdir(parents=True)
                (case_paths.repo_dir / "README.md").write_text("ok\n", encoding="utf-8")
                (case_paths.repo_dir / "src" / "main.py").write_text("x = 1\n", encoding="utf-8")

            with mock.patch("swe_gpu_bench.evaluate.use_cases.oracle.clone_or_update_repo", side_effect=fake_clone), \
                 mock.patch("swe_gpu_bench.evaluate.use_cases.oracle.generate_text", return_value="diff --git a/src/main.py b/src/main.py\n"), \
                 mock.patch("swe_gpu_bench.evaluate.use_cases.oracle.shutil.rmtree", wraps=shutil.rmtree) as mocked_rmtree:
                oracle.generate_oracle_patch(
                    instance_id=record["instance_id"],
                    dataset_path=dataset_path,
                    output_root=root / "OraclePatches",
                    prompt_template_path=prompt_template,
                    model_spec="gpt-default",
                )

            self.assertEqual(calls, ["clone", "clone"])
            removed_paths = [Path(call.args[0]) for call in mocked_rmtree.call_args_list]
            self.assertIn(root / "OraclePatches" / record["instance_id"] / "case" / "repo", removed_paths)

    def test_generate_oracle_patch_saves_prompt_for_debugging(self):
        from swe_gpu_bench.evaluate.use_cases import oracle

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = make_oracle_record()
            dataset_path = root / "dataset.jsonl"
            dataset_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            prompt_template = root / "oracle_prompt.md"
            prompt_template.write_text("Template header.\n", encoding="utf-8")

            def fake_clone(case_paths, repo_slug, head_sha, pr_number=None, cache_root=None, strict_commit=False):
                (case_paths.repo_dir / "src").mkdir(parents=True)
                (case_paths.repo_dir / "README.md").write_text("ok\n", encoding="utf-8")
                (case_paths.repo_dir / "src" / "main.py").write_text("x = 1\n", encoding="utf-8")

            with mock.patch("swe_gpu_bench.evaluate.use_cases.oracle.clone_or_update_repo", side_effect=fake_clone), \
                 mock.patch("swe_gpu_bench.evaluate.use_cases.oracle.generate_text", return_value="diff --git a/src/main.py b/src/main.py\n"):
                metadata = oracle.generate_oracle_patch(
                    instance_id=record["instance_id"],
                    dataset_path=dataset_path,
                    output_root=root / "OraclePatches",
                    prompt_template_path=prompt_template,
                    model_spec="gpt-default",
                )

            prompt_path = Path(metadata["prompt_path"])
            self.assertTrue(prompt_path.exists())
            self.assertIn("Template header.", prompt_path.read_text(encoding="utf-8"))
            self.assertEqual(prompt_path.name, "prompt.md")

    def test_generate_oracle_patches_runs_all_instances_under_model_scoped_dir(self):
        from swe_gpu_bench.evaluate.use_cases import oracle

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path = root / "dataset.jsonl"
            rows = [
                {**make_oracle_record(), "instance_id": "org__repo__101"},
                {**make_oracle_record(), "instance_id": "org__repo__102"},
            ]
            dataset_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            prompt_template = root / "oracle_prompt.md"
            prompt_template.write_text("Template header.\n", encoding="utf-8")

            def fake_generate(instance_id, dataset_path, output_root, prompt_template_path, model_spec, model, env_file, max_tokens=8192):
                return {
                    "instance_id": instance_id,
                    "pred_patch_path": str(output_root / instance_id / "pred.diff"),
                    "model_spec": model_spec,
                }

            with mock.patch("swe_gpu_bench.evaluate.use_cases.oracle.generate_oracle_patch", side_effect=fake_generate):
                results = oracle.generate_oracle_patches(
                    dataset_path=dataset_path,
                    output_root=root / "OraclePatches",
                    prompt_template_path=prompt_template,
                    model_spec="kimi",
                    all_instances=True,
                )

        self.assertEqual([row["instance_id"] for row in results], ["org__repo__101", "org__repo__102"])
        self.assertTrue(all("/OraclePatches/kimi/" in row["pred_patch_path"] for row in results))

    def test_generate_oracle_patches_skips_terminal_metadata_without_pred_diff(self):
        from swe_gpu_bench.evaluate.use_cases import oracle

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path = root / "dataset.jsonl"
            row = {**make_oracle_record(), "instance_id": "org__repo__101"}
            dataset_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            prompt_template = root / "oracle_prompt.md"
            prompt_template.write_text("Template header.\n", encoding="utf-8")
            model_root = root / "OraclePatches" / "gpt-default" / row["instance_id"]
            model_root.mkdir(parents=True)
            (model_root / "metadata.json").write_text(
                json.dumps(
                    {
                        "instance_id": row["instance_id"],
                        "structured_edit_status": "skipped",
                        "skip_reason": "context_length_exceeded",
                        "pred_patch_apply_check_status": "skipped",
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch("swe_gpu_bench.evaluate.use_cases.oracle.generate_oracle_patch") as mocked_generate:
                results = oracle.generate_oracle_patches(
                    dataset_path=dataset_path,
                    output_root=root / "OraclePatches",
                    prompt_template_path=prompt_template,
                    model_spec="gpt-default",
                    all_instances=True,
                )

        mocked_generate.assert_not_called()
        self.assertEqual(results, [{"instance_id": "org__repo__101", "skipped": True, "reason": "context_length_exceeded"}])


class OracleSummaryTests(unittest.TestCase):
    def test_summarize_oracle_results_groups_by_model(self):
        from swe_gpu_bench.evaluate.use_cases import oracle

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for model, rows in {
                "gpt-default": [("a", "passed"), ("b", "failed")],
                "kimi": [("a", "passed")],
            }.items():
                for instance_id, status in rows:
                    case_dir = root / model / instance_id
                    case_dir.mkdir(parents=True)
                    (case_dir / "result.json").write_text(
                        json.dumps({"instance_id": instance_id, "status": status, "failure_category": None}),
                        encoding="utf-8",
                    )

            summary = oracle.summarize_oracle_results(root, ["gpt-default", "kimi"])

        self.assertEqual(summary["models"]["gpt-default"]["passed"], 1)
        self.assertEqual(summary["models"]["gpt-default"]["total"], 2)
        self.assertEqual(summary["models"]["kimi"]["pass_rate"], 1.0)




if __name__ == "__main__":
    unittest.main()

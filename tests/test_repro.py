import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from swe_gpu_bench.generate.repro import (
    CodexInvocationResult,
    build_case_name,
    build_result_from_invocation,
    build_case_sample,
    execute_repro,
    load_sample,
    load_base_prompt,
    parse_codex_session_stats,
    parse_result_markers,
    prepare_case_directory,
    render_prompt,
    invoke_codex,
)
from swe_gpu_bench.cli import main
from swe_gpu_bench.core.git import clone_or_update_repo


class LoadSampleTests(unittest.TestCase):
    def test_load_sample_merges_scored_record_with_pr_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scored_dir = root / "data" / "fused_scored" / "Dao-AILab__flash-attention"
            prs_dir = root / "data" / "prs" / "Dao-AILab__flash-attention"
            scored_dir.mkdir(parents=True)
            prs_dir.mkdir(parents=True)

            gpt_record = {
                "repo": "Dao-AILab/flash-attention",
                "number": 1365,
                "title": "Sample title",
                "head_sha": "from-gpt",
            }
            pr_record = {
                "repo": "Dao-AILab/flash-attention",
                "number": 1365,
                "head_sha": "from-pr-json",
                "body": "sample body",
            }

            (scored_dir / "fused_scored.jsonl").write_text(
                json.dumps(gpt_record) + "\n", encoding="utf-8"
            )
            (prs_dir / "1365.json").write_text(
                json.dumps(pr_record), encoding="utf-8"
            )

            sample = load_sample(root, "Dao-AILab/flash-attention", 1365)

            self.assertEqual(sample["repo"], "Dao-AILab/flash-attention")
            self.assertEqual(sample["number"], 1365)
            self.assertEqual(sample["pr_record"]["head_sha"], "from-pr-json")
            self.assertEqual(sample["head_sha"], "from-pr-json")

    def test_build_case_sample_strips_large_prompt_inputs(self):
        sample = {
            "repo": "Dao-AILab/flash-attention",
            "number": 1365,
            "patch_text": "x" * 100,
            "raw_response": {"huge": True},
            "head_sha": "abc",
        }

        case_sample = build_case_sample(sample)

        self.assertNotIn("patch_text", case_sample)
        self.assertNotIn("raw_response", case_sample)
        self.assertEqual(case_sample["head_sha"], "abc")


class PromptAndCaseTests(unittest.TestCase):
    def test_load_base_prompt_strips_embedded_sample_json_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "agent_explore.md"
            path.write_text(
                "header\n\n<SAMPLE_JSON>\n{\"number\": 1}\n</SAMPLE_JSON>\n\nPlease execute.\n",
                encoding="utf-8",
            )

            prompt = load_base_prompt(path)

            self.assertEqual(prompt, "header\n\n")

    def test_render_prompt_includes_case_constraints_and_markers(self):
        template = "Base template.\n"
        sample = {
            "repo": "Dao-AILab/flash-attention",
            "number": 1365,
            "head_sha": "deadbeef",
        }

        prompt = render_prompt(template, sample)

        self.assertIn("BEGIN_FINAL_RESULT_JSON", prompt)
        self.assertIn("END_FINAL_RESULT_JSON", prompt)
        self.assertIn('"number": 1365', prompt)
        self.assertIn('"head_sha": "deadbeef"', prompt)
        self.assertIn("Do not edit repository source code", prompt)
        self.assertIn("Create any `uv` virtual environment inside the current case directory", prompt)
        self.assertIn("Do not give up after the first failed install or build attempt", prompt)

    def test_prepare_case_directory_creates_expected_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_root = root / "TestCases"
            template_dir = root / "template_codex"
            template_dir.mkdir(parents=True)
            (template_dir / "config.toml").write_text("model = 'x'\n", encoding="utf-8")

            sample = {
                "repo": "Dao-AILab/flash-attention",
                "number": 1365,
                "head_sha": "deadbeef",
            }

            paths = prepare_case_directory(
                case_root=case_root,
                repo_slug="Dao-AILab/flash-attention",
                pr_number=1365,
                template_codex_dir=template_dir,
                sample=sample,
                prompt_text="prompt body",
            )

            self.assertEqual(paths.case_dir.name, "flash-attention-pr1365")
            self.assertTrue((paths.case_dir / ".codex" / "config.toml").exists())
            self.assertTrue((paths.case_dir / "inputs" / "sample.json").exists())
            self.assertEqual(
                json.loads((paths.case_dir / "inputs" / "sample.json").read_text()),
                sample,
            )
            self.assertEqual(
                (paths.case_dir / "inputs" / "prompt.md").read_text(encoding="utf-8"),
                "prompt body",
            )


class ResultParsingTests(unittest.TestCase):
    def test_parse_codex_session_stats_aggregates_turns_and_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path = Path(tmp) / "codex.session.jsonl"
            jsonl_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "type": "turn.completed",
                                "usage": {
                                    "input_tokens": 100,
                                    "cached_input_tokens": 25,
                                    "output_tokens": 10,
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "type": "item.completed",
                                "item": {"type": "command_execution"},
                            }
                        ),
                        json.dumps(
                            {
                                "type": "turn.completed",
                                "usage": {
                                    "input_tokens": 40,
                                    "cached_input_tokens": 5,
                                    "output_tokens": 7,
                                },
                            }
                        ),
                    ])
                + "\n",
                encoding="utf-8",
            )

            stats = parse_codex_session_stats(jsonl_path)

            self.assertEqual(
                stats,
                {
                    "input_tokens": 140,
                    "cached_input_tokens": 30,
                    "output_tokens": 17,
                    "turns": 2,
                    "command_executions": 1,
                },
            )

    def test_parse_result_markers_extracts_json_payload(self):
        output = """
preface
BEGIN_FINAL_RESULT_JSON
{"status":"reproduced","failure_category":null,"test_commands":["pytest x"]}
END_FINAL_RESULT_JSON
"""
        result = parse_result_markers(output)
        self.assertEqual(result["status"], "reproduced")
        self.assertEqual(result["test_commands"], ["pytest x"])

    def test_missing_codex_binary_returns_configuration_failure(self):
        invocation = CodexInvocationResult(
            command=["codex", "exec"],
            exit_code=127,
            stdout_path=Path("stdout.log"),
            stderr_path=Path("stderr.log"),
            parsed_result=None,
            error_message="codex executable not found",
        )

        result = build_result_from_invocation(
            sample={"repo": "Dao-AILab/flash-attention", "number": 1365, "head_sha": "x"},
            invocation=invocation,
        )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure_category"], "AUTOMATION_CONFIGURATION_FAILURE")
        self.assertIn("codex executable not found", result["summary"])
        self.assertEqual(
            result["codex_stats"],
            {
                "elapsed_seconds": None,
                "input_tokens": 0,
                "cached_input_tokens": 0,
                "output_tokens": 0,
                "turns": 0,
                "command_executions": 0,
            },
        )

    def test_build_result_from_invocation_preserves_codex_stats(self):
        invocation = CodexInvocationResult(
            command=["codex", "exec"],
            exit_code=0,
            stdout_path=Path("stdout.log"),
            stderr_path=Path("stderr.log"),
            parsed_result={
                "status": "reproduced",
                "failure_category": None,
                "summary": "ok",
                "setup_commands": [],
                "test_commands": [],
                "environment_summary": "",
                "evidence": [],
            },
            elapsed_seconds=12.34,
            codex_stats={
                "input_tokens": 123,
                "cached_input_tokens": 45,
                "output_tokens": 67,
                "turns": 2,
                "command_executions": 9,
            },
        )

        result = build_result_from_invocation(
            sample={"repo": "Dao-AILab/flash-attention", "number": 1365, "head_sha": "x"},
            invocation=invocation,
        )

        self.assertEqual(result["status"], "reproduced")
        self.assertEqual(
            result["codex_stats"],
            {
                "elapsed_seconds": 12.34,
                "input_tokens": 123,
                "cached_input_tokens": 45,
                "output_tokens": 67,
                "turns": 2,
                "command_executions": 9,
            },
        )


class NamingTests(unittest.TestCase):
    def test_build_case_name_uses_repo_tail_and_pr_number(self):
        self.assertEqual(
            build_case_name("Dao-AILab/flash-attention", 1365),
            "flash-attention-pr1365",
        )


class ExecuteReproTests(unittest.TestCase):
    def test_execute_repro_writes_result_even_when_codex_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scored_dir = root / "data" / "fused_scored" / "Dao-AILab__flash-attention"
            prs_dir = root / "data" / "prs" / "Dao-AILab__flash-attention"
            prompt_path = root / "prompts" / "agent_explore.md"
            template_codex_dir = root / "template_codex"
            case_root = root / "TestCases"

            scored_dir.mkdir(parents=True)
            prs_dir.mkdir(parents=True)
            prompt_path.parent.mkdir(parents=True)
            template_codex_dir.mkdir(parents=True)

            (scored_dir / "fused_scored.jsonl").write_text(
                json.dumps(
                    {
                        "repo": "Dao-AILab/flash-attention",
                        "number": 1365,
                        "title": "x",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (prs_dir / "1365.json").write_text(
                json.dumps(
                    {
                        "repo": "Dao-AILab/flash-attention",
                        "number": 1365,
                        "head_sha": "8ae74ea8f165f4bd5da01f1604daf112bfd2d4fc",
                    }
                ),
                encoding="utf-8",
            )
            prompt_path.write_text("Base prompt.\n", encoding="utf-8")
            (template_codex_dir / "config.toml").write_text(
                "model = 'x'\n", encoding="utf-8"
            )

            def fake_clone(case_paths, repo_slug, head_sha, pr_number, cache_root=None):
                (case_paths.repo_dir / "README.md").write_text(
                    "%s@%s\n" % (repo_slug, head_sha), encoding="utf-8"
                )

            with mock.patch(
                "swe_gpu_bench.generate.repro.invoke_codex",
                return_value=CodexInvocationResult(
                    command=["codex", "exec"],
                    exit_code=127,
                    stdout_path=case_root / "flash-attention-pr1365" / "logs" / "codex.session.jsonl",
                    stderr_path=case_root / "flash-attention-pr1365" / "logs" / "codex.stderr.log",
                    parsed_result=None,
                    error_message="codex executable not found",
                ),
            ):
                result = execute_repro(
                    data_root=root,
                    repo_slug="Dao-AILab/flash-attention",
                    pr_number=1365,
                    case_root=case_root,
                    prompt_template_path=prompt_path,
                    template_codex_dir=template_codex_dir,
                    clone_repo_fn=fake_clone,
                )

            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["failure_category"], "AUTOMATION_CONFIGURATION_FAILURE")
            result_path = case_root / "flash-attention-pr1365" / "logs" / "result.json"
            self.assertTrue(result_path.exists())
            written = json.loads(result_path.read_text())
            self.assertEqual(written["pr_number"], 1365)
            self.assertIn("codex_stats", written)
            self.assertEqual(written["codex_stats"]["turns"], 0)

    def test_execute_repro_converts_checkout_failure_into_result_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scored_dir = root / "data" / "fused_scored" / "Dao-AILab__flash-attention"
            prs_dir = root / "data" / "prs" / "Dao-AILab__flash-attention"
            prompt_path = root / "prompts" / "agent_explore.md"
            template_codex_dir = root / "template_codex"
            case_root = root / "TestCases"

            scored_dir.mkdir(parents=True)
            prs_dir.mkdir(parents=True)
            prompt_path.parent.mkdir(parents=True)
            template_codex_dir.mkdir(parents=True)

            (scored_dir / "fused_scored.jsonl").write_text(
                json.dumps(
                    {
                        "repo": "Dao-AILab/flash-attention",
                        "number": 1365,
                        "title": "x",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (prs_dir / "1365.json").write_text(
                json.dumps(
                    {
                        "repo": "Dao-AILab/flash-attention",
                        "number": 1365,
                        "head_sha": "missing-sha",
                    }
                ),
                encoding="utf-8",
            )
            prompt_path.write_text("Base prompt.\n", encoding="utf-8")
            (template_codex_dir / "config.toml").write_text(
                "model = 'x'\n", encoding="utf-8"
            )

            def failing_clone(case_paths, repo_slug, head_sha, pr_number):
                raise subprocess.CalledProcessError(
                    returncode=128, cmd=["git", "checkout", head_sha]
                )

            result = execute_repro(
                data_root=root,
                repo_slug="Dao-AILab/flash-attention",
                pr_number=1365,
                case_root=case_root,
                prompt_template_path=prompt_path,
                template_codex_dir=template_codex_dir,
                clone_repo_fn=failing_clone,
            )

            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["failure_category"], "REPOSITORY_PREPARATION_FAILURE")
            self.assertIn("git checkout", result["summary"])

    def test_execute_repro_reports_progress_messages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scored_dir = root / "data" / "fused_scored" / "Dao-AILab__flash-attention"
            prs_dir = root / "data" / "prs" / "Dao-AILab__flash-attention"
            prompt_path = root / "prompts" / "agent_explore.md"
            template_codex_dir = root / "template_codex"
            case_root = root / "TestCases"

            scored_dir.mkdir(parents=True)
            prs_dir.mkdir(parents=True)
            prompt_path.parent.mkdir(parents=True)
            template_codex_dir.mkdir(parents=True)

            (scored_dir / "fused_scored.jsonl").write_text(
                json.dumps(
                    {
                        "repo": "Dao-AILab/flash-attention",
                        "number": 1365,
                        "title": "x",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (prs_dir / "1365.json").write_text(
                json.dumps(
                    {
                        "repo": "Dao-AILab/flash-attention",
                        "number": 1365,
                        "head_sha": "8ae74ea8f165f4bd5da01f1604daf112bfd2d4fc",
                    }
                ),
                encoding="utf-8",
            )
            prompt_path.write_text("Base prompt.\n", encoding="utf-8")
            (template_codex_dir / "config.toml").write_text(
                "model = 'x'\n", encoding="utf-8"
            )

            events = []

            def fake_clone(case_paths, repo_slug, head_sha, pr_number, cache_root=None):
                (case_paths.repo_dir / "README.md").write_text(
                    "%s@%s\n" % (repo_slug, head_sha), encoding="utf-8"
                )

            def fake_invoke_codex(case_paths):
                return CodexInvocationResult(
                    command=["codex", "exec"],
                    exit_code=127,
                    stdout_path=case_paths.stdout_log,
                    stderr_path=case_paths.stderr_log,
                    parsed_result=None,
                    error_message="codex executable not found",
                )

            result = execute_repro(
                data_root=root,
                repo_slug="Dao-AILab/flash-attention",
                pr_number=1365,
                case_root=case_root,
                prompt_template_path=prompt_path,
                template_codex_dir=template_codex_dir,
                clone_repo_fn=fake_clone,
                invoke_codex_fn=fake_invoke_codex,
                progress_callback=events.append,
            )

            self.assertEqual(result["status"], "failed")
            self.assertEqual(
                events,
                [
                    "Loading sample metadata",
                    "Rendering Codex prompt",
                    "Preparing case directory",
                    "Preparing repository checkout",
                    "Running Codex",
                    "Writing normalized result",
                ],
            )


class CliTests(unittest.TestCase):
    def test_cli_rejects_removed_codex_bin_flag(self):
        with self.assertRaises(SystemExit):
            main(["generate", "repro", "--repo", "Dao-AILab/flash-attention", "--pr", "1365", "--codex-bin", "x"])

    def test_cli_forwards_arguments_to_execute_repro(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch("swe_gpu_bench.generate.repro.execute_repro") as execute:
                execute.return_value = {"status": "failed"}
                exit_code = main(
                    [
                        "generate",
                        "repro",
                        "--repo", "Dao-AILab/flash-attention",
                        "--pr", "1365",
                        "--data-root", str(root),
                        "--case-root", "cases",
                        "--template-codex-dir", "template",
                    ]
                )

            self.assertEqual(exit_code, 0)
            kwargs = execute.call_args.kwargs
            self.assertEqual(kwargs["repo_slug"], "Dao-AILab/flash-attention")
            self.assertEqual(kwargs["pr_number"], 1365)
            self.assertEqual(kwargs["case_root"], (root / "cases").resolve())
            self.assertEqual(kwargs["template_codex_dir"], (root / "template").resolve())
            self.assertEqual(kwargs["prompt_template_path"], (root / "prompts" / "agent_explore.md").resolve())


class GitPreparationTests(unittest.TestCase):
    def test_clone_or_update_repo_reuses_local_cache_when_configured(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "repo-cache"
            case_one = prepare_case_directory(
                case_root=root / "cases-one",
                repo_slug="Dao-AILab/flash-attention",
                pr_number=1365,
                template_codex_dir=root / "missing-template",
                sample={"repo": "Dao-AILab/flash-attention", "number": 1365},
                prompt_text="x",
            )
            case_two = prepare_case_directory(
                case_root=root / "cases-two",
                repo_slug="Dao-AILab/flash-attention",
                pr_number=1366,
                template_codex_dir=root / "missing-template",
                sample={"repo": "Dao-AILab/flash-attention", "number": 1366},
                prompt_text="x",
            )

            calls = []

            def fake_run_git(args, cwd):
                args = list(args)
                calls.append((args, cwd))
                if args[:3] == ["git", "clone", "https://github.com/Dao-AILab/flash-attention.git"]:
                    target = cwd / args[-1]
                    target.mkdir(parents=True, exist_ok=True)
                    (target / ".git").mkdir()
                    return
                if args[:3] == ["git", "clone", "--shared"]:
                    target = cwd / args[-1]
                    target.mkdir(parents=True, exist_ok=True)
                    (target / ".git").mkdir()
                    return

            with mock.patch("swe_gpu_bench.core.git.run_git", side_effect=fake_run_git):
                clone_or_update_repo(
                    case_one,
                    "Dao-AILab/flash-attention",
                    "base123",
                    pr_number=1365,
                    cache_root=cache_root,
                )
                clone_or_update_repo(
                    case_two,
                    "Dao-AILab/flash-attention",
                    "base123",
                    pr_number=1366,
                    cache_root=cache_root,
                )

            remote_clones = [call for call in calls if call[0][:3] == ["git", "clone", "https://github.com/Dao-AILab/flash-attention.git"]]
            shared_clones = [call for call in calls if call[0][:3] == ["git", "clone", "--shared"]]
            self.assertEqual(len(remote_clones), 1)
            self.assertEqual(len(shared_clones), 2)

    def test_clone_or_update_repo_fetches_pull_ref_when_head_sha_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_dir = root / "flash-attention-pr1365"
            repo_dir = case_dir / "repo"
            repo_dir.mkdir(parents=True)
            (repo_dir / ".git").mkdir()
            case_paths = prepare_case_directory(
                case_root=root,
                repo_slug="Dao-AILab/flash-attention",
                pr_number=1365,
                template_codex_dir=root / "missing-template",
                sample={"repo": "Dao-AILab/flash-attention", "number": 1365},
                prompt_text="x",
            )

            calls = []
            fetched = []

            def fake_run_git(args, cwd):
                calls.append((list(args), cwd))
                if args[:2] == ["git", "fetch"] and args[-1] == "pull/1365/head":
                    fetched.append(True)
                if args[:3] == ["git", "checkout", "--force"] and args[-1] == "missing-sha" and not fetched:
                    raise subprocess.CalledProcessError(returncode=128, cmd=args)

            with mock.patch("swe_gpu_bench.core.git.run_git", side_effect=fake_run_git):
                clone_or_update_repo(
                    case_paths,
                    "Dao-AILab/flash-attention",
                    "missing-sha",
                    pr_number=1365,
                )

            self.assertEqual(
                [args for args, _ in calls[-2:]],
                [
                    ["git", "fetch", "origin", "pull/1365/head"],
                    ["git", "checkout", "--force", "missing-sha"],
                ],
            )
            self.assertNotIn("FETCH_HEAD", [args[-1] for args, _ in calls])


class CodexInvocationTests(unittest.TestCase):
    def test_invoke_codex_runs_from_case_root_with_prompt_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_paths = prepare_case_directory(
                case_root=root,
                repo_slug="Dao-AILab/flash-attention",
                pr_number=1365,
                template_codex_dir=root / "missing-template",
                sample={"repo": "Dao-AILab/flash-attention", "number": 1365},
                prompt_text="prompt body",
            )

            recorded = {}

            class Completed(object):
                returncode = 0

            def fake_run(command, cwd, text, stdout, stderr, env, check):
                recorded["command"] = command
                recorded["cwd"] = cwd
                recorded["uv_cache_dir"] = env.get("UV_CACHE_DIR")
                last_message_path = Path(
                    command[command.index("--output-last-message") + 1]
                )
                last_message_path.write_text(
                    "BEGIN_FINAL_RESULT_JSON\n"
                    '{"status":"failed","failure_category":"X","summary":"x","setup_commands":[],"test_commands":[],"environment_summary":"","evidence":[]}\n'
                    "END_FINAL_RESULT_JSON\n",
                    encoding="utf-8",
                )
                stdout.write(
                    json.dumps({"type": "turn.completed", "usage": {"input_tokens": 100, "cached_input_tokens": 20, "output_tokens": 5}})
                    + "\n"
                )
                stdout.write(
                    json.dumps({"type": "item.completed", "item": {"type": "command_execution"}})
                    + "\n"
                )
                return Completed()

            with mock.patch("swe_gpu_bench.core.agent.subprocess.run", side_effect=fake_run):
                result = invoke_codex(case_paths)

            self.assertEqual(recorded["cwd"], str(case_paths.case_dir))
            self.assertEqual(
                recorded["command"],
                [
                    "codex",
                    "exec",
                    "--dangerously-bypass-approvals-and-sandbox",
                    "--skip-git-repo-check",
                    "--json",
                    "--output-last-message",
                    str(case_paths.last_message_path),
                    "prompt body",
                ],
            )
            self.assertEqual(
                recorded["uv_cache_dir"],
                str(case_paths.artifacts_dir / ".uv-cache"),
            )
            self.assertEqual(result.parsed_result["failure_category"], "X")
            self.assertEqual(
                result.codex_stats,
                {
                    "elapsed_seconds": result.elapsed_seconds,
                    "input_tokens": 100,
                    "cached_input_tokens": 20,
                    "output_tokens": 5,
                    "turns": 1,
                    "command_executions": 1,
                },
            )
            self.assertIsNotNone(result.elapsed_seconds)

    def test_invoke_codex_places_extra_args_before_prompt_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_paths = prepare_case_directory(
                case_root=root,
                repo_slug="Dao-AILab/flash-attention",
                pr_number=1365,
                template_codex_dir=root / "missing-template",
                sample={"repo": "Dao-AILab/flash-attention", "number": 1365},
                prompt_text="prompt body",
            )

            recorded = {}

            class Completed(object):
                returncode = 0

            def fake_run(command, cwd, text, stdout, stderr, env, check):
                recorded["command"] = command
                last_message_path = Path(command[command.index("--output-last-message") + 1])
                last_message_path.write_text(
                    "BEGIN_FINAL_RESULT_JSON\n"
                    '{"status":"failed","failure_category":"X","summary":"x","setup_commands":[],"test_commands":[],"environment_summary":"","evidence":[]}\n'
                    "END_FINAL_RESULT_JSON\n",
                    encoding="utf-8",
                )
                return Completed()

            with mock.patch("swe_gpu_bench.core.agent.subprocess.run", side_effect=fake_run):
                invoke_codex(case_paths, extra_args=["--model", "gpt-5.1-codex-2025-11-13"])

            self.assertEqual(recorded["command"][-3:], ["--model", "gpt-5.1-codex-2025-11-13", "prompt body"])


if __name__ == "__main__":
    unittest.main()

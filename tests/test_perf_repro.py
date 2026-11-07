import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from swe_gpu_bench.generate.repro import CodexInvocationResult


class PerfReproLoadSampleTests(unittest.TestCase):
    def test_load_perf_sample_merges_perf_review_record_with_pr_metadata(self):
        from swe_gpu_bench.generate import perf_repro

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            perf_dir = root / "data" / "perf_reviewed" / "deepspeedai__DeepSpeed"
            prs_dir = root / "data" / "prs" / "deepspeedai__DeepSpeed"
            perf_dir.mkdir(parents=True)
            prs_dir.mkdir(parents=True)

            perf_record = {
                "repo": "deepspeedai/DeepSpeed",
                "number": 7886,
                "title": "Fix hook count performance regression",
                "gpt_result": {
                    "perf_kind": "latency",
                    "measurement_hint": "latency_ms",
                    "workload_hint": "backward pass with ZeRO hooks",
                    "search_hint": "inspect stage_1_and_2.py and test_zero_hook_count_regression.py",
                },
            }
            pr_record = {
                "repo": "deepspeedai/DeepSpeed",
                "number": 7886,
                "base_sha": "base123",
                "head_sha": "head456",
                "body": "body",
            }

            (perf_dir / "perf_accepted.jsonl").write_text(json.dumps(perf_record) + "\n", encoding="utf-8")
            (prs_dir / "7886.json").write_text(json.dumps(pr_record), encoding="utf-8")

            sample = perf_repro.load_perf_sample(root, "deepspeedai/DeepSpeed", 7886)

            self.assertEqual(sample["base_sha"], "base123")
            self.assertEqual(sample["head_sha"], "head456")
            self.assertEqual(sample["gpt_result"]["perf_kind"], "latency")

    def test_render_perf_prompt_includes_strict_output_contract(self):
        from swe_gpu_bench.generate import perf_repro

        prompt = perf_repro.render_perf_prompt(
            "Base prompt.\n",
            {
                "repo": "deepspeedai/DeepSpeed",
                "number": 7886,
                "base_sha": "base123",
                "head_sha": "head456",
                "gpt_result": {"perf_kind": "latency"},
            },
        )

        self.assertIn("BEGIN_FINAL_RESULT_JSON", prompt)
        self.assertIn("END_FINAL_RESULT_JSON", prompt)
        self.assertIn("perf_setup_commands", prompt)
        self.assertIn("perf_commands", prompt)
        self.assertIn("status", prompt)


class PerfReproExecutionTests(unittest.TestCase):
    def test_execute_perf_repro_replays_base_and_head_and_writes_ready_result(self):
        from swe_gpu_bench.generate import perf_repro

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            perf_dir = root / "data" / "perf_reviewed" / "deepspeedai__DeepSpeed"
            prs_dir = root / "data" / "prs" / "deepspeedai__DeepSpeed"
            prompt_path = root / "prompts" / "agent_perf_repro.md"
            template_codex_dir = root / "template_codex"
            case_root = root / "PerfCases"

            perf_dir.mkdir(parents=True)
            prs_dir.mkdir(parents=True)
            prompt_path.parent.mkdir(parents=True)
            template_codex_dir.mkdir(parents=True)
            prompt_path.write_text("Base prompt.\n", encoding="utf-8")

            (perf_dir / "perf_accepted.jsonl").write_text(
                json.dumps(
                    {
                        "repo": "deepspeedai/DeepSpeed",
                        "number": 7886,
                        "title": "Fix hook count performance regression",
                        "gpt_result": {
                            "perf_kind": "latency",
                            "measurement_hint": "latency_ms",
                            "workload_hint": "backward pass with ZeRO hooks",
                            "search_hint": "inspect tests/unit/v1/zero/test_zero_hook_count_regression.py",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (prs_dir / "7886.json").write_text(
                json.dumps(
                    {
                        "repo": "deepspeedai/DeepSpeed",
                        "number": 7886,
                        "base_sha": "base123",
                        "head_sha": "head456",
                    }
                ),
                encoding="utf-8",
            )

            def fake_clone(case_paths, repo_slug, head_sha, pr_number=None, cache_root=None):
                (case_paths.repo_dir / "README.md").write_text(f"{repo_slug}@{head_sha}\n", encoding="utf-8")

            def fake_invoke(_case_paths, extra_args=None):
                return CodexInvocationResult(
                    command=["codex", "exec"],
                    exit_code=0,
                    stdout_path=case_root / "DeepSpeed-pr7886" / "logs" / "codex.session.jsonl",
                    stderr_path=case_root / "DeepSpeed-pr7886" / "logs" / "codex.stderr.log",
                    parsed_result={
                        "status": "reproduced",
                        "failure_category": None,
                        "summary": "ok",
                        "setup_commands": ["setup"],
                        "test_commands": ["test"],
                        "perf_setup_commands": ["perf-setup"],
                        "perf_commands": ["perf"],
                        "environment_summary": "env",
                        "perf_plan_summary": "plan",
                        "evidence": ["e1"],
                    },
                )

            def fake_run_recorded(case_paths, commands_by_phase, checkout_label):
                return {
                    "checkout": checkout_label,
                    "setup_records": [{"phase": "setup", "command": "setup", "exit_code": 0}],
                    "test_records": [{"phase": "test", "command": "test", "exit_code": 0}],
                    "perf_setup_records": [{"phase": "perf_setup", "command": "perf-setup", "exit_code": 0}],
                    "perf_records": [{"phase": "perf", "command": "perf", "exit_code": 0}],
                }

            with mock.patch("swe_gpu_bench.generate.perf_repro.invoke_codex", side_effect=fake_invoke), \
                 mock.patch("swe_gpu_bench.generate.perf_repro.run_perf_replay", side_effect=fake_run_recorded):
                result = perf_repro.execute_perf_repro(
                    data_root=root,
                    repo_slug="deepspeedai/DeepSpeed",
                    pr_number=7886,
                    case_root=case_root,
                    prompt_template_path=prompt_path,
                    template_codex_dir=template_codex_dir,
                    clone_repo_fn=fake_clone,
                    checkout_commit_fn=lambda *args: None,
                )

            self.assertEqual(result["status"], "ready")
            self.assertEqual(result["base_commit"], "base123")
            self.assertEqual(result["head_commit"], "head456")
            self.assertEqual(result["setup_commands"], ["setup"])
            self.assertEqual(result["perf_setup_commands"], ["perf-setup"])
            self.assertEqual(result["perf_commands"], ["perf"])
            self.assertEqual(result["base_perf_results"][0]["command"], "perf")
            self.assertEqual(result["head_perf_results"][0]["command"], "perf")

    def test_execute_perf_repro_blocks_when_base_correctness_fails(self):
        from swe_gpu_bench.generate import perf_repro

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            perf_dir = root / "data" / "perf_reviewed" / "deepspeedai__DeepSpeed"
            prs_dir = root / "data" / "prs" / "deepspeedai__DeepSpeed"
            prompt_path = root / "prompts" / "agent_perf_repro.md"
            template_codex_dir = root / "template_codex"
            case_root = root / "PerfCases"

            perf_dir.mkdir(parents=True)
            prs_dir.mkdir(parents=True)
            prompt_path.parent.mkdir(parents=True)
            template_codex_dir.mkdir(parents=True)
            prompt_path.write_text("Base prompt.\n", encoding="utf-8")

            (perf_dir / "perf_accepted.jsonl").write_text(
                json.dumps(
                    {
                        "repo": "deepspeedai/DeepSpeed",
                        "number": 7886,
                        "title": "Fix hook count performance regression",
                        "gpt_result": {"perf_kind": "latency", "measurement_hint": "latency_ms"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (prs_dir / "7886.json").write_text(
                json.dumps(
                    {
                        "repo": "deepspeedai/DeepSpeed",
                        "number": 7886,
                        "base_sha": "base123",
                        "head_sha": "head456",
                    }
                ),
                encoding="utf-8",
            )

            def fake_clone(case_paths, repo_slug, head_sha, pr_number=None, cache_root=None):
                (case_paths.repo_dir / "README.md").write_text(f"{repo_slug}@{head_sha}\n", encoding="utf-8")

            def fake_invoke(_case_paths, extra_args=None):
                return CodexInvocationResult(
                    command=["codex", "exec"],
                    exit_code=0,
                    stdout_path=case_root / "DeepSpeed-pr7886" / "logs" / "codex.session.jsonl",
                    stderr_path=case_root / "DeepSpeed-pr7886" / "logs" / "codex.stderr.log",
                    parsed_result={
                        "status": "reproduced",
                        "failure_category": None,
                        "summary": "ok",
                        "setup_commands": ["setup"],
                        "test_commands": ["test"],
                        "perf_setup_commands": ["perf-setup"],
                        "perf_commands": ["perf"],
                        "environment_summary": "env",
                        "perf_plan_summary": "plan",
                        "evidence": ["e1"],
                    },
                )

            def fake_run_recorded(case_paths, commands_by_phase, checkout_label):
                test_exit = 1 if checkout_label == "base" else 0
                return {
                    "checkout": checkout_label,
                    "setup_records": [{"phase": "setup", "command": "setup", "exit_code": 0}],
                    "test_records": [{"phase": "test", "command": "test", "exit_code": test_exit}],
                    "perf_setup_records": [],
                    "perf_records": [],
                }

            with mock.patch("swe_gpu_bench.generate.perf_repro.invoke_codex", side_effect=fake_invoke), \
                 mock.patch("swe_gpu_bench.generate.perf_repro.run_perf_replay", side_effect=fake_run_recorded):
                result = perf_repro.execute_perf_repro(
                    data_root=root,
                    repo_slug="deepspeedai/DeepSpeed",
                    pr_number=7886,
                    case_root=case_root,
                    prompt_template_path=prompt_path,
                    template_codex_dir=template_codex_dir,
                    clone_repo_fn=fake_clone,
                    checkout_commit_fn=lambda *args: None,
                )

            self.assertEqual(result["status"], "blocked")
            self.assertEqual(result["failure_category"], "BASE_CORRECTNESS_FAILURE")


class PerfReproCliTests(unittest.TestCase):
    def test_build_parser_defaults_target_model_to_gpt51_codex(self):
        from swe_gpu_bench.generate import perf_repro

        parser = argparse.ArgumentParser()
        perf_repro.add_arguments(parser)
        args = parser.parse_args(
            [
                "--repo",
                "NVIDIA/cccl",
                "--pr",
                "7381",
            ]
        )

        self.assertEqual(args.target_model, "gpt-5.1-codex-2025-11-13")

    def test_execute_perf_repro_passes_target_model_to_codex(self):
        from swe_gpu_bench.generate import perf_repro

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            perf_dir = root / "data" / "perf_reviewed" / "NVIDIA__cccl"
            prs_dir = root / "data" / "prs" / "NVIDIA__cccl"
            prompt_path = root / "prompts" / "agent_perf_repro.md"
            template_codex_dir = root / "template_codex"
            case_root = root / "PerfCases"

            perf_dir.mkdir(parents=True)
            prs_dir.mkdir(parents=True)
            prompt_path.parent.mkdir(parents=True)
            template_codex_dir.mkdir(parents=True)
            prompt_path.write_text("Base prompt.\n", encoding="utf-8")
            (perf_dir / "perf_accepted.jsonl").write_text(
                json.dumps({"repo": "NVIDIA/cccl", "number": 7381, "gpt_result": {}}) + "\n",
                encoding="utf-8",
            )
            (prs_dir / "7381.json").write_text(
                json.dumps({"repo": "NVIDIA/cccl", "number": 7381, "base_sha": "base", "head_sha": "head"}),
                encoding="utf-8",
            )

            seen = {}

            def fake_clone(case_paths, repo_slug, head_sha, pr_number=None, cache_root=None):
                (case_paths.repo_dir / "README.md").write_text(f"{repo_slug}@{head_sha}\n", encoding="utf-8")

            def fake_invoke(_case_paths, extra_args=None):
                seen["extra_args"] = extra_args
                return CodexInvocationResult(
                    command=["codex", "exec"],
                    exit_code=0,
                    stdout_path=case_root / "cccl-pr7381" / "logs" / "codex.session.jsonl",
                    stderr_path=case_root / "cccl-pr7381" / "logs" / "codex.stderr.log",
                    parsed_result={"status": "blocked", "failure_category": "X", "summary": "x"},
                )

            with mock.patch("swe_gpu_bench.generate.perf_repro.invoke_codex", side_effect=fake_invoke):
                perf_repro.execute_perf_repro(
                    data_root=root,
                    repo_slug="NVIDIA/cccl",
                    pr_number=7381,
                    case_root=case_root,
                    prompt_template_path=prompt_path,
                    template_codex_dir=template_codex_dir,
                    target_model="gpt-5.1-codex-2025-11-13",
                    clone_repo_fn=fake_clone,
                    checkout_commit_fn=lambda *args: None,
                )

        self.assertEqual(seen["extra_args"], ["--model", "gpt-5.1-codex-2025-11-13"])


def replay(setup=0, test=(0,), perf_setup=(), perf=(0,)):
    return {
        "setup_records": [{"command": "setup", "exit_code": setup}],
        "test_records": [{"command": "test", "exit_code": code} for code in test],
        "perf_setup_records": [{"command": "perf-setup", "exit_code": code} for code in perf_setup],
        "perf_records": [{"command": "perf", "exit_code": code} for code in perf],
    }


class PerfReproResultTests(unittest.TestCase):
    sample = {"repo": "NVIDIA/cccl", "number": 7381, "base_sha": "base", "head_sha": "head"}

    def category(self, base, head):
        from swe_gpu_bench.generate import perf_repro

        result = perf_repro.build_perf_result(self.sample, {"status": "reproduced"}, base, head)
        return result["status"], result["failure_category"]

    def test_ready_needs_setup_correctness_and_perf_to_pass_on_both_commits(self):
        self.assertEqual(self.category(replay(), replay()), ("ready", None))
        self.assertEqual(self.category(replay(setup=1), replay()), ("blocked", "BASE_SETUP_FAILURE"))
        self.assertEqual(self.category(replay(), replay(perf_setup=(0, 2))), ("blocked", "HEAD_SETUP_FAILURE"))
        self.assertEqual(self.category(replay(test=(1, 0)), replay()), ("blocked", "BASE_CORRECTNESS_FAILURE"))
        self.assertEqual(self.category(replay(test=()), replay()), ("blocked", "BASE_CORRECTNESS_FAILURE"))
        self.assertEqual(self.category(replay(), replay(perf=(1,))), ("blocked", "HEAD_PERF_FAILURE"))
        self.assertEqual(self.category(replay(), replay(perf=())), ("blocked", "HEAD_PERF_FAILURE"))


if __name__ == "__main__":
    unittest.main()

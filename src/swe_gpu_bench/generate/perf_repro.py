"""``generate perf-repro``: discover correctness + perf commands, then replay them on base and head.

Reads ``data/perf_reviewed/<owner__repo>/perf_accepted.jsonl`` (from ``collect perf-review``) and
writes ``<case-root>/<repo-tail>-pr<N>/logs/result.json`` with ``status`` ``ready`` when, on both
commits, every setup command passes and the correctness and perf commands exist and all pass.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from swe_gpu_bench.core.git import checkout_commit, clone_or_update_repo
from swe_gpu_bench.core.io import repo_dir, write_json
from swe_gpu_bench.core.shell import all_commands_succeeded, run_logged_shell_commands
from swe_gpu_bench.generate.repro import (
    RESULT_END_MARKER,
    RESULT_START_MARKER,
    CasePaths,
    build_repository_preparation_failure,
    build_result_from_invocation,
    invoke_codex,
    load_base_prompt,
    load_scored_sample,
    prepare_case_directory,
    sample_json_block,
)

DEFAULT_TARGET_MODEL = "gpt-5.1-codex-2025-11-13"
PHASES = ("setup", "test", "perf_setup", "perf")


def load_perf_sample(data_root: Path, repo_slug: str, pr_number: int) -> Dict[str, Any]:
    perf_path = repo_dir(data_root / "data" / "perf_reviewed", repo_slug) / "perf_accepted.jsonl"
    if not perf_path.exists():
        raise FileNotFoundError(f"Missing perf review file for {repo_slug}: {perf_path}")
    return load_scored_sample(perf_path, data_root / "data" / "prs", repo_slug, pr_number)


def render_perf_prompt(base_template: str, sample: Dict[str, Any]) -> str:
    contract = """
==================================================
Perf Discovery Contract
==================================================

Use the perf review hints embedded in `gpt_result` to discover:
- setup_commands
- test_commands
- perf_setup_commands
- perf_commands

Return those fields in the same result JSON contract as repro, plus:
- perf_plan_summary

Your final response must include exactly one JSON object between these marker lines:
%s
<json object here>
%s

The JSON object must contain these keys:
- `status`: one of `reproduced`, `blocked`, `failed`
- `failure_category`: string or null
- `summary`: short human-readable verdict
- `setup_commands`: array of exact shell commands actually used to prepare the environment
- `test_commands`: array of exact shell commands actually used to run correctness checks
- `perf_setup_commands`: array of exact shell commands actually used to prepare perf-only dependencies
- `perf_commands`: array of exact shell commands actually used to measure the workload
- `environment_summary`: short explanation of the chosen environment/build flow
- `perf_plan_summary`: short explanation of the measured workload and metric
- `evidence`: array of short evidence strings grounded in actual execution

If correctness and perf commands were discovered and executed successfully, set:
- `status` = `reproduced`
- `failure_category` = null

Do not include failed commands, exploratory/debugging commands, superseded commands, or absolute paths in command arrays.

==================================================
Sample JSON
==================================================

```json
%s
```
""" % (RESULT_START_MARKER, RESULT_END_MARKER, sample_json_block(sample))
    return base_template.rstrip() + "\n\n" + contract.lstrip()


def run_perf_replay(case_paths: CasePaths, commands_by_phase: Dict[str, List[str]], checkout_label: str) -> Dict[str, Any]:
    replay: Dict[str, Any] = {"checkout": checkout_label}
    for phase in PHASES:
        replay[f"{phase}_records"] = run_logged_shell_commands(
            commands_by_phase.get(f"{phase}_commands", []),
            phase=phase,
            cwd=case_paths.case_dir,
            artifacts_dir=case_paths.artifacts_dir,
            command_logs_dir=case_paths.logs_dir,
            log_prefix=f"{checkout_label}-",
            include_cwd=False,
        )
    return replay


def build_perf_result(
    sample: Dict[str, Any],
    invocation_result: Dict[str, Any],
    base_replay: Optional[Dict[str, Any]],
    head_replay: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    result = dict(invocation_result)
    result.update(
        {
            "repo": sample["repo"],
            "pr_number": sample["number"],
            "base_commit": sample.get("base_sha"),
            "head_commit": sample.get("head_sha"),
            "perf_context": dict(sample.get("gpt_result") or {}),
            "base_correctness_results": [],
            "head_correctness_results": [],
            "base_perf_results": [],
            "head_perf_results": [],
        }
    )
    if base_replay is None or head_replay is None:
        result.setdefault("status", "failed")
        return result

    result["base_correctness_results"] = [*base_replay["setup_records"], *base_replay["test_records"]]
    result["head_correctness_results"] = [*head_replay["setup_records"], *head_replay["test_records"]]
    result["base_perf_results"] = list(base_replay["perf_records"])
    result["head_perf_results"] = list(head_replay["perf_records"])

    for label, replay in (("base", base_replay), ("head", head_replay)):
        failure = _replay_failure(replay)
        if failure is not None:
            result.update(status="blocked", failure_category=f"{label.upper()}_{failure}_FAILURE",
                          summary=f"{label.capitalize()} commit {failure.lower()} commands did not pass.")
            return result
    result.update(status="ready", failure_category=None,
                  summary="Perf workflow recorded for base/head comparison.")
    return result


def _replay_failure(replay: Dict[str, Any]) -> Optional[str]:
    if any(record.get("exit_code") != 0 for record in [*replay["setup_records"], *replay["perf_setup_records"]]):
        return "SETUP"
    if not all_commands_succeeded(replay["test_records"]):
        return "CORRECTNESS"
    if not all_commands_succeeded(replay["perf_records"]):
        return "PERF"
    return None


def execute_perf_repro(
    data_root: Path,
    repo_slug: str,
    pr_number: int,
    case_root: Path,
    prompt_template_path: Path,
    template_codex_dir: Optional[Path],
    target_model: str = DEFAULT_TARGET_MODEL,
    clone_repo_fn=None,
    checkout_commit_fn: Callable[..., None] = checkout_commit,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    emit = progress_callback or (lambda message: None)
    emit("Loading perf sample metadata")
    sample = load_perf_sample(data_root, repo_slug, pr_number)
    emit("Preparing perf case directory")
    case_paths = prepare_case_directory(
        case_root=case_root,
        repo_slug=repo_slug,
        pr_number=pr_number,
        template_codex_dir=template_codex_dir,
        sample=sample,
        prompt_text=render_perf_prompt(load_base_prompt(prompt_template_path), sample),
    )

    def finish(result: Dict[str, Any]) -> Dict[str, Any]:
        write_json(case_paths.result_path, result)
        return result

    emit("Preparing repository checkout")
    try:
        (clone_repo_fn or clone_or_update_repo)(case_paths, repo_slug, sample.get("head_sha"), sample["number"])
    except subprocess.CalledProcessError as exc:
        return finish(build_perf_result(sample, build_repository_preparation_failure(sample, case_paths, exc), None, None))

    emit("Running Codex for perf discovery")
    invocation = build_result_from_invocation(sample, invoke_codex(case_paths, extra_args=["--model", target_model]))
    if invocation.get("status") != "reproduced":
        return finish(build_perf_result(sample, invocation, None, None))

    commands_by_phase = {f"{phase}_commands": list(invocation.get(f"{phase}_commands") or []) for phase in PHASES}
    replays = {}
    for label, commit in (("base", sample["base_sha"]), ("head", sample["head_sha"])):
        emit(f"Replaying commands on {label} commit")
        checkout_commit_fn(case_paths.repo_dir, commit, sample["number"])
        replays[label] = run_perf_replay(case_paths, commands_by_phase, label)
    return finish(build_perf_result(sample, invocation, replays["base"], replays["head"]))


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo", required=True)
    parser.add_argument("--pr", required=True, type=int)
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--case-root", default="results/perf_repro")
    parser.add_argument("--prompt-template", default="prompts/agent_perf_repro.md")
    parser.add_argument("--template-codex-dir", default=None, help="Optional .codex template directory to copy.")
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL)


def run(args: argparse.Namespace) -> int:
    data_root = Path(args.data_root).resolve()
    result = execute_perf_repro(
        data_root=data_root,
        repo_slug=args.repo,
        pr_number=args.pr,
        case_root=(data_root / args.case_root).resolve(),
        prompt_template_path=(data_root / args.prompt_template).resolve(),
        template_codex_dir=(data_root / args.template_codex_dir).resolve() if args.template_codex_dir else None,
        target_model=args.target_model,
        progress_callback=lambda message: print(f"[perf-repro] {message}", flush=True),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0

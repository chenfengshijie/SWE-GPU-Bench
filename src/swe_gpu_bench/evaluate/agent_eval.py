"""``evaluate agent``: let an agent solve one dataset instance, then grade it with the hidden tests.

Stages: clone at ``base_commit`` -> ``setup_commands`` -> hide later git history -> agent ->
capture ``git diff`` -> re-run setup if it builds the repo -> apply ``test_patch`` -> ``test_commands``.
Writes ``<eval-root>/<instance_id>/result.json``.

The helpers here (case layout, command phases, post-patch grading) are shared by the oracle and
perf evaluators.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from swe_gpu_bench.core.agent import AgentInvocationResult, AgentRunner, agent_runner_from_args
from swe_gpu_bench.core.config import add_profile_arguments
from swe_gpu_bench.core.git import clone_or_update_repo, hide_future_history, repo_cache_root
from swe_gpu_bench.core.io import find_record, write_json, write_text
from swe_gpu_bench.core.patch import apply_test_patch, capture_diff, is_test_patch_failure_status
from swe_gpu_bench.core.shell import (
    did_commands_succeed,
    format_called_process_error,
    run_logged_shell_commands,
    setup_builds_repo,
    strip_phase,
)

AGENT_VISIBLE_KEYS = (
    "instance_id",
    "repo",
    "base_commit",
    "problem_statement",
    "task_type",
    "setup_commands",
    "docker_image",
)
DEFAULT_DATASET = "data/dataset/swe_gpu_bench.jsonl"


@dataclass
class EvalCasePaths:
    case_dir: Path          # cwd for every command
    repo_dir: Path
    inputs_dir: Path
    artifacts_dir: Path
    command_logs_dir: Path
    instance_path: Path     # inputs/instance.json (agent-visible fields only)
    prompt_path: Path       # inputs/prompt.md
    agent_stdout_log: Path
    agent_stderr_log: Path
    result_path: Path


def load_eval_record(dataset_path: Path, instance_id: str) -> Dict[str, Any]:
    return find_record(dataset_path, "instance_id", instance_id)


def agent_visible_record(record: Dict[str, Any], keys: Sequence[str] = AGENT_VISIBLE_KEYS) -> Dict[str, Any]:
    """Allowlisted fields only: new dataset fields (gold patch, head commit, test commands, ...) stay hidden by default."""
    return {key: record[key] for key in keys if key in record}


def build_evaluation_stats(
    setup_records: List[Dict[str, Any]],
    post_patch_setup_records: List[Dict[str, Any]],
    test_records: List[Dict[str, Any]],
    elapsed_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    all_records = [*setup_records, *post_patch_setup_records, *test_records]
    return {
        "elapsed_seconds": elapsed_seconds,
        "total_commands": len(all_records),
        "setup_commands": len(setup_records),
        "post_patch_setup_commands": len(post_patch_setup_records),
        "test_commands": len(test_records),
        "failed_commands": sum(1 for record in all_records if record["exit_code"] != 0),
        "last_exit_code": all_records[-1]["exit_code"] if all_records else None,
    }


def render_eval_prompt(template_text: str, record: Dict[str, Any]) -> str:
    setup_commands = "\n".join(f"  {i + 1}. `{cmd}`" for i, cmd in enumerate(record.get("setup_commands", [])))
    dynamic = f"""
==================================================
Task
==================================================

{record.get("problem_statement", "")}

==================================================
Repository
==================================================

The repository is located at `repo/`.
The workflow has already executed the following setup commands and prepared the environment:

{setup_commands}

Reuse the prepared environment under `artifacts/` instead of rebuilding it from scratch.
If your code changes require reinstalling or recompiling the repository, do the minimum necessary
validation steps using the existing environment layout.

==================================================
Working Directory Layout
==================================================

- `repo/`         — repository source code (make your changes here)
- `artifacts/`    — use this for virtual environments, build cache, and outputs
  - `artifacts/.venv/`     — recommended venv location
  - `artifacts/.uv-cache/` — uv cache
  - `artifacts/.cache/`    — XDG cache

Do not create files outside the current working directory.
"""
    return template_text.rstrip() + "\n\n" + dynamic.lstrip()


def prepare_eval_case_dir(
    record: Dict[str, Any],
    eval_root: Path,
    prompt_template_path: Optional[Path] = None,
    force: bool = False,
    *,
    work_root: Optional[Path] = None,
) -> EvalCasePaths:
    """Create ``<eval_root>/<instance_id>``.

    With ``work_root`` the checkout and environment live in ``<work_root>/<instance_id>`` (always
    wiped) while inputs, logs and ``result.json`` stay under ``eval_root``.
    """
    instance_id: str = record["instance_id"]
    result_dir = eval_root / instance_id
    if result_dir.exists():
        if not force:
            raise FileExistsError(f"Eval case already exists: {result_dir}")
        shutil.rmtree(result_dir)
    work_dir = (work_root / instance_id) if work_root is not None else result_dir
    if work_root is not None and work_dir.exists():
        shutil.rmtree(work_dir)

    inputs_dir = result_dir / "inputs"
    artifacts_dir = work_dir / "artifacts"
    command_logs_dir = (result_dir / "command-logs") if work_root is not None else artifacts_dir / "command-logs"
    paths = EvalCasePaths(
        case_dir=work_dir,
        repo_dir=work_dir / "repo",
        inputs_dir=inputs_dir,
        artifacts_dir=artifacts_dir,
        command_logs_dir=command_logs_dir,
        instance_path=inputs_dir / "instance.json",
        prompt_path=inputs_dir / "prompt.md",
        agent_stdout_log=command_logs_dir / "agent.stdout.jsonl",
        agent_stderr_log=command_logs_dir / "agent.stderr.log",
        result_path=result_dir / "result.json",
    )
    for path in (paths.repo_dir, inputs_dir, artifacts_dir, command_logs_dir):
        path.mkdir(parents=True, exist_ok=True)
    write_json(paths.instance_path, agent_visible_record(record))
    if prompt_template_path is not None:
        write_text(paths.prompt_path, render_eval_prompt(prompt_template_path.read_text(encoding="utf-8"), record))
    return paths


def run_commands(commands: List[str], phase: str, case_paths: EvalCasePaths) -> List[Dict[str, Any]]:
    return run_logged_shell_commands(
        commands,
        phase=phase,
        cwd=case_paths.case_dir,
        artifacts_dir=case_paths.artifacts_dir,
        command_logs_dir=case_paths.command_logs_dir,
        include_cwd=False,
    )


def clone_record_repo(case_paths: EvalCasePaths, record: Dict[str, Any], cache_default: Path) -> None:
    clone_or_update_repo(
        case_paths,
        record["repo"],
        record.get("base_commit"),
        pr_number=record.get("pr_number"),
        cache_root=repo_cache_root(cache_default),
    )


def grade_patched_repo(
    record: Dict[str, Any],
    case_paths: EvalCasePaths,
    *,
    rerun_setup: bool = True,
    emit: Callable[[str], None] = lambda message: None,
) -> tuple[List[Dict[str, Any]], str, List[Dict[str, Any]]]:
    """Refresh the build if needed, apply ``test_patch`` and run ``test_commands``.

    Returns ``(post_patch_setup_records, test_patch_status, test_records)``.
    """
    setup_commands = record.get("setup_commands", [])
    post_patch_setup_records: List[Dict[str, Any]] = []
    if rerun_setup and setup_builds_repo(setup_commands):
        emit("Re-running setup_commands to refresh the repo build after patching")
        post_patch_setup_records = run_commands(setup_commands, "post_patch_setup", case_paths)

    test_patch = record.get("test_patch") or ""
    test_patch_status = apply_test_patch(test_patch, case_paths.repo_dir) if test_patch.strip() else "no_test_patch"
    emit(f"test_patch status: {test_patch_status}")
    if is_test_patch_failure_status(test_patch_status):
        return post_patch_setup_records, test_patch_status, []

    emit("Running test_commands")
    return post_patch_setup_records, test_patch_status, run_commands(record.get("test_commands", []), "test", case_paths)


def test_outcome(test_records: List[Dict[str, Any]]) -> tuple[str, Optional[str]]:
    if not test_records:
        return "error", "TEST_EXECUTION_FAILURE"
    return ("passed", None) if all(record["exit_code"] == 0 for record in test_records) else ("failed", None)


def _determine_status(
    setup_records: List[Dict[str, Any]],
    post_patch_setup_records: List[Dict[str, Any]],
    test_records: List[Dict[str, Any]],
    test_patch_status: str,
) -> tuple[str, Optional[str]]:
    """Setup failures only matter when no test ran; passing tests still count as passed."""
    if is_test_patch_failure_status(test_patch_status):
        return "error", "TEST_PATCH_APPLY_FAILURE"
    if not test_records and not (did_commands_succeed(setup_records) and did_commands_succeed(post_patch_setup_records)):
        return "error", "SETUP_FAILURE"
    return test_outcome(test_records)


def build_eval_result(
    record: Dict[str, Any],
    case_paths: EvalCasePaths,
    agent_name: str,
    started_at: datetime,
    *,
    agent_result: Optional[AgentInvocationResult] = None,
    agent_diff: str = "",
    setup_records: Sequence[Dict[str, Any]] = (),
    post_patch_setup_records: Sequence[Dict[str, Any]] = (),
    test_patch_status: Optional[str] = None,
    test_records: Sequence[Dict[str, Any]] = (),
    failure_category: Optional[str] = None,
    summary: Optional[str] = None,
) -> Dict[str, Any]:
    """Final result; pass ``failure_category`` (+ ``summary``) for an early error."""
    setup_records, post_patch_setup_records, test_records = list(setup_records), list(post_patch_setup_records), list(test_records)
    if failure_category is not None:
        status = "error"
    else:
        status, failure_category = _determine_status(setup_records, post_patch_setup_records, test_records, test_patch_status or "")
    finished_at = datetime.now(timezone.utc)
    elapsed = round((finished_at - started_at).total_seconds(), 2)
    result = {
        "instance_id": record.get("instance_id", ""),
        "repo": record.get("repo", ""),
        "base_commit": record.get("base_commit", ""),
        "head_commit": record.get("head_commit", ""),
        "agent": agent_name,
        "status": status,
        "failure_category": failure_category,
        "agent_exit_code": agent_result.exit_code if agent_result else None,
        "agent_command": agent_result.command if agent_result else [],
        "agent_error_message": agent_result.error_message if agent_result else None,
        "agent_diff": agent_diff,
        "agent_stats": agent_result.agent_stats if agent_result else {},
        "evaluation_stats": build_evaluation_stats(setup_records, post_patch_setup_records, test_records, elapsed_seconds=elapsed),
        "test_patch_status": test_patch_status,
        "setup_commands": record.get("setup_commands", []),
        "setup_results": strip_phase(setup_records),
        "post_patch_setup_results": strip_phase(post_patch_setup_records),
        "test_commands": record.get("test_commands", []),
        "test_results": strip_phase(test_records),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "elapsed_seconds": elapsed,
        "case_dir": str(case_paths.case_dir),
        "agent_stdout_log": str(case_paths.agent_stdout_log),
        "agent_stderr_log": str(case_paths.agent_stderr_log),
    }
    if summary is not None:
        result["summary"] = summary
    return result


def execute_eval(
    instance_id: str,
    dataset_path: Path,
    eval_root: Path,
    prompt_template_path: Path,
    agent_runner: AgentRunner,
    force: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
    keep_git_history: bool = False,
) -> Dict[str, Any]:
    emit = progress_callback or (lambda message: None)
    started_at = datetime.now(timezone.utc)
    agent_name = agent_runner.name
    record = load_eval_record(dataset_path, instance_id)
    case_paths = prepare_eval_case_dir(record, eval_root, prompt_template_path, force=force)

    def finish(**kwargs: Any) -> Dict[str, Any]:
        result = build_eval_result(record, case_paths, agent_name, started_at, **kwargs)
        write_json(case_paths.result_path, result)
        emit(f"Done — status: {result['status']}")
        return result

    emit(f"Cloning repo at base_commit {str(record.get('base_commit', ''))[:12]}")
    try:
        clone_record_repo(case_paths, record, eval_root / "_repo_cache")
    except subprocess.CalledProcessError as exc:
        return finish(
            failure_category="REPOSITORY_PREPARATION_FAILURE",
            summary="Repository preparation failed: " + format_called_process_error(exc),
        )

    emit("Running setup_commands")
    setup_records = run_commands(record.get("setup_commands", []), "setup", case_paths)
    if not did_commands_succeed(setup_records):
        emit("setup_commands failed; skipping the agent")
        return finish(setup_records=setup_records, test_patch_status="not_run")

    if not keep_git_history:
        try:
            hide_future_history(case_paths.repo_dir)
        except subprocess.CalledProcessError as exc:
            return finish(
                setup_records=setup_records,
                failure_category="REPOSITORY_PREPARATION_FAILURE",
                summary="Hiding git history failed: " + format_called_process_error(exc),
            )

    emit(f"Invoking agent ({agent_name})")
    agent_result = agent_runner.invoke(case_paths)
    if agent_result.error_message:
        emit(f"Agent warning: {agent_result.error_message}")
    agent_diff = capture_diff(case_paths.repo_dir)
    write_text(case_paths.artifacts_dir / "agent.diff", agent_diff)

    post_patch_setup_records, test_patch_status, test_records = grade_patched_repo(record, case_paths, emit=emit)
    return finish(
        agent_result=agent_result,
        agent_diff=agent_diff,
        setup_records=setup_records,
        post_patch_setup_records=post_patch_setup_records,
        test_patch_status=test_patch_status,
        test_records=test_records,
    )


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--eval-root", default="results/agent_eval")
    parser.add_argument("--prompt-template", default="prompts/agent_eval.md")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--keep-git-history",
        action="store_true",
        help="Leave branches, tags and remotes in the checkout (by default commits after base_commit are unreachable)",
    )
    add_profile_arguments(parser, kind="agent")


def run(args: argparse.Namespace) -> int:
    result = execute_eval(
        instance_id=args.instance_id,
        dataset_path=Path(args.dataset).resolve(),
        eval_root=Path(args.eval_root).resolve(),
        prompt_template_path=Path(args.prompt_template).resolve(),
        agent_runner=agent_runner_from_args(args),
        force=args.force,
        progress_callback=lambda message: print(f"[eval] {message}", flush=True),
        keep_git_history=args.keep_git_history,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0

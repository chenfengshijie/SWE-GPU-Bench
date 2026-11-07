"""``evaluate perf-agent``: restore a perf instance's environment and let an agent write a patch.

Only the patch is produced here (``<eval-root>/<instance_id>/artifacts/agent.diff``); grade it with
``evaluate perf-validate --patch-source agent``. The checkout and environment live under
``--work-root`` so the result directory stays small.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from swe_gpu_bench.core.agent import AgentInvocationResult, AgentRunner, agent_runner_from_args
from swe_gpu_bench.core.config import add_profile_arguments
from swe_gpu_bench.core.git import clone_or_update_repo, hide_future_history, repo_cache_root
from swe_gpu_bench.core.io import write_json, write_text
from swe_gpu_bench.core.patch import capture_diff
from swe_gpu_bench.core.shell import did_commands_succeed, format_called_process_error, strip_phase
from swe_gpu_bench.evaluate.agent_eval import (
    EvalCasePaths,
    agent_visible_record,
    load_eval_record,
    prepare_eval_case_dir,
    run_commands,
)

DEFAULT_DATASET = "data/perf_dataset/swe_gpu_bench_perf.jsonl"
LIGHTWEIGHT_ARTIFACT_LIMIT_BYTES = 10 * 1024 * 1024
_EXCLUDED_ARTIFACT_DIRS = frozenset({".venv", ".uv-cache", ".cache", "command-logs", "__pycache__"})
PERF_AGENT_VISIBLE_KEYS = ("instance_id", "repo", "base_commit", "problem_statement", "task_type", "docker_image")
_MANIFEST_PATHS = (
    "repo",
    "artifacts",
    "artifacts/.venv",
    "artifacts/venv",
    "artifacts/.uv-cache",
    "artifacts/.cache",
    "artifacts/overlay",
    "artifacts/bazelisk",
    "artifacts/bazelisk_cache",
    "artifacts/cccl-release-build",
    "artifacts/cub-build",
)
_RESTORE_PHASES = ("setup", "test", "perf_setup", "perf")
_HARD_RESTORE_PHASES = frozenset({"setup", "perf_setup"})


@dataclass
class PerfRestoreState:
    mode: str
    records: Dict[str, List[Dict[str, Any]]] = field(default_factory=lambda: {phase: [] for phase in _RESTORE_PHASES})
    failed_phase: Optional[str] = None
    soft_failed_phases: List[str] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return self.failed_phase is None

    def serialize(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "succeeded": self.succeeded,
            "failed_phase": self.failed_phase,
            "soft_failed_phases": list(self.soft_failed_phases),
            **{f"{phase}_results": strip_phase(self.records[phase]) for phase in _RESTORE_PHASES},
        }


def render_perf_eval_prompt(template_text: str, record: Dict[str, Any]) -> str:
    guidance = [
        "The repository is located at `repo/`.",
        "The workflow has already reconstructed the reusable environment under `artifacts/` before you start.",
        "Inspect `artifacts/environment_manifest.json` for the prepared directory layout and reusable outputs.",
        "Prefer reusing existing build directories, virtual environments, overlays, caches, and generated artifacts instead of rebuilding them.",
        "Do not ask the harness to rerun correctness or benchmark commands; your job here is to produce the code patch only.",
    ]
    environment_summary = (record.get("environment_summary") or "").strip()
    if environment_summary:
        guidance.append(f"Recorded environment summary: {environment_summary}")
    dynamic = f"""
==================================================
Task
==================================================

{record.get("problem_statement", "")}

==================================================
Repository
==================================================

{chr(10).join(guidance)}

verification_status: `{record.get("verification_status") or "unknown"}`

==================================================
Working Directory Layout
==================================================

- `repo/`         — repository source code (make your changes here)
- `artifacts/`    — reconstructed environment, build cache, overlays, and outputs
- `artifacts/environment_manifest.json` — manifest of prepared paths and restore status

Do not create files outside the current working directory.
"""
    return template_text.rstrip() + "\n\n" + dynamic.lstrip()


def prepare_perf_eval_case_dir(
    record: Dict[str, Any],
    eval_root: Path,
    work_root: Path,
    prompt_template_path: Path,
    force: bool = False,
    template_codex_dir: Optional[Path] = None,
) -> EvalCasePaths:
    case_paths = prepare_eval_case_dir(record, eval_root, None, force, work_root=work_root)
    safe_record = agent_visible_record(record, PERF_AGENT_VISIBLE_KEYS)
    safe_record["environment_manifest_path"] = "artifacts/environment_manifest.json"
    write_json(case_paths.instance_path, safe_record)
    write_text(case_paths.prompt_path, render_perf_eval_prompt(prompt_template_path.read_text(encoding="utf-8"), record))
    if template_codex_dir is not None and template_codex_dir.is_dir():
        shutil.copytree(template_codex_dir, case_paths.case_dir / ".codex", dirs_exist_ok=True)
    return case_paths


def result_dir_of(case_paths: EvalCasePaths) -> Path:
    return case_paths.result_path.parent


def preserve_lightweight_artifacts(
    source_dir: Path,
    destination_dir: Path,
    size_limit_bytes: int = LIGHTWEIGHT_ARTIFACT_LIMIT_BYTES,
) -> List[str]:
    preserved: List[str] = []
    if not source_dir.exists():
        return preserved
    for path in sorted(source_dir.rglob("*")):
        relative = path.relative_to(source_dir)
        if any(part in _EXCLUDED_ARTIFACT_DIRS for part in relative.parts) or path.is_dir() or path.is_symlink():
            continue
        try:
            if path.stat().st_size > size_limit_bytes:
                continue
        except OSError:
            continue
        target = destination_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        preserved.append(relative.as_posix())
    return preserved


def build_perf_evaluation_stats(
    setup_records: List[Dict[str, Any]],
    test_records: List[Dict[str, Any]],
    perf_setup_records: List[Dict[str, Any]],
    perf_records: List[Dict[str, Any]],
    elapsed_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    all_records = [*setup_records, *test_records, *perf_setup_records, *perf_records]
    return {
        "elapsed_seconds": elapsed_seconds,
        "total_commands": len(all_records),
        "setup_commands": len(setup_records),
        "post_patch_setup_commands": 0,
        "test_commands": len(test_records),
        "perf_setup_commands": len(perf_setup_records),
        "perf_commands": len(perf_records),
        "failed_commands": sum(1 for record in all_records if record["exit_code"] != 0),
        "last_exit_code": all_records[-1]["exit_code"] if all_records else None,
    }


def determine_restore_mode(record: Dict[str, Any]) -> str:
    """``ready`` perf cases replay correctness + perf on base; others only rebuild the environment."""
    if (record.get("verification_status") or "").strip().lower() == "ready":
        return "full_base_replay"
    return "setup_and_perf_setup_only"


def restore_perf_eval_environment(
    record: Dict[str, Any],
    case_paths: EvalCasePaths,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> PerfRestoreState:
    emit = progress_callback or (lambda message: None)
    state = PerfRestoreState(mode=determine_restore_mode(record))
    for phase in _RESTORE_PHASES:
        if phase in ("test", "perf") and state.mode != "full_base_replay":
            continue
        commands = list(record.get(f"{phase}_commands", []))
        if not commands:
            continue
        emit(f"Restoring environment via {phase}_commands")
        state.records[phase] = run_commands(commands, phase, case_paths)
        if not did_commands_succeed(state.records[phase]):
            if phase in _HARD_RESTORE_PHASES:
                state.failed_phase = phase
                break
            state.soft_failed_phases.append(phase)
    return state


def write_environment_manifest(record: Dict[str, Any], case_paths: EvalCasePaths, state: PerfRestoreState) -> Path:
    work_dir = case_paths.case_dir
    manifest_path = case_paths.artifacts_dir / "environment_manifest.json"
    write_json(
        manifest_path,
        {
            "instance_id": record["instance_id"],
            "verification_status": record.get("verification_status"),
            "environment_restore_mode": state.mode,
            "environment_restore_succeeded": state.succeeded,
            "failed_phase": state.failed_phase,
            "soft_failed_phases": list(state.soft_failed_phases),
            "environment_summary": record.get("environment_summary"),
            "prepared_paths": [
                {"path": rel, "exists": (work_dir / rel).exists(), "is_dir": (work_dir / rel).is_dir()}
                for rel in _MANIFEST_PATHS
            ],
            "restore_counts": {f"{phase}_commands": len(state.records[phase]) for phase in _RESTORE_PHASES},
        },
    )
    return manifest_path


def build_perf_eval_result(
    record: Dict[str, Any],
    case_paths: EvalCasePaths,
    agent_name: str,
    started_at: datetime,
    restore_state: PerfRestoreState,
    *,
    agent_result: Optional[AgentInvocationResult] = None,
    agent_diff: str = "",
    preserved_artifacts: Optional[List[str]] = None,
    failure_category: Optional[str] = None,
    agent_error_message: Optional[str] = None,
) -> Dict[str, Any]:
    finished_at = datetime.now(timezone.utc)
    elapsed = round((finished_at - started_at).total_seconds(), 2)
    result_dir = result_dir_of(case_paths)
    if failure_category is None and agent_result is not None and agent_result.exit_code != 0:
        failure_category = "AGENT_EXECUTION_FAILURE"
    if agent_result is None:
        patch_capture_status = "not_run"
    else:
        patch_capture_status = "captured" if agent_diff.strip() else "empty"
    records = restore_state.records
    return {
        "instance_id": record["instance_id"],
        "repo": record.get("repo", ""),
        "base_commit": record.get("base_commit", ""),
        "head_commit": record.get("head_commit", ""),
        "agent": agent_name,
        "status": "error" if failure_category else "generated",
        "failure_category": failure_category,
        "perf_capture_status": "not_run",
        "perf_verdict": "not_judged",
        "perf_judgment": None,
        "reference_compare_mode": "recorded_base_head",
        "patch_capture_status": patch_capture_status,
        "agent_patch_path": str(result_dir / "artifacts" / "agent.diff"),
        "agent_exit_code": agent_result.exit_code if agent_result else None,
        "agent_command": agent_result.command if agent_result else [],
        "agent_error_message": agent_result.error_message if agent_result else agent_error_message,
        "agent_diff": agent_diff,
        "agent_stats": agent_result.agent_stats if agent_result else {},
        "agent_trajectory_path": str(case_paths.agent_stdout_log),
        "evaluation_stats": build_perf_evaluation_stats(
            records["setup"], records["test"], records["perf_setup"], records["perf"], elapsed_seconds=elapsed
        ),
        "environment_restore_mode": restore_state.mode,
        "environment_manifest_path": str(result_dir / "artifacts" / "environment_manifest.json"),
        "environment_restore_results": restore_state.serialize(),
        "test_patch_status": "not_run",
        "setup_commands": list(record.get("setup_commands", [])),
        "test_commands": [],
        "perf_setup_commands": [],
        "perf_commands": [],
        "setup_results": strip_phase(records["setup"]),
        "post_patch_setup_results": [],
        "test_results": strip_phase(records["test"]),
        "perf_setup_results": strip_phase(records["perf_setup"]),
        "candidate_perf_results": [],
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "elapsed_seconds": elapsed,
        "case_dir": str(result_dir),
        "result_dir": str(result_dir),
        "work_dir": str(case_paths.case_dir),
        "agent_stdout_log": str(case_paths.agent_stdout_log),
        "agent_stderr_log": str(case_paths.agent_stderr_log),
        "preserved_artifacts": list(preserved_artifacts or []),
    }


def execute_perf_eval(
    instance_id: str,
    dataset_path: Path,
    eval_root: Path,
    work_root: Path,
    prompt_template_path: Path,
    agent_runner: Optional[AgentRunner] = None,
    force: bool = False,
    template_codex_dir: Optional[Path] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    keep_git_history: bool = False,
) -> Dict[str, Any]:
    emit = progress_callback or (lambda message: None)
    started_at = datetime.now(timezone.utc)
    record = load_eval_record(dataset_path, instance_id)
    case_paths = prepare_perf_eval_case_dir(record, eval_root, work_root, prompt_template_path, force, template_codex_dir)
    result_artifacts_dir = result_dir_of(case_paths) / "artifacts"
    agent_name = agent_runner.name if agent_runner is not None else "codex"

    def finish(restore_state: PerfRestoreState, **kwargs: Any) -> Dict[str, Any]:
        result = build_perf_eval_result(record, case_paths, agent_name, started_at, restore_state, **kwargs)
        write_json(case_paths.result_path, result)
        return result

    emit(f"Cloning repo at base_commit {str(record.get('base_commit', ''))[:12]}")
    try:
        clone_or_update_repo(
            case_paths,
            record["repo"],
            record.get("base_commit"),
            pr_number=record.get("pr_number"),
            cache_root=repo_cache_root(work_root / "_repo_cache"),
        )
    except subprocess.CalledProcessError as exc:
        return finish(
            PerfRestoreState(mode=determine_restore_mode(record), failed_phase="clone"),
            failure_category="REPOSITORY_PREPARATION_FAILURE",
            agent_error_message="Repository preparation failed: " + format_called_process_error(exc),
        )

    emit("Restoring perf environment")
    restore_state = restore_perf_eval_environment(record, case_paths, progress_callback=emit)
    manifest_path = write_environment_manifest(record, case_paths, restore_state)
    result_artifacts_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, result_artifacts_dir / manifest_path.name)
    if not restore_state.succeeded:
        return finish(
            restore_state,
            failure_category="ENVIRONMENT_RESTORE_FAILURE",
            agent_error_message=f"Environment restore failed during phase: {restore_state.failed_phase}",
        )
    if agent_runner is None:
        raise ValueError("agent_runner is required once the environment is restored")
    if not keep_git_history:
        try:
            hide_future_history(case_paths.repo_dir)
        except subprocess.CalledProcessError as exc:
            return finish(
                restore_state,
                failure_category="REPOSITORY_PREPARATION_FAILURE",
                agent_error_message="Hiding git history failed: " + format_called_process_error(exc),
            )

    emit(f"Invoking agent ({agent_name})")
    agent_result = agent_runner.invoke(case_paths)
    agent_diff = capture_diff(case_paths.repo_dir)
    write_text(result_artifacts_dir / "agent.diff", agent_diff)
    preserved = preserve_lightweight_artifacts(case_paths.artifacts_dir, result_artifacts_dir)
    return finish(restore_state, agent_result=agent_result, agent_diff=agent_diff, preserved_artifacts=preserved)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--eval-root", default="results/perf_agent_eval")
    parser.add_argument("--work-root", default="tmp/perf_agent_eval")
    parser.add_argument("--prompt-template", default="prompts/agent_perf_eval.md")
    parser.add_argument("--template-codex-dir", default=None, help="Optional .codex directory copied into the work dir")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--keep-git-history",
        action="store_true",
        help="Leave branches, tags and remotes in the checkout (by default commits after base_commit are unreachable)",
    )
    add_profile_arguments(parser, kind="agent")


def run(args: argparse.Namespace) -> int:
    result = execute_perf_eval(
        instance_id=args.instance_id,
        dataset_path=Path(args.dataset).resolve(),
        eval_root=Path(args.eval_root).resolve(),
        work_root=Path(args.work_root).resolve(),
        prompt_template_path=Path(args.prompt_template).resolve(),
        agent_runner=agent_runner_from_args(args),
        force=args.force,
        template_codex_dir=Path(args.template_codex_dir).resolve() if args.template_codex_dir else None,
        progress_callback=lambda message: print(f"[perf-agent] {message}", flush=True),
        keep_git_history=args.keep_git_history,
    )
    print(json.dumps({key: result.get(key) for key in ("instance_id", "status", "failure_category", "patch_capture_status")}, indent=2))
    return 0

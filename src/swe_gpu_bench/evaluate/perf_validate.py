"""``evaluate perf-validate``: check a perf patch for correctness and measure base vs patched perf.

Flow: clone at ``base_commit`` -> setup -> perf_setup + perf on base -> apply the patch ->
(re-run setup if it builds the repo) -> apply ``test_patch`` -> tests -> perf_setup + perf on the
patched tree -> versioned numeric parser compares matched workloads without a model judge.

The patch is ``<patch-root>/<instance_id>/pred.diff`` (``--patch-source oracle``) or
``<patch-root>/<instance_id>/artifacts/agent.diff`` (``--patch-source agent``, i.e. the
``evaluate perf-agent`` eval root). Results go to ``<eval-root>/<instance_id>/result.json``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from swe_gpu_bench.core.config import ModelProfile, add_profile_arguments
from swe_gpu_bench.core.git import clone_or_update_repo, prepare_record_checkout
from swe_gpu_bench.core.io import find_record, load_text, write_json, write_text
from swe_gpu_bench.core.patch import (
    PatchCheckResult,
    apply_pred_patch,
    check_patch_applies,
    is_test_patch_failure_status,
)
from swe_gpu_bench.core.shell import did_commands_succeed, strip_phase
from swe_gpu_bench.evaluate import agent_eval
from swe_gpu_bench.evaluate.perf_eval import DEFAULT_DATASET

DEFAULT_JUDGE_PROMPT = Path("prompts/perf_judge.md")
PATCH_SOURCES = {
    # source: (relative patch path under <patch-root>/<instance_id>, result field, failure prefix)
    "oracle": ("pred.diff", "pred_patch_path", "PRED_PATCH"),
    "agent": ("artifacts/agent.diff", "agent_patch_path", "AGENT_PATCH"),
}
PERF_VERDICTS = ("improved", "no_improvement", "unclear")
_RECORD_PHASES = (
    "setup",
    "base_perf_setup",
    "base_perf",
    "post_patch_setup",
    "test",
    "patched_perf_setup",
    "patched_perf",
)
_MAX_PATCH_CHARS = 4000
_MAX_LOG_CHARS = 6000


def _perf_hint(record: Dict[str, Any], key: str) -> Any:
    return record.get(key) or (record.get("perf_context") or {}).get(key)


def _read_command_log(path_str: str, max_chars: int = _MAX_LOG_CHARS) -> str:
    if not path_str:
        return "(no log path)"
    path = Path(path_str)
    if not path.exists():
        return "(log file missing)"
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        half = max_chars // 2
        return text[:half] + "\n\n... [TRUNCATED] ...\n\n" + text[-half:]
    return text


def build_perf_validate_stats(records: Dict[str, List[Dict[str, Any]]], elapsed_seconds: Optional[float] = None) -> Dict[str, Any]:
    all_records = [record for phase in _RECORD_PHASES for record in records[phase]]
    counts = {f"{phase}_commands": len(records[phase]) for phase in _RECORD_PHASES}
    return {
        "elapsed_seconds": elapsed_seconds,
        "total_commands": len(all_records),
        **counts,
        "perf_setup_commands": counts["base_perf_setup_commands"] + counts["patched_perf_setup_commands"],
        "perf_commands": counts["base_perf_commands"] + counts["patched_perf_commands"],
        "failed_commands": sum(1 for record in all_records if record["exit_code"] != 0),
        "last_exit_code": all_records[-1]["exit_code"] if all_records else None,
    }


# ---------------------------------------------------------------------------
# Legacy prompt formatting retained for callers; numeric evaluation never uses it.
# ---------------------------------------------------------------------------

def build_perf_judge_prompt(
    template_text: str,
    record: Dict[str, Any],
    patch_text: str,
    base_perf_records: List[Dict[str, Any]],
    patched_perf_records: List[Dict[str, Any]],
) -> str:
    if len(patch_text) > _MAX_PATCH_CHARS:
        patch_text = patch_text[:_MAX_PATCH_CHARS] + "\n... [patch truncated] ..."

    def section(label: str, records: List[Dict[str, Any]]) -> str:
        if not records:
            return f"[{label}]\n(no commands ran)\n"
        lines = [f"[{label}]"]
        for index, item in enumerate(records, 1):
            lines.append(f"\nCommand {index}: {item.get('command', '?')}")
            lines.append(f"Exit code: {item.get('exit_code', '?')}")
            lines.append(f"stdout:\n{_read_command_log(item.get('stdout_log', ''))}")
            stderr = _read_command_log(item.get("stderr_log", ""))
            if stderr.strip() and stderr.strip() != "(log file missing)":
                lines.append(f"stderr:\n{stderr}")
        return "\n".join(lines)

    perf_commands = "\n".join(f"  {i + 1}. {cmd}" for i, cmd in enumerate(record.get("perf_commands", []))) or "  (none)"
    dynamic = f"""
==================================================
INSTANCE CONTEXT
==================================================
Repository: {record.get("repo", "unknown")}
Instance ID: {record.get("instance_id", "unknown")}
Performance kind: {_perf_hint(record, "perf_kind") or "unknown"}
Measurement hint: {_perf_hint(record, "measurement_hint") or "unknown"}

==================================================
PATCH APPLIED
==================================================
```diff
{patch_text}
```

==================================================
PERFORMANCE BENCHMARK COMMANDS
==================================================
{perf_commands}

==================================================
{section("BASELINE PERFORMANCE (before patch)", base_perf_records)}

==================================================
{section("PATCHED PERFORMANCE (after patch)", patched_perf_records)}

==================================================
JUDGMENT
==================================================
Based on the benchmark output above, produce the JSON judgment.
"""
    return template_text.strip() + "\n" + dynamic


def call_perf_judge(
    record: Dict[str, Any],
    patch_text: str,
    base_perf_records: List[Dict[str, Any]],
    patched_perf_records: List[Dict[str, Any]],
    output_dir: Path,
    profile: Optional[ModelProfile] = None,
    env_file: Optional[Path] = None,
    prompt_path: Path = DEFAULT_JUDGE_PROMPT,
) -> Dict[str, Any]:
    """Compatibility signature; comparison is deterministic and never calls a model."""
    from swe_gpu_bench.evaluate.metrics import MetricError, recipe, parse, compare
    try:
        spec = recipe(record)
        def measurements(rows):
            if not rows or any(r.get("exit_code") != 0 for r in rows):
                raise MetricError("measurement_command_failed")
            if spec.get("artifact"):
                path = rows[-1].get("metric_artifact_log")
                if not path:
                    raise MetricError("missing_metric_artifact")
                text = Path(path).read_text()
            else:
                text = "\n".join(Path(r["stdout_log"]).read_text() for r in rows)
            return parse(spec, text)
        result = compare(spec, measurements(base_perf_records), measurements(patched_perf_records), correctness_passed=True)
        return {**result, "verdict": "improved" if result["improved"] else "no_improvement",
                "method": "deterministic", "judge_profile": None, "raw_response": None}
    except (ValueError, OSError, KeyError) as exc:
        return {"verdict": "unclear", "method": "deterministic", "failure_reason": str(exc),
                "judge_profile": None, "raw_response": None}


def patch_path_for(patch_root: Path, instance_id: str, patch_source: str) -> Path:
    return patch_root / instance_id / PATCH_SOURCES[patch_source][0]


def execute_perf_validate(
    instance_id: str,
    dataset_path: Path,
    eval_root: Path,
    patch_root: Path,
    work_root: Path,
    patch_source: str = "oracle",
    judge_profile: Optional[ModelProfile] = None,
    env_file: Optional[Path] = None,
    judge_prompt_path: Path = DEFAULT_JUDGE_PROMPT,
    force: bool = False,
) -> Dict[str, Any]:
    _, patch_field, failure_prefix = PATCH_SOURCES[patch_source]
    patch_path = patch_path_for(patch_root, instance_id, patch_source)
    record = find_record(dataset_path, "instance_id", instance_id)
    case_paths = agent_eval.prepare_eval_case_dir(record, eval_root, None, force, work_root=work_root)
    result_dir = case_paths.result_path.parent
    started_at = datetime.now(timezone.utc)
    records: Dict[str, List[Dict[str, Any]]] = {phase: [] for phase in _RECORD_PHASES}
    patch_check: Optional[PatchCheckResult] = None
    patch_apply: Optional[PatchCheckResult] = None
    test_patch_status = "not_run"
    llm_judgment: Optional[Dict[str, Any]] = None

    def run(phase: str, commands: List[str]) -> List[Dict[str, Any]]:
        records[phase] = agent_eval.run_commands(commands, phase, case_paths)
        if phase in ("base_perf", "patched_perf") and records[phase]:
            from swe_gpu_bench.evaluate.metrics import recipe, MetricError
            try:
                spec = recipe(record)
            except MetricError:
                spec = {}
            if spec.get("artifact"):
                source = case_paths.case_dir / spec["artifact"]
                if source.is_file():
                    target = result_dir / (phase + "-metric.json")
                    write_text(target, source.read_text())
                    records[phase][-1]["metric_artifact_log"] = str(target)

        return records[phase]

    def finish(status: str, failure_category: Optional[str], perf_status: str) -> Dict[str, Any]:
        finished_at = datetime.now(timezone.utc)
        elapsed = round((finished_at - started_at).total_seconds(), 2)
        result = {
            "instance_id": instance_id,
            "repo": record.get("repo", ""),
            "base_commit": record.get("base_commit", ""),
            "head_commit": record.get("head_commit", ""),
            "status": status,
            "failure_category": failure_category,
            "patch_source": patch_source,
            patch_field: str(patch_path),
            "pred_patch_path": str(patch_path),
            "pred_patch_apply_check_status": patch_check.status if patch_check else "not_run",
            "pred_patch_apply_check_error": patch_check.stderr if patch_check else "",
            "pred_patch_apply_status": patch_apply.status if patch_apply else "not_run",
            "pred_patch_apply_error": patch_apply.stderr if patch_apply else "",
            "test_patch_status": test_patch_status,
            **{f"{key}_commands": record.get(f"{key}_commands", []) for key in ("setup", "test", "perf_setup", "perf")},
            **{f"{phase}_results": strip_phase(records[phase]) for phase in _RECORD_PHASES},
            "perf_status": perf_status,
            "llm_judgment": llm_judgment,
            "perf_kind": _perf_hint(record, "perf_kind"),
            "measurement_hint": _perf_hint(record, "measurement_hint"),
            "judge_profile": None,
            "evaluation_stats": build_perf_validate_stats(records, elapsed_seconds=elapsed),
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "elapsed_seconds": elapsed,
            "work_dir": str(case_paths.case_dir),
            "result_dir": str(result_dir),
        }
        write_json(case_paths.result_path, result)
        return result

    def error(category: str) -> Dict[str, Any]:
        return finish("error", category, "error")

    if not patch_path.exists():
        patch_check = patch_apply = PatchCheckResult(status="missing")
        return error(f"{failure_prefix}_MISSING")
    patch_text = load_text(patch_path)
    if not patch_text.strip():
        patch_check = patch_apply = PatchCheckResult(status="empty")
        return error(f"{failure_prefix}_EMPTY")

    try:
        prepare_record_checkout(case_paths, record, cache_root=work_root / "_repo_cache", clone_repo_fn=clone_or_update_repo)
    except subprocess.CalledProcessError:
        return error("REPOSITORY_PREPARATION_FAILURE")
    patch_check = check_patch_applies(case_paths.repo_dir, patch_text)
    if patch_check.status != "valid":
        patch_apply = PatchCheckResult(status="invalid", stderr=patch_check.stderr)
        return error(f"{failure_prefix}_APPLY_FAILURE")

    if not did_commands_succeed(run("setup", record.get("setup_commands", []))):
        return error("SETUP_FAILURE")
    run("base_perf_setup", record.get("perf_setup_commands", []))
    run("base_perf", record.get("perf_commands", []))

    patch_apply = apply_pred_patch(case_paths.repo_dir, patch_text)
    if patch_apply.status != "applied":
        return error(f"{failure_prefix}_APPLY_FAILURE")

    records["post_patch_setup"], test_patch_status, records["test"] = agent_eval.grade_patched_repo(record, case_paths)
    if is_test_patch_failure_status(test_patch_status):
        return error("TEST_PATCH_APPLY_FAILURE")
    status, failure_category = agent_eval.test_outcome(records["test"])
    if status != "passed":
        return finish(status, failure_category, "test_failed")

    run("patched_perf_setup", record.get("perf_setup_commands", []))
    run("patched_perf", record.get("perf_commands", []))
    perf_phases = ("base_perf_setup", "base_perf", "patched_perf_setup", "patched_perf")
    perf_ran = (
        all(all(r.get("exit_code") == 0 for r in records[phase]) for phase in perf_phases)
        and bool(records["base_perf"])
        and bool(records["patched_perf"])
    )
    if not perf_ran:
        return finish(status, failure_category, "unclear")
    llm_judgment = call_perf_judge(
        record, patch_text, records["base_perf"], records["patched_perf"], result_dir, judge_profile, env_file, judge_prompt_path
    )
    write_json(result_dir / "llm_judgment.json", llm_judgment)
    verdict = llm_judgment.get("verdict")
    return finish(status, failure_category, verdict if verdict in PERF_VERDICTS else "unclear")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--patch-source", choices=sorted(PATCH_SOURCES), default="oracle")
    parser.add_argument(
        "--patch-root",
        required=True,
        help="oracle: <root>/<instance_id>/pred.diff; agent: the perf-agent eval root (<root>/<instance_id>/artifacts/agent.diff)",
    )
    parser.add_argument("--eval-root", default="results/perf_validate")
    parser.add_argument("--work-root", default="tmp/perf_validate")
    parser.add_argument("--judge-prompt", default=str(DEFAULT_JUDGE_PROMPT), help="Deprecated compatibility option; numeric parsing does not use a judge")
    parser.add_argument("--force", action="store_true")
    add_profile_arguments(parser, kind="text")


def run(args: argparse.Namespace) -> int:
    result = execute_perf_validate(
        instance_id=args.instance_id,
        dataset_path=Path(args.dataset).resolve(),
        eval_root=Path(args.eval_root).resolve(),
        patch_root=Path(args.patch_root).resolve(),
        work_root=Path(args.work_root).resolve(),
        patch_source=args.patch_source,
        judge_profile=None,
        env_file=Path(args.env_file).resolve() if args.env_file else None,
        judge_prompt_path=Path(args.judge_prompt).resolve(),
        force=args.force,
    )
    print(json.dumps({key: result.get(key) for key in ("instance_id", "status", "perf_status", "failure_category")}, indent=2))
    return 0

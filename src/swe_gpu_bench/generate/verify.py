"""``generate verify``: replay recorded commands in a fresh checkout.

Two modes:

* ``--case-dir``: re-run a ``generate repro`` case's ``setup_commands`` + ``test_commands`` at the
  PR head (every command must succeed) and write ``<benchmark-root>/<case>/artifacts/result.json``.
* ``--dataset``: check each record exactly as the benchmark grades it. On ``base_commit``: setup,
  apply ``test_patch``, run the tests (they must fail); then apply the gold ``patch``, re-run setup if
  it builds the repo, and run the tests again (they must all pass). Writes
  ``<eval-root>/<instance_id>/result.json``.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

from swe_gpu_bench.core.git import checkout_commit, clone_or_update_repo, repo_cache_root
from swe_gpu_bench.core.io import load_json, write_json
from swe_gpu_bench.core.patch import PatchCheckResult, apply_pred_patch, apply_test_patch, is_test_patch_failure_status
from swe_gpu_bench.core.selection import add_selection_arguments, has_selection, select_records
from swe_gpu_bench.core.shell import format_called_process_error, run_logged_shell_command, setup_builds_repo

DEFAULT_DATASET_PATH = Path("data/dataset/swe_gpu_bench.jsonl")
DEFAULT_CONCURRENCY = 8
SETUP_PHASES = ("setup", "post_patch_setup")
TEST_PHASES = ("test", "base_test", "gold_test")
DATASET_PHASES = ("setup", "base_test", "post_patch_setup", "gold_test")
CASE_ROOT_PATTERN = re.compile(
    r"(?P<case_root>/(?:[^\s\"'=]+/)*TestCases/(?P<case>[A-Za-z0-9_.-]+))(?=$|[\s\"';)&|/])"
)


# ---------------------------------------------------------------------------
# Recorded-command normalization: rewrite absolute .../TestCases/<case> paths to $PWD
# ---------------------------------------------------------------------------

def _iter_heredoc_ranges(command: str) -> Iterator[tuple[int, int, int, int]]:
    position = 0
    while True:
        match = re.search(r"<<(?P<quote>['\"]?)(?P<tag>[A-Za-z_][A-Za-z0-9_]*)?(?P=quote)", command[position:])
        if match is None or match.group("tag") is None:
            return
        start = position + match.start()
        header_end = command.find("\n", start)
        if header_end == -1:
            return
        body_start = header_end + 1
        terminator = re.search(r"(?m)^%s\s*$" % re.escape(match.group("tag")), command[body_start:])
        if terminator is None:
            return
        yield start, body_start, body_start + terminator.start(), body_start + terminator.end()
        position = body_start + terminator.end()


def _replace_case_root(text: str, case_root: str, replacement: str, bare_replacement: str | None = None) -> str:
    if bare_replacement is None:
        bare_replacement = replacement.rstrip("/") or "."
    return re.sub(
        rf"{re.escape(case_root)}(?:(?P<trailing>/)|(?=$|[\s\"';)&|]))",
        lambda match: replacement if match.group("trailing") else bare_replacement,
        text,
    )


def normalize_recorded_command(command: str) -> str:
    match = CASE_ROOT_PATTERN.search(command)
    if match is None:
        return command
    case_root = match.group("case_root")
    parts: list[str] = []
    last = 0
    for _, body_start, body_end, terminator_end in _iter_heredoc_ranges(command):
        parts.append(_replace_case_root(command[last:body_start], case_root, "$PWD/", "$PWD/"))
        parts.append(_replace_case_root(command[body_start:body_end], case_root, ""))
        parts.append(command[body_end:terminator_end])
        last = terminator_end
    parts.append(_replace_case_root(command[last:], case_root, "$PWD/", "$PWD/"))
    return "".join(parts)


def normalize_command_result(result: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(result)
    for key in ("setup_commands", "test_commands"):
        if isinstance(normalized.get(key), list):
            normalized[key] = [normalize_recorded_command(command) for command in normalized[key]]
    return normalized


# ---------------------------------------------------------------------------
# Case directories and command execution
# ---------------------------------------------------------------------------

@dataclass
class VerificationCasePaths:
    case_dir: Path
    repo_dir: Path
    inputs_dir: Path
    artifacts_dir: Path
    command_logs_dir: Path
    sample_path: Optional[Path]
    original_result_path: Optional[Path]
    result_path: Path
    result_dir: Optional[Path] = None
    work_dir: Optional[Path] = None

    def __post_init__(self) -> None:
        self.result_dir = self.result_dir or self.case_dir
        self.work_dir = self.work_dir or self.case_dir


class CheckoutFailure(Exception):
    def __init__(self, stage: str, error: subprocess.CalledProcessError, phase_results: Dict[str, Any]):
        super().__init__(stage)
        self.stage = stage
        self.error = error
        self.phase_results = phase_results


def empty_verification_stats() -> Dict[str, Any]:
    return {
        "elapsed_seconds": None,
        "total_commands": 0,
        "setup_commands": 0,
        "test_commands": 0,
        "failed_commands": 0,
        "last_exit_code": None,
    }


def build_verification_stats(command_records: List[Dict[str, Any]], elapsed_seconds: Optional[float] = None) -> Dict[str, Any]:
    stats = empty_verification_stats()
    stats["elapsed_seconds"] = elapsed_seconds
    stats["total_commands"] = len(command_records)
    stats["setup_commands"] = sum(1 for record in command_records if record["phase"] in SETUP_PHASES)
    stats["test_commands"] = sum(1 for record in command_records if record["phase"] in TEST_PHASES)
    stats["failed_commands"] = sum(1 for record in command_records if record["exit_code"] != 0)
    if command_records:
        stats["last_exit_code"] = command_records[-1]["exit_code"]
    return stats


def _record_instance_id(record: Dict[str, Any], fallback_index: Optional[int] = None) -> str:
    value = str(record.get("instance_id") or "").strip()
    if value:
        return value
    return f"missing-instance-id-{fallback_index}" if fallback_index is not None else "missing-instance-id"


def load_case_inputs(source_case_dir: Path) -> tuple[Dict[str, Any], Dict[str, Any]]:
    sample_path = source_case_dir / "inputs" / "sample.json"
    result_path = source_case_dir / "logs" / "result.json"
    if not result_path.exists():
        result_path = source_case_dir / "artifacts" / "result.json"
    if not sample_path.exists():
        raise ValueError("Missing inputs/sample.json")
    if not result_path.exists():
        raise ValueError("Missing logs/result.json")

    sample = load_json(sample_path)
    original_result = normalize_command_result(load_json(result_path))
    for key in ("repo", "number", "head_sha"):
        if sample.get(key) in (None, ""):
            raise ValueError("sample.json missing required field: %s" % key)
    for key in ("setup_commands", "test_commands"):
        if not isinstance(original_result.get(key), list):
            raise ValueError("result.json missing required field: %s" % key)
    return sample, original_result


def prepare_verification_case_directory(
    benchmark_root: Path,
    source_case_dir: Path,
    sample: Dict[str, Any],
    original_result: Dict[str, Any],
    force: bool,
) -> VerificationCasePaths:
    case_dir = benchmark_root / source_case_dir.name
    if case_dir.exists():
        if not force:
            raise FileExistsError("Benchmark case already exists: %s" % case_dir)
        shutil.rmtree(case_dir)
    paths = VerificationCasePaths(
        case_dir=case_dir,
        repo_dir=case_dir / "repo",
        inputs_dir=case_dir / "inputs",
        artifacts_dir=case_dir / "artifacts",
        command_logs_dir=case_dir / "artifacts" / "command-logs",
        sample_path=case_dir / "inputs" / "sample.json",
        original_result_path=case_dir / "inputs" / "original_result.json",
        result_path=case_dir / "artifacts" / "result.json",
    )
    for path in (paths.repo_dir, paths.inputs_dir, paths.artifacts_dir, paths.command_logs_dir):
        path.mkdir(parents=True, exist_ok=True)
    write_json(paths.sample_path, sample)
    write_json(paths.original_result_path, original_result)
    return paths


def prepare_dataset_case_directory(
    record: Dict[str, Any],
    eval_root: Path,
    work_root: Path,
    force: bool,
) -> VerificationCasePaths:
    instance_id = _record_instance_id(record, record.get("__line_number__"))
    result_dir = eval_root / instance_id
    work_dir = work_root / instance_id
    result_path = result_dir / "result.json"
    if result_path.exists() and not force:
        raise FileExistsError("Verification result already exists: %s" % result_path)
    if force:
        shutil.rmtree(result_dir, ignore_errors=True)
        shutil.rmtree(work_dir, ignore_errors=True)
    paths = VerificationCasePaths(
        case_dir=work_dir,
        repo_dir=work_dir / "repo",
        inputs_dir=result_dir / "inputs",
        artifacts_dir=work_dir / "artifacts",
        command_logs_dir=result_dir / "command-logs",
        sample_path=result_dir / "inputs" / "record.json",
        original_result_path=None,
        result_path=result_path,
        result_dir=result_dir,
        work_dir=work_dir,
    )
    for path in (paths.repo_dir, paths.artifacts_dir, paths.inputs_dir, paths.command_logs_dir):
        path.mkdir(parents=True, exist_ok=True)
    write_json(paths.sample_path, record)
    return paths


def _parse_plain_cd_command(command: str) -> Optional[str]:
    stripped = command.strip()
    if not stripped.startswith("cd ") or any(token in stripped for token in ("&&", "||", ";", "\n")):
        return None
    try:
        parts = shlex.split(stripped)
    except ValueError:
        return None
    return parts[1] if len(parts) == 2 and parts[0] == "cd" else None


def _resolve_cd(current_cwd: Path, target: str) -> Path:
    if target == "-":
        return current_cwd
    candidate = Path(target)
    return (candidate if candidate.is_absolute() else current_cwd / candidate).resolve()


def _run_command_sequence(case_paths: VerificationCasePaths, phase: str, commands: Sequence[str]) -> List[Dict[str, Any]]:
    """Run commands one by one, carrying a bare ``cd <dir>`` over to the following commands."""
    current_cwd = case_paths.work_dir
    records: List[Dict[str, Any]] = []
    for index, command in enumerate(commands, start=1):
        target = _parse_plain_cd_command(command)
        if target is None:
            records.append(
                run_logged_shell_command(
                    phase=phase,
                    index=index,
                    command=command,
                    cwd=current_cwd,
                    artifacts_dir=case_paths.artifacts_dir,
                    command_logs_dir=case_paths.command_logs_dir,
                )
            )
            continue
        current_cwd = _resolve_cd(current_cwd, target)
        stdout_log = case_paths.command_logs_dir / ("%s-%02d.stdout.log" % (phase, index))
        stderr_log = case_paths.command_logs_dir / ("%s-%02d.stderr.log" % (phase, index))
        stdout_log.write_text("", encoding="utf-8")
        stderr_log.write_text("", encoding="utf-8")
        records.append(
            {
                "phase": phase,
                "command": command,
                "cwd": str(current_cwd),
                "exit_code": 0,
                "elapsed_seconds": 0.0,
                "stdout_log": str(stdout_log),
                "stderr_log": str(stderr_log),
            }
        )
    return records


def run_recorded_commands(case_paths: VerificationCasePaths, original_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        *_run_command_sequence(case_paths, "setup", original_result.get("setup_commands", [])),
        *_run_command_sequence(case_paths, "test", original_result.get("test_commands", [])),
    ]


def empty_phase_results() -> Dict[str, Any]:
    return {
        **{f"{phase}_results": [] for phase in DATASET_PHASES},
        "test_patch_status": "not_run",
        "patch_status": "not_run",
    }


def run_commit_aware_commands(
    case_paths: VerificationCasePaths,
    record: Dict[str, Any],
    checkout_commit_fn: Optional[Callable[[Path, str], None]] = None,
    apply_test_patch_fn: Optional[Callable[[str, Path], str]] = None,
    apply_patch_fn: Optional[Callable[[Path, str], PatchCheckResult]] = None,
) -> Dict[str, Any]:
    """Grade the gold patch the way ``evaluate agent`` grades a model patch; stop at the first decisive step."""
    if checkout_commit_fn is None:
        checkout_commit_fn = lambda repo_dir, commit: checkout_commit(repo_dir, commit, record.get("pr_number"))
    apply_test_patch_fn = apply_test_patch_fn or apply_test_patch
    apply_patch_fn = apply_patch_fn or apply_pred_patch
    setup_commands = list(record.get("setup_commands", []))
    test_commands = list(record.get("test_commands", []))
    phase_results = empty_phase_results()

    try:
        checkout_commit_fn(case_paths.repo_dir, str(record["base_commit"]))
    except subprocess.CalledProcessError as exc:
        raise CheckoutFailure("base", exc, phase_results) from exc
    phase_results["setup_results"] = _run_command_sequence(case_paths, "setup", setup_commands)
    if _any_failed(phase_results["setup_results"]):
        return phase_results

    test_patch = record.get("test_patch") or ""
    phase_results["test_patch_status"] = (
        apply_test_patch_fn(test_patch, case_paths.repo_dir) if test_patch.strip() else "no_test_patch"
    )
    if is_test_patch_failure_status(phase_results["test_patch_status"]):
        return phase_results
    phase_results["base_test_results"] = _run_command_sequence(case_paths, "base_test", test_commands)
    if not _any_failed(phase_results["base_test_results"]):
        return phase_results

    phase_results["patch_status"] = apply_patch_fn(case_paths.repo_dir, record.get("patch") or "").status
    if phase_results["patch_status"] != "applied":
        return phase_results
    if setup_builds_repo(setup_commands):
        phase_results["post_patch_setup_results"] = _run_command_sequence(case_paths, "post_patch_setup", setup_commands)
        if _any_failed(phase_results["post_patch_setup_results"]):
            return phase_results
    phase_results["gold_test_results"] = _run_command_sequence(case_paths, "gold_test", test_commands)
    return phase_results


# ---------------------------------------------------------------------------
# Single-case mode (--case-dir)
# ---------------------------------------------------------------------------

def _any_failed(records: Sequence[Dict[str, Any]]) -> bool:
    return any(record["exit_code"] != 0 for record in records)


def _determine_single_case_status(command_records: List[Dict[str, Any]]) -> str:
    """Every recorded command must succeed: a failing step that a later command papers over still fails."""
    if _any_failed([record for record in command_records if record["phase"] == "setup"]):
        return "setup_failed"
    test_records = [record for record in command_records if record["phase"] in TEST_PHASES]
    if not test_records or _any_failed(test_records):
        return "test_failed"
    return "verified"


def _single_case_result(source_case_dir: Path, **overrides: Any) -> Dict[str, Any]:
    result = {
        "instance_id": source_case_dir.name,
        "source_case_dir": str(source_case_dir),
        "benchmark_case_dir": None,
        "repo": None,
        "pr_number": None,
        "base_commit": None,
        "head_commit": None,
        "head_sha": None,
        "original_status": None,
        "setup_commands": [],
        "test_commands": [],
        "setup_results": [],
        "test_results": [],
        "cleanup": {"repo_removed": False, "artifacts_removed": False},
        "verification_status": "",
        "commands": [],
        "verification_stats": empty_verification_stats(),
        "comparison": {
            "matches_original_setup_commands": False,
            "matches_original_test_commands": False,
            "matches_original_verdict": False,
        },
        "summary": "",
    }
    result.update(overrides)
    return result


def _sample_fields(sample: Dict[str, Any], benchmark_case_dir: Path, original_result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "benchmark_case_dir": str(benchmark_case_dir),
        "repo": sample["repo"],
        "pr_number": sample["number"],
        "base_commit": sample.get("base_sha"),
        "head_commit": sample["head_sha"],
        "head_sha": sample["head_sha"],
        "original_status": original_result.get("status"),
        "setup_commands": list(original_result.get("setup_commands", [])),
        "test_commands": list(original_result.get("test_commands", [])),
    }


def build_verification_result(
    sample: Dict[str, Any],
    source_case_dir: Path,
    benchmark_case_dir: Path,
    original_result: Dict[str, Any],
    command_records: List[Dict[str, Any]],
    elapsed_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    setup_records = [record for record in command_records if record["phase"] == "setup"]
    test_records = [record for record in command_records if record["phase"] == "test"]
    status = _determine_single_case_status(command_records)
    reproduced = original_result.get("status") == "reproduced"
    return _single_case_result(
        source_case_dir,
        **_sample_fields(sample, benchmark_case_dir, original_result),
        setup_results=setup_records,
        test_results=test_records,
        verification_status=status,
        commands=command_records,
        verification_stats=build_verification_stats(command_records, elapsed_seconds=elapsed_seconds),
        comparison={
            "matches_original_setup_commands": [r["command"] for r in setup_records] == original_result.get("setup_commands", []),
            "matches_original_test_commands": [r["command"] for r in test_records] == original_result.get("test_commands", []),
            "matches_original_verdict": reproduced == (status == "verified"),
        },
        summary="Verification status: %s" % status,
    )


def execute_verification(
    source_case_dir: Path,
    benchmark_root: Path,
    force: bool,
    clone_repo_fn: Optional[Callable[..., None]] = None,
) -> Dict[str, Any]:
    started_at = time.monotonic()
    source_case_dir = source_case_dir.resolve()
    try:
        sample, original_result = load_case_inputs(source_case_dir)
    except (ValueError, json.JSONDecodeError) as exc:
        return _single_case_result(source_case_dir, verification_status="invalid_source_case", summary=str(exc))

    case_paths = prepare_verification_case_directory(benchmark_root.resolve(), source_case_dir, sample, original_result, force)
    try:
        (clone_repo_fn or clone_or_update_repo)(case_paths, sample["repo"], sample["head_sha"], sample["number"])
    except subprocess.CalledProcessError as exc:
        stats = empty_verification_stats()
        stats["elapsed_seconds"] = round(time.monotonic() - started_at, 2)
        result = _single_case_result(
            source_case_dir,
            **_sample_fields(sample, case_paths.case_dir, original_result),
            verification_status="repository_preparation_failed",
            verification_stats=stats,
            comparison={
                "matches_original_setup_commands": True,
                "matches_original_test_commands": True,
                "matches_original_verdict": original_result.get("status") != "reproduced",
            },
            summary="Repository preparation failed while running: %s" % format_called_process_error(exc),
        )
    else:
        result = build_verification_result(
            sample=sample,
            source_case_dir=source_case_dir,
            benchmark_case_dir=case_paths.case_dir,
            original_result=original_result,
            command_records=run_recorded_commands(case_paths, original_result),
            elapsed_seconds=round(time.monotonic() - started_at, 2),
        )
    write_json(case_paths.result_path, result)
    return result


# ---------------------------------------------------------------------------
# Dataset mode (--dataset)
# ---------------------------------------------------------------------------

def _determine_dataset_status(phase_results: Dict[str, Any]) -> str:
    if _any_failed(phase_results["setup_results"]):
        return "setup_failed"
    if is_test_patch_failure_status(phase_results["test_patch_status"]):
        return "test_patch_apply_failed"
    if not _any_failed(phase_results["base_test_results"]):
        return "base_test_not_failing"
    if phase_results["patch_status"] != "applied":
        return "patch_apply_failed"
    if _any_failed(phase_results["post_patch_setup_results"]):
        return "post_patch_setup_failed"
    gold_test_results = phase_results["gold_test_results"]
    if not gold_test_results or _any_failed(gold_test_results):
        return "gold_test_failed"
    return "verified"


def _validate_dataset_record(record: Dict[str, Any]) -> Optional[str]:
    for key in ("instance_id", "repo", "base_commit", "patch"):
        if str(record.get(key) or "").strip() == "":
            return "record missing required field: %s" % key
    for key in ("setup_commands", "test_commands"):
        value = record.get(key, [])
        if value is None:
            record[key] = []
        elif not isinstance(value, list):
            return "record field must be a list: %s" % key
    if not record["test_commands"]:
        return "record has no test_commands"
    return None


def _build_dataset_result(
    record: Dict[str, Any],
    case_paths: VerificationCasePaths,
    *,
    verification_status: str,
    summary: str,
    elapsed_seconds: Optional[float],
    phase_results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    phase_results = {**empty_phase_results(), **(phase_results or {})}
    command_records = [record for phase in DATASET_PHASES for record in phase_results[f"{phase}_results"]]
    return {
        "instance_id": _record_instance_id(record, record.get("__line_number__")),
        "dataset": record.get("dataset"),
        "result_dir": str(case_paths.result_dir),
        "work_dir": str(case_paths.work_dir),
        "repo": record.get("repo"),
        "pr_number": record.get("pr_number"),
        "base_commit": record.get("base_commit"),
        "head_commit": record.get("head_commit"),
        "setup_commands": list(record.get("setup_commands", [])),
        "test_commands": list(record.get("test_commands", [])),
        **{f"{phase}_results": list(phase_results[f"{phase}_results"]) for phase in DATASET_PHASES},
        "test_patch_status": phase_results["test_patch_status"],
        "patch_status": phase_results["patch_status"],
        "cleanup": {"repo_removed": False, "artifacts_removed": False},
        "verification_status": verification_status,
        "commands": command_records,
        "verification_stats": build_verification_stats(command_records, elapsed_seconds=elapsed_seconds),
        "summary": summary,
    }


def _cleanup_dataset_workdir(case_paths: VerificationCasePaths) -> Dict[str, Any]:
    cleanup: Dict[str, Any] = {}
    errors: List[str] = []
    for key, path in (("repo_removed", case_paths.repo_dir), ("artifacts_removed", case_paths.artifacts_dir)):
        try:
            if path.exists():
                shutil.rmtree(path)
        except OSError as exc:
            errors.append("%s: %s" % (path, exc))
        cleanup[key] = not path.exists()
    if errors:
        cleanup["errors"] = errors
    return cleanup


def execute_dataset_verification(
    record: Dict[str, Any],
    eval_root: Path,
    work_root: Path,
    force: bool,
    clone_repo_fn: Optional[Callable[..., None]] = None,
    checkout_commit_fn: Optional[Callable[[Path, str], None]] = None,
) -> Dict[str, Any]:
    started_at = time.monotonic()
    record = dict(record)
    case_paths = prepare_dataset_case_directory(record, eval_root.resolve(), work_root.resolve(), force)

    def finish(status: str, summary: str, phase_results=None) -> Dict[str, Any]:
        return _build_dataset_result(
            record,
            case_paths,
            verification_status=status,
            summary=summary,
            elapsed_seconds=round(time.monotonic() - started_at, 2),
            phase_results=phase_results,
        )

    try:
        invalid_reason = _validate_dataset_record(record)
        if invalid_reason is not None:
            result = finish("invalid_input", invalid_reason)
        else:
            try:
                (clone_repo_fn or clone_or_update_repo)(
                    case_paths,
                    str(record["repo"]),
                    str(record["base_commit"]),
                    pr_number=record.get("pr_number"),
                    cache_root=repo_cache_root(work_root.resolve() / "_repo_cache"),
                )
                phase_results = run_commit_aware_commands(case_paths, record, checkout_commit_fn)
            except subprocess.CalledProcessError as exc:
                result = finish(
                    "repository_preparation_failed",
                    "Repository preparation failed while running: %s" % format_called_process_error(exc),
                )
            except CheckoutFailure as exc:
                result = finish(
                    "repository_preparation_failed",
                    "Repository preparation failed while running: %s" % format_called_process_error(exc.error),
                    exc.phase_results,
                )
            else:
                status = _determine_dataset_status(phase_results)
                result = finish(status, "Verification status: %s" % status, phase_results)
    except Exception as exc:
        result = finish("unexpected_error", str(exc))

    result["cleanup"] = _cleanup_dataset_workdir(case_paths)
    write_json(case_paths.result_path, result)
    return result


def run_dataset_verification_batch(
    dataset_path: Path,
    eval_root: Path,
    work_root: Path,
    *,
    instance_ids: Sequence[str] = (),
    instance_id_file: Optional[Path] = None,
    all_instances: bool = False,
    concurrency: int = DEFAULT_CONCURRENCY,
    force: bool = False,
    clone_repo_fn: Optional[Callable[..., None]] = None,
    checkout_commit_fn: Optional[Callable[[Path, str], None]] = None,
) -> Dict[str, Any]:
    records, skipped = select_records(
        dataset_path,
        instance_ids=instance_ids,
        instance_id_file=instance_id_file,
        all_instances=all_instances,
        done=None if force else (lambda record: (eval_root / str(record.get("instance_id")) / "result.json").exists()),
    )
    results: List[Dict[str, Any]] = []
    if records:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(concurrency, len(records)))) as executor:
            results = list(
                executor.map(
                    lambda record: execute_dataset_verification(
                        record, eval_root, work_root, force, clone_repo_fn, checkout_commit_fn
                    ),
                    records,
                )
            )
    status_counts: Dict[str, int] = {}
    for result in results:
        status = str(result.get("verification_status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "dataset_path": str(dataset_path),
        "eval_root": str(eval_root),
        "work_root": str(work_root),
        "counts": {"processed": len(results), "skipped": skipped},
        "verification_statuses": status_counts,
        "results": results,
    }


def add_arguments(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--case-dir", help="Replay one `generate repro` case directory")
    source.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET_PATH),
        help="Verify dataset records: base+test_patch fails, base+patch+test_patch passes",
    )
    parser.add_argument("--benchmark-root", default="results/verify", help="--case-dir output root")
    parser.add_argument("--eval-root", default="results/verify_dataset", help="--dataset output root")
    parser.add_argument("--work-root", default="tmp/verify", help="--dataset scratch checkouts (removed after each run)")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    add_selection_arguments(parser)
    parser.add_argument("--force", action="store_true")


def run(args: argparse.Namespace) -> int:
    if args.case_dir:
        if has_selection(args):
            raise SystemExit("--instance-id/--instance-id-file/--all are only valid with --dataset")
        result = execute_verification(
            source_case_dir=Path(args.case_dir), benchmark_root=Path(args.benchmark_root), force=args.force
        )
    else:
        if not has_selection(args):
            raise SystemExit("One of --instance-id, --instance-id-file, or --all is required")
        result = run_dataset_verification_batch(
            dataset_path=Path(args.dataset),
            eval_root=Path(args.eval_root),
            work_root=Path(args.work_root),
            instance_ids=args.instance_id,
            instance_id_file=Path(args.instance_id_file) if args.instance_id_file else None,
            all_instances=args.all,
            concurrency=args.concurrency,
            force=args.force,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from swe_gpu_bench.adapters.agent import (
    AgentInvocationResult,
    AgentRunner,
    build_agent_runner,
    get_agent_runner,
)
from swe_gpu_bench.core.campaign_repo import clone_or_update_repo
from swe_gpu_bench.shared.config import resolve_profile
from swe_gpu_bench.shared.io import load_jsonl, write_json, write_text
from swe_gpu_bench.shared.runtime import did_commands_succeed, run_logged_shell_commands
from swe_gpu_bench.test_patch_utils import (
    is_test_patch_failure_status,
    restore_test_patch_targets,
)


# Withheld solutions and their measured outcomes are evaluator-only evidence.
# The public command contract and metric definition remain useful solver context.
_HIDDEN_KEYS = frozenset([
    "patch", "test_patch", "gold_patch_performance", "gold_patch_improvement_pct", "head_commit", "head_ref",
    "test_commands", "perf_commands", "perf_setup_commands", "oracle_patch", "expected_results",
])
_SOLVER_KEYS = frozenset([
    'instance_id', 'repo', 'base_commit', 'problem_statement', 'docker_image',
    'task_type', 'command_contract', 'performance_metric', 'better',
])
REPO_CACHE_ROOT_ENV = "SWEBENCH_REPO_CACHE_ROOT"


@dataclass
class EvalCasePaths:
    case_dir: Path
    repo_dir: Path
    inputs_dir: Path
    artifacts_dir: Path
    command_logs_dir: Path
    instance_path: Path   # inputs/instance.json  (hidden fields excluded)
    prompt_path: Path     # inputs/prompt.md
    agent_stdout_log: Path
    agent_stderr_log: Path
    result_path: Path     # result.json


def empty_evaluation_stats() -> Dict[str, Any]:
    return {
        "elapsed_seconds": None,
        "total_commands": 0,
        "setup_commands": 0,
        "post_patch_setup_commands": 0,
        "test_commands": 0,
        "failed_commands": 0,
        "last_exit_code": None,
    }


def build_evaluation_stats(
    setup_records: List[Dict[str, Any]],
    post_patch_setup_records: List[Dict[str, Any]],
    test_records: List[Dict[str, Any]],
    elapsed_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    all_records = [*setup_records, *post_patch_setup_records, *test_records]
    stats = empty_evaluation_stats()
    stats["elapsed_seconds"] = elapsed_seconds
    stats["total_commands"] = len(all_records)
    stats["setup_commands"] = len(setup_records)
    stats["post_patch_setup_commands"] = len(post_patch_setup_records)
    stats["test_commands"] = len(test_records)
    stats["failed_commands"] = sum(1 for record in all_records if record["exit_code"] != 0)
    if all_records:
        stats["last_exit_code"] = all_records[-1]["exit_code"]
    return stats


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_eval_record(dataset_path: Path, instance_id: str) -> Dict[str, Any]:
    """Load a single instance from a JSONL dataset file."""
    for record in load_jsonl(dataset_path):
        if record.get("instance_id") == instance_id:
            return record
    raise KeyError(f"instance_id {instance_id!r} not found in {dataset_path}")


def resolve_repo_cache_root() -> Path | None:
    raw = os.getenv(REPO_CACHE_ROOT_ENV)
    if not raw:
        return None
    return Path(raw).resolve()


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------

def render_eval_prompt(template_text: str, record: Dict[str, Any]) -> str:
    """Append dynamic instance sections to the static template."""
    setup_cmds_formatted = "\n".join(
        f"  {i + 1}. `{cmd}`" for i, cmd in enumerate(record.get("setup_commands", []))
    )

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

{setup_cmds_formatted}

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
Solve from the supplied base source and task. Do not retrieve the upstream solution,
PR patch, future commits, or evaluator files outside this working directory.
"""
    return template_text.rstrip() + "\n\n" + dynamic.lstrip()


# ---------------------------------------------------------------------------
# Case directory
# ---------------------------------------------------------------------------

def prepare_eval_case_dir(
    record: Dict[str, Any],
    eval_root: Path,
    prompt_template_path: Path,
    force: bool = False,
) -> EvalCasePaths:
    instance_id: str = record["instance_id"]
    case_dir = eval_root / instance_id
    if case_dir.exists():
        if not force:
            raise FileExistsError(f"Eval case already exists: {case_dir}")
        shutil.rmtree(case_dir)

    repo_dir = case_dir / "repo"
    inputs_dir = case_dir / "inputs"
    artifacts_dir = case_dir / "artifacts"
    command_logs_dir = artifacts_dir / "command-logs"
    for path in (repo_dir, inputs_dir, artifacts_dir, command_logs_dir):
        path.mkdir(parents=True, exist_ok=True)

    # Write instance.json with hidden keys removed.
    hidden = _HIDDEN_KEYS | ({'setup_commands'} if os.getenv('SWEBENCH_SOLVER_ISOLATION') == '1' else set())
    safe_record = {k: v for k, v in record.items() if k not in hidden}
    if os.getenv('SWEBENCH_SOLVER_ISOLATION') == '1':
        # New evaluator metadata must not silently become solver input.
        safe_record = {k: v for k, v in safe_record.items() if k in _SOLVER_KEYS}
        if isinstance(safe_record.get('performance_metric'), dict):
            safe_record['performance_metric'] = {k: v for k, v in safe_record['performance_metric'].items()
                                                if k in ('name', 'unit', 'workload', 'better')}
    instance_path = inputs_dir / "instance.json"
    write_text(instance_path, json.dumps(safe_record, ensure_ascii=False, indent=2) + "\n")

    # Render and write prompt.
    template_text = prompt_template_path.read_text(encoding="utf-8")
    prompt_record = {**record, 'setup_commands': []} if os.getenv('SWEBENCH_SOLVER_ISOLATION') == '1' else record
    prompt_text = render_eval_prompt(template_text, prompt_record)
    prompt_path = inputs_dir / "prompt.md"
    write_text(prompt_path, prompt_text)

    agent_stdout_log = command_logs_dir / "agent.stdout.jsonl"
    agent_stderr_log = command_logs_dir / "agent.stderr.log"
    result_path = case_dir / "result.json"

    return EvalCasePaths(
        case_dir=case_dir,
        repo_dir=repo_dir,
        inputs_dir=inputs_dir,
        artifacts_dir=artifacts_dir,
        command_logs_dir=command_logs_dir,
        instance_path=instance_path,
        prompt_path=prompt_path,
        agent_stdout_log=agent_stdout_log,
        agent_stderr_log=agent_stderr_log,
        result_path=result_path,
    )


# ---------------------------------------------------------------------------
# Command execution helpers
# ---------------------------------------------------------------------------

def run_commands(
    commands: List[str],
    phase: str,
    case_paths: EvalCasePaths,
) -> List[Dict[str, Any]]:
    deferred = []
    deferred_to_test = []
    setup_overrides = {}
    if os.getenv('SWEBENCH_VERIFY_SOURCE') == '1' and phase in {'setup', 'post_patch_setup'}:
        setup_overrides = json.loads(os.getenv('SWEBENCH_SETUP_COMMAND_OVERRIDES', '{}'))
        if not set(setup_overrides) <= {hashlib.sha256(c.encode()).hexdigest() for c in commands}:
            raise ValueError('Setup environment repair does not match the frozen commands')
        hashes = set(json.loads(os.getenv('SWEBENCH_DEFER_SETUP_TO_TEST_HASHES', '[]')))
        if not hashes <= {hashlib.sha256(c.encode()).hexdigest() for c in commands}:
            raise ValueError('Test-phase deferral does not match the frozen setup commands')
        deferred_to_test = [c for c in commands if hashlib.sha256(c.encode()).hexdigest() in hashes]
        commands = [c for c in commands if hashlib.sha256(c.encode()).hexdigest() not in hashes]
    if os.getenv('SWEBENCH_VERIFY_SOURCE') == '1' and phase == 'test':
        setup_overrides = json.loads(os.getenv('SWEBENCH_TEST_COMMAND_OVERRIDES', '{}'))
        if not set(setup_overrides) <= {hashlib.sha256(c.encode()).hexdigest() for c in commands}:
            raise ValueError('Test environment repair does not match the frozen commands')
    if os.getenv('SWEBENCH_VERIFY_SOURCE') == '1' and phase == 'setup':
        hashes = set(json.loads(os.getenv('SWEBENCH_DEFER_SETUP_COMMAND_HASHES', '[]')))
        command_hashes = {hashlib.sha256(c.encode()).hexdigest() for c in commands}
        if not hashes <= command_hashes:
            raise ValueError('Deferred setup repair does not match the frozen commands')
        deferred = [c for c in commands if hashlib.sha256(c.encode()).hexdigest() in hashes]
        commands = [c for c in commands if hashlib.sha256(c.encode()).hexdigest() not in hashes]
    original_commands = commands
    commands = [setup_overrides.get(hashlib.sha256(c.encode()).hexdigest(), c) for c in commands]
    if os.getenv('SWEBENCH_VERIFY_SOURCE') == '1' and phase == 'post_patch_setup':
        # A rebuild repeats environment preparation in the same isolated case.
        # uv otherwise aborts before reinstalling the patched local package.
        commands = [re.sub(r'\buv\s+venv\b(?!\s+--allow-existing)',
                           'uv venv --allow-existing', command) for command in commands]
    observer = None
    validation_env = None
    if os.getenv('SWEBENCH_VERIFY_SOURCE') == '1' and phase in {'post_patch_setup', 'test'} and commands:
        from swe_gpu_bench.shared.source_access import SourceAccess
        candidates_path = case_paths.artifacts_dir / 'source-candidates.json'
        candidate_names = json.loads(candidates_path.read_text())['paths'] if candidates_path.exists() else None
        observer = SourceAccess(case_paths.repo_dir, candidate_names=candidate_names)
        # Agent self-tests may have populated valid .pyc files. A fresh cache
        # prefix makes validation import the candidate source again, so source
        # read evidence cannot be lost to bytecode produced during generation.
        cache_prefix = tempfile.mkdtemp(prefix='.validation-pycache-', dir=case_paths.artifacts_dir)
        validation_env = {**os.environ, 'PYTHONPYCACHEPREFIX': cache_prefix}
        observer.start()
    try:
        records = run_logged_shell_commands(
            commands, phase=phase, cwd=case_paths.case_dir,
            artifacts_dir=case_paths.artifacts_dir,
            command_logs_dir=case_paths.command_logs_dir, include_cwd=False,
            **({'base_env': validation_env} if validation_env is not None else {}))
    finally:
        observed = observer.finish() if observer else None
    if records and observed is not None:
        observed['python_cache_prefix'] = validation_env['PYTHONPYCACHEPREFIX']
        records[0]['source_access'] = observed
    if records and deferred:
        records[0]['deferred_setup_commands'] = [
            {'command': c, 'sha256': hashlib.sha256(c.encode()).hexdigest(), 'execute_in': 'post_patch_setup'} for c in deferred]
    if records and deferred_to_test:
        records[0].setdefault('deferred_setup_commands', []).extend(
            {'command': c, 'sha256': hashlib.sha256(c.encode()).hexdigest(), 'execute_in': 'test'} for c in deferred_to_test)
    for record, original in zip(records, original_commands):
        if record.get('command') != original:
            record['original_command'] = original
            record['environment_repair'] = ('documented_test_environment_override' if phase == 'test' and hashlib.sha256(original.encode()).hexdigest() in setup_overrides
                                            else 'documented_setup_command_override' if hashlib.sha256(original.encode()).hexdigest() in setup_overrides
                                            else 'preserve_existing_uv_environment_on_rebuild')
    from swe_gpu_bench.shared.artifacts import atomic_json
    atomic_json(case_paths.command_logs_dir / (phase + '-records.json'), records)
    return records


def should_rerun_setup_after_agent(commands: List[str]) -> bool:
    if os.getenv('SWEBENCH_VERIFY_SOURCE') == '1' and json.loads(os.getenv('SWEBENCH_DEFER_SETUP_COMMAND_HASHES', '[]')):
        return True
    """Detect setup commands that install or build repo code and must be refreshed."""
    patterns = (
        r"(?:uv\s+)?pip\s+install\b[^\n]*?(?:-e|--editable)\s+(?:\$PWD/)?repo\b",
        r"(?:uv\s+)?pip\s+install\b[^\n]*?(?:\$PWD/)?repo\b",
        r"python(?:\d+(?:\.\d+)*)?\s+setup\.py\s+(?:install|develop|build|build_ext)\b",
        r"python(?:\d+(?:\.\d+)*)?\s+-m\s+pip\s+install\b[^\n]*?(?:-e|--editable)?[^\n]*?(?:\$PWD/)?repo\b",
        r"cmake\s+-S\s+(?:\$PWD/)?repo\b",
        r"cmake\s+--build\b",
        r"(^|[;&]|\&\&)\s*make(?:\s|$)",
        r"(^|[;&]|\&\&)\s*ninja(?:\s|$)",
        r"(^|[;&]|\&\&)\s*nvcc(?:\s|$)",
    )
    combined = "\n".join(commands)
    return any(re.search(pattern, combined, re.MULTILINE) for pattern in patterns)


# ---------------------------------------------------------------------------
# test_patch application
# ---------------------------------------------------------------------------

def apply_test_patch(test_patch: str, case_paths: EvalCasePaths) -> str:
    """Apply test_patch to repo_dir with three fallback strategies.

    Returns one of:
      "restore_failed"  — reset of touched files to HEAD failed
      "applied_clean"   — git apply --3way succeeded
      "applied_fuzz"    — patch --fuzz=3 succeeded
      "applied_additive"— pure new-file patch written directly
      "apply_failed"    — all strategies failed
    """
    repo_dir = case_paths.repo_dir
    restore_result = restore_test_patch_targets(test_patch, repo_dir)
    if restore_result.status == "restore_failed":
        return restore_result.status

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".patch", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(test_patch)
        patch_file = Path(tmp.name)

    try:
        # Strategy 1: git apply --3way
        result = subprocess.run(
            ["git", "apply", "--3way", str(patch_file)],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return "applied_clean"

        # Strategy 2: patch --fuzz=3
        result = subprocess.run(
            ["patch", "--fuzz=3", "-p1", "--input", str(patch_file)],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return "applied_fuzz"

        # Strategy 3: pure additive (new files only)
        if _apply_additive_only(test_patch, repo_dir):
            return "applied_additive"

        return "apply_failed"
    finally:
        try:
            patch_file.unlink()
        except OSError:
            pass


def _apply_additive_only(patch_text: str, repo_dir: Path) -> bool:
    """Write new files from a purely additive unified diff.

    Returns True if all hunks were new-file additions and were written
    successfully, False otherwise.
    """
    lines = patch_text.splitlines(keepends=True)
    current_file: Optional[Path] = None
    content_lines: List[str] = []
    is_new_file = False
    all_new = True
    files: List[tuple[Path, str]] = []

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("diff --git "):
            if current_file is not None and is_new_file:
                files.append((current_file, "".join(content_lines)))
            current_file = None
            content_lines = []
            is_new_file = False
        elif line.startswith("new file mode"):
            is_new_file = True
        elif line.startswith("--- /dev/null"):
            pass  # confirms new file
        elif line.startswith("+++ b/"):
            if is_new_file:
                rel_path = line[6:].rstrip("\n")
                current_file = repo_dir / rel_path
        elif line.startswith("@@"):
            pass  # hunk header
        elif line.startswith("+") and current_file is not None and is_new_file:
            content_lines.append(line[1:])
        elif line.startswith("-") and not line.startswith("---"):
            # Deletion in a non-new-file context → not purely additive
            all_new = False
            break
        i += 1

    if current_file is not None and is_new_file:
        files.append((current_file, "".join(content_lines)))

    if not all_new or not files:
        return False

    for file_path, content in files:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# Agent diff capture
# ---------------------------------------------------------------------------

def untracked_paths(repo: Path) -> set[str]:
    output = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard', '-z'], cwd=repo)
    return {name for name in output.decode().split('\0') if name}


def capture_agent_diff(case_paths: EvalCasePaths, untracked_before: set[str] | None = None) -> str:
    """Capture new answer files as well as tracked edits, excluding setup fixtures."""
    if untracked_before is not None:
        added = sorted(untracked_paths(case_paths.repo_dir) - untracked_before)
        if added:
            # Intent-to-add makes both patch export and source-read observation
            # see the new files without changing their contents.
            subprocess.run(['git', 'add', '--intent-to-add', '--', *added], cwd=case_paths.repo_dir, check=True)
    result = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=str(case_paths.repo_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    if hasattr(case_paths, 'artifacts_dir'):
        names = subprocess.check_output(
            ['git', 'diff', '--name-only', 'HEAD', '-z'], cwd=case_paths.repo_dir).decode().split('\0')
        write_json(case_paths.artifacts_dir / 'source-candidates.json', {
            'paths': sorted(n for n in names if n),
            'captured_before_test_patch': True,
            'candidate_diff_sha256': hashlib.sha256(result.stdout.encode()).hexdigest(),
        })
    return result.stdout


# ---------------------------------------------------------------------------
# Result construction
# ---------------------------------------------------------------------------

def _determine_status(
    setup_records: List[Dict[str, Any]],
    post_patch_setup_records: List[Dict[str, Any]],
    test_records: List[Dict[str, Any]],
    test_patch_status: str,
    strict_commands: bool = False,
) -> tuple[str, Optional[str]]:
    if not did_commands_succeed(setup_records):
        return "error", "SETUP_FAILURE"
    if not did_commands_succeed(post_patch_setup_records):
        return "error", "POST_PATCH_SETUP_FAILURE"
    if strict_commands:
        from swe_gpu_bench.shared.artifacts import skipped_test_output
        for record in test_records:
            text = "\n".join(Path(record[k]).read_text(errors="replace")
                             for k in ("stdout_log", "stderr_log")
                             if record.get(k) and Path(record[k]).is_file())
            if skipped_test_output(text):
                return "error", "NO_TESTS_EXECUTED"
    if is_test_patch_failure_status(test_patch_status):
        return "error", "TEST_PATCH_APPLY_FAILURE"
    if not test_records:
        return "error", "TEST_EXECUTION_FAILURE"
    all_passed = all(r["exit_code"] == 0 for r in test_records)
    if all_passed:
        return "passed", None
    return "failed", None


def build_eval_result(
    record: Dict[str, Any],
    case_paths: EvalCasePaths,
    agent_name: str,
    agent_result: AgentInvocationResult,
    agent_diff: str,
    setup_records: List[Dict[str, Any]],
    post_patch_setup_records: List[Dict[str, Any]],
    test_patch_status: str,
    test_records: List[Dict[str, Any]],
    started_at: datetime,
    finished_at: datetime,
    strict_commands: bool = False,
) -> Dict[str, Any]:
    setup_cmds = record.get("setup_commands", [])
    test_cmds = record.get("test_commands", [])
    test_results = [
        {k: v for k, v in r.items() if k != "phase"}
        for r in test_records
    ]
    setup_results = [
        {k: v for k, v in r.items() if k != "phase"}
        for r in setup_records
    ]
    post_patch_setup_results = [
        {k: v for k, v in r.items() if k != "phase"}
        for r in post_patch_setup_records
    ]

    status, failure_category = _determine_status(
        setup_records,
        post_patch_setup_records,
        test_records,
        test_patch_status,
        strict_commands=strict_commands,
    )
    if strict_commands and (agent_result.exit_code != 0 or agent_result.error_message):
        status, failure_category = "error", "AGENT_EXECUTION_FAILURE"

    elapsed = (finished_at - started_at).total_seconds()
    evaluation_stats = build_evaluation_stats(
        setup_records,
        post_patch_setup_records,
        test_records,
        elapsed_seconds=round(elapsed, 2),
    )
    return {
        "instance_id": record["instance_id"],
        "repo": record.get("repo", ""),
        "base_commit": record.get("base_commit", ""),
        "head_commit": record.get("head_commit", ""),
        "agent": agent_name,
        "status": status,
        "failure_category": failure_category,
        "agent_exit_code": agent_result.exit_code,
        "agent_command": agent_result.command,
        "agent_error_message": agent_result.error_message,
        "agent_diff": agent_diff,
        "agent_stats": agent_result.agent_stats,
        "evaluation_stats": evaluation_stats,
        "test_patch_status": test_patch_status,
        "setup_commands": setup_cmds,
        "setup_results": setup_results,
        "post_patch_setup_results": post_patch_setup_results,
        "test_commands": test_cmds,
        "test_results": test_results,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "elapsed_seconds": round(elapsed, 2),
        "case_dir": str(case_paths.case_dir),
        "agent_stdout_log": str(case_paths.agent_stdout_log),
        "agent_stderr_log": str(case_paths.agent_stderr_log),
    }


def _error_result(
    record: Dict[str, Any],
    case_paths: EvalCasePaths,
    agent_name: str,
    failure_category: str,
    summary: str,
    started_at: datetime,
) -> Dict[str, Any]:
    finished_at = datetime.now(timezone.utc)
    elapsed_seconds = round((finished_at - started_at).total_seconds(), 2)
    return {
        "instance_id": record.get("instance_id", ""),
        "repo": record.get("repo", ""),
        "base_commit": record.get("base_commit", ""),
        "head_commit": record.get("head_commit", ""),
        "agent": agent_name,
        "status": "error",
        "failure_category": failure_category,
        "summary": summary,
        "agent_exit_code": None,
        "agent_command": [],
        "agent_error_message": None,
        "agent_diff": "",
        "agent_stats": {},
        "evaluation_stats": build_evaluation_stats([], [], [], elapsed_seconds=elapsed_seconds),
        "test_patch_status": None,
        "setup_commands": [],
        "setup_results": [],
        "post_patch_setup_results": [],
        "test_commands": record.get("test_commands", []),
        "test_results": [],
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "case_dir": str(case_paths.case_dir),
        "agent_stdout_log": str(case_paths.agent_stdout_log),
        "agent_stderr_log": str(case_paths.agent_stderr_log),
    }


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def execute_eval(
    instance_id: str,
    dataset_path: Path,
    eval_root: Path,
    prompt_template_path: Path,
    agent_name: str = "codex",
    agent_runner: Optional[AgentRunner] = None,
    force: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
    isolate_history: bool = False,
    strict_commands: bool = False,
    candidate_patch: Optional[Path] = None,
    generate_only: bool = False,
) -> Dict[str, Any]:
    def emit(msg: str) -> None:
        if progress_callback is not None:
            progress_callback(msg)

    started_at = datetime.now(timezone.utc)

    emit(f"Loading instance {instance_id!r}")
    record = load_eval_record(dataset_path, instance_id)

    emit("Preparing case directory")
    case_paths = prepare_eval_case_dir(
        record=record,
        eval_root=eval_root,
        prompt_template_path=prompt_template_path,
        force=force,
    )

    if agent_runner is None:
        agent_runner = get_agent_runner(agent_name)

    # Stage 1: Clone and checkout base_commit
    emit(f"Cloning repo and checking out base_commit {record.get('base_commit', '')[:12]}")
    try:
        clone_or_update_repo(
            case_paths,  # duck-typed: only repo_dir is used internally
            record["repo"],
            record.get("base_commit"),
            pr_number=record.get("pr_number"),
            cache_root=resolve_repo_cache_root(),
            **({"strict_commit": True} if strict_commands or isolate_history else {}),
        )
    except subprocess.CalledProcessError as exc:
        summary = "Repository preparation failed: " + " ".join(
            exc.cmd if isinstance(exc.cmd, (list, tuple)) else [str(exc.cmd)]
        )
        result = _error_result(record, case_paths, agent_name, "REPOSITORY_PREPARATION_FAILURE", summary, started_at)
        write_json(case_paths.result_path, result)
        return result

    # Stage 2: Build the base environment before invoking the agent.
    emit("Running setup_commands to prepare environment")
    setup_cmds: List[str] = record.get("setup_commands", [])
    setup_records = run_commands(setup_cmds, "setup", case_paths)
    if not did_commands_succeed(setup_records):
        emit("setup_commands did not finish successfully; skipping agent invocation")
        finished_at = datetime.now(timezone.utc)
        result = build_eval_result(
            record=record,
            case_paths=case_paths,
            agent_name=agent_name,
            agent_result=AgentInvocationResult(
                command=[],
                exit_code=0,
                stdout_path=case_paths.agent_stdout_log,
                stderr_path=case_paths.agent_stderr_log,
            ),
            agent_diff="",
            setup_records=setup_records,
            post_patch_setup_records=[],
            test_patch_status="not_run",
            test_records=[],
            started_at=started_at,
            finished_at=finished_at,
            strict_commands=strict_commands,
        )
        write_json(case_paths.result_path, result)
        return result

    # Stage 3: Invoke agent against the prepared environment.
    if isolate_history:
        from swe_gpu_bench.evaluate.use_cases.solver_isolation import isolate_git_history
        isolate_git_history(case_paths.repo_dir, record["base_commit"])
    untracked_before = untracked_paths(case_paths.repo_dir) if os.getenv('SWEBENCH_VERIFY_SOURCE') == '1' else None
    emit(f"Invoking agent ({agent_name})")
    if candidate_patch is not None:
        from swe_gpu_bench.evaluate.use_cases.oracle import apply_pred_patch
        applied = apply_pred_patch(case_paths.repo_dir, candidate_patch.read_text())
        if applied.status != "applied":
            result = _error_result(record, case_paths, agent_name, "PRED_PATCH_APPLY_FAILURE",
                                   "Saved candidate did not apply", started_at)
            write_json(case_paths.result_path, result)
            return result
        agent_result = AgentInvocationResult(command=["resume_saved_candidate"], exit_code=0,
            stdout_path=case_paths.agent_stdout_log, stderr_path=case_paths.agent_stderr_log)
    else:
        agent_result = agent_runner.invoke(case_paths)
    if agent_result.error_message:
        emit(f"Agent warning: {agent_result.error_message}")

    # Capture what the agent changed
    emit("Capturing agent diff")
    agent_diff = capture_agent_diff(case_paths, untracked_before)
    diff_path = case_paths.artifacts_dir / "agent.diff"
    write_text(diff_path, agent_diff)

    if generate_only:
        generation_result = {"status": "generated", "agent_exit_code": agent_result.exit_code,
                "agent_error_message": agent_result.error_message,
                "candidate_path": str(diff_path), "setup_results": setup_records,
                "agent_stdout_log": str(case_paths.agent_stdout_log),
                "agent_stderr_log": str(case_paths.agent_stderr_log)}
        write_json(case_paths.artifacts_dir / 'generation-result.json', generation_result)
        return generation_result

    # Stage 4: Apply trusted tests after capturing the answer, before configuring targets
    test_patch: str = record.get("test_patch", "")
    if test_patch.strip():
        emit("Applying test_patch")
        test_patch_status = apply_test_patch(test_patch, case_paths)
        emit(f"test_patch status: {test_patch_status}")
    else:
        test_patch_status = "no_test_patch"
        emit("No test_patch in record, skipping")

    # Stage 5: Conditionally refresh repo installs/build artifacts after patching.
    post_patch_setup_records: List[Dict[str, Any]] = []
    if (not is_test_patch_failure_status(test_patch_status)
            and record.get("command_contract") != "portable_unified_v1"
            and should_rerun_setup_after_agent(setup_cmds)):
        emit("Re-running setup_commands to refresh repo install/build state after agent changes")
        post_patch_setup_records = run_commands(setup_cmds, "post_patch_setup", case_paths)
        if not did_commands_succeed(post_patch_setup_records):
            emit("Post-patch setup failed; skipping test commands")
    else:
        emit("Skipping post-patch setup rerun; no repo install/build step detected")

    # Stage 6: Run test_commands
    emit("Running test_commands")
    test_cmds: List[str] = record.get("test_commands", [])
    can_test = (did_commands_succeed(post_patch_setup_records)
                and not is_test_patch_failure_status(test_patch_status)
                and (not strict_commands or (agent_result.exit_code == 0 and not agent_result.error_message)))
    test_records = run_commands(test_cmds, "test", case_paths) if can_test else []

    finished_at = datetime.now(timezone.utc)
    result = build_eval_result(
        record=record,
        case_paths=case_paths,
        agent_name=agent_name,
        agent_result=agent_result,
        agent_diff=agent_diff,
        setup_records=setup_records,
        post_patch_setup_records=post_patch_setup_records,
        test_patch_status=test_patch_status,
        test_records=test_records,
        started_at=started_at,
        finished_at=finished_at,
        strict_commands=strict_commands,
    )
    write_json(case_paths.result_path, result)
    emit(f"Done — status: {result['status']}")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run agent eval on a single dataset instance.")
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--dataset", default="data/dataset/swe_gpu_bench_cuda126.jsonl")
    parser.add_argument("--eval-root", default="../EvalResults")
    parser.add_argument("--agent", default=None)
    parser.add_argument("--prompt-template", default="prompts/agent_eval.md")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--config", default="swe_gpu_bench.toml")
    parser.add_argument("--framework", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--isolate-history", action="store_true",
                        help="Expose only the pinned base snapshot to the solver")
    parser.add_argument("--strict-commands", action="store_true",
                        help="Reject any setup/rebuild failure, agent error or all-skipped tests")
    parser.add_argument("--env-file", default=None)
    return parser


def cli_main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cwd = Path(".").resolve()
    profile = None
    agent_runner = None
    agent_name = args.agent or args.framework or "codex"
    if args.profile or args.framework or args.model or args.base_url or args.api_key_env:
        profile = resolve_profile(
            profile_name=args.profile,
            profile_kind="agent",
            config_path=(cwd / args.config).resolve() if args.config else None,
            framework=args.framework,
            model=args.model,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
        )
        agent_name = profile.framework or agent_name
        if args.env_file and profile.api_key_env:
            from swe_gpu_bench.adapters.llm import _profile_api_key
            key = _profile_api_key(profile, env_file=Path(args.env_file).resolve())
            profile = replace(profile, extra_env={**profile.extra_env, profile.api_key_env: key})
        agent_runner = build_agent_runner(profile=profile)
    result = execute_eval(
        instance_id=args.instance_id,
        dataset_path=(cwd / args.dataset).resolve(),
        eval_root=Path(args.eval_root).resolve(),
        prompt_template_path=(cwd / args.prompt_template).resolve(),
        agent_name=agent_name,
        agent_runner=agent_runner,
        force=args.force,
        isolate_history=args.isolate_history,
        strict_commands=args.strict_commands,
        progress_callback=lambda msg: print(f"[eval] {msg}", flush=True),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0

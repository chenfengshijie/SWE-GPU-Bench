"""Run shell commands inside a case directory with per-command log files."""
from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

_REBUILD_PATTERNS = (
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


def case_env(artifacts_dir: Path, extra_env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Process env with uv/XDG caches redirected into the case's ``artifacts/``."""
    env = dict(os.environ)
    env["UV_CACHE_DIR"] = str(artifacts_dir / ".uv-cache")
    env["XDG_CACHE_HOME"] = str(artifacts_dir / ".cache")
    if extra_env:
        env.update(extra_env)
    return env


def run_logged_shell_command(
    *,
    phase: str,
    index: int,
    command: str,
    cwd: Path,
    artifacts_dir: Path,
    command_logs_dir: Path,
    log_prefix: str = "",
    include_cwd: bool = True,
) -> dict[str, Any]:
    stem = f"{log_prefix}{phase}-{index:02d}"
    stdout_log = command_logs_dir / f"{stem}.stdout.log"
    stderr_log = command_logs_dir / f"{stem}.stderr.log"
    command_logs_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.monotonic()
    with stdout_log.open("w", encoding="utf-8") as out_fh, stderr_log.open("w", encoding="utf-8") as err_fh:
        completed = subprocess.run(
            ["bash", "-lc", command],
            cwd=str(cwd),
            env=case_env(artifacts_dir),
            stdout=out_fh,
            stderr=err_fh,
            text=True,
            check=False,
        )
    record: dict[str, Any] = {
        "phase": phase,
        "command": command,
        "exit_code": completed.returncode,
        "elapsed_seconds": round(time.monotonic() - started_at, 2),
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
    }
    if include_cwd:
        record["cwd"] = str(cwd)
    return record


def run_logged_shell_commands(
    commands: Sequence[str],
    *,
    phase: str,
    cwd: Path,
    artifacts_dir: Path,
    command_logs_dir: Path,
    log_prefix: str = "",
    include_cwd: bool = True,
) -> list[dict[str, Any]]:
    return [
        run_logged_shell_command(
            phase=phase,
            index=index,
            command=command,
            cwd=cwd,
            artifacts_dir=artifacts_dir,
            command_logs_dir=command_logs_dir,
            log_prefix=log_prefix,
            include_cwd=include_cwd,
        )
        for index, command in enumerate(commands, start=1)
    ]


def did_commands_succeed(records: Sequence[Mapping[str, Any]]) -> bool:
    """Optional phases may be empty. Every recorded command must succeed."""
    return all(record.get("exit_code") == 0 for record in records)


def all_commands_succeeded(records: Sequence[Mapping[str, Any]]) -> bool:
    """Every command ran and exited 0; an empty phase does not count as success."""
    return bool(records) and did_commands_succeed(records)


def setup_builds_repo(commands: Sequence[str]) -> bool:
    """True when setup installs or builds repo code, so it must be re-run after the repo is patched."""
    combined = "\n".join(commands)
    return any(re.search(pattern, combined, re.MULTILINE) for pattern in _REBUILD_PATTERNS)


def strip_phase(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in record.items() if k != "phase"} for record in records]


def format_called_process_error(exc: subprocess.CalledProcessError) -> str:
    return " ".join(exc.cmd if isinstance(exc.cmd, (list, tuple)) else [str(exc.cmd)])

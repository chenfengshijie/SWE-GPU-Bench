from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from swe_gpu_bench.core.shell import did_commands_succeed as did_commands_succeed
from swe_gpu_bench.shared.artifacts import atomic_json


def build_command_env(
    artifacts_dir: Path,
    base_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    env["UV_CACHE_DIR"] = env.get("SWEBENCH_UV_CACHE_ROOT") or str(artifacts_dir / ".uv-cache")
    env["XDG_CACHE_HOME"] = str(artifacts_dir / ".cache")
    return env


def run_logged_shell_command(
    *,
    phase: str,
    index: int,
    command: str,
    cwd: Path,
    artifacts_dir: Path,
    command_logs_dir: Path,
    base_env: Mapping[str, str] | None = None,
    include_cwd: bool = True,
) -> dict[str, Any]:
    stdout_log = command_logs_dir / f"{phase}-{index:02d}.stdout.log"
    stderr_log = command_logs_dir / f"{phase}-{index:02d}.stderr.log"
    metadata_log = command_logs_dir / f"{phase}-{index:02d}.json"
    atomic_json(metadata_log, {'phase': phase, 'command': command, 'cwd': str(cwd),
                              'started_at': time.time(), 'status': 'running', 'exit_code': None,
                              'stdout_log': str(stdout_log), 'stderr_log': str(stderr_log)})
    started_at = time.monotonic()
    with stdout_log.open("w", encoding="utf-8") as out_fh:
        with stderr_log.open("w", encoding="utf-8") as err_fh:
            completed = subprocess.run(
                ["bash", "-c", command],
                cwd=str(cwd),
                env=build_command_env(artifacts_dir, base_env),
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
    atomic_json(metadata_log, {**record, 'status': 'completed'})
    return record


def run_logged_shell_commands(
    commands: Sequence[str],
    *,
    phase: str,
    cwd: Path,
    artifacts_dir: Path,
    command_logs_dir: Path,
    base_env: Mapping[str, str] | None = None,
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
            base_env=base_env,
            include_cwd=include_cwd,
        )
        for index, command in enumerate(commands, start=1)
    ]


def strip_phase(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in record.items() if k != "phase"} for record in records]

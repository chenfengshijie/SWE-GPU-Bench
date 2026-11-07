from pathlib import Path

from swe_gpu_bench.core.shell import case_env, run_logged_shell_command


def test_case_env_points_caches_into_artifacts(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"

    env = case_env(artifacts_dir)

    assert env["UV_CACHE_DIR"] == str(artifacts_dir / ".uv-cache")
    assert env["XDG_CACHE_HOME"] == str(artifacts_dir / ".cache")


def test_run_logged_shell_command_writes_logs_and_captures_cwd(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    command_logs_dir = tmp_path / "logs"
    work_dir = tmp_path / "work"
    artifacts_dir.mkdir()
    command_logs_dir.mkdir()
    work_dir.mkdir()

    record = run_logged_shell_command(
        phase="setup",
        index=1,
        command="pwd",
        cwd=work_dir,
        artifacts_dir=artifacts_dir,
        command_logs_dir=command_logs_dir,
    )

    assert record["phase"] == "setup"
    assert record["cwd"] == str(work_dir)
    assert record["exit_code"] == 0
    assert Path(record["stdout_log"]).read_text(encoding="utf-8").strip() == str(work_dir)
    assert Path(record["stderr_log"]).read_text(encoding="utf-8") == ""

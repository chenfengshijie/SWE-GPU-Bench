import subprocess
from pathlib import Path

from swe_gpu_bench.core.patch import check_patch_applies


def test_check_patch_applies_accepts_valid_git_diff(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    subprocess.run(["git", "init"], cwd=repo_dir, check=True, capture_output=True, text=True)
    (repo_dir / "sample.txt").write_text("before\n", encoding="utf-8")
    subprocess.run(["git", "add", "sample.txt"], cwd=repo_dir, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
        env={
            "GIT_AUTHOR_NAME": "Test",
            "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "Test",
            "GIT_COMMITTER_EMAIL": "test@example.com",
        },
    )
    (repo_dir / "sample.txt").write_text("after\n", encoding="utf-8")
    patch_text = subprocess.run(
        ["git", "diff", "HEAD"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=repo_dir, check=True, capture_output=True, text=True)

    result = check_patch_applies(repo_dir, patch_text)

    assert result.status == "valid"

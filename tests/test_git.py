import subprocess
from pathlib import Path
from unittest import mock

import pytest

from swe_gpu_bench.core.git import checkout_commit, hide_future_history


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def commit_file(repo: Path, content: str, tag: str) -> str:
    (repo / "kernel.cu").write_text(content, encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-m", tag)
    git(repo, "tag", tag)
    return git(repo, "rev-parse", "HEAD")


def test_checkout_commit_fetches_pr_ref_and_retries_the_same_commit():
    calls = []
    failed = []

    def fake_run_git(args, cwd):
        calls.append(list(args))
        if args[:3] == ["git", "checkout", "--force"] and not failed:
            failed.append(True)
            raise subprocess.CalledProcessError(1, args)

    with mock.patch("swe_gpu_bench.core.git.run_git", side_effect=fake_run_git):
        checkout_commit(Path("/tmp/repo"), "head123", pr_number=7)

    assert calls == [
        ["git", "checkout", "--force", "head123"],
        ["git", "fetch", "origin", "pull/7/head"],
        ["git", "checkout", "--force", "head123"],
    ]


def test_checkout_commit_raises_instead_of_using_current_pr_head():
    calls = []

    def fake_run_git(args, cwd):
        calls.append(list(args))
        if args[:2] == ["git", "checkout"]:
            raise subprocess.CalledProcessError(1, args)

    with mock.patch("swe_gpu_bench.core.git.run_git", side_effect=fake_run_git):
        with pytest.raises(subprocess.CalledProcessError):
            checkout_commit(Path("/tmp/repo"), "gone123", pr_number=7)

    assert ["git", "fetch", "origin", "gone123"] in calls
    assert all("FETCH_HEAD" not in call for call in calls)


def test_checkout_commit_rejects_missing_commit():
    with mock.patch("swe_gpu_bench.core.git.run_git") as run_git:
        with pytest.raises(subprocess.CalledProcessError):
            checkout_commit(Path("/tmp/repo"), None, pr_number=7)
    run_git.assert_not_called()


def test_hide_future_history_leaves_only_base_ancestry(tmp_path):
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    git(upstream, "init", "-b", "main")
    base = commit_file(upstream, "v1\n", "v1")
    future = commit_file(upstream, "v2\n", "v2")
    git(tmp_path, "clone", "--quiet", str(upstream), "repo")
    repo = tmp_path / "repo"
    git(repo, "checkout", "--quiet", "--force", base)

    hide_future_history(repo)

    assert git(repo, "rev-parse", "HEAD") == base
    assert git(repo, "log", "--all", "--format=%H").splitlines() == [base]
    assert git(repo, "tag").splitlines() == ["v1"]
    assert git(repo, "remote") == ""
    assert future not in git(repo, "log", "--all", "--reflog", "--format=%H")
    with pytest.raises(subprocess.CalledProcessError):
        git(repo, "show", "main:kernel.cu")


def test_hide_future_history_skips_non_git_directories(tmp_path):
    hide_future_history(tmp_path)

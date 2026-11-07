"""Pinned evaluation checkouts and private clone caches."""
from __future__ import annotations
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Optional, Sequence

from swe_gpu_bench.shared.repo import repo_to_slug

def run_git(args: Sequence[str], cwd: Path) -> None:
    subprocess.run(list(args), cwd=str(cwd), check=True, text=True,
                   env={**os.environ, 'GIT_TERMINAL_PROMPT': '0'},
                   timeout=int(os.getenv('SWEBENCH_GIT_TIMEOUT_SECONDS', '600')))


def _repo_cache_path(cache_root: Path, repo_slug: str) -> Path:
    return cache_root / repo_to_slug(repo_slug)


def _repo_cache_lock_path(cache_root: Path, repo_slug: str) -> Path:
    return cache_root / f"{repo_to_slug(repo_slug)}.lock"


def _remove_stale_repo_cache_lock(lock_path: Path, stale_after_seconds: float = 1800.0) -> bool:
    try:
        age_seconds = time.time() - lock_path.stat().st_mtime
    except FileNotFoundError:
        return False
    if age_seconds < stale_after_seconds:
        return False
    shutil.rmtree(lock_path, ignore_errors=True)
    return True


def _refresh_repo_cache_remote(cache_repo: Path, *, fetch_attempts: int = 3) -> bool:
    last_error: subprocess.CalledProcessError | None = None
    for attempt in range(1, fetch_attempts + 1):
        try:
            run_git(["git", "fetch", "--all", "--tags"], cwd=cache_repo)
            return True
        except subprocess.CalledProcessError as exc:
            last_error = exc
            if attempt == fetch_attempts:
                break
            time.sleep(float(attempt))
    if last_error is not None:
        raise last_error
    return False


def _refresh_repo_cache(cache_root: Path, repo_slug: str, remote_url: str, target_commit: str | None = None) -> Path:
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_repo = _repo_cache_path(cache_root, repo_slug)
    lock_path = _repo_cache_lock_path(cache_root, repo_slug)
    while True:
        try:
            lock_path.mkdir(parents=False, exist_ok=False)
            break
        except FileExistsError:
            if _remove_stale_repo_cache_lock(lock_path):
                continue
            time.sleep(0.2)
    try:
        if target_commit:
            if not (cache_repo / '.git').exists():
                cache_repo.mkdir(exist_ok=True)
                run_git(['git', 'init', '-q'], cwd=cache_repo)
                run_git(['git', 'remote', 'add', 'origin', remote_url], cwd=cache_repo)
            present = subprocess.run(['git', 'cat-file', '-e', target_commit + '^{commit}'],
                                     cwd=cache_repo, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                     timeout=int(os.getenv('SWEBENCH_GIT_TIMEOUT_SECONDS', '600')))
            if present.returncode:
                run_git(['git', 'fetch', '--depth=1', '--no-tags', 'origin', target_commit], cwd=cache_repo)
            run_git(['git', 'update-ref', 'refs/heads/evaluation-base', target_commit], cwd=cache_repo)
            run_git(['git', 'symbolic-ref', 'HEAD', 'refs/heads/evaluation-base'], cwd=cache_repo)
        elif not (cache_repo / ".git").exists():
            if cache_repo.exists():
                shutil.rmtree(cache_repo)
            run_git(["git", "clone", remote_url, cache_repo.name], cwd=cache_repo.parent)
            _refresh_repo_cache_remote(cache_repo)
        else:
            try:
                _refresh_repo_cache_remote(cache_repo)
            except subprocess.CalledProcessError as exc:
                print(f"[warn] repo cache refresh failed for {repo_slug}: {exc}. Reusing existing cache.")
    finally:
        shutil.rmtree(lock_path, ignore_errors=True)
    return cache_repo


def clone_or_update_repo(
    case_paths: Any,
    repo_slug: str,
    head_sha: str | None,
    pr_number: Optional[int] = None,
    cache_root: Path | None = None,
    strict_commit: bool = False,
) -> None:
    remote_url = f"https://github.com/{repo_slug}.git"
    git_dir = case_paths.repo_dir / ".git"
    if cache_root is not None:
        cache_repo = _refresh_repo_cache(cache_root, repo_slug, remote_url,
                                         target_commit=head_sha if strict_commit else None)
        if not git_dir.exists():
            run_git(
                ["git", "clone", "--shared", str(cache_repo), case_paths.repo_dir.name],
                cwd=case_paths.repo_dir.parent,
            )
            run_git(["git", "remote", "set-url", "origin", remote_url], cwd=case_paths.repo_dir)
    else:
        if not git_dir.exists():
            run_git(["git", "clone", remote_url, case_paths.repo_dir.name], cwd=case_paths.repo_dir.parent)
        run_git(["git", "fetch", "--all", "--tags"], cwd=case_paths.repo_dir)
    checked_out = False
    if head_sha is not None:
        try:
            run_git(["git", "checkout", "--force", head_sha], cwd=case_paths.repo_dir)
            checked_out = True
        except subprocess.CalledProcessError:
            pass
    if not checked_out:
        if pr_number is None:
            raise subprocess.CalledProcessError(1, "git checkout", "head_sha is None and no pr_number provided")
        run_git(["git", "fetch", "origin", f"pull/{pr_number}/head"], cwd=case_paths.repo_dir)
        # Evaluation must never substitute the solved PR head for a missing base.
        target = head_sha if strict_commit and head_sha is not None else "FETCH_HEAD"
        run_git(["git", "checkout", "--force", target], cwd=case_paths.repo_dir)

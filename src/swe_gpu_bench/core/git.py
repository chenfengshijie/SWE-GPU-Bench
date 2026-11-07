"""Repository checkouts backed by an optional shared clone cache."""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Protocol, Sequence

from swe_gpu_bench.core.io import repo_to_slug

REPO_CACHE_ROOT_ENV = "SWEBENCH_REPO_CACHE_ROOT"
STALE_LOCK_SECONDS = 1800.0


class HasRepoDir(Protocol):
    repo_dir: Path


def run_git(args: Sequence[str], cwd: Path) -> None:
    subprocess.run(list(args), cwd=str(cwd), check=True, text=True)


def _git_lines(args: Sequence[str], cwd: Path) -> list[str]:
    completed = subprocess.run(list(args), cwd=str(cwd), check=True, text=True, capture_output=True)
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def repo_cache_root(default: Path) -> Path:
    """``$SWEBENCH_REPO_CACHE_ROOT`` if set, otherwise ``default``."""
    raw = os.getenv(REPO_CACHE_ROOT_ENV)
    return Path(raw).resolve() if raw else default


def _acquire_lock(lock_path: Path) -> None:
    while True:
        try:
            lock_path.mkdir(parents=False, exist_ok=False)
            return
        except FileExistsError:
            try:
                stale = time.time() - lock_path.stat().st_mtime >= STALE_LOCK_SECONDS
            except FileNotFoundError:
                continue
            if stale:
                shutil.rmtree(lock_path, ignore_errors=True)
                continue
            time.sleep(0.2)


def _fetch_with_retry(repo: Path, attempts: int = 3) -> None:
    for attempt in range(1, attempts + 1):
        try:
            run_git(["git", "fetch", "--all", "--tags"], cwd=repo)
            return
        except subprocess.CalledProcessError:
            if attempt == attempts:
                raise
            time.sleep(float(attempt))


def _refresh_repo_cache(cache_root: Path, repo_slug: str, remote_url: str) -> Path:
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_repo = cache_root / repo_to_slug(repo_slug)
    lock_path = cache_root / f"{repo_to_slug(repo_slug)}.lock"
    _acquire_lock(lock_path)
    try:
        if not (cache_repo / ".git").exists():
            shutil.rmtree(cache_repo, ignore_errors=True)
            run_git(["git", "clone", remote_url, cache_repo.name], cwd=cache_repo.parent)
            _fetch_with_retry(cache_repo)
        else:
            try:
                _fetch_with_retry(cache_repo)
            except subprocess.CalledProcessError as exc:
                print(f"[warn] repo cache refresh failed for {repo_slug}: {exc}. Reusing existing cache.")
    finally:
        shutil.rmtree(lock_path, ignore_errors=True)
    return cache_repo


def checkout_commit(repo_dir: Path, commit: str | None, pr_number: int | None = None) -> None:
    """Check out exactly ``commit``.

    A commit missing from the clone is fetched (``refs/pull/<pr>/head``, then the SHA itself) and retried;
    it is never replaced by whatever those refs point to now, since a force-pushed PR head is different code.
    """
    if not commit:
        raise subprocess.CalledProcessError(1, ["git", "checkout"], "no commit recorded")
    try:
        run_git(["git", "checkout", "--force", commit], cwd=repo_dir)
        return
    except subprocess.CalledProcessError as exc:
        error = exc
    fetch_refs = ([f"pull/{pr_number}/head"] if pr_number is not None else []) + [commit]
    for ref in fetch_refs:
        try:
            run_git(["git", "fetch", "origin", ref], cwd=repo_dir)
            run_git(["git", "checkout", "--force", commit], cwd=repo_dir)
            return
        except subprocess.CalledProcessError as exc:
            error = exc
    raise error


def hide_future_history(repo_dir: Path) -> None:
    """Leave only HEAD's ancestry reachable before an agent works in ``repo_dir``.

    HEAD is detached, remotes and branches are removed, tags are kept only if HEAD contains them (so
    ``git describe``-based versioning still works), and reflogs are expired. Objects of later commits may
    still sit in the object store or a ``--shared`` cache, but no ref, log or ``git show <ref>:path`` reaches them.
    """
    git_dir = repo_dir / ".git"
    if not git_dir.is_dir():
        return
    run_git(["git", "checkout", "--quiet", "--detach"], cwd=repo_dir)
    for remote in _git_lines(["git", "remote"], repo_dir):
        run_git(["git", "remote", "remove", remote], cwd=repo_dir)
    keep = set(_git_lines(["git", "tag", "--merged", "HEAD", "--format=%(refname)"], repo_dir))
    doomed = [ref for ref in _git_lines(["git", "for-each-ref", "--format=%(refname)"], repo_dir) if ref not in keep]
    if doomed:
        subprocess.run(
            ["git", "update-ref", "--no-deref", "--stdin"],
            cwd=str(repo_dir),
            check=True,
            text=True,
            input="".join(f"delete {ref}\n" for ref in doomed),
        )
    run_git(["git", "reflog", "expire", "--expire=now", "--all"], cwd=repo_dir)
    for name in ("FETCH_HEAD", "ORIG_HEAD"):
        (git_dir / name).unlink(missing_ok=True)


def clone_or_update_repo(
    case_paths: HasRepoDir,
    repo_slug: str,
    commit: str | None,
    pr_number: int | None = None,
    cache_root: Path | None = None,
) -> None:
    """Clone ``repo_slug`` into ``case_paths.repo_dir`` (via the cache when given) and check out ``commit``."""
    remote_url = f"https://github.com/{repo_slug}.git"
    repo_dir = case_paths.repo_dir
    if cache_root is not None:
        cache_repo = _refresh_repo_cache(cache_root, repo_slug, remote_url)
        if not (repo_dir / ".git").exists():
            run_git(["git", "clone", "--shared", str(cache_repo), repo_dir.name], cwd=repo_dir.parent)
            run_git(["git", "remote", "set-url", "origin", remote_url], cwd=repo_dir)
    else:
        if not (repo_dir / ".git").exists():
            run_git(["git", "clone", remote_url, repo_dir.name], cwd=repo_dir.parent)
        run_git(["git", "fetch", "--all", "--tags"], cwd=repo_dir)
    checkout_commit(repo_dir, commit, pr_number)


def prepare_record_checkout(
    case_paths: HasRepoDir,
    record: dict[str, Any],
    *,
    cache_root: Path | None,
    clone_repo_fn=clone_or_update_repo,
    attempts: int = 2,
) -> None:
    """Check out a dataset record's ``base_commit``, wiping and re-cloning once on git failure.

    ``cache_root`` is the default cache location; ``$SWEBENCH_REPO_CACHE_ROOT`` overrides it.
    """
    if cache_root is not None:
        cache_root = repo_cache_root(cache_root)
    for attempt in range(1, attempts + 1):
        try:
            clone_repo_fn(
                case_paths,
                record["repo"],
                record.get("base_commit"),
                pr_number=record.get("pr_number"),
                cache_root=cache_root,
            )
            return
        except subprocess.CalledProcessError:
            if attempt == attempts:
                raise
            shutil.rmtree(case_paths.repo_dir, ignore_errors=True)
            case_paths.repo_dir.mkdir(parents=True, exist_ok=True)

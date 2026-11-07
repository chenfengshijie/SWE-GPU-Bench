from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from swe_gpu_bench.shared.diff import parse_diff_sections

TEST_PATCH_FAILURE_STATUSES = frozenset({"apply_failed", "restore_failed"})


@dataclass(frozen=True)
class TestPatchRestoreResult:
    status: str
    paths: tuple[str, ...]
    error: str = ""


def is_test_patch_failure_status(status: str) -> bool:
    return status in TEST_PATCH_FAILURE_STATUSES


def extract_test_patch_paths(test_patch: str) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for section in parse_diff_sections(test_patch):
        path = section.new_path if section.new_path != "/dev/null" else section.old_path
        path = path.strip()
        if not path or path == "/dev/null" or path in seen:
            continue
        seen.add(path)
        paths.append(path)
    return paths


def restore_test_patch_targets(test_patch: str, repo_dir: Path) -> TestPatchRestoreResult:
    paths = tuple(extract_test_patch_paths(test_patch))
    if not paths:
        return TestPatchRestoreResult(status="not_needed", paths=())

    errors: list[str] = []
    for rel_path in paths:
        try:
            repo_path = _resolve_repo_path(repo_dir, rel_path)
            tracked_in_head = _is_tracked_in_head(repo_dir, rel_path)
        except (RuntimeError, ValueError) as exc:
            errors.append(str(exc))
            continue

        if tracked_in_head:
            restore = subprocess.run(
                ["git", "checkout", "HEAD", "--", rel_path],
                cwd=str(repo_dir),
                capture_output=True,
                text=True,
                check=False,
            )
            if restore.returncode != 0:
                errors.append(_format_git_error(f"restore {rel_path}", restore))
            continue

        if repo_path.exists() or repo_path.is_symlink():
            try:
                if repo_path.is_dir() and not repo_path.is_symlink():
                    shutil.rmtree(repo_path)
                else:
                    repo_path.unlink()
            except OSError as exc:
                errors.append(f"Failed to remove untracked {rel_path}: {exc}")

    if errors:
        return TestPatchRestoreResult(
            status="restore_failed",
            paths=paths,
            error="; ".join(errors),
        )
    return TestPatchRestoreResult(status="restored", paths=paths)


def _resolve_repo_path(repo_dir: Path, rel_path: str) -> Path:
    candidate = Path(rel_path)
    if candidate.is_absolute() or not rel_path.strip() or ".." in candidate.parts:
        raise ValueError(f"Invalid test_patch path: {rel_path!r}")
    return repo_dir / candidate


def _is_tracked_in_head(repo_dir: Path, rel_path: str) -> bool:
    result = subprocess.run(
        ["git", "ls-tree", "--name-only", "HEAD", "--", rel_path],
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(_format_git_error(f"inspect {rel_path}", result))
    return bool(result.stdout.strip())


def _format_git_error(action: str, result: subprocess.CompletedProcess[str]) -> str:
    detail = (result.stderr or result.stdout or "").strip()
    suffix = f": {detail}" if detail else f" (exit {result.returncode})"
    return f"Failed to {action}{suffix}"

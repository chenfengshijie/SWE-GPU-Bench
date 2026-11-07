"""Applying, checking and restoring unified diffs inside a repository checkout."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from swe_gpu_bench.core.diff import parse_diff_sections

TEST_PATCH_FAILURE_STATUSES = frozenset({"apply_failed", "restore_failed"})


@dataclass(frozen=True)
class PatchCheckResult:
    status: str
    stderr: str = ""


def extract_minimal_patch(patch_text: str) -> str:
    lines = patch_text.splitlines(keepends=True)
    output: list[str] = []
    i = 0
    total_delta = 0

    while i < len(lines):
        line = lines[i]
        if (
            line.startswith("diff --git")
            or line.startswith("index ")
            or line.startswith("--- ")
            or line.startswith("+++ ")
            or line.startswith("new file")
            or line.startswith("deleted file")
        ):
            if line.startswith("diff --git"):
                total_delta = 0
            output.append(line)
            i += 1
            continue

        if line.startswith("@@"):
            match = re.match(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@(.*)", line)
            if not match:
                output.append(line)
                i += 1
                continue

            old_start = int(match.group(1))
            hunk_tail = match.group(3)
            i += 1
            hunk_lines: list[str] = []
            while i < len(lines) and not lines[i].startswith(("@@", "diff ")):
                hunk_lines.append(lines[i])
                i += 1

            while hunk_lines and hunk_lines[-1].startswith(" "):
                hunk_lines.pop()
            if not hunk_lines:
                continue

            old_count = sum(1 for entry in hunk_lines if not entry.startswith("+"))
            new_count = sum(1 for entry in hunk_lines if not entry.startswith("-"))
            new_start = old_start + total_delta

            added = sum(1 for entry in hunk_lines if entry.startswith("+"))
            removed = sum(1 for entry in hunk_lines if entry.startswith("-"))
            total_delta += added - removed

            old_str = f"{old_start},{old_count}" if old_count != 1 else str(old_start)
            new_str = f"{new_start},{new_count}" if new_count != 1 else str(new_start)
            output.append(f"@@ -{old_str} +{new_str} @@{hunk_tail}\n")
            output.extend(hunk_lines)
            continue

        output.append(line)
        i += 1

    return "".join(output)


@contextmanager
def _patch_file(patch_text: str) -> Iterator[str]:
    with tempfile.NamedTemporaryFile("w", suffix=".diff", delete=False, encoding="utf-8") as handle:
        handle.write(patch_text)
        path = handle.name
    try:
        yield path
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _run(cmd: Sequence[str], repo_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(cmd), cwd=str(repo_dir), capture_output=True, text=True, check=False)


def _first_success(repo_dir: Path, patch_text: str, commands: Sequence[Sequence[str]]) -> tuple[int | None, str]:
    """Run each ``{patch}`` command template until one succeeds; return its index and the last error."""
    last_error = ""
    with _patch_file(patch_text) as path:
        for index, template in enumerate(commands):
            result = _run([path if item == "{patch}" else item for item in template], repo_dir)
            if result.returncode == 0:
                return index, ""
            last_error = (result.stderr or result.stdout or "").strip()
    return None, last_error


def check_patch_applies(repo_dir: Path, patch_text: str) -> PatchCheckResult:
    candidates = [patch_text]
    normalized = extract_minimal_patch(patch_text)
    if normalized != patch_text:
        candidates.append(normalized)
    last_error = ""
    for candidate in candidates:
        index, last_error = _first_success(repo_dir, candidate, [["git", "apply", "--check", "{patch}"]])
        if index is not None:
            return PatchCheckResult(status="valid")
    return PatchCheckResult(status="invalid", stderr=last_error)


def _try_apply_patch_text(repo_dir: Path, patch_text: str, commands: Sequence[Sequence[str]]) -> PatchCheckResult:
    index, last_error = _first_success(repo_dir, patch_text, commands)
    if index is not None:
        return PatchCheckResult(status="applied")
    return PatchCheckResult(status="apply_failed", stderr=last_error)


def apply_pred_patch(repo_dir: Path, patch_text: str) -> PatchCheckResult:
    raw_result = _try_apply_patch_text(repo_dir, patch_text, [["git", "apply", "{patch}"]])
    if raw_result.status == "applied":
        return raw_result

    normalized = extract_minimal_patch(patch_text)
    if normalized == patch_text:
        return raw_result

    normalized_result = _try_apply_patch_text(
        repo_dir,
        normalized,
        [
            ["git", "apply", "{patch}"],
            ["git", "apply", "--reject", "{patch}"],
            ["patch", "--batch", "--fuzz=5", "-p1", "-i", "{patch}"],
        ],
    )
    return normalized_result


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


def apply_test_patch(test_patch: str, repo_dir: Path) -> str:
    """Reset files touched by ``test_patch`` and apply it with progressively looser strategies.

    Returns ``restore_failed``, ``applied_clean`` (git apply --3way), ``applied_fuzz``
    (patch --fuzz=3), ``applied_additive`` (new-file-only patch written directly) or
    ``apply_failed``.
    """
    restore_result = restore_test_patch_targets(test_patch, repo_dir)
    if restore_result.status == "restore_failed":
        return restore_result.status
    index, _ = _first_success(
        repo_dir,
        test_patch,
        [["git", "apply", "--3way", "{patch}"], ["patch", "--fuzz=3", "-p1", "--input", "{patch}"]],
    )
    if index is not None:
        return ("applied_clean", "applied_fuzz")[index]
    if _apply_additive_only(test_patch, repo_dir):
        return "applied_additive"
    return "apply_failed"


def _apply_additive_only(patch_text: str, repo_dir: Path) -> bool:
    """Write files from a diff that only creates new files; False if any section edits existing code."""
    files: list[tuple[Path, str]] = []
    for section in parse_diff_sections(patch_text):
        lines = section.raw_text.splitlines(keepends=True)
        if not any(line.startswith("new file mode") for line in lines):
            return False
        content = [line[1:] for line in lines if line.startswith("+") and not line.startswith("+++ ")]
        files.append((repo_dir / section.new_path, "".join(content)))
    if not files:
        return False
    for file_path, content in files:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
    return True


def capture_diff(repo_dir: Path) -> str:
    """Return ``git diff HEAD`` for the working tree."""
    return _run(["git", "diff", "HEAD"], repo_dir).stdout

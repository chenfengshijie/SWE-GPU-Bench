from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


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


def check_patch_applies(repo_dir: Path, patch_text: str) -> PatchCheckResult:
    candidates = [patch_text]
    normalized = extract_minimal_patch(patch_text)
    if normalized != patch_text:
        candidates.append(normalized)
    last_error = ""
    for candidate in candidates:
        with tempfile.NamedTemporaryFile("w", suffix=".diff", delete=False, encoding="utf-8") as handle:
            handle.write(candidate)
            patch_path = handle.name
        try:
            completed = subprocess.run(
                ["git", "apply", "--check", patch_path],
                cwd=str(repo_dir),
                capture_output=True,
                text=True,
                check=False,
            )
        finally:
            try:
                os.unlink(patch_path)
            except OSError:
                pass
        if completed.returncode == 0:
            return PatchCheckResult(status="valid")
        last_error = (completed.stderr or completed.stdout or "").strip()
    return PatchCheckResult(status="invalid", stderr=last_error)


def _try_apply_patch_text(
    repo_dir: Path,
    patch_text: str,
    cmds_template: Sequence[Sequence[str]],
) -> PatchCheckResult:
    with tempfile.NamedTemporaryFile("w", suffix=".diff", delete=False, encoding="utf-8") as handle:
        handle.write(patch_text)
        patch_path = handle.name
    try:
        last_stderr = ""
        for template in cmds_template:
            cmd = [patch_path if item == "{patch}" else item for item in template]
            result = subprocess.run(
                cmd,
                cwd=str(repo_dir),
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                return PatchCheckResult(status="applied")
            last_stderr = (result.stderr or result.stdout or "").strip()
    finally:
        try:
            os.unlink(patch_path)
        except OSError:
            pass
    return PatchCheckResult(status="apply_failed", stderr=last_stderr)


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
    if normalized_result.status == "applied":
        return normalized_result
    return normalized_result

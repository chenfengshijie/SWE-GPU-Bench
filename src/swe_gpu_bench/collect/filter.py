"""``collect annotate`` / ``collect filter``: deterministic PR signals and the candidate filter.

``annotate`` writes the signals into each PR JSON. ``filter`` keeps PRs that pass every enabled
criterion (resolved issue, test change, merged, default-branch base, GPU code touched, bounded
size, no binary diffs) and reports how many PRs each criterion rejected.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from swe_gpu_bench.core.diff import is_test_like_path, parse_diff_sections, summarize_diff
from swe_gpu_bench.core.io import load_json, load_text, write_json

logger = logging.getLogger(__name__)

ANNOTATION_VERSION = 2
ISSUE_PAT = re.compile(
    r"(?i)\b(close|closes|closed|fix|fixes|fixed|resolve|resolves|resolved)\b"
    r"(?:\s+[^#\n\r:;,.()]*)?\s+"
    r"(?:"
    r"#(?P<local>\d+)"
    r"|(?P<repo>[\w.\-]+/[\w.\-]+)#(?P<cross>\d+)"
    r"|https?://github\.com/(?P<url_repo>[\w.\-]+/[\w.\-]+)/issues/(?P<url_num>\d+)"
    r")"
)
DEFAULT_BRANCH_NAMES = frozenset({"main", "master", "develop", "dev", "trunk"})
NON_DEFAULT_BASE_PAT = re.compile(
    r"^(?:release|releases|branch|stable|hotfix|backport)[/_-]|^v?\d+(?:\.\d+)+(?:\.x)?$|\.x$|^gh/[^/]+/\d+/base$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FilterCriteria:
    require_issue: bool = True
    require_merged: bool = True
    require_default_branch: bool = True
    require_gpu_signal: bool = True
    max_code_lines: int = 1000
    max_code_files: int = 30


def metadata_to_patch_path(metadata_path: Path, patches_dir: Path) -> Path:
    return patches_dir / metadata_path.parent.name / f"{metadata_path.stem}.diff"


def iter_pr_metadata(prs_dir: Path):
    yield from sorted(prs_dir.glob("*/*.json"))


def extract_resolved_issues(payload: dict) -> list[str]:
    text = "\n".join(
        [payload.get("title") or "", payload.get("body") or "", "\n".join(payload.get("commit_messages") or [])]
    )
    resolved = {
        number
        for match in ISSUE_PAT.finditer(text)
        if (number := match.group("local") or match.group("cross") or match.group("url_num"))
    }
    return sorted(resolved, key=int)


def has_test_patch(diff_text: str) -> bool:
    return any(
        is_test_like_path(section.old_path) or is_test_like_path(section.new_path)
        for section in parse_diff_sections(diff_text)
    )


def targets_default_branch(payload: dict[str, Any]) -> bool | None:
    """``None`` when unknown; release, backport and stacked-PR bases (``gh/<user>/<n>/base``) are not default."""
    base_ref = payload.get("base_ref")
    if not base_ref:
        return None
    if payload.get("default_branch"):
        return base_ref == payload["default_branch"]
    return base_ref in DEFAULT_BRANCH_NAMES or not NON_DEFAULT_BASE_PAT.search(base_ref)


def pr_signals(payload: dict[str, Any], diff_text: str | None) -> dict[str, Any]:
    issues = extract_resolved_issues(payload)
    summary = summarize_diff(diff_text) if diff_text is not None else None
    return {
        "annotation_version": ANNOTATION_VERSION,
        "resolves_issues": bool(issues),
        "resolved_issues": issues,
        "has_test_patch": bool(summary and summary.test_files),
        "merged": (payload["merged_at"] is not None) if "merged_at" in payload else None,
        "targets_default_branch": targets_default_branch(payload),
        "gpu_signal": bool(summary and summary.gpu_signal),
        "code_files": len(summary.code_files) if summary else 0,
        "code_changed_lines": summary.code_changed_lines if summary else 0,
        "binary_files": list(summary.binary_files) if summary else [],
        "has_patch": summary is not None,
    }


def rejection_reasons(signals: dict[str, Any], criteria: FilterCriteria = FilterCriteria()) -> list[str]:
    reasons: list[str] = []
    if not signals["has_patch"]:
        return ["missing_patch"]
    if criteria.require_issue and not signals["resolves_issues"]:
        reasons.append("no_resolved_issue")
    if not signals["has_test_patch"]:
        reasons.append("no_test_change")
    if criteria.require_merged and signals["merged"] is False:
        reasons.append("not_merged")
    if criteria.require_default_branch and signals["targets_default_branch"] is False:
        reasons.append("non_default_base")
    if signals["code_files"] == 0:
        reasons.append("no_code_change")
    elif criteria.require_gpu_signal and not signals["gpu_signal"]:
        reasons.append("no_gpu_signal")
    if signals["code_changed_lines"] > criteria.max_code_lines:
        reasons.append("too_many_changed_lines")
    if signals["code_files"] > criteria.max_code_files:
        reasons.append("too_many_files")
    if signals["binary_files"]:
        reasons.append("binary_diff")
    return reasons


def _signals(metadata_path: Path, patches_dir: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    payload = load_json(metadata_path)
    patch_file = metadata_to_patch_path(metadata_path, patches_dir)
    diff_text = load_text(patch_file) if patch_file.exists() else None
    return payload, pr_signals(payload, diff_text), patch_file


def annotate_pr_file(metadata_path: Path, patches_dir: Path) -> tuple[bool, bool]:
    """Write the PR signals into the PR JSON; return ``(resolves_issues, has_test_patch)``."""
    payload, signals, _ = _signals(metadata_path, patches_dir)
    payload.update(signals)
    write_json(metadata_path, payload)
    return signals["resolves_issues"], signals["has_test_patch"]


def _annotate_one(task: tuple[Path, Path]) -> tuple[bool, bool, bool]:
    metadata_path, patches_dir = task
    payload = load_json(metadata_path)
    if payload.get("annotation_version") == ANNOTATION_VERSION:
        return True, bool(payload.get("resolves_issues")), bool(payload["has_test_patch"])
    return (False, *annotate_pr_file(metadata_path, patches_dir))


def annotate_prs(prs_dir: Path, patches_dir: Path, workers: int = 1) -> dict[str, int]:
    tasks = [(path, patches_dir) for path in iter_pr_metadata(prs_dir)]
    if workers <= 1:
        results = list(map(_annotate_one, tasks))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_annotate_one, tasks, chunksize=64))
    summary = {
        "total": len(results),
        "skipped_already_annotated": sum(already for already, _, _ in results),
        "resolves_issues": sum(issues for _, issues, _ in results),
        "has_test_patch": sum(tests for _, _, tests in results),
        "matched_either": sum(issues or tests for _, issues, tests in results),
    }
    logger.info("Annotation summary: %s", summary)
    return summary


def filter_prs(
    prs_dir: Path, patches_dir: Path, criteria: FilterCriteria = FilterCriteria()
) -> tuple[list[dict], Counter[str]]:
    """Candidate PRs plus a count of rejections per reason (a PR can fail several criteria)."""
    matches: list[dict] = []
    rejections: Counter[str] = Counter()
    for metadata_path in iter_pr_metadata(prs_dir):
        payload, signals, patch_file = _signals(metadata_path, patches_dir)
        reasons = rejection_reasons(signals, criteria)
        if reasons:
            rejections.update(reasons)
            continue
        matches.append(
            {
                "repo": payload.get("repo"),
                "number": payload.get("number"),
                "title": payload.get("title"),
                "html_url": payload.get("html_url"),
                "resolved_issues": signals["resolved_issues"],
                "code_files": signals["code_files"],
                "code_changed_lines": signals["code_changed_lines"],
                "metadata_path": str(metadata_path),
                "patch_path": str(patch_file),
            }
        )
    return matches, rejections


def add_annotate_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prs-dir", default="data/prs")
    parser.add_argument("--patches-dir", default="data/patches")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))


def run_annotate(args: argparse.Namespace) -> int:
    annotate_prs(Path(args.prs_dir), Path(args.patches_dir), workers=args.workers)
    return 0


def add_filter_arguments(parser: argparse.ArgumentParser) -> None:
    defaults = FilterCriteria()
    parser.add_argument("--prs-dir", default="data/prs")
    parser.add_argument("--patches-dir", default="data/patches")
    parser.add_argument("--output", default="data/filtered_prs_with_tests.json")
    parser.add_argument("--allow-no-issue", action="store_true", help="Keep PRs that reference no resolved issue")
    parser.add_argument("--include-unmerged", action="store_true", help="Keep PRs closed without merging")
    parser.add_argument("--include-non-default-base", action="store_true", help="Keep release/backport/stacked PRs")
    parser.add_argument("--allow-no-gpu-signal", action="store_true", help="Keep PRs whose code changes show no GPU signal")
    parser.add_argument("--max-code-lines", type=int, default=defaults.max_code_lines, help="Changed non-test lines")
    parser.add_argument("--max-code-files", type=int, default=defaults.max_code_files, help="Changed non-test files")


def run_filter(args: argparse.Namespace) -> int:
    criteria = FilterCriteria(
        require_issue=not args.allow_no_issue,
        require_merged=not args.include_unmerged,
        require_default_branch=not args.include_non_default_base,
        require_gpu_signal=not args.allow_no_gpu_signal,
        max_code_lines=args.max_code_lines,
        max_code_files=args.max_code_files,
    )
    matches, rejections = filter_prs(Path(args.prs_dir), Path(args.patches_dir), criteria)
    output = Path(args.output)
    write_json(output, matches)
    summary_path = output.with_name(f"{output.stem}.summary.json")
    write_json(summary_path, {"kept": len(matches), "rejections": dict(rejections.most_common())})
    logger.info("Wrote %s filtered PRs to %s", len(matches), output)
    for reason, count in rejections.most_common():
        logger.info("  rejected %-24s %s", reason, count)
    return 0

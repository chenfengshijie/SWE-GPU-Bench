"""``generate dataset-check``: offline quality gate for dataset rows.

Each row gets a list of issues. An ``error`` means the row is not usable as a benchmark task as it
stands; a ``warning`` needs a human look. The checks read only the rows (and, with ``--verify-root``,
the ``generate verify --dataset`` results), so they need no GPU, network or model.
``generate dataset-build`` writes the same report next to its output and ``generate export`` drops
rows with errors.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from swe_gpu_bench.core.diff import (
    changed_lines,
    diff_files,
    is_test_like_path,
    new_names_used_by_tests,
    parse_diff_sections,
    summarize_diff,
)
from swe_gpu_bench.core.io import load_json, load_jsonl, write_json, write_jsonl
from swe_gpu_bench.generate.split_patch import (
    TEST_TARGET_PAT,
    SplitResult,
    extract_test_signals,
    is_support_file,
    split_patch_text,
)

logger = logging.getLogger(__name__)

ERROR = "error"
WARNING = "warning"
SEVERITY = {
    "missing_commit": ERROR,
    "empty_patch": ERROR,
    "no_test_commands": ERROR,
    "unverified": ERROR,
    "unmerged_pr": ERROR,
    "split_mismatch": ERROR,
    "unappliable_binary": ERROR,
    "cpu_only_tests": ERROR,
    "host_path": ERROR,
    "duplicate": ERROR,
    "no_test_patch": WARNING,
    "large_patch": WARNING,
    "cuda_path": WARNING,
    "network_in_tests": WARNING,
    "unpinned_torch": WARNING,
    "blackwell_signal": WARNING,
    "underspecified_interface": WARNING,
    "harness_entangled": WARNING,
    "no_gpu_signal": WARNING,
}
MAX_CODE_LINES = 1000
MAX_CODE_FILES = 30
MIN_FINGERPRINT_LINES = 3

COMMIT_PAT = re.compile(r"^[0-9a-f]{40}$")
CPU_ONLY_PAT = re.compile(
    r"\bNO_CUDA=1\b|\bUSE_CUDA=0\b|\bJAX_PLATFORMS=['\"]?cpu\b|\bTRITON_INTERPRET=1\b|--config[= ]cpu\b"
    r"|\bCUDA_VISIBLE_DEVICES=(?:''|\"\"|-1\b|(?=\s|$))"
)
TVM_TARGETS_PAT = re.compile(r"\bTVM_TEST_TARGETS=(['\"]?)(?P<targets>[^'\"\s]*)\1")
BAZEL_TEST_PAT = re.compile(r"\bbazel(?:isk)?\b[^\n;&|]*\btest\b")
BAZEL_GPU_PAT = re.compile(r"--config[= ](?:cuda|rocm)|\bTF_NEED_(?:CUDA|ROCM)=1")
HOST_PATH_PAT = re.compile(r"(?<![\w.~$}-])/(?:home/[\w.-]+|Users/[\w.-]+|opt/tiger|mnt|data\d*)/[^\s'\";|&)]*")
CUDA_PATH_PAT = re.compile(r"/usr/local/cuda-\d[\d.]*")
NETWORK_PAT = re.compile(
    r"\b(?:pip3?|uv\s+pip|conda|mamba|apt(?:-get)?)\s+install\b|\bgit\s+(?:clone|fetch|pull|submodule\s+update)\b"
    r"|\b(?:curl|wget)\s|\b(?:huggingface-cli|hf)\s+download\b"
)
PIP_INSTALL_PAT = re.compile(r"\b(?:pip3?|uv\s+pip)\s+install\b(?P<args>[^\n;&|]*)")
REQUIREMENT_NAME_SPLIT_PAT = re.compile(r"[\[=<>~!@;]")
BLACKWELL_PAT = re.compile(r"\bsm_?1[0-2]\d[af]?\b|\bcompute_1[0-2]\d|\bblackwell\b|\btcgen05\b|\bG?B[23]00\b", re.IGNORECASE)
SOURCE_REF_PAT = re.compile(r"[\w/.-]+\.(?:cu|cuh|cpp|cc|cxx|c|h|hpp|py)\b")


@dataclass(frozen=True)
class Issue:
    code: str
    detail: str = ""

    @property
    def severity(self) -> str:
        return SEVERITY[self.code]

    def to_json(self) -> dict[str, str]:
        return {"code": self.code, "detail": self.detail} if self.detail else {"code": self.code}


def combined_patch(row: dict[str, Any]) -> str:
    patch = row.get("patch") or ""
    if patch and not patch.endswith("\n"):
        patch += "\n"
    return patch + (row.get("test_patch") or "")


def resplit(row: dict[str, Any]) -> SplitResult:
    """Split ``patch`` + ``test_patch`` again with the current rules and the row's test commands."""
    return split_patch_text(combined_patch(row), extract_test_signals(list(row.get("test_commands") or [])))


def change_fingerprint(diff_text: str) -> str | None:
    """Hash of the changed lines regardless of file paths, so mirrored or re-landed changes collide."""
    lines = sorted({f"{sign}{line.strip()}" for sign in "+-" for line in changed_lines(diff_text, (sign,)) if line.strip()})
    if len(lines) < MIN_FINGERPRINT_LINES:
        return None
    return hashlib.sha1("\n".join(lines).encode("utf-8")).hexdigest()


def cpu_only_reason(commands_text: str) -> str | None:
    if match := CPU_ONLY_PAT.search(commands_text):
        return match.group(0).strip()
    for match in TVM_TARGETS_PAT.finditer(commands_text):
        targets = match.group("targets").lower()
        if "cuda" not in targets and "rocm" not in targets:
            return match.group(0)
    if BAZEL_TEST_PAT.search(commands_text) and not BAZEL_GPU_PAT.search(commands_text):
        return "bazel test without --config=cuda"
    return None


def installs_unpinned_torch(commands_text: str) -> bool:
    for match in PIP_INSTALL_PAT.finditer(commands_text):
        for token in match.group("args").split():
            token = token.strip("'\"")
            if REQUIREMENT_NAME_SPLIT_PAT.split(token, 1)[0].lower() == "torch" and "==" not in token and "@" not in token:
                return True
    return False


def entangled_build_files(patch: str, test_patch: str) -> list[str]:
    """Build files whose changes mix library and test wiring across the patch / test_patch boundary."""
    flagged = [
        section.new_path
        for section in parse_diff_sections(test_patch)
        if is_support_file(section.new_path)
        and not is_test_like_path(section.new_path)
        and any(SOURCE_REF_PAT.search(line) and not TEST_TARGET_PAT.search(line) for line in changed_lines(section.raw_text))
    ]
    flagged += [
        section.new_path
        for section in parse_diff_sections(patch)
        if is_support_file(section.new_path) and any(TEST_TARGET_PAT.search(line) for line in changed_lines(section.raw_text))
    ]
    return flagged


def load_verify_results(verify_root: Path) -> dict[str, dict[str, Any]]:
    """``generate verify --dataset`` results keyed by instance id."""
    results: dict[str, dict[str, Any]] = {}
    for path in sorted(verify_root.glob("*/result.json")):
        result = load_json(path)
        results[str(result.get("instance_id") or path.parent.name)] = result
    return results


def row_status(row: dict[str, Any], verify_results: dict[str, dict[str, Any]] | None) -> str | None:
    """Dataset verification wins when results are given; a row without a result then counts as unverified."""
    if verify_results is None:
        return row.get("verification_status")
    result = verify_results.get(str(row.get("instance_id")))
    return result.get("verification_status") if result else None


def check_row(row: dict[str, Any], status: str | None) -> list[Issue]:
    issues: list[Issue] = []
    patch, test_patch = row.get("patch") or "", row.get("test_patch") or ""
    test_commands = list(row.get("test_commands") or [])
    commands = "\n".join(list(row.get("setup_commands") or []) + test_commands)
    combined = combined_patch(row)
    summary = summarize_diff(combined)

    if not COMMIT_PAT.match(str(row.get("base_commit") or "")):
        issues.append(Issue("missing_commit", f"base_commit={row.get('base_commit')!r}"))
    if not patch.strip():
        issues.append(Issue("empty_patch"))
    if not test_commands:
        issues.append(Issue("no_test_commands"))
    if status != "verified":
        issues.append(Issue("unverified", f"verification_status={status!r}"))
    if "merged_at" in row and row["merged_at"] is None:
        issues.append(Issue("unmerged_pr"))

    split = resplit(row)
    to_patch = sorted(set(diff_files(test_patch)) - set(diff_files(split.test_patch_text)))
    to_tests = sorted(set(diff_files(patch)) - set(diff_files(split.patch_text)))
    if to_patch or to_tests:
        moves = [f"test_patch->patch: {', '.join(to_patch)}" if to_patch else "", f"patch->test_patch: {', '.join(to_tests)}" if to_tests else ""]
        issues.append(Issue("split_mismatch", "; ".join(move for move in moves if move)))

    if summary.binary_files:
        issues.append(Issue("unappliable_binary", ", ".join(summary.binary_files)))
    if reason := cpu_only_reason(commands):
        issues.append(Issue("cpu_only_tests", reason))
    if match := HOST_PATH_PAT.search(commands):
        issues.append(Issue("host_path", match.group(0)))

    if not test_patch.strip():
        issues.append(Issue("no_test_patch"))
    if summary.code_changed_lines > MAX_CODE_LINES or len(summary.code_files) > MAX_CODE_FILES:
        issues.append(Issue("large_patch", f"{summary.code_changed_lines} changed lines in {len(summary.code_files)} files"))
    if match := CUDA_PATH_PAT.search(commands):
        issues.append(Issue("cuda_path", match.group(0)))
    if match := NETWORK_PAT.search("\n".join(test_commands)):
        issues.append(Issue("network_in_tests", match.group(0).strip()))
    if installs_unpinned_torch(commands):
        issues.append(Issue("unpinned_torch"))
    if match := BLACKWELL_PAT.search("\n".join(changed_lines(combined)) + "\n" + commands):
        issues.append(Issue("blackwell_signal", match.group(0)))
    problem_statement = row.get("problem_statement") or ""
    if unnamed := [name for name in new_names_used_by_tests(patch, test_patch) if name not in problem_statement]:
        issues.append(Issue("underspecified_interface", ", ".join(unnamed[:10])))
    if entangled := entangled_build_files(patch, test_patch):
        issues.append(Issue("harness_entangled", ", ".join(entangled)))
    if not summary.gpu_signal:
        issues.append(Issue("no_gpu_signal"))
    return issues


def check_dataset(
    rows: list[dict[str, Any]],
    *,
    verify_results: dict[str, dict[str, Any]] | None = None,
    ignore: Iterable[str] = (),
) -> list[list[Issue]]:
    """Issues per row, in row order.

    A repeated instance id is a duplicate. Rows with the same change (mirrored or re-landed PRs) are
    duplicates of one kept row: the first without other errors, else the first of the group.
    """
    ignored = set(ignore)
    reports = [
        [issue for issue in check_row(row, row_status(row, verify_results)) if issue.code not in ignored] for row in rows
    ]
    if "duplicate" in ignored:
        return reports
    seen_ids: set[str] = set()
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        instance_id = str(row.get("instance_id"))
        if instance_id in seen_ids:
            reports[index].append(Issue("duplicate", "instance_id appears earlier in the dataset"))
        seen_ids.add(instance_id)
        if fingerprint := change_fingerprint(combined_patch(row)):
            groups.setdefault(fingerprint, []).append(index)
    for members in groups.values():
        kept = next((index for index in members if not has_error(reports[index])), members[0])
        for index in members:
            if index != kept and all(issue.code != "duplicate" for issue in reports[index]):
                reports[index].append(Issue("duplicate", f"same change as {rows[kept].get('instance_id')}"))
    return reports


def has_error(issues: list[Issue]) -> bool:
    return any(issue.severity == ERROR for issue in issues)


def clean_rows(rows: list[dict[str, Any]], reports: list[list[Issue]]) -> list[dict[str, Any]]:
    return [row for row, issues in zip(rows, reports) if not has_error(issues)]


def apply_verify_status(
    rows: list[dict[str, Any]], verify_results: dict[str, dict[str, Any]] | None
) -> list[dict[str, Any]]:
    if verify_results is None:
        return rows
    return [{**row, "verification_status": row_status(row, verify_results)} for row in rows]


def build_report(dataset: str, rows: list[dict[str, Any]], reports: list[list[Issue]]) -> dict[str, Any]:
    counts = Counter(code for issues in reports for code in {issue.code for issue in issues})
    with_errors = sum(has_error(issues) for issues in reports)
    return {
        "dataset": dataset,
        "rows": len(rows),
        "rows_with_errors": with_errors,
        "clean_rows": len(rows) - with_errors,
        "issue_counts": {
            code: {"severity": SEVERITY[code], "rows": counts[code]}
            for code in sorted(counts, key=lambda code: (SEVERITY[code] != ERROR, -counts[code], code))
        },
        "instances": [
            {
                "instance_id": row.get("instance_id"),
                "repo": row.get("repo"),
                "errors": [issue.to_json() for issue in issues if issue.severity == ERROR],
                "warnings": [issue.to_json() for issue in issues if issue.severity == WARNING],
            }
            for row, issues in zip(rows, reports)
            if issues
        ],
    }


def log_report(report: dict[str, Any], report_path: Path) -> None:
    logger.info(
        "Checked %d rows: %d with errors, %d clean -> %s",
        report["rows"], report["rows_with_errors"], report["clean_rows"], report_path,
    )
    for code, entry in report["issue_counts"].items():
        logger.info("  %-7s %-26s %d", entry["severity"], code, entry["rows"])


def write_check_report(dataset: str, rows: list[dict[str, Any]], reports: list[list[Issue]], report_path: Path) -> dict[str, Any]:
    report = build_report(dataset, rows, reports)
    write_json(report_path, report)
    log_report(report, report_path)
    return report


def add_ignore_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ignore", nargs="+", default=[], choices=sorted(SEVERITY), metavar="CODE", help="Issue codes to skip")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", default="data/dataset/swe_gpu_bench.jsonl")
    parser.add_argument("--report", default=None, help="Report JSON (default: <dataset>.check.json)")
    parser.add_argument("--output", default=None, help="Also write the rows without errors to this JSONL")
    parser.add_argument(
        "--verify-root", default=None, help="`generate verify --dataset` results; their status replaces verification_status"
    )
    add_ignore_argument(parser)


def run(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    dataset_path = Path(args.dataset)
    rows = load_jsonl(dataset_path)
    verify_results = load_verify_results(Path(args.verify_root)) if args.verify_root else None
    reports = check_dataset(rows, verify_results=verify_results, ignore=args.ignore)
    report_path = Path(args.report) if args.report else dataset_path.with_suffix(".check.json")
    write_check_report(str(dataset_path), rows, reports, report_path)
    if args.output:
        kept = apply_verify_status(clean_rows(rows, reports), verify_results)
        write_jsonl(Path(args.output), kept)
        logger.info("Wrote %d clean rows to %s", len(kept), args.output)
    return 0

"""Dataset assembly commands.

* ``generate dataset-build``: verified PRs + queries + split patches -> ``data/dataset/swe_gpu_bench.jsonl``
  (plus one ``<dataset>.jsonl`` per source dataset and a ``dataset-check`` report).
* ``generate dataset-finalize``: re-split ``patch`` + ``test_patch`` with the current rules (and optionally
  regenerate problem statements) for an existing JSONL; other fields are kept.
* ``generate perf-dataset-build``: ``ready`` perf-repro results -> ``data/perf_dataset/swe_gpu_bench_perf.jsonl``.
* ``generate export``: project dataset rows onto the published Hugging Face schema
  (``Froze/SWE-GPU-Bench``), one ``<split>.jsonl`` per call; rows with ``dataset-check`` errors are
  dropped unless ``--keep-invalid``.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
from pathlib import Path
from typing import Any

from swe_gpu_bench.core.config import add_profile_arguments, profile_from_args
from swe_gpu_bench.core.diff import new_names_used_by_tests
from swe_gpu_bench.core.io import (
    find_pr_record,
    load_json,
    load_jsonl,
    load_text,
    load_verified_records,
    patch_path,
    pr_metadata_path,
    query_output_path,
    repo_dir,
    repo_to_slug,
    write_jsonl,
)
from swe_gpu_bench.core.selection import parse_instance_ids
from swe_gpu_bench.generate.check import (
    add_ignore_argument,
    apply_verify_status,
    check_dataset,
    clean_rows,
    combined_patch,
    load_verify_results,
    resplit,
    write_check_report,
)
from swe_gpu_bench.generate.query import DEFAULT_PROMPT, build_prompt, generate_problem_statement
from swe_gpu_bench.generate.split_patch import extract_test_signals, split_patch_text

logger = logging.getLogger(__name__)

PERF_DATASET_NAME = "swe_gpu_bench_perf"
PERF_PROBLEM_STATEMENT = "Recover the performance-oriented change and preserve correctness."
TASK_TYPE_BY_CATEGORY = {"bugfix": "bug", "perf": "perf"}
DEFAULT_TASK_TYPE = "feature"
EXPORT_FIELDS = (
    "instance_id",
    "repo",
    "pr_number",
    "base_commit",
    "head_commit",
    "base_ref",
    "head_ref",
    "created_at",
    "merged_at",
    "problem_statement",
    "task_type",
    "patch",
    "test_patch",
    "setup_commands",
    "test_commands",
    "verification_status",
    "docker_image",
)
PERF_EXPORT_FIELDS = ("perf_setup_commands", "perf_commands")


def instance_id_for(repo: str, pr_number: int) -> str:
    return f"{repo_to_slug(repo)}__{pr_number}"


def task_type_for(scored_row: dict[str, Any] | None) -> str | None:
    if not scored_row:
        return None
    primary = ((scored_row.get("gpt_result") or {}).get("pr_category") or {}).get("primary")
    return TASK_TYPE_BY_CATEGORY.get(primary, DEFAULT_TASK_TYPE) if primary else None


def _pr_fields(pr_meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "pr_title": pr_meta.get("title"),
        "pr_body": pr_meta.get("body"),
        "html_url": pr_meta.get("html_url"),
        "base_ref": pr_meta.get("base_ref"),
        "head_ref": pr_meta.get("head_ref"),
        "created_at": pr_meta.get("created_at"),
        "merged_at": pr_meta.get("merged_at"),
        "commit_messages": pr_meta.get("commit_messages", []),
    }


# ---------------------------------------------------------------------------
# dataset-build
# ---------------------------------------------------------------------------

def build_instance(
    verified_rec: dict[str, Any],
    pr_meta: dict[str, Any],
    query: str,
    patch: str,
    test_patch: str,
    task_type: str | None = None,
) -> dict[str, Any]:
    repo, pr_number = verified_rec["repo"], verified_rec["pr_number"]
    return {
        "instance_id": instance_id_for(repo, pr_number),
        "dataset": verified_rec.get("dataset"),
        "repo": repo,
        "pr_number": pr_number,
        "base_commit": pr_meta.get("base_sha"),
        "head_commit": verified_rec.get("head_sha"),
        **_pr_fields(pr_meta),
        "problem_statement": query,
        "task_type": task_type,
        "patch": patch,
        "test_patch": test_patch,
        "setup_commands": verified_rec.get("setup_commands", []),
        "test_commands": verified_rec.get("test_commands", []),
        "verification_status": verified_rec.get("verification_status"),
    }


def build_dataset(
    verified_dir: Path,
    queries_dir: Path,
    prs_dir: Path,
    split_patch_dir: Path,
    scored_dir: Path,
) -> tuple[list[dict[str, Any]], int]:
    instances: list[dict[str, Any]] = []
    skipped = 0
    for record in load_verified_records(verified_dir):
        repo, pr_number = record["repo"], record["pr_number"]
        query_path = query_output_path(queries_dir, repo, pr_number)
        pr_path = pr_metadata_path(prs_dir, repo, pr_number)
        split_dir = repo_dir(split_patch_dir, repo)
        patch_file, test_patch_file = split_dir / f"{pr_number}.diff", split_dir / f"{pr_number}.test.diff"
        missing = [path for path in (query_path, pr_path, patch_file, test_patch_file) if not path.exists()]
        query = load_json(query_path).get("query") if not missing else None
        if missing or query is None:
            logger.warning("Skipping %s#%s: missing %s", repo, pr_number, missing[0] if missing else "query text")
            skipped += 1
            continue
        scored = find_pr_record(repo_dir(scored_dir, repo) / "fused_scored.jsonl", pr_number)
        instances.append(
            build_instance(
                record,
                load_json(pr_path),
                query,
                load_text(patch_file),
                load_text(test_patch_file),
                task_type=task_type_for(scored),
            )
        )
    return instances, skipped


def add_build_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--verified-dir", default="data/verified")
    parser.add_argument("--queries-dir", default="data/queries")
    parser.add_argument("--prs-dir", default="data/prs")
    parser.add_argument("--split-patch-dir", default="data/split_patch")
    parser.add_argument("--scored-dir", default="data/fused_scored", help="Source of task_type (pr_category)")
    parser.add_argument("--output-dir", default="data/dataset")


def run_build(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    output_dir = Path(args.output_dir)
    instances, skipped = build_dataset(
        Path(args.verified_dir), Path(args.queries_dir), Path(args.prs_dir), Path(args.split_patch_dir), Path(args.scored_dir)
    )
    dataset_path = output_dir / "swe_gpu_bench.jsonl"
    write_jsonl(dataset_path, instances)
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for instance in instances:
        by_dataset.setdefault(instance.get("dataset") or "unknown", []).append(instance)
    for name, rows in sorted(by_dataset.items()):
        write_jsonl(output_dir / f"{name}.jsonl", rows)
        logger.info("  %s: %d instances", name, len(rows))
    logger.info("Dataset built: %d instances, %d skipped -> %s", len(instances), skipped, output_dir)
    write_check_report(str(dataset_path), instances, check_dataset(instances), dataset_path.with_suffix(".check.json"))
    return 0


# ---------------------------------------------------------------------------
# dataset-finalize
# ---------------------------------------------------------------------------

def finalize_entry(entry: dict[str, Any], args: argparse.Namespace, prompt_template: str | None, profile) -> dict[str, Any]:
    split = resplit(entry)
    problem_statement = entry.get("problem_statement") or ""
    if (args.regenerate_queries or not problem_statement) and prompt_template is not None:
        pr_obj = {
            "repo": entry.get("repo", ""),
            "title": entry.get("pr_title", ""),
            "body": entry.get("pr_body", ""),
            "head_sha": entry.get("head_commit", ""),
            "commit_messages": entry.get("commit_messages", []),
        }
        interface = new_names_used_by_tests(split.patch_text, split.test_patch_text)
        try:
            problem_statement = generate_problem_statement(
                build_prompt(prompt_template, pr_obj, combined_patch(entry), args.max_patch_chars, interface),
                profile=profile,
                env_file=Path(args.env_file) if args.env_file else None,
                attempts=args.max_retries,
                retry_delay_seconds=args.retry_delay_seconds,
            )
        except Exception as exc:
            logger.warning("%s: problem statement generation failed (%s); keeping existing", entry.get("instance_id"), exc)

    return {
        **entry,
        "problem_statement": problem_statement,
        "patch": split.patch_text,
        "test_patch": split.test_patch_text,
        "setup_commands": list(entry.get("setup_commands") or []),
        "test_commands": list(entry.get("test_commands") or []),
    }


def add_finalize_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", default="data/dataset/reproduced_dataset.jsonl")
    parser.add_argument("--output-dir", default="data/dataset")
    parser.add_argument("--output-file", default="swe_gpu_bench.jsonl")
    parser.add_argument("--regenerate-queries", action="store_true", help="Regenerate every problem_statement via LLM")
    parser.add_argument("--prompt-file", default=str(DEFAULT_PROMPT))
    parser.add_argument("--max-patch-chars", type=int, default=50000)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay-seconds", type=float, default=5.0)
    add_profile_arguments(parser, kind="text")


def run_finalize(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        return 1
    prompt_path = Path(args.prompt_file)
    if args.regenerate_queries and not prompt_path.exists():
        logger.error("Prompt file not found: %s", prompt_path)
        return 1
    prompt_template = load_text(prompt_path) if prompt_path.exists() else None
    profile = profile_from_args(args, kind="text")

    entries = load_jsonl(input_path)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(args.workers, len(entries) or 1))) as executor:
        finalized = list(executor.map(lambda entry: finalize_entry(entry, args, prompt_template, profile), entries))

    output_path = Path(args.output_dir) / args.output_file
    write_jsonl(output_path, finalized)
    with_test_patch = sum(1 for record in finalized if record.get("test_patch"))
    logger.info("Wrote %d instances to %s (%d with non-empty test_patch)", len(finalized), output_path, with_test_patch)
    return 0


# ---------------------------------------------------------------------------
# perf-dataset-build
# ---------------------------------------------------------------------------

def load_ready_perf_results(case_root: Path) -> list[dict[str, Any]]:
    """``status == "ready"`` results from ``<case_root>/<case>/logs/result.json``."""
    results = []
    for path in sorted(case_root.glob("*/logs/result.json")):
        result = load_json(path)
        if result.get("status") == "ready":
            results.append(result)
    return results


def build_perf_instance(perf_result: dict[str, Any], pr_meta: dict[str, Any], patch: str, test_patch: str) -> dict[str, Any]:
    repo, pr_number = perf_result["repo"], perf_result["pr_number"]
    return {
        "instance_id": instance_id_for(repo, pr_number),
        "dataset": PERF_DATASET_NAME,
        "repo": repo,
        "pr_number": pr_number,
        "base_commit": perf_result.get("base_commit"),
        "head_commit": perf_result.get("head_commit"),
        **_pr_fields(pr_meta),
        "problem_statement": PERF_PROBLEM_STATEMENT,
        "task_type": "perf",
        "patch": patch,
        "test_patch": test_patch,
        "setup_commands": perf_result.get("setup_commands", []),
        "test_commands": perf_result.get("test_commands", []),
        "perf_setup_commands": perf_result.get("perf_setup_commands", []),
        "perf_commands": perf_result.get("perf_commands", []),
        "perf_context": perf_result.get("perf_context", {}),
    }


def build_perf_dataset(case_root: Path, prs_dir: Path, patches_dir: Path) -> list[dict[str, Any]]:
    instances = []
    for result in load_ready_perf_results(case_root):
        repo, pr_number = result["repo"], result["pr_number"]
        pr_path, diff_path = pr_metadata_path(prs_dir, repo, pr_number), patch_path(patches_dir, repo, pr_number)
        if not (pr_path.exists() and diff_path.exists()):
            logger.warning("Skipping %s#%s: missing PR metadata or patch", repo, pr_number)
            continue
        split = split_patch_text(load_text(diff_path), extract_test_signals(result.get("test_commands") or []))
        instances.append(build_perf_instance(result, load_json(pr_path), split.patch_text, split.test_patch_text))
    return instances


def add_perf_build_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--case-root", default="results/perf_repro", help="`generate perf-repro` case root")
    parser.add_argument("--prs-dir", default="data/prs")
    parser.add_argument("--patches-dir", default="data/patches")
    parser.add_argument("--output-dir", default="data/perf_dataset")


def run_perf_build(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    instances = build_perf_dataset(Path(args.case_root), Path(args.prs_dir), Path(args.patches_dir))
    output_path = Path(args.output_dir) / f"{PERF_DATASET_NAME}.jsonl"
    write_jsonl(output_path, instances)
    logger.info("Perf dataset built: %d instances -> %s", len(instances), output_path)
    return 0


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

def export_record(record: dict[str, Any], docker_image: str | None) -> dict[str, Any]:
    row = {field: record.get(field) for field in EXPORT_FIELDS}
    row["setup_commands"] = list(row["setup_commands"] or [])
    row["test_commands"] = list(row["test_commands"] or [])
    row["docker_image"] = row["docker_image"] or docker_image
    for field in PERF_EXPORT_FIELDS:
        if record.get(field):
            row[field] = list(record[field])
    return row


def add_export_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", default="data/dataset/swe_gpu_bench.jsonl")
    parser.add_argument("--output-dir", default="data/export")
    parser.add_argument("--split", default="test", help="Output file name (<split>.jsonl)")
    parser.add_argument("--docker-image", default=None, help="docker_image for rows that do not carry one")
    parser.add_argument("--instance-id-file", default=None, help="Export only these instance ids (one per line)")
    parser.add_argument(
        "--verify-root", default=None, help="`generate verify --dataset` results; their status replaces verification_status"
    )
    parser.add_argument("--keep-invalid", action="store_true", help="Also export rows with dataset-check errors")
    add_ignore_argument(parser)


def run_export(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    records = load_jsonl(Path(args.dataset))
    if args.instance_id_file:
        wanted = set(parse_instance_ids(instance_id_file=Path(args.instance_id_file)))
        records = [record for record in records if record.get("instance_id") in wanted]
    output_dir = Path(args.output_dir)
    verify_results = load_verify_results(Path(args.verify_root)) if args.verify_root else None
    reports = check_dataset(records, verify_results=verify_results, ignore=args.ignore)
    report_path = output_dir / f"{args.split}.check.json"
    write_check_report(args.dataset, records, reports, report_path)
    if not args.keep_invalid:
        records = clean_rows(records, reports)
    rows = [export_record(record, args.docker_image) for record in apply_verify_status(records, verify_results)]
    output_path = output_dir / f"{args.split}.jsonl"
    write_jsonl(output_path, rows)
    missing_image = sum(1 for row in rows if not row["docker_image"])
    dropped = len(reports) - len(rows)
    print(
        f"Exported {len(rows)} rows -> {output_path} ({missing_image} without docker_image, "
        f"{dropped} dropped for dataset-check errors; see {report_path})"
    )
    return 0

"""``collect fused-review`` / ``collect perf-review``: LLM rubric scoring of candidate PRs.

Both commands send the rubric prompt plus PR metadata and diff to a text model and
write, per repository::

    <output-dir>/<owner__repo>/<stem>_scored.jsonl    every scored row
    <output-dir>/<owner__repo>/<stem>_accepted.jsonl  rows whose verdict is "accept"
    <output-dir>/<owner__repo>/<stem>_scored.parquet  checkpointed table of all rows
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from swe_gpu_bench.core.config import ModelProfile, add_profile_arguments, profile_from_args
from swe_gpu_bench.core.io import append_jsonl, load_json, load_jsonl, load_text, patch_path, repo_dir
from swe_gpu_bench.core.llm import complete, parse_json_response, resolve_model, trim_text, write_parquet

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReviewKind:
    stem: str
    verdict_key: str
    default_prompt: str
    default_output_dir: str


FUSED = ReviewKind("fused", "benchmark_fit", "prompts/fuse_filter.md", "data/fused_scored")
PERF = ReviewKind("perf", "benchmarkability", "prompts/perf_review.md", "data/perf_reviewed")

PROMPT_FIELDS = (
    "repo", "number", "title", "body", "state", "draft", "html_url",
    "base_ref", "head_ref", "commit_messages", "resolves_issues", "has_test_patch",
)


def output_paths(output_dir: Path, repo_name: str, kind: ReviewKind) -> tuple[Path, Path, Path]:
    base = repo_dir(output_dir, repo_name)
    return (
        base / f"{kind.stem}_scored.jsonl",
        base / f"{kind.stem}_accepted.jsonl",
        base / f"{kind.stem}_scored.parquet",
    )


def build_prompt(rubric: str, pr_obj: dict[str, Any], patch_text: str, max_patch_chars: int) -> str:
    payload = {key: pr_obj.get(key) for key in PROMPT_FIELDS}
    return (
        f"{rubric.strip()}\n\nPR to evaluate:\n"
        "Evaluate this GitHub PR using the provided rubric.\n\n"
        f"PR metadata and text:\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        f"Patch / diff:\n{trim_text(patch_text, max_patch_chars)}"
    )


def _base_row(pr_obj: dict[str, Any], model: str) -> dict[str, Any]:
    return {
        "repo": pr_obj["repo"],
        "number": pr_obj["number"],
        "title": pr_obj.get("title"),
        "html_url": pr_obj.get("html_url"),
        "resolves_issues": pr_obj.get("resolves_issues"),
        "has_test_patch": pr_obj.get("has_test_patch"),
        "pr_record": pr_obj,
        "gpt_model": model,
    }


def judge_record(
    pr_obj: dict[str, Any],
    *,
    rubric: str,
    profile: ModelProfile | None,
    patches_dir: Path,
    max_patch_chars: int,
    env_file: Path | None = None,
) -> dict[str, Any]:
    patch_file = patch_path(patches_dir, pr_obj["repo"], pr_obj["number"])
    if not patch_file.exists():
        raise FileNotFoundError(f"Missing patch file: {patch_file}")
    patch_text = load_text(patch_file)
    response_text = complete(build_prompt(rubric, pr_obj, patch_text, max_patch_chars), profile=profile, env_file=env_file)
    parsed = parse_json_response(response_text)
    if parsed.get("parse_error"):
        raise ValueError(f"Failed to parse model response: {parsed['parse_error']}")
    return {
        **_base_row(pr_obj, resolve_model(profile)),
        "patch_text": patch_text,
        "gpt_result": parsed,
        "raw_response": response_text,
    }


def review_one(
    pr_obj: dict[str, Any],
    *,
    rubric: str,
    profile: ModelProfile | None,
    patches_dir: Path,
    max_patch_chars: int,
    max_retries: int,
    retry_delay_seconds: float,
    env_file: Path | None = None,
) -> dict[str, Any]:
    for attempt in range(1, max_retries + 1):
        try:
            result = judge_record(
                pr_obj,
                rubric=rubric,
                profile=profile,
                patches_dir=patches_dir,
                max_patch_chars=max_patch_chars,
                env_file=env_file,
            )
            if attempt > 1:
                result["retry_attempts"] = attempt - 1
            return result
        except Exception:
            if attempt >= max_retries:
                return {
                    **_base_row(pr_obj, resolve_model(profile)),
                    "patch_text": None,
                    "gpt_result": None,
                    "raw_response": None,
                    "retry_attempts": attempt - 1,
                    "error": traceback.format_exc(),
                }
            time.sleep(retry_delay_seconds)
    raise AssertionError("unreachable")


def is_accepted(row: dict[str, Any], kind: ReviewKind) -> bool:
    gpt_result = row.get("gpt_result")
    return (
        isinstance(gpt_result, dict)
        and not gpt_result.get("parse_error")
        and not row.get("error")
        and gpt_result.get(kind.verdict_key) == "accept"
    )


def load_fused_candidates(input_path: Path) -> list[dict[str, Any]]:
    """Accept ``collect filter`` output (JSON array with ``metadata_path``) or a JSONL of PR records."""
    text = input_path.read_text(encoding="utf-8")
    if not text.lstrip().startswith("["):
        return load_jsonl(input_path)
    return [load_json(Path(item["metadata_path"])) if "metadata_path" in item else item for item in json.loads(text)]


def is_perf_row(row: dict[str, Any]) -> bool:
    category = (row.get("gpt_result") or {}).get("pr_category") or {}
    return category.get("primary") == "perf" or "perf" in (category.get("secondary") or [])


def load_perf_candidates(fused_dir: Path) -> list[dict[str, Any]]:
    """PR records from fused-review output whose category is (or includes) ``perf``."""
    return [
        row.get("pr_record") or row
        for scored_path in sorted(fused_dir.glob(f"*/{FUSED.stem}_scored.jsonl"))
        for row in load_jsonl(scored_path)
        if is_perf_row(row)
    ]


def run_review(records: list[dict[str, Any]], args: argparse.Namespace, kind: ReviewKind) -> int:
    if args.limit is not None:
        records = records[: args.limit]
    rubric = load_text(Path(args.prompt_file))
    output_dir = Path(args.output_dir)
    profile = profile_from_args(args, kind="text")
    env_file = Path(args.env_file) if args.env_file else None

    rows_by_repo: dict[str, list[dict[str, Any]]] = {}
    if args.resume:
        for scored_path in output_dir.glob(f"*/{kind.stem}_scored.jsonl"):
            for row in load_jsonl(scored_path):
                rows_by_repo.setdefault(row["repo"], []).append({**row, "accepted": is_accepted(row, kind)})
    seen = {(row["repo"], row["number"]) for rows in rows_by_repo.values() for row in rows}
    pending = [pr for pr in records if (pr["repo"], pr["number"]) not in seen]
    if not pending:
        logger.info("No pending PRs to score.")
        return 0
    logger.info("Scoring %d PRs (%d already scored)", len(pending), len(seen))

    def score(pr_obj: dict[str, Any]) -> dict[str, Any]:
        result = review_one(
            pr_obj,
            rubric=rubric,
            profile=profile,
            patches_dir=Path(args.patches_dir),
            max_patch_chars=args.max_patch_chars,
            max_retries=args.max_retries,
            retry_delay_seconds=args.retry_delay_seconds,
            env_file=env_file,
        )
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)
        return result

    scored = accepted = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(args.workers, len(pending)))) as executor:
        for result in executor.map(score, pending):
            repo = result["repo"]
            scored_path, accepted_path, parquet_path = output_paths(output_dir, repo, kind)
            append_jsonl(scored_path, result)
            row = {**result, "accepted": is_accepted(result, kind)}
            rows_by_repo.setdefault(repo, []).append(row)
            scored += 1
            if row["accepted"]:
                append_jsonl(accepted_path, result)
                accepted += 1
            if args.save_every > 0 and scored % args.save_every == 0:
                write_parquet(parquet_path, rows_by_repo[repo])
            logger.info(
                "Scored %s#%s %s=%s error=%s",
                repo, result["number"], kind.verdict_key,
                (result.get("gpt_result") or {}).get(kind.verdict_key), bool(result.get("error")),
            )

    for repo, rows in rows_by_repo.items():
        write_parquet(output_paths(output_dir, repo, kind)[2], rows)
    logger.info("Completed %s review: scored=%s accepted=%s output_dir=%s", kind.stem, scored, accepted, output_dir)
    return 0


def _add_common_arguments(parser: argparse.ArgumentParser, kind: ReviewKind) -> None:
    parser.add_argument("--patches-dir", default="data/patches")
    parser.add_argument("--prompt-file", default=kind.default_prompt)
    parser.add_argument("--output-dir", default=kind.default_output_dir)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-patch-chars", type=int, default=50000)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay-seconds", type=float, default=5.0)
    parser.add_argument("--resume", action="store_true", help="Skip PRs already present in --output-dir")
    parser.add_argument("--save-every", type=int, default=20, help="Rewrite the parquet checkpoint every N rows")
    add_profile_arguments(parser, kind="text")


def add_fused_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", default="data/filtered_prs_with_tests.json", help="`collect filter` output or PR JSONL")
    _add_common_arguments(parser, FUSED)


def run_fused(args: argparse.Namespace) -> int:
    return run_review(load_fused_candidates(Path(args.input)), args, FUSED)


def add_perf_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-dir", default=FUSED.default_output_dir, help="fused-review output to pick perf PRs from")
    _add_common_arguments(parser, PERF)


def run_perf(args: argparse.Namespace) -> int:
    return run_review(load_perf_candidates(Path(args.input_dir)), args, PERF)

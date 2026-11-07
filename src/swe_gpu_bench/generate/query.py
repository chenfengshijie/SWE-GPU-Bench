"""``generate query``: write an agent-facing problem statement for every verified PR.

Reads ``data/verified/**/<pr>.json`` plus ``data/prs`` / ``data/patches`` and writes
``data/queries/<owner__repo>/<pr>.json`` (and an append-only ``query_generated.jsonl``).
Names the PR introduces and its tests call are passed to the prompt as ``{interface}``: the hidden
tests fail unless the agent uses exactly those names, so the statement has to mention them.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
import traceback
from pathlib import Path
from typing import Any, Sequence

from swe_gpu_bench.core.config import ModelProfile, add_profile_arguments, profile_from_args
from swe_gpu_bench.core.diff import new_names_used_by_tests
from swe_gpu_bench.core.io import (
    append_jsonl,
    load_json,
    load_text,
    load_verified_records,
    patch_path,
    pr_metadata_path,
    query_output_path,
    write_json,
)
from swe_gpu_bench.core.llm import complete, resolve_model, trim_text
from swe_gpu_bench.generate.split_patch import extract_test_signals, split_patch_text

logger = logging.getLogger(__name__)
DEFAULT_PROMPT = Path("prompts/query_generate.md")


def interface_names(diff_text: str, test_commands: Sequence[str]) -> list[str]:
    split = split_patch_text(diff_text, extract_test_signals(list(test_commands)))
    return new_names_used_by_tests(split.patch_text, split.test_patch_text)


def build_prompt(
    prompt_template: str,
    pr_obj: dict[str, Any],
    patch_text: str,
    max_patch_chars: int,
    interface: Sequence[str] = (),
) -> str:
    context_parts = []
    if pr_obj.get("head_sha"):
        context_parts.append(f"head_sha: {pr_obj['head_sha']}")
    commit_messages = pr_obj.get("commit_messages") or []
    if commit_messages:
        context_parts.append("commit messages:\n" + "\n".join(f"- {message}" for message in commit_messages))
    return (
        prompt_template.replace("{repo}", pr_obj.get("repo") or "")
        .replace("{title}", pr_obj.get("title") or "")
        .replace("{body}", pr_obj.get("body") or "")
        .replace("{context}", "\n".join(context_parts) or "(none)")
        .replace("{interface}", "\n".join(f"- {name}" for name in interface) or "(none)")
        .replace("{patch}", trim_text(patch_text, max_patch_chars))
    )


def generate_problem_statement(
    prompt: str,
    *,
    profile: ModelProfile | None,
    env_file: Path | None,
    attempts: int,
    retry_delay_seconds: float,
) -> str:
    text = complete(prompt, profile=profile, env_file=env_file, attempts=attempts, retry_delay_seconds=retry_delay_seconds)
    if not text:
        raise ValueError("LLM returned an empty response")
    return text


def generate_one(record: dict[str, Any], args: argparse.Namespace, prompt_template: str, profile: ModelProfile) -> dict[str, Any]:
    repo, pr_number = record["repo"], record["pr_number"]
    result: dict[str, Any] = {
        "dataset": record.get("dataset"),
        "repo": repo,
        "pr_number": pr_number,
        "head_sha": record.get("head_sha"),
        "verification_status": record.get("verification_status"),
        "query": None,
        "interface": [],
        "model": resolve_model(profile),
        "error": None,
    }
    try:
        pr_file = pr_metadata_path(Path(args.prs_dir), repo, pr_number)
        patch_file = patch_path(Path(args.patches_dir), repo, pr_number)
        for path in (pr_file, patch_file):
            if not path.exists():
                raise FileNotFoundError(f"Missing input: {path}")
        diff_text = load_text(patch_file)
        result["interface"] = interface_names(diff_text, record.get("test_commands") or [])
        prompt = build_prompt(prompt_template, load_json(pr_file), diff_text, args.max_patch_chars, result["interface"])
        result["query"] = generate_problem_statement(
            prompt,
            profile=profile,
            env_file=Path(args.env_file) if args.env_file else None,
            attempts=args.max_retries,
            retry_delay_seconds=args.retry_delay_seconds,
        )
    except Exception:
        result["error"] = traceback.format_exc()
    return result


def load_seen(output_dir: Path) -> set[tuple[str, int]]:
    seen: set[tuple[str, int]] = set()
    for path in output_dir.glob("*/*.json"):
        try:
            record = load_json(path)
        except ValueError:
            continue
        if record.get("query") is not None:
            seen.add((record["repo"], record["pr_number"]))
    return seen


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--verified-dir", default="data/verified")
    parser.add_argument("--prs-dir", default="data/prs")
    parser.add_argument("--patches-dir", default="data/patches")
    parser.add_argument("--prompt-file", default=str(DEFAULT_PROMPT))
    parser.add_argument("--output-dir", default="data/queries")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-patch-chars", type=int, default=50000)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay-seconds", type=float, default=5.0)
    parser.add_argument("--resume", action="store_true", help="Skip PRs that already have a non-null query")
    add_profile_arguments(parser, kind="text")


def run(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    output_dir = Path(args.output_dir)
    records = load_verified_records(Path(args.verified_dir))
    logger.info("Found %d verified PRs", len(records))

    seen = load_seen(output_dir) if args.resume and output_dir.exists() else set()
    pending = [record for record in records if (record["repo"], record["pr_number"]) not in seen]
    if args.limit is not None:
        pending = pending[: args.limit]
    if not pending:
        logger.info("No pending PRs to generate queries for.")
        return 0

    profile = profile_from_args(args, kind="text")
    prompt_template = load_text(Path(args.prompt_file))
    errors = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(args.workers, len(pending)))) as executor:
        futures = [executor.submit(generate_one, record, args, prompt_template, profile) for record in pending]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            out_path = query_output_path(output_dir, result["repo"], result["pr_number"])
            write_json(out_path, result)
            append_jsonl(output_dir / "query_generated.jsonl", result)
            if result["error"]:
                errors += 1
                logger.warning("Error %s#%s: %s", result["repo"], result["pr_number"], result["error"].splitlines()[-1])
            else:
                logger.info("Generated query for %s#%s -> %s", result["repo"], result["pr_number"], out_path)
    logger.info("Done: generated=%d errors=%d output_dir=%s", len(pending), errors, output_dir)
    return 0

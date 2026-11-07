"""``collect patches``: download the unified diff for every collected PR."""
from __future__ import annotations

import argparse
import concurrent.futures
import logging
from pathlib import Path

from swe_gpu_bench.core.github import GitHubAPIError, GitHubClient, parse_tokens
from swe_gpu_bench.core.io import load_json, write_text

logger = logging.getLogger(__name__)

DIFF_ACCEPT = "application/vnd.github.v3.diff"


def iter_pr_metadata_files(input_dir: Path):
    direct_files = sorted(input_dir.glob("*.json"))
    if direct_files:
        yield from direct_files
        return
    yield from sorted(input_dir.glob("*/*.json"))


def patch_output_path(output_dir: Path, metadata_path: Path) -> Path:
    return output_dir / metadata_path.parent.name / f"{metadata_path.stem}.diff"


def metadata_to_diff_api_url(payload: dict) -> str | None:
    repo = payload.get("repo")
    number = payload.get("number")
    if not repo or number is None:
        return None
    return f"https://api.github.com/repos/{repo}/pulls/{number}"


def download_patches(
    input_dir: Path,
    output_dir: Path,
    *,
    client: GitHubClient | None = None,
    overwrite: bool = False,
    jobs: int = 8,
) -> tuple[int, int, int]:
    """Download missing diffs; return ``(downloaded, skipped, failed)``."""
    client = client or GitHubClient(retry_delay=5)
    pending: list[tuple[Path, str, str, Path]] = []
    skipped = 0
    for metadata_path in iter_pr_metadata_files(input_dir):
        payload = load_json(metadata_path)
        api_url = metadata_to_diff_api_url(payload)
        source_url = api_url or payload.get("diff_url")
        if not source_url:
            logger.warning("Skipping %s: missing diff_url", metadata_path)
            skipped += 1
            continue
        destination = patch_output_path(output_dir, metadata_path)
        if destination.exists() and not overwrite:
            skipped += 1
            continue
        pending.append((metadata_path, source_url, DIFF_ACCEPT if api_url else "text/plain", destination))

    def download_one(task: tuple[Path, str, str, Path]) -> str | None:
        _, source_url, accept, destination = task
        try:
            write_text(destination, client.get_text(source_url, accept=accept))
            return None
        except GitHubAPIError as exc:
            return f"{source_url}: {exc}"

    downloaded = failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, jobs)) as executor:
        for task, error in zip(pending, executor.map(download_one, pending)):
            if error is None:
                downloaded += 1
            else:
                failed += 1
                logger.warning("Failed to download %s from %s", task[0], error)

    logger.info(
        "Patch download completed: downloaded=%s skipped=%s failed=%s output_dir=%s",
        downloaded, skipped, failed, output_dir,
    )
    return downloaded, skipped, failed


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-dir", default="data/prs")
    parser.add_argument("--output-dir", default="data/patches")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--token", default=None, help="GitHub token (default: $GITHUB_TOKEN)")
    parser.add_argument("--tokens", default=None, help="Comma-separated tokens to rotate through")
    parser.add_argument("--jobs", type=int, default=8)


def run(args: argparse.Namespace) -> int:
    client = GitHubClient(tokens=parse_tokens(args.token, args.tokens), retry_delay=5)
    download_patches(Path(args.input_dir), Path(args.output_dir), client=client, overwrite=args.overwrite, jobs=args.jobs)
    return 0

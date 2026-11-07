"""``collect fetch``: download merged pull requests and their commit messages.

Closed-but-unmerged PRs are skipped unless ``--include-unmerged``: their diff never landed, so it is
not a trustworthy reference solution.
"""
from __future__ import annotations

import argparse
import logging
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from swe_gpu_bench.core.github import GitHubAPIError, GitHubClient, parse_tokens
from swe_gpu_bench.core.io import repo_dir, write_json

logger = logging.getLogger(__name__)

DEFAULT_REPOS_CONFIG = Path("configs/repos.toml")


def iter_paginated(client: GitHubClient, path: str, params: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    page = 1
    while True:
        items = client.get_json(path, params={**(params or {}), "per_page": 100, "page": page})
        if not items:
            return
        yield from items
        page += 1


def iter_closed_pull_requests(client: GitHubClient, owner: str, repo: str, cutoff_date: str | None = None):
    """Yield closed PRs newest first, stopping at the first one created before ``cutoff_date`` (YYYYMMDD)."""
    cutoff = datetime.strptime(cutoff_date, "%Y%m%d") if cutoff_date else None
    for pull in iter_paginated(client, f"/repos/{owner}/{repo}/pulls", {"state": "closed"}):
        if cutoff is not None and datetime.strptime(pull["created_at"], "%Y-%m-%dT%H:%M:%SZ") < cutoff:
            return
        yield pull


def collect_commit_messages(client: GitHubClient, owner: str, repo: str, pull_number: int) -> list[str]:
    commits = iter_paginated(client, f"/repos/{owner}/{repo}/pulls/{pull_number}/commits")
    return [message for commit in commits if (message := (commit.get("commit") or {}).get("message")) is not None]


def build_pull_payload(client: GitHubClient, owner: str, repo: str, pull: dict[str, Any]) -> dict[str, Any]:
    base = pull.get("base") or {}
    head = pull.get("head") or {}
    return {
        "repo": f"{owner}/{repo}",
        "number": pull["number"],
        "title": pull.get("title"),
        "body": pull.get("body"),
        "state": pull.get("state"),
        "draft": pull.get("draft"),
        "created_at": pull.get("created_at"),
        "updated_at": pull.get("updated_at"),
        "closed_at": pull.get("closed_at"),
        "merged_at": pull.get("merged_at"),
        "html_url": pull.get("html_url"),
        "user_login": (pull.get("user") or {}).get("login"),
        "default_branch": (base.get("repo") or {}).get("default_branch"),
        "base_ref": base.get("ref"),
        "base_sha": base.get("sha"),
        "head_ref": head.get("ref"),
        "head_sha": head.get("sha"),
        "commits_count": pull.get("commits"),
        "comments_count": pull.get("comments"),
        "review_comments_count": pull.get("review_comments"),
        "additions": pull.get("additions"),
        "deletions": pull.get("deletions"),
        "changed_files": pull.get("changed_files"),
        "diff_url": pull.get("diff_url"),
        "patch_url": pull.get("patch_url"),
        "commit_messages": collect_commit_messages(client, owner, repo, pull["number"]),
    }


def collect_repo_pull_requests(
    client: GitHubClient,
    repo_name: str,
    output_dir: Path,
    max_pulls: int | None = None,
    cutoff_date: str | None = None,
    progress_every: int = 25,
    include_unmerged: bool = False,
) -> int:
    """Write ``<output_dir>/<owner__repo>/<number>.json`` for each new merged PR; return how many were written."""
    owner, repo = repo_name.split("/", 1)
    destination = repo_dir(output_dir, repo_name)
    destination.mkdir(parents=True, exist_ok=True)
    logger.info("Collecting %s into %s (max_pulls=%s cutoff_date=%s)", repo_name, destination, max_pulls, cutoff_date)

    written = inspected = skipped = unmerged = 0
    for pull in iter_closed_pull_requests(client, owner, repo, cutoff_date=cutoff_date):
        if max_pulls is not None and inspected >= max_pulls:
            break
        inspected += 1
        output_path = destination / f"{pull['number']}.json"
        if not include_unmerged and pull.get("merged_at") is None:
            unmerged += 1
        elif output_path.exists():
            skipped += 1
        else:
            write_json(output_path, build_pull_payload(client, owner, repo, pull))
            written += 1
        if progress_every > 0 and inspected % progress_every == 0:
            logger.info(
                "%s: inspected=%s written=%s skipped=%s unmerged=%s", repo_name, inspected, written, skipped, unmerged
            )

    logger.info(
        "Completed %s: inspected=%s written=%s skipped=%s unmerged=%s", repo_name, inspected, written, skipped, unmerged
    )
    return written


def load_default_repos(config_path: Path = DEFAULT_REPOS_CONFIG) -> list[str]:
    if not config_path.exists():
        return []
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    return list((config.get("collection") or {}).get("default_repos") or [])


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repos", nargs="+", default=None, help=f"owner/repo list (default: {DEFAULT_REPOS_CONFIG})")
    parser.add_argument("--output-dir", default="data/prs")
    parser.add_argument("--token", default=None, help="GitHub token (default: $GITHUB_TOKEN)")
    parser.add_argument("--tokens", default=None, help="Comma-separated tokens to rotate through")
    parser.add_argument("--max-pulls", type=int, default=None)
    parser.add_argument("--cutoff-date", default=None, help="Stop at PRs created before YYYYMMDD")
    parser.add_argument("--include-unmerged", action="store_true", help="Also keep PRs that were closed without merging")


def run(args: argparse.Namespace) -> int:
    tokens = parse_tokens(args.token, args.tokens)
    if not tokens:
        raise SystemExit("Missing GitHub token. Pass --token/--tokens or set GITHUB_TOKEN.")
    repos = args.repos or load_default_repos()
    if not repos:
        raise SystemExit(f"No repositories given and {DEFAULT_REPOS_CONFIG} has no default_repos.")
    client = GitHubClient(tokens=tokens)
    exit_code = 0
    for repo_name in repos:
        try:
            collect_repo_pull_requests(
                client,
                repo_name,
                Path(args.output_dir),
                max_pulls=args.max_pulls,
                cutoff_date=args.cutoff_date,
                include_unmerged=args.include_unmerged,
            )
        except (GitHubAPIError, ValueError) as exc:
            logger.error("Failed to collect %s: %s", repo_name, exc)
            exit_code = 1
    return exit_code

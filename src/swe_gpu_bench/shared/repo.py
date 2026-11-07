from __future__ import annotations


def repo_to_slug(repo_name: str) -> str:
    return repo_name.replace("/", "__")


def slug_to_repo(repo_slug: str) -> str:
    return repo_slug.replace("__", "/", 1)


def split_repo_name(repo_name: str) -> tuple[str, str]:
    return repo_name.split("/", 1)


def repo_tail(repo_name: str) -> str:
    return repo_name.rsplit("/", 1)[-1]

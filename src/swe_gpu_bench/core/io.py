"""JSON/JSONL/text helpers and the on-disk ``owner__repo`` layout."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def repo_to_slug(repo_name: str) -> str:
    return repo_name.replace("/", "__")


def slug_to_repo(repo_slug: str) -> str:
    return repo_slug.replace("__", "/", 1)


def repo_tail(repo_name: str) -> str:
    return repo_name.rsplit("/", 1)[-1]


def repo_dir(root: Path, repo_name: str) -> Path:
    return root / repo_to_slug(repo_name)


def pr_metadata_path(prs_dir: Path, repo_name: str, pr_number: int) -> Path:
    return repo_dir(prs_dir, repo_name) / f"{pr_number}.json"


def patch_path(patches_dir: Path, repo_name: str, pr_number: int) -> Path:
    return repo_dir(patches_dir, repo_name) / f"{pr_number}.diff"


def query_output_path(queries_dir: Path, repo_name: str, pr_number: int) -> Path:
    return repo_dir(queries_dir, repo_name) / f"{pr_number}.json"


def load_verified_records(verified_dir: Path) -> list[dict[str, Any]]:
    """Every ``verification_status == "verified"`` JSON under ``verified_dir``, one per (repo, pr).

    Accepts ``generate verify --case-dir`` output (``<case>/artifacts/result.json``) as well as
    ``<dataset>/<owner__repo>/<pr>.json``; in the latter layout ``dataset`` falls back to the
    ``<dataset>`` directory name when the record does not carry one.
    """
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for path in sorted(verified_dir.rglob("*.json")):
        record = load_json(path)
        if record.get("verification_status") != "verified" or not record.get("repo"):
            continue
        key = (record["repo"], int(record["pr_number"]))
        if key in seen:
            continue
        seen.add(key)
        relative = path.relative_to(verified_dir)
        if not record.get("dataset") and len(relative.parts) > 2 and relative.parts[-2] == repo_to_slug(record["repo"]):
            record["dataset"] = relative.parts[0]
        records.append(record)
    return records


def find_record(path: Path, key: str, value: Any) -> dict[str, Any]:
    """Return the first JSONL row whose ``key`` equals ``value``."""
    for record in load_jsonl(path):
        if record.get(key) == value:
            return record
    raise KeyError(f"{key}={value!r} not found in {path}")


def find_pr_record(path: Path, pr_number: int) -> dict[str, Any] | None:
    if not path.exists():
        return None
    for record in load_jsonl(path):
        if int(record.get("number", -1)) == int(pr_number):
            return record
    return None

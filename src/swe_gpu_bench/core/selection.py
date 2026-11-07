"""Pick dataset records by ``--instance-id`` / ``--instance-id-file`` / ``--all``."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from swe_gpu_bench.core.io import load_jsonl


def add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--instance-id",
        action="append",
        default=[],
        help="Instance id to run; repeatable and comma-separated values are accepted",
    )
    parser.add_argument("--instance-id-file", default=None, help="File with one instance id per line")
    parser.add_argument("--all", action="store_true", help="Run every record in the dataset")


def has_selection(args: argparse.Namespace) -> bool:
    return bool(args.all or args.instance_id or args.instance_id_file)


def parse_instance_ids(values: Iterable[str] = (), instance_id_file: Optional[Path] = None) -> list[str]:
    raw: list[str] = []
    for value in values:
        raw.extend(value.split(","))
    if instance_id_file and instance_id_file.exists():
        raw.extend(instance_id_file.read_text(encoding="utf-8").splitlines())
    ordered: list[str] = []
    for item in (value.strip() for value in raw):
        if item and item not in ordered:
            ordered.append(item)
    return ordered


def select_records(
    dataset_path: Path,
    *,
    instance_ids: Iterable[str] = (),
    instance_id_file: Optional[Path] = None,
    all_instances: bool = False,
    done: Optional[Callable[[dict[str, Any]], bool]] = None,
) -> tuple[list[dict[str, Any]], int]:
    """Return ``(records, skipped)``; records for which ``done(record)`` is true are skipped.

    Each returned record carries ``__line_number__`` (1-based) so records without an
    ``instance_id`` still get a stable output directory.
    """
    records = [dict(record, __line_number__=index) for index, record in enumerate(load_jsonl(dataset_path), start=1)]
    if not all_instances:
        requested = parse_instance_ids(instance_ids, instance_id_file)
        if not requested:
            raise ValueError("One of --instance-id, --instance-id-file, or --all is required")
        by_id = {record["instance_id"]: record for record in records if record.get("instance_id")}
        missing = [instance_id for instance_id in requested if instance_id not in by_id]
        if missing:
            raise KeyError(f"instance_id {missing[0]!r} not found in {dataset_path}")
        records = [by_id[instance_id] for instance_id in requested]

    selected: list[dict[str, Any]] = []
    skipped = 0
    for record in records:
        if done is not None and done(record):
            skipped += 1
            print(f"[skip] {record.get('instance_id')} already has a result (use --force to override)", file=sys.stderr)
            continue
        selected.append(record)
    return selected, skipped

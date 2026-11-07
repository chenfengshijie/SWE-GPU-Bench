"""``evaluate summary``: aggregate ``<eval-root>/[<group>/]<instance_id>/result.json`` files.

Works for every evaluator (agent, oracle-validate, perf-agent, perf-validate). Pass ``--groups``
when the eval root holds one subdirectory per model/profile; perf fields are filled when present.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence

from swe_gpu_bench.core.io import load_json, write_json

ROW_FIELDS = (
    "group",
    "instance_id",
    "status",
    "failure_category",
    "failed_commands",
    "elapsed_seconds",
    "perf_status",
    "llm_verdict",
    "delta_pct",
)
STATUSES = ("passed", "failed", "error", "missing")


def _row(group: str, instance_id: str, payload: Optional[dict[str, Any]]) -> dict[str, Any]:
    row = dict.fromkeys(ROW_FIELDS)
    row.update(group=group, instance_id=instance_id, status="missing")
    if payload is None:
        return row
    judgment = payload.get("llm_judgment") or {}
    row.update(
        instance_id=payload.get("instance_id") or instance_id,
        status=payload.get("status", "unknown"),
        failure_category=payload.get("failure_category"),
        failed_commands=(payload.get("evaluation_stats") or {}).get("failed_commands"),
        elapsed_seconds=(payload.get("agent_stats") or {}).get("elapsed_seconds", payload.get("elapsed_seconds")),
        perf_status=payload.get("perf_status"),
        llm_verdict=judgment.get("verdict"),
        delta_pct=judgment.get("delta_pct"),
    )
    return row


def collect_rows(eval_root: Path, groups: Sequence[str] = ()) -> list[dict[str, Any]]:
    """One row per case directory; directories starting with ``_`` (caches, summaries) are skipped."""
    rows: list[dict[str, Any]] = []
    for group in groups or [""]:
        root = eval_root / group if group else eval_root
        if not root.is_dir():
            continue
        for case_dir in sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith("_")):
            result_path = case_dir / "result.json"
            payload = load_json(result_path) if result_path.exists() else None
            rows.append(_row(group or eval_root.name, case_dir.name, payload))
    return rows


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    for group in dict.fromkeys(row["group"] for row in rows):
        group_rows = [row for row in rows if row["group"] == group]
        counts = Counter(row["status"] for row in group_rows)
        total = len(group_rows)
        totals: dict[str, Any] = {"total": total, **{status: counts.get(status, 0) for status in STATUSES}}
        totals["other"] = total - sum(totals[status] for status in STATUSES)
        totals["pass_rate"] = round(totals["passed"] / total, 4) if total else 0.0
        perf_counts = Counter(row["perf_status"] for row in group_rows if row["perf_status"])
        if perf_counts:
            totals["perf_statuses"] = dict(perf_counts)
            totals["improvement_rate"] = round(perf_counts.get("improved", 0) / total, 4)
        groups[group] = totals
    return {"groups": groups, "rows": rows}


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ROW_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--groups", default="", help="Comma-separated subdirectories (one per model/profile)")
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--csv-output", default=None)


def run(args: argparse.Namespace) -> int:
    groups = [item.strip() for item in args.groups.split(",") if item.strip()]
    rows = collect_rows(Path(args.eval_root), groups)
    summary = build_summary(rows)
    if args.json_output:
        write_json(Path(args.json_output), summary)
    if args.csv_output:
        write_csv(rows, Path(args.csv_output))
    print(json.dumps(summary["groups"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0

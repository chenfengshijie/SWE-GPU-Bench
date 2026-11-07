"""Exact scalar measurements shared by collection and patch evaluation.

Metadata travels with the record; stdout contains only the measured value. Neither
the command nor this parser receives a reference revision's measured baseline.
"""
from __future__ import annotations

import math
import re
from typing import Any

SCALAR_COMMAND_CONTRACT = "portable_unified_v1"
_NUMBER = re.compile(r"[+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
METRIC_FIELDS = ("name", "unit", "workload", "better")


def validate_performance_metric(metric: Any) -> dict[str, str]:
    if not isinstance(metric, dict) or set(metric) != set(METRIC_FIELDS):
        raise ValueError("Scalar performance_metric requires exactly name, unit, workload and better")
    if any(not isinstance(metric[key], str) or not metric[key].strip() or metric[key] != metric[key].strip()
           for key in METRIC_FIELDS):
        raise ValueError("Scalar performance_metric fields must be nonempty strings")
    if metric["better"] not in ("lower", "higher"):
        raise ValueError("Scalar performance_metric better must be lower or higher")
    conventional = {"latency": "lower", "throughput": "higher"}.get(metric["name"])
    if conventional and metric["better"] != conventional:
        raise ValueError("Scalar performance_metric direction contradicts its metric name")
    return dict(metric)


def parse_scalar(text: str) -> float:
    """Accept one finite, positive decimal scalar, never guess a number from logs."""
    value = text.strip()
    if not _NUMBER.fullmatch(value):
        raise ValueError("Performance stdout must contain exactly one positive numeric scalar")
    measured = float(value)
    if not math.isfinite(measured) or measured <= 0:
        raise ValueError("Performance scalar must be finite and positive")
    return measured


def scalar_comparison(base: float, head: float, metric: dict[str, str]) -> dict[str, Any]:
    metadata = validate_performance_metric(metric)
    if any(type(value) not in (float, int) or not math.isfinite(value) or value <= 0 for value in (base, head)):
        raise ValueError("Performance scalar must be finite and positive")
    delta = 100 * ((base - head) / base if metadata["better"] == "lower" else (head - base) / base)
    speedup = base / head if metadata["better"] == "lower" else head / base
    if not math.isfinite(delta) or not math.isfinite(speedup):
        raise ValueError("Non-finite performance comparison")
    return {**metadata, "base_value": base, "head_value": head,
            "improvement_pct": delta, "speedup": speedup}

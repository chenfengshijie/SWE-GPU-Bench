from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BenchmarkRecord:
    instance_id: str
    repo: str
    pr_number: int | None = None
    base_commit: str | None = None
    head_commit: str | None = None
    problem_statement: str = ""
    patch: str | None = None
    test_patch: str | None = None
    setup_commands: list[str] = field(default_factory=list)
    test_commands: list[str] = field(default_factory=list)
    perf_setup_commands: list[str] = field(default_factory=list)
    perf_commands: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelProfile:
    name: str
    kind: str
    framework: str | None = None
    provider: str | None = None
    model: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    extra_env: dict[str, str] = field(default_factory=dict)
    reasoning_effort: str | None = None

    def __post_init__(self):
        if self.reasoning_effort is not None and self.reasoning_effort not in (
            "none", "minimal", "low", "medium", "high", "xhigh"
        ):
            raise ValueError("Invalid reasoning_effort")


@dataclass(frozen=True)
class CommandRunResult:
    phase: str
    command: str
    exit_code: int
    elapsed_seconds: float
    stdout_log: str
    stderr_log: str

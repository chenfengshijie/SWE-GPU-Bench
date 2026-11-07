"""Storage contracts for immutable sources/releases and resumable pipeline runs."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import fcntl
import json
import os
from pathlib import Path
import re
import secrets
import tempfile
import time
import tomllib
from typing import Any, Iterator

STAGES = ("collect", "review", "reproduce", "perf-reproduce", "build", "evaluate")
IDENTIFIER = re.compile(r"[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}\Z")
COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
VERSION = re.compile(r"v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\Z")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + secrets.token_hex(4)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_reference(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def verify_reference(reference: dict[str, Any]) -> Path:
    path = Path(reference["path"])
    if sha256(path) != reference["sha256"]:
        raise ValueError(f"Input checksum changed: {path}")
    return path


def read_json(path: Path) -> dict[str, Any]:
    # Shared mounts can briefly hide a renamed index after a successful stat.
    # Retry boundedly, then fail closed rather than inventing empty state.
    for attempt in range(3):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
            break
        except (FileNotFoundError, json.JSONDecodeError):
            if attempt == 2:
                raise
            time.sleep(0.05 * (attempt + 1))
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return result


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{number}")
            rows.append(row)
    return rows


def skipped_test_output(text: str) -> bool:
    """Recognize zero-test/all-skipped summaries from pytest, GoogleTest and unittest."""
    if re.search(r"\b(?:Running|Ran|Executed) 0 tests?\b|\bNo tests were found\b|\bcollected 0 items\b", text, re.I):
        return True
    if re.search(r"\b[1-9][0-9]* skipped\b", text, re.I) and not re.search(r"\b[1-9][0-9]* passed\b", text, re.I):
        return True
    if re.search(r"\[\s*SKIPPED\s*\]\s+[1-9][0-9]* tests?", text) and not re.search(r"\[\s*PASSED\s*\]\s+[1-9][0-9]* tests?", text):
        return True
    total = re.search(r"\bRan ([0-9]+) tests?\b", text)
    skipped = re.search(r"\bOK \(skipped=([0-9]+)\)", text)
    return bool(total and skipped and int(total.group(1)) <= int(skipped.group(1)))


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_text(path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def artifact_root(explicit: str | Path | None = None, start: Path | None = None) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    start = (start or Path.cwd()).resolve()
    for root in (start, *start.parents):
        config = root / "workspace.toml"
        if config.is_file():
            paths = tomllib.loads(config.read_text()).get("paths", {})
            return (root / paths.get("artifacts", "artifacts")).resolve()
    return start / "artifacts"


@dataclass(frozen=True)
class ArtifactLayout:
    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", self.root.resolve())

    def path(self, *parts: str) -> Path:
        for part in parts:
            if not COMPONENT.fullmatch(part) or part in (".", ".."):
                raise ValueError(f"Invalid path component: {part!r}")
        path = self.root.joinpath(*parts)
        if not path.resolve().is_relative_to(self.root):
            raise ValueError("Artifact path escapes its configured root")
        return path

    def run(self, stage: str, run_id: str) -> Path:
        if stage not in STAGES or not IDENTIFIER.fullmatch(run_id):
            raise ValueError("Invalid stage or run ID")
        return self.path("runs", stage, run_id)

    def snapshot(self, snapshot_id: str) -> Path:
        if not IDENTIFIER.fullmatch(snapshot_id):
            raise ValueError("Invalid snapshot ID")
        return self.path("sources", "github", snapshot_id)

    def release(self, name: str, version: str) -> Path:
        if not VERSION.fullmatch(version):
            raise ValueError("Dataset version must use vMAJOR.MINOR.PATCH")
        return self.path("datasets", name, version)


@contextmanager
def directory_lock(path: Path) -> Iterator[None]:
    """A nonblocking process lock, automatically released if a worker exits."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise FileExistsError(f"Another process owns {path}") from None
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def create_run(layout: ArtifactLayout, stage: str, *, inputs: list[dict[str, Any]] | None = None,
               config: dict[str, Any] | None = None, run_id: str | None = None,
               code: dict[str, Any] | None = None) -> Path:
    run_id = run_id or new_id()
    path = layout.run(stage, run_id)
    path.mkdir(parents=True, exist_ok=False)
    (path / "logs").mkdir()
    atomic_json(path / "run.json", {
        "schema_version": 1, "run_id": run_id, "stage": stage,
        "created_at": utc_now(), "updated_at": utc_now(), "status": "running",
        "inputs": inputs or [], "config": config or {}, "code": code or {}, "outputs": [],
    })
    return path


def resume_run(layout: ArtifactLayout, stage: str, run_id: str,
               *, config: dict[str, Any] | None = None) -> Path:
    path = layout.run(stage, run_id)
    run = read_json(path / "run.json")
    if run["stage"] != stage or run["run_id"] != run_id:
        raise ValueError("Run identity mismatch")
    if config is not None and run["config"] != config:
        raise ValueError("Resume configuration differs; create a new run")
    for reference in run["inputs"]:
        verify_reference(reference)
    return path


def update_run(path: Path, **fields: Any) -> None:
    with directory_lock(path / ".run-lock"):
        run = read_json(path / "run.json")
        if set(fields) & {"run_id", "stage", "created_at", "inputs", "config"}:
            raise ValueError("Run identity and inputs cannot be replaced")
        run.update(fields, updated_at=utc_now())
        atomic_json(path / "run.json", run)


def create_attempt(run: Path, instance_id: str) -> Path:
    if not COMPONENT.fullmatch(instance_id) or instance_id in (".", ".."):
        raise ValueError("Invalid instance ID")
    case = run / "cases" / instance_id
    if not case.resolve().is_relative_to(run.resolve()):
        raise ValueError("Case path escapes run")
    case.mkdir(parents=True, exist_ok=True)
    number = 1
    while True:
        path = case / f"attempt-{number:04d}"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            number += 1


def seal_snapshot(layout: ArtifactLayout, staging: Path, snapshot_id: str,
                  *, run: Path, status: str = "complete") -> Path:
    target = layout.snapshot(snapshot_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    with directory_lock(target.parent / f".{snapshot_id}.lock"):
        if target.exists():
            raise FileExistsError(target)
        files = {}
        for path in sorted(staging.rglob("*")):
            if path.is_file() and path.name != "manifest.json":
                if path.is_symlink():
                    raise ValueError("Snapshot inputs cannot be symlinks")
                files[str(path.relative_to(staging))] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        atomic_json(staging / "manifest.json", {
            "schema_version": 1, "snapshot_id": snapshot_id, "source": "github",
            "created_at": utc_now(), "run": str(run.resolve()), "status": status, "files": files,
        })
        staging.rename(target)
    return target


def verify_file_map(root: Path, files: dict[str, Any]) -> None:
    for relative, expected in files.items():
        path = root / relative
        if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
            raise ValueError("Manifest path escapes its directory")
        if sha256(path) != expected["sha256"]:
            raise ValueError(f"Artifact checksum mismatch: {path}")

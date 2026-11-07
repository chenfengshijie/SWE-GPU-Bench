"""Oracle baseline: give an LLM the files the reference patch touches and ask for JSON edits.

* ``evaluate oracle-generate``: write ``<output-root>/<profile>/<instance_id>/pred.diff`` (+ prompt,
  raw response, edits, metadata).
* ``evaluate oracle-validate``: apply ``pred.diff`` at ``base_commit`` and grade it with the hidden
  tests, writing ``<eval-root>/<instance_id>/result.json``.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from swe_gpu_bench.core.config import ModelProfile, add_profile_arguments, profile_from_args
from swe_gpu_bench.core.diff import is_test_like_path, parse_diff_sections
from swe_gpu_bench.core.git import clone_or_update_repo, prepare_record_checkout
from swe_gpu_bench.core.io import find_record, load_json, load_jsonl, load_text, write_json, write_text
from swe_gpu_bench.core.llm import ContextLengthExceededError, complete, profile_label, resolve_model
from swe_gpu_bench.core.patch import (
    PatchCheckResult,
    apply_pred_patch,
    check_patch_applies,
    is_test_patch_failure_status,
)
from swe_gpu_bench.core.selection import parse_instance_ids
from swe_gpu_bench.core.shell import did_commands_succeed, strip_phase
from swe_gpu_bench.evaluate.agent_eval import (
    DEFAULT_DATASET,
    build_evaluation_stats,
    grade_patched_repo,
    prepare_eval_case_dir,
    run_commands,
    test_outcome,
)

README_CANDIDATES = ("README.md", "README.rst", "README.txt", "README")
TERMINAL_SKIP_REASONS = frozenset({"context_length_exceeded"})


@dataclass(frozen=True)
class OracleContext:
    readme_path: str | None
    oracle_files: list[str]
    code_block: str


@dataclass(frozen=True)
class StructuredPatchResult:
    status: str
    patch_text: str = ""
    error: str = ""
    edits_payload: dict[str, Any] | None = None


def oracle_patch_dir(pred_patch_root: Path, instance_id: str) -> Path:
    return pred_patch_root / instance_id


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def extract_oracle_files(patch_text: str) -> list[str]:
    files: list[str] = []
    for section in parse_diff_sections(patch_text):
        path = section.old_path if section.new_path == "/dev/null" else section.new_path
        if path not in files and not is_test_like_path(path):
            files.append(path)
    return files


def _format_file_block(label: str, text: str) -> str:
    numbered = "\n".join(f"{index:04d}| {line}" for index, line in enumerate(text.splitlines(), start=1))
    if text.endswith("\n"):
        numbered += "\n"
    return (
        f"[start of {label}]\n"
        "[line numbers are reference only; do not copy them into patches]\n"
        f"{numbered}"
        f"[end of {label}]\n"
    )


def build_oracle_context(record: dict[str, Any], repo_dir: Path) -> OracleContext:
    parts: list[str] = []
    readme_path = next((repo_dir / name for name in README_CANDIDATES if (repo_dir / name).exists()), None)
    if readme_path is not None:
        parts.append(_format_file_block(readme_path.name, load_text(readme_path)))
    oracle_files: list[str] = []
    for rel_path in extract_oracle_files(record.get("patch") or ""):
        file_path = repo_dir / rel_path
        if file_path.is_file():
            oracle_files.append(rel_path)
            parts.append(_format_file_block(rel_path, load_text(file_path)))
    return OracleContext(
        readme_path=readme_path.name if readme_path is not None else None,
        oracle_files=oracle_files,
        code_block="\n".join(part.rstrip("\n") for part in parts).rstrip() + ("\n" if parts else ""),
    )


def render_oracle_prompt(template_text: str, problem_statement: str, code_block: str) -> str:
    contract = f"""
    You will be provided with a partial code base and an task statement
explaining a problem to resolve.
<task>
{problem_statement}
</task>

The code context below is split into file blocks:
- `[start of path]` and `[end of path]` mark the exact file path for each block.
- Lines are prefixed as `0001| source text` for reference only.
- The `0001|` prefixes are not part of the files. Do not include them in the patch.
- Use the reference line numbers only to locate edits and produce accurate unified diff hunks.

<code>
{code_block}
</code>

Return exactly one JSON object and nothing else. Do not return a unified diff.
The JSON object must have this shape:

```json
{{
  "edits": [
    {{
      "path": "relative/path/to/file.py",
      "old": "exact existing text block to replace",
      "new": "replacement text block"
    }}
  ]
}}
```

Rules:
- `path` must be one of the provided file paths unless you are adding a new helper file.
- `old` must be copied exactly from the file content, without the `0001|` line prefixes.
- For an existing file, `old` must match exactly once in that file.
- To add a new file, use `"old": ""` and put the complete file content in `new`.
- Do not include code-context line prefixes such as `0001|` in `old` or `new`.
- Prefer the smallest set of edits that solves the task.

The system will apply your JSON edits locally and generate the final unified diff with `git diff`.
Response:
"""
    return template_text.rstrip() + "\n\n" + contract.lstrip()


# ---------------------------------------------------------------------------
# Structured edits -> unified diff
# ---------------------------------------------------------------------------

def extract_json_from_response(response_text: str) -> dict[str, Any]:
    fenced = re.search(r"```(?:json)?\s*\n(?P<json>\{[\s\S]*?\})\s*```", response_text)
    if fenced:
        return json.loads(fenced.group("json"))
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", response_text):
        try:
            payload, _ = decoder.raw_decode(response_text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("No JSON object found in model response")


def _safe_repo_path(repo_dir: Path, rel_path: str) -> Path:
    path = Path(rel_path)
    if path.is_absolute() or ".." in path.parts or not rel_path.strip():
        raise ValueError(f"Invalid edit path: {rel_path!r}")
    return repo_dir / path


def _apply_one_structured_edit(repo_dir: Path, edit: dict[str, Any], index: int) -> None:
    rel_path, old, new = edit.get("path"), edit.get("old"), edit.get("new")
    if not isinstance(rel_path, str) or not isinstance(old, str) or not isinstance(new, str):
        raise ValueError(f"Edit {index} must contain string path, old, and new fields")
    if "0001|" in old or "0001|" in new:
        raise ValueError(f"Edit {index} includes code-context line prefixes")
    file_path = _safe_repo_path(repo_dir, rel_path)
    if old == "":
        if file_path.exists():
            raise ValueError(f"Edit {index} cannot create existing file: {rel_path}")
        write_text(file_path, new)
        return
    if not file_path.is_file():
        raise ValueError(f"Edit {index} target file does not exist: {rel_path}")
    text = load_text(file_path)
    occurrences = text.count(old)
    if occurrences == 0:
        raise ValueError(f"Edit {index} old block not found in {rel_path}")
    if occurrences > 1:
        raise ValueError(f"Edit {index} old block is ambiguous in {rel_path}: {occurrences} matches")
    file_path.write_text(text.replace(old, new, 1), encoding="utf-8")


def _git(repo_dir: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(repo_dir), capture_output=True, text=True, check=False)


def build_patch_from_structured_response(repo_dir: Path, response_text: str) -> StructuredPatchResult:
    """Apply the JSON edits, diff them against HEAD, then reset the worktree."""
    payload: dict[str, Any] | None = None
    try:
        payload = extract_json_from_response(response_text)
        edits = payload.get("edits")
        if not isinstance(edits, list) or not edits:
            return StructuredPatchResult(status="invalid", error="JSON response must contain a non-empty edits list", edits_payload=payload)
        edited_paths: list[str] = []
        for index, edit in enumerate(edits, start=1):
            if not isinstance(edit, dict):
                raise ValueError(f"Edit {index} must be an object")
            _apply_one_structured_edit(repo_dir, edit, index)
            edited_paths.append(edit["path"])
        for args in (("add", "-A", "--", *edited_paths), ("diff", "--cached", "--binary", "HEAD", "--", *edited_paths)):
            completed = _git(repo_dir, *args)
            if completed.returncode != 0:
                error = (completed.stderr or completed.stdout or f"git {args[0]} failed").strip()
                return StructuredPatchResult(status="invalid", error=error, edits_payload=payload)
        if not completed.stdout.strip():
            return StructuredPatchResult(status="invalid", error="Structured edits produced an empty diff", edits_payload=payload)
        return StructuredPatchResult(status="ok", patch_text=completed.stdout, edits_payload=payload)
    except (ValueError, json.JSONDecodeError, OSError) as exc:
        return StructuredPatchResult(status="invalid", error=str(exc), edits_payload=payload)
    finally:
        _git(repo_dir, "reset", "--hard", "HEAD")
        _git(repo_dir, "clean", "-fd")


# ---------------------------------------------------------------------------
# oracle-generate
# ---------------------------------------------------------------------------

def _load_terminal_skip_reason(output_root: Path, instance_id: str) -> str | None:
    metadata_path = oracle_patch_dir(output_root, instance_id) / "metadata.json"
    try:
        metadata = load_json(metadata_path)
    except (OSError, json.JSONDecodeError):
        return None
    reason = metadata.get("skip_reason")
    if metadata.get("structured_edit_status") == "skipped" and reason in TERMINAL_SKIP_REASONS:
        return str(reason)
    return None


def generate_oracle_patch(
    instance_id: str,
    dataset_path: Path,
    output_root: Path,
    prompt_template_path: Path,
    profile: ModelProfile | None = None,
    env_file: Path | None = None,
    max_tokens: int = 8192,
) -> dict[str, Any]:
    record = find_record(dataset_path, "instance_id", instance_id)
    repo_dir = output_root / instance_id / "case" / "repo"
    repo_dir.mkdir(parents=True, exist_ok=True)
    prepare_record_checkout(
        type("CasePaths", (), {"repo_dir": repo_dir}),
        record,
        cache_root=output_root / "_repo_cache",
        clone_repo_fn=clone_or_update_repo,
    )

    context = build_oracle_context(record, repo_dir)
    prompt = render_oracle_prompt(load_text(prompt_template_path), record.get("problem_statement", ""), context.code_block)
    patch_dir = oracle_patch_dir(output_root, instance_id)
    paths = {name: patch_dir / name for name in ("prompt.md", "raw_response.txt", "pred.diff", "edits.json", "metadata.json")}
    write_text(paths["prompt.md"], prompt)
    skip_reason = None
    try:
        response = complete(prompt, profile=profile, max_tokens=max_tokens, env_file=env_file) or ""
        structured = build_patch_from_structured_response(repo_dir, response)
        if structured.status == "ok":
            apply_check = check_patch_applies(repo_dir, structured.patch_text)
        else:
            apply_check = PatchCheckResult(status="invalid", stderr=structured.error)
        write_text(paths["raw_response.txt"], response)
        write_text(paths["pred.diff"], structured.patch_text)
        write_json(paths["edits.json"], structured.edits_payload or {"error": structured.error})
    except ContextLengthExceededError as exc:
        skip_reason = "context_length_exceeded"
        structured = StructuredPatchResult(status="skipped", error=str(exc))
        apply_check = PatchCheckResult(status="skipped", stderr=str(exc))
        write_text(paths["raw_response.txt"], "")
        write_json(paths["edits.json"], {"error": str(exc), "skip_reason": skip_reason})

    metadata = {
        "instance_id": instance_id,
        "repo": record.get("repo"),
        "base_commit": record.get("base_commit"),
        "profile": profile_label(profile),
        "model": resolve_model(profile),
        "readme_path": context.readme_path,
        "oracle_files": context.oracle_files,
        "prompt_template": str(prompt_template_path),
        "prompt_path": str(paths["prompt.md"]),
        "raw_response_path": str(paths["raw_response.txt"]),
        "edits_path": str(paths["edits.json"]),
        "pred_patch_path": str(paths["pred.diff"]),
        "structured_edit_status": structured.status,
        "structured_edit_error": structured.error,
        "pred_patch_apply_check_status": apply_check.status,
        "pred_patch_apply_check_error": apply_check.stderr,
        "skip_reason": skip_reason,
    }
    write_json(paths["metadata.json"], metadata)
    return metadata


def generate_oracle_patches(
    dataset_path: Path,
    output_root: Path,
    prompt_template_path: Path,
    instance_ids: List[str],
    profile: ModelProfile | None = None,
    env_file: Path | None = None,
    max_tokens: int = 8192,
) -> list[dict[str, Any]]:
    """Generate under ``output_root`` (already scoped to the profile); existing outputs are skipped."""
    results: list[dict[str, Any]] = []
    for instance_id in instance_ids:
        pred_path = oracle_patch_dir(output_root, instance_id) / "pred.diff"
        if pred_path.exists():
            print(f"[skip] {instance_id} already has pred.diff", file=sys.stderr)
            results.append({"instance_id": instance_id, "skipped": True, "pred_patch_path": str(pred_path)})
            continue
        skip_reason = _load_terminal_skip_reason(output_root, instance_id)
        if skip_reason is not None:
            print(f"[skip] {instance_id} already marked skipped: {skip_reason}", file=sys.stderr)
            results.append({"instance_id": instance_id, "skipped": True, "reason": skip_reason})
            continue
        try:
            results.append(
                generate_oracle_patch(instance_id, dataset_path, output_root, prompt_template_path, profile, env_file, max_tokens)
            )
        except Exception as exc:
            print(f"[error] {instance_id} failed: {exc}", file=sys.stderr)
            results.append({"instance_id": instance_id, "error": str(exc)})
    return results


def add_generate_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--instance-id", action="append", default=[], help="Repeatable; comma-separated allowed")
    parser.add_argument("--instance-id-file", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output-root", default="results/oracle_patches", help="Patches go to <output-root>/<profile>/")
    parser.add_argument("--prompt-template", default="prompts/oracle_generate.md")
    parser.add_argument("--max-tokens", type=int, default=8192)
    add_profile_arguments(parser, kind="text")


def run_generate(args: argparse.Namespace) -> int:
    dataset_path = Path(args.dataset).resolve()
    if args.all:
        instance_ids = [record["instance_id"] for record in load_jsonl(dataset_path)]
    else:
        instance_ids = parse_instance_ids(args.instance_id, Path(args.instance_id_file) if args.instance_id_file else None)
    if not instance_ids:
        raise SystemExit("One of --instance-id, --instance-id-file, or --all is required")
    profile = profile_from_args(args, kind="text")
    results = generate_oracle_patches(
        dataset_path=dataset_path,
        output_root=Path(args.output_root).resolve() / profile_label(profile),
        prompt_template_path=Path(args.prompt_template).resolve(),
        instance_ids=instance_ids,
        profile=profile,
        env_file=Path(args.env_file).resolve() if args.env_file else None,
        max_tokens=args.max_tokens,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


# ---------------------------------------------------------------------------
# oracle-validate
# ---------------------------------------------------------------------------

def execute_oracle_validate(
    instance_id: str,
    dataset_path: Path,
    eval_root: Path,
    pred_patch_root: Path,
    force: bool = False,
) -> dict[str, Any]:
    record = find_record(dataset_path, "instance_id", instance_id)
    case_paths = prepare_eval_case_dir(record, eval_root, None, force)
    started_at = datetime.now(timezone.utc)
    setup_commands = record.get("setup_commands", [])
    pred_path = oracle_patch_dir(pred_patch_root, instance_id) / "pred.diff"
    setup_records: List[Dict[str, Any]] = []
    pred_check: Optional[PatchCheckResult] = None
    pred_apply: Optional[PatchCheckResult] = None
    post_patch_setup_records: List[Dict[str, Any]] = []
    test_patch_status = "not_run"
    test_records: List[Dict[str, Any]] = []
    checkout_failed = False

    if pred_path.exists():
        try:
            prepare_record_checkout(case_paths, record, cache_root=eval_root / "_repo_cache", clone_repo_fn=clone_or_update_repo)
        except subprocess.CalledProcessError:
            checkout_failed = True
        else:
            setup_records = run_commands(setup_commands, "setup", case_paths)

    if not pred_path.exists():
        status, failure_category = "error", "PRED_PATCH_MISSING"
        pred_check = pred_apply = PatchCheckResult(status="missing")
    elif checkout_failed:
        status, failure_category = "error", "REPOSITORY_PREPARATION_FAILURE"
    elif not did_commands_succeed(setup_records):
        status, failure_category = "error", "SETUP_FAILURE"
    else:
        pred_patch_text = load_text(pred_path)
        pred_check = check_patch_applies(case_paths.repo_dir, pred_patch_text)
        if pred_check.status == "valid":
            pred_apply = apply_pred_patch(case_paths.repo_dir, pred_patch_text)
        else:
            pred_apply = PatchCheckResult(status="invalid", stderr=pred_check.stderr)
        post_patch_setup_records, test_patch_status, test_records = grade_patched_repo(
            record, case_paths, rerun_setup=pred_apply.status == "applied"
        )
        if pred_apply.status != "applied":
            status, failure_category = "error", "PRED_PATCH_APPLY_FAILURE"
        elif is_test_patch_failure_status(test_patch_status):
            status, failure_category = "error", "TEST_PATCH_APPLY_FAILURE"
        else:
            status, failure_category = test_outcome(test_records)

    finished_at = datetime.now(timezone.utc)
    elapsed = round((finished_at - started_at).total_seconds(), 2)
    result = {
        "instance_id": instance_id,
        "repo": record.get("repo", ""),
        "base_commit": record.get("base_commit", ""),
        "head_commit": record.get("head_commit", ""),
        "status": status,
        "failure_category": failure_category,
        "pred_patch_path": str(pred_path),
        "pred_patch_apply_check_status": pred_check.status if pred_check else "not_run",
        "pred_patch_apply_check_error": pred_check.stderr if pred_check else "",
        "pred_patch_apply_status": pred_apply.status if pred_apply else "not_run",
        "pred_patch_apply_error": pred_apply.stderr if pred_apply else "",
        "test_patch_status": test_patch_status,
        "setup_commands": setup_commands,
        "setup_results": strip_phase(setup_records),
        "post_patch_setup_results": strip_phase(post_patch_setup_records),
        "test_commands": record.get("test_commands", []),
        "test_results": strip_phase(test_records),
        "evaluation_stats": build_evaluation_stats(setup_records, post_patch_setup_records, test_records, elapsed_seconds=elapsed),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "elapsed_seconds": elapsed,
        "case_dir": str(case_paths.case_dir),
    }
    write_json(case_paths.result_path, result)
    return result


def add_validate_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--eval-root", default="results/oracle_eval")
    parser.add_argument("--pred-patch-root", required=True, help="Directory with <instance_id>/pred.diff")
    parser.add_argument("--force", action="store_true")


def run_validate(args: argparse.Namespace) -> int:
    result = execute_oracle_validate(
        instance_id=args.instance_id,
        dataset_path=Path(args.dataset).resolve(),
        eval_root=Path(args.eval_root).resolve(),
        pred_patch_root=Path(args.pred_patch_root).resolve(),
        force=args.force,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0

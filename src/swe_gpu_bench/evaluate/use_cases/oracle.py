from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from swe_gpu_bench.adapters.llm import (
    ContextLengthExceededError,
    DEFAULT_EVAL_ENV_FILE,
    generate_text,
    generate_text_from_profile,
)
from swe_gpu_bench.evaluate.use_cases.eval import (
    apply_test_patch,
    capture_agent_diff,
    untracked_paths,
    build_evaluation_stats,
    did_commands_succeed,
    prepare_eval_case_dir,
    run_commands,
    should_rerun_setup_after_agent,
    resolve_repo_cache_root,
    _determine_status,
)
from swe_gpu_bench.core.campaign_repo import clone_or_update_repo
from swe_gpu_bench.shared.config import resolve_profile
from swe_gpu_bench.shared.diff import is_test_like_path, parse_diff_sections
from swe_gpu_bench.shared.io import load_jsonl, load_text, write_json, write_text
from swe_gpu_bench.shared.models import ModelProfile
from swe_gpu_bench.shared.patch_utils import (
    PatchCheckResult,
    apply_pred_patch,
    check_patch_applies,
)
from swe_gpu_bench.test_patch_utils import is_test_patch_failure_status


DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2-2025-12-11")
DEFAULT_MODEL_SPEC = "gpt-default"


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


README_CANDIDATES = ("README.md", "README.rst", "README.txt", "README")
TERMINAL_SKIP_REASONS = frozenset({"context_length_exceeded"})


def load_oracle_record(dataset_path: Path, instance_id: str) -> dict[str, Any]:
    for record in load_jsonl(dataset_path):
        if record.get("instance_id") == instance_id:
            return record
    raise KeyError(f"instance_id {instance_id!r} not found in {dataset_path}")


def load_oracle_records(dataset_path: Path) -> list[dict[str, Any]]:
    return load_jsonl(dataset_path)


def extract_oracle_files(patch_text: str) -> list[str]:
    files: list[str] = []
    seen: set[str] = set()
    for section in parse_diff_sections(patch_text):
        path = section.new_path
        if path == "/dev/null":
            path = section.old_path
        if path not in seen and not is_test_like_path(path):
            seen.add(path)
            files.append(path)
    return files


def _find_readme_path(repo_dir: Path) -> Path | None:
    for name in README_CANDIDATES:
        candidate = repo_dir / name
        if candidate.exists():
            return candidate
    return None


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
    readme_path = _find_readme_path(repo_dir)
    if readme_path is not None:
        parts.append(_format_file_block(readme_path.name, load_text(readme_path)))

    oracle_files: list[str] = []
    for rel_path in extract_oracle_files(record.get("patch") or ""):
        file_path = repo_dir / rel_path
        if not file_path.exists() or not file_path.is_file():
            continue
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


def extract_patch_from_response(response_text: str) -> str:
    fenced = re.search(r"```diff\s*\n(?P<patch>diff --git[\s\S]*?)```", response_text)
    if fenced:
        return fenced.group("patch").strip() + "\n"
    raw = re.search(r"(?P<patch>diff --git[\s\S]*)", response_text)
    if raw:
        return raw.group("patch").strip() + "\n"
    return response_text.strip() + ("\n" if response_text.strip() else "")


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
    rel_path = edit.get("path")
    old = edit.get("old")
    new = edit.get("new")
    if not isinstance(rel_path, str) or not isinstance(old, str) or not isinstance(new, str):
        raise ValueError(f"Edit {index} must contain string path, old, and new fields")
    if "0001|" in old or "0001|" in new:
        raise ValueError(f"Edit {index} includes code-context line prefixes")

    file_path = _safe_repo_path(repo_dir, rel_path)
    if old == "":
        if file_path.exists():
            raise ValueError(f"Edit {index} cannot create existing file: {rel_path}")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(new, encoding="utf-8")
        return

    if not file_path.exists() or not file_path.is_file():
        raise ValueError(f"Edit {index} target file does not exist: {rel_path}")
    text = load_text(file_path)
    occurrences = text.count(old)
    if occurrences == 0:
        raise ValueError(f"Edit {index} old block not found in {rel_path}")
    if occurrences > 1:
        raise ValueError(f"Edit {index} old block is ambiguous in {rel_path}: {occurrences} matches")
    file_path.write_text(text.replace(old, new, 1), encoding="utf-8")


def _cleanup_repo_worktree(repo_dir: Path) -> None:
    subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=str(repo_dir), capture_output=True, text=True, check=False)
    subprocess.run(["git", "clean", "-fd"], cwd=str(repo_dir), capture_output=True, text=True, check=False)


def build_patch_from_structured_response(repo_dir: Path, response_text: str) -> StructuredPatchResult:
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
        add = subprocess.run(
            ["git", "add", "-A", "--", *edited_paths],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            check=False,
        )
        if add.returncode != 0:
            return StructuredPatchResult(
                status="invalid",
                error=(add.stderr or add.stdout or "git add failed").strip(),
                edits_payload=payload,
            )
        diff = subprocess.run(
            ["git", "diff", "--cached", "--binary", "HEAD", "--", *edited_paths],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            check=False,
        )
        if diff.returncode != 0:
            return StructuredPatchResult(
                status="invalid",
                error=(diff.stderr or diff.stdout or "git diff failed").strip(),
                edits_payload=payload,
            )
        patch_text = diff.stdout
        if not patch_text.strip():
            return StructuredPatchResult(status="invalid", error="Structured edits produced an empty diff", edits_payload=payload)
        return StructuredPatchResult(status="ok", patch_text=patch_text, edits_payload=payload)
    except (ValueError, json.JSONDecodeError, OSError) as exc:
        try:
            payload
        except UnboundLocalError:
            payload = None
        return StructuredPatchResult(status="invalid", error=str(exc), edits_payload=payload)
    finally:
        _cleanup_repo_worktree(repo_dir)
def oracle_patch_dir(pred_patch_root: Path, instance_id: str) -> Path:
    return pred_patch_root / instance_id


def scoped_output_root(output_root: Path, model_spec: str) -> Path:
    return output_root / model_spec


def _terminal_skip_metadata_path(output_root: Path, instance_id: str) -> Path:
    return oracle_patch_dir(output_root, instance_id) / "metadata.json"


def _load_terminal_skip_reason(output_root: Path, instance_id: str) -> str | None:
    metadata_path = _terminal_skip_metadata_path(output_root, instance_id)
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if metadata.get("structured_edit_status") != "skipped":
        return None
    reason = metadata.get("skip_reason")
    if reason in TERMINAL_SKIP_REASONS:
        return str(reason)
    return None


def _prepare_oracle_repo(
    repo_dir: Path,
    repo_cache_root: Path,
    record: dict[str, Any],
) -> None:
    case_paths = type("CasePaths", (), {"repo_dir": repo_dir})
    for attempt in range(2):
        try:
            clone_or_update_repo(
                case_paths,
                record["repo"],
                record.get("base_commit"),
                pr_number=record.get("pr_number"),
                cache_root=repo_cache_root,
                strict_commit=True,
            )
            return
        except subprocess.CalledProcessError:
            if attempt == 1:
                raise
            shutil.rmtree(repo_dir, ignore_errors=True)
            repo_dir.mkdir(parents=True, exist_ok=True)


def generate_oracle_patch(
    instance_id: str,
    dataset_path: Path,
    output_root: Path,
    prompt_template_path: Path,
    model_spec: str = DEFAULT_MODEL_SPEC,
    profile: ModelProfile | None = None,
    model: str | None = None,
    env_file: Path | None = None,
    max_tokens: int = 8192,
    request_timeout: float | None = None,
    isolate_history: bool = False,
    request_attempts: int = 3,
) -> dict[str, Any]:
    record = load_oracle_record(dataset_path, instance_id)
    case_dir = output_root / instance_id / "case"
    repo_dir = case_dir / "repo"
    repo_cache_root = output_root / "_repo_cache"
    repo_dir.mkdir(parents=True, exist_ok=True)
    _prepare_oracle_repo(repo_dir, repo_cache_root, record)

    if isolate_history:
        from swe_gpu_bench.evaluate.use_cases.solver_isolation import isolate_git_history
        isolate_git_history(repo_dir, record["base_commit"])
    context = build_oracle_context(record, repo_dir)
    prompt = render_oracle_prompt(
        load_text(prompt_template_path),
        problem_statement=record.get("problem_statement", ""),
        code_block=context.code_block,
    )
    resolved_model = model or (profile.model if profile is not None else None)
    patch_dir = oracle_patch_dir(output_root, instance_id)
    raw_path = patch_dir / "raw_response.txt"
    pred_path = patch_dir / "pred.diff"
    prompt_path = patch_dir / "prompt.md"
    edits_path = patch_dir / "edits.json"
    metadata_path = patch_dir / "metadata.json"
    write_text(prompt_path, prompt)
    from swe_gpu_bench.shared.usage import summarize_usage
    usage_events = []
    def record_usage(event):
        usage_events.append(event)
        with (patch_dir / 'usage.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(event) + '\n')
            handle.flush()
    try:
        if profile is not None:
            response = generate_text_from_profile(
                prompt,
                profile=replace(profile, model=resolved_model),
                max_tokens=max_tokens,
                env_file=env_file,
                usage_callback=record_usage,
                max_attempts=request_attempts,
                **({"request_timeout": request_timeout} if request_timeout is not None else {}),
            ) or ""
        else:
            response = generate_text(
                prompt,
                model_spec=model_spec,
                model_name=resolved_model,
                max_tokens=max_tokens,
                env_file=env_file,
            ) or ""
        structured_result = build_patch_from_structured_response(repo_dir, response)
        patch_text = structured_result.patch_text
        if structured_result.status == "ok":
            apply_check = check_patch_applies(repo_dir, patch_text)
        else:
            apply_check = PatchCheckResult(status="invalid", stderr=structured_result.error)
        write_text(raw_path, response)
        write_text(pred_path, patch_text)
        write_json(edits_path, structured_result.edits_payload or {"error": structured_result.error})
        skip_reason = None
    except ContextLengthExceededError as exc:
        response = ""
        structured_result = StructuredPatchResult(status="skipped", error=str(exc))
        apply_check = PatchCheckResult(status="skipped", stderr=str(exc))
        skip_reason = "context_length_exceeded"
        write_text(raw_path, "")
        write_json(edits_path, {"error": str(exc), "skip_reason": skip_reason})
    metadata = {
        "instance_id": instance_id,
        "repo": record.get("repo"),
        "base_commit": record.get("base_commit"),
        "model_spec": model_spec,
        "model": resolved_model,
        "reasoning_effort": profile.reasoning_effort if profile is not None else None,
        "readme_path": context.readme_path,
        "oracle_files": context.oracle_files,
        "prompt_template": str(prompt_template_path),
        "prompt_path": str(prompt_path),
        "raw_response_path": str(raw_path),
        "edits_path": str(edits_path),
        "pred_patch_path": str(pred_path),
        "structured_edit_status": structured_result.status,
        "structured_edit_error": structured_result.error,
        "pred_patch_apply_check_status": apply_check.status,
        "pred_patch_apply_check_error": apply_check.stderr,
        "skip_reason": skip_reason,
        "usage": summarize_usage(usage_events),
    }
    write_json(metadata_path, metadata)
    return metadata


def _resolve_generation_ids(
    dataset_path: Path,
    *,
    instance_id: str | None = None,
    instance_id_file: Path | None = None,
    all_instances: bool = False,
) -> list[str]:
    if all_instances:
        return [record["instance_id"] for record in load_oracle_records(dataset_path)]

    ids: list[str] = []
    if instance_id:
        ids.append(instance_id)
    if instance_id_file and instance_id_file.exists():
        for raw in instance_id_file.read_text(encoding="utf-8").splitlines():
            value = raw.strip()
            if value and value not in ids:
                ids.append(value)
    if not ids:
        raise ValueError("One of --instance-id, --instance-id-file, or --all is required")
    return ids


def generate_oracle_patches(
    dataset_path: Path,
    output_root: Path,
    prompt_template_path: Path,
    model_spec: str,
    profile: ModelProfile | None = None,
    *,
    instance_id: str | None = None,
    instance_id_file: Path | None = None,
    all_instances: bool = False,
    model: str | None = None,
    env_file: Path | None = None,
    max_tokens: int = 8192,
) -> list[dict[str, Any]]:
    import sys

    ids = _resolve_generation_ids(
        dataset_path,
        instance_id=instance_id,
        instance_id_file=instance_id_file,
        all_instances=all_instances,
    )
    model_root = scoped_output_root(output_root, model_spec)
    results: list[dict[str, Any]] = []
    for current_id in ids:
        pred_path = oracle_patch_dir(model_root, current_id) / "pred.diff"
        if pred_path.exists():
            print(f"[skip] {current_id} already has pred.diff", file=sys.stderr)
            results.append({"instance_id": current_id, "skipped": True, "pred_patch_path": str(pred_path)})
            continue
        skip_reason = _load_terminal_skip_reason(model_root, current_id)
        if skip_reason is not None:
            print(f"[skip] {current_id} already marked skipped: {skip_reason}", file=sys.stderr)
            results.append({"instance_id": current_id, "skipped": True, "reason": skip_reason})
            continue
        try:
            kwargs = {
                "instance_id": current_id,
                "dataset_path": dataset_path,
                "output_root": model_root,
                "prompt_template_path": prompt_template_path,
                "model_spec": model_spec,
                "model": model,
                "env_file": env_file,
                "max_tokens": max_tokens,
            }
            if profile is not None:
                kwargs["profile"] = profile
            results.append(generate_oracle_patch(**kwargs))
        except Exception as exc:
            print(f"[error] {current_id} failed: {exc}", file=sys.stderr)
            results.append({"instance_id": current_id, "error": str(exc)})
    return results


def execute_oracle_validate(
    instance_id: str,
    dataset_path: Path,
    eval_root: Path,
    pred_patch_root: Path,
    prompt_template_path: Path,
    force: bool = False,
) -> dict[str, Any]:
    record = load_oracle_record(dataset_path, instance_id)
    case_paths = prepare_eval_case_dir(
        record=record,
        eval_root=eval_root,
        prompt_template_path=prompt_template_path,
        force=force,
    )

    started_at = datetime.now(timezone.utc)
    clone_or_update_repo(case_paths, record["repo"], record.get("base_commit"), pr_number=record.get("pr_number"),
                         cache_root=resolve_repo_cache_root(), strict_commit=True)

    setup_commands = record.get("setup_commands", [])
    setup_records = run_commands(setup_commands, "setup", case_paths)
    pred_path = oracle_patch_dir(pred_patch_root, instance_id) / "pred.diff"
    if not pred_path.exists():
        finished_at = datetime.now(timezone.utc)
        result = {
            "instance_id": instance_id,
            "repo": record.get("repo", ""),
            "base_commit": record.get("base_commit", ""),
            "head_commit": record.get("head_commit", ""),
            "status": "error",
            "failure_category": "PRED_PATCH_MISSING",
            "pred_patch_path": str(pred_path),
            "pred_patch_apply_check_status": "missing",
            "pred_patch_apply_check_error": "",
            "pred_patch_apply_status": "missing",
            "pred_patch_apply_error": "",
            "test_patch_status": "not_run",
            "setup_commands": setup_commands,
            "setup_results": [{k: v for k, v in item.items() if k != "phase"} for item in setup_records],
            "post_patch_setup_results": [],
            "test_commands": record.get("test_commands", []),
            "test_results": [],
            "evaluation_stats": build_evaluation_stats(setup_records, [], [], elapsed_seconds=round((finished_at - started_at).total_seconds(), 2)),
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "elapsed_seconds": round((finished_at - started_at).total_seconds(), 2),
            "case_dir": str(case_paths.case_dir),
        }
        write_json(case_paths.result_path, result)
        return result
    if not all(item.get("exit_code") == 0 for item in setup_records):
        finished_at = datetime.now(timezone.utc)
        result = {
            "instance_id": instance_id,
            "repo": record.get("repo", ""),
            "base_commit": record.get("base_commit", ""),
            "head_commit": record.get("head_commit", ""),
            "status": "error",
            "failure_category": "SETUP_FAILURE",
            "pred_patch_path": str(pred_path),
            "pred_patch_apply_check_status": "not_run",
            "pred_patch_apply_check_error": "",
            "pred_patch_apply_status": "not_run",
            "pred_patch_apply_error": "",
            "test_patch_status": "not_run",
            "setup_commands": setup_commands,
            "setup_results": [{k: v for k, v in item.items() if k != "phase"} for item in setup_records],
            "post_patch_setup_results": [],
            "test_commands": record.get("test_commands", []),
            "test_results": [],
            "evaluation_stats": build_evaluation_stats(setup_records, [], [], elapsed_seconds=round((finished_at - started_at).total_seconds(), 2)),
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "elapsed_seconds": round((finished_at - started_at).total_seconds(), 2),
            "case_dir": str(case_paths.case_dir),
        }
        write_json(case_paths.result_path, result)
        return result

    pred_patch_text = load_text(pred_path)
    untracked_before = untracked_paths(case_paths.repo_dir) if os.getenv('SWEBENCH_VERIFY_SOURCE') == '1' else None
    pred_check = check_patch_applies(case_paths.repo_dir, pred_patch_text)
    if pred_check.status == "valid":
        pred_apply = apply_pred_patch(case_paths.repo_dir, pred_patch_text)
    else:
        pred_apply = PatchCheckResult(status="invalid", stderr=pred_check.stderr)
    if pred_apply.status == 'applied' and untracked_before is not None:
        capture_agent_diff(case_paths, untracked_before)

    test_patch = record.get("test_patch", "")
    if test_patch.strip():
        test_patch_status = apply_test_patch(test_patch, case_paths)
    else:
        test_patch_status = "no_test_patch"

    post_patch_setup_records: List[Dict[str, Any]] = []
    if (pred_apply.status == "applied" and not is_test_patch_failure_status(test_patch_status)
            and record.get("command_contract") != "portable_unified_v1"
            and should_rerun_setup_after_agent(setup_commands)):
        post_patch_setup_records = run_commands(setup_commands, "post_patch_setup", case_paths)

    can_test = (pred_apply.status == "applied"
                and not is_test_patch_failure_status(test_patch_status)
                and all(item.get("exit_code") == 0 for item in post_patch_setup_records))
    test_records = run_commands(record.get("test_commands", []), "test", case_paths) if can_test else []
    finished_at = datetime.now(timezone.utc)

    if pred_apply.status != "applied":
        status = "error"
        failure_category = "PRED_PATCH_APPLY_FAILURE"
    elif is_test_patch_failure_status(test_patch_status):
        status = "error"
        failure_category = "TEST_PATCH_APPLY_FAILURE"
    elif not all(item.get("exit_code") == 0 for item in post_patch_setup_records):
        status = "error"
        failure_category = "POST_PATCH_SETUP_FAILURE"
    else:
        status, failure_category = _determine_status(
            setup_records, post_patch_setup_records, test_records, test_patch_status, strict_commands=True)

    elapsed = round((finished_at - started_at).total_seconds(), 2)
    result = {
        "instance_id": instance_id,
        "repo": record.get("repo", ""),
        "base_commit": record.get("base_commit", ""),
        "head_commit": record.get("head_commit", ""),
        "status": status,
        "failure_category": failure_category,
        "pred_patch_path": str(pred_path),
        "pred_patch_apply_check_status": pred_check.status,
        "pred_patch_apply_check_error": pred_check.stderr,
        "pred_patch_apply_status": pred_apply.status,
        "pred_patch_apply_error": pred_apply.stderr,
        "test_patch_status": test_patch_status,
        "setup_commands": setup_commands,
        "setup_results": [{k: v for k, v in item.items() if k != "phase"} for item in setup_records],
        "post_patch_setup_results": [{k: v for k, v in item.items() if k != "phase"} for item in post_patch_setup_records],
        "test_commands": record.get("test_commands", []),
        "test_results": [{k: v for k, v in item.items() if k != "phase"} for item in test_records],
        "evaluation_stats": build_evaluation_stats(
            setup_records,
            post_patch_setup_records,
            test_records,
            elapsed_seconds=elapsed,
        ),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "elapsed_seconds": elapsed,
        "case_dir": str(case_paths.case_dir),
    }
    write_json(case_paths.result_path, result)
    return result


def summarize_oracle_results(eval_root: Path, models: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {"models": {}, "rows": []}
    for model in models:
        model_root = eval_root / model
        rows: list[dict[str, Any]] = []
        if model_root.exists():
            for case_dir in sorted(path for path in model_root.iterdir() if path.is_dir()):
                result_path = case_dir / "result.json"
                if not result_path.exists():
                    rows.append(
                        {
                            "model_spec": model,
                            "instance_id": case_dir.name,
                            "status": "missing",
                            "failure_category": None,
                        }
                    )
                    continue
                payload = json.loads(result_path.read_text(encoding="utf-8"))
                rows.append(
                    {
                        "model_spec": model,
                        "instance_id": payload.get("instance_id", case_dir.name),
                        "status": payload.get("status", "unknown"),
                        "failure_category": payload.get("failure_category"),
                    }
                )
        passed = sum(1 for row in rows if row["status"] == "passed")
        total = len(rows)
        summary["models"][model] = {
            "total": total,
            "passed": passed,
            "failed": sum(1 for row in rows if row["status"] == "failed"),
            "error": sum(1 for row in rows if row["status"] == "error"),
            "missing": sum(1 for row in rows if row["status"] == "missing"),
            "pass_rate": (passed / total) if total else 0.0,
        }
        summary["rows"].extend(rows)
    return summary


def write_oracle_summary_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model_spec", "instance_id", "status", "failure_category"])
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Oracle patch generation and validation.")
    subparsers = parser.add_subparsers(dest="oracle_command", required=True)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--instance-id", default=None)
    generate.add_argument("--instance-id-file", default=None)
    generate.add_argument("--all", action="store_true")
    generate.add_argument("--dataset", default="data/dataset/swe_gpu_bench_v1_100_with_hw.jsonl")
    generate.add_argument("--output-root", default="../OraclePatches")
    generate.add_argument("--prompt-template", default="prompts/oracle_generate.md")
    generate.add_argument("--model-spec", default=DEFAULT_MODEL_SPEC)
    generate.add_argument("--profile", default=None)
    generate.add_argument("--config", default="swe_gpu_bench.toml")
    generate.add_argument("--provider", default=None)
    generate.add_argument("--model", default=None)
    generate.add_argument("--base-url", default=None)
    generate.add_argument("--api-key-env", default=None)
    generate.add_argument("--env-file", default=str(DEFAULT_EVAL_ENV_FILE))
    generate.add_argument("--max-tokens", type=int, default=8192)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--instance-id", required=True)
    validate.add_argument("--dataset", default="data/dataset/swe_gpu_bench_v1_100_with_hw.jsonl")
    validate.add_argument("--eval-root", default="../OracleEvalResults")
    validate.add_argument("--pred-patch-root", required=True)
    validate.add_argument("--prompt-template", default="prompts/oracle_generate.md")
    validate.add_argument("--force", action="store_true")

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--eval-root", required=True)
    summarize.add_argument("--models", required=True)
    summarize.add_argument("--json-output", default=None)
    summarize.add_argument("--csv-output", default=None)
    return parser


def cli_main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cwd = Path(".").resolve()
    if args.oracle_command == "generate":
        profile = None
        if args.profile or args.provider or args.model or args.base_url or args.api_key_env:
            profile = resolve_profile(
                profile_name=args.profile,
                profile_kind="text",
                config_path=(cwd / args.config).resolve() if args.config else None,
                provider=args.provider,
                model=args.model,
                base_url=args.base_url,
                api_key_env=args.api_key_env,
            )
        results = generate_oracle_patches(
            dataset_path=(cwd / args.dataset).resolve(),
            output_root=Path(args.output_root).resolve(),
            prompt_template_path=(cwd / args.prompt_template).resolve(),
            model_spec=args.model_spec,
            profile=profile,
            instance_id=args.instance_id,
            instance_id_file=Path(args.instance_id_file).resolve() if args.instance_id_file else None,
            all_instances=args.all,
            model=args.model,
            env_file=Path(args.env_file).resolve() if args.env_file else None,
            max_tokens=args.max_tokens,
        )
        print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.oracle_command == "summarize":
        models = [item.strip() for item in args.models.split(",") if item.strip()]
        summary = summarize_oracle_results(Path(args.eval_root).resolve(), models)
        if args.json_output:
            write_json(Path(args.json_output).resolve(), summary)
        if args.csv_output:
            write_oracle_summary_csv(summary["rows"], Path(args.csv_output).resolve())
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    result = execute_oracle_validate(
        instance_id=args.instance_id,
        dataset_path=(cwd / args.dataset).resolve(),
        eval_root=Path(args.eval_root).resolve(),
        pred_patch_root=Path(args.pred_patch_root).resolve(),
        prompt_template_path=(cwd / args.prompt_template).resolve(),
        force=args.force,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0

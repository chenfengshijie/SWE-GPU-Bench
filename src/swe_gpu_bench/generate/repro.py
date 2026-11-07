"""``generate repro``: let a Codex agent discover setup/test commands for one PR.

Case layout under ``<case-root>/<repo-tail>-pr<N>/``::

    .codex/          copied from --template-codex-dir (Codex auth/config)
    repo/            checkout at the PR head
    inputs/          sample.json, prompt.md
    artifacts/       venvs and caches created by the agent
    logs/            codex.session.jsonl, codex.stderr.log, codex.last-message.txt, result.json
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from swe_gpu_bench.core.agent import CodexAgentRunner, parse_codex_session_stats
from swe_gpu_bench.core.git import clone_or_update_repo
from swe_gpu_bench.core.io import find_pr_record, load_json, pr_metadata_path, repo_dir, repo_tail, write_json, write_text
from swe_gpu_bench.core.shell import format_called_process_error

RESULT_START_MARKER = "BEGIN_FINAL_RESULT_JSON"
RESULT_END_MARKER = "END_FINAL_RESULT_JSON"
EXCLUDED_CASE_SAMPLE_KEYS = frozenset(["patch_text", "raw_response"])
PR_METADATA_KEYS = (
    "repo", "number", "title", "body", "html_url", "base_ref", "head_ref", "commit_messages", "has_test_patch",
)
DEFAULT_SCORED_DIR = Path("data/fused_scored")


@dataclass
class CasePaths:
    case_dir: Path
    codex_dir: Path
    repo_dir: Path
    inputs_dir: Path
    artifacts_dir: Path
    logs_dir: Path
    sample_path: Path
    prompt_path: Path
    stdout_log: Path
    stderr_log: Path
    last_message_path: Path
    result_path: Path

    @property
    def agent_stdout_log(self) -> Path:
        return self.stdout_log

    @property
    def agent_stderr_log(self) -> Path:
        return self.stderr_log


@dataclass
class CodexInvocationResult:
    command: List[str]
    exit_code: int
    stdout_path: Path
    stderr_path: Path
    parsed_result: Optional[Dict[str, Any]]
    error_message: Optional[str] = None
    elapsed_seconds: Optional[float] = None
    codex_stats: Dict[str, Any] = field(default_factory=dict)


def empty_codex_stats() -> Dict[str, Any]:
    return {
        "elapsed_seconds": None,
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "turns": 0,
        "command_executions": 0,
    }


def build_case_name(repo_slug: str, pr_number: int) -> str:
    return f"{repo_tail(repo_slug)}-pr{pr_number}"


def load_scored_sample(scored_path: Path, prs_dir: Path, repo_slug: str, pr_number: int) -> Dict[str, Any]:
    """Merge a scored-review row with the PR metadata JSON; commit SHAs always come from the PR JSON."""
    matched = find_pr_record(scored_path, pr_number)
    if matched is None:
        raise FileNotFoundError(f"PR #{pr_number} not found in {scored_path}")
    pr_path = pr_metadata_path(prs_dir, repo_slug, pr_number)
    if not pr_path.exists():
        raise FileNotFoundError(f"Missing PR metadata file: {pr_path}")
    pr_record = load_json(pr_path)
    sample = dict(matched)
    sample["pr_record"] = pr_record
    for key in PR_METADATA_KEYS:
        if key in pr_record and key not in sample:
            sample[key] = pr_record[key]
    for key in ("base_sha", "head_sha"):
        if key in pr_record:
            sample[key] = pr_record[key]
    return sample


def load_sample(
    data_root: Path,
    repo_slug: str,
    pr_number: int,
    scored_dir: Path = DEFAULT_SCORED_DIR,
) -> Dict[str, Any]:
    scored_path = repo_dir(data_root / scored_dir, repo_slug) / "fused_scored.jsonl"
    return load_scored_sample(scored_path, data_root / "data" / "prs", repo_slug, pr_number)


def load_base_prompt(prompt_template_path: Path) -> str:
    text = prompt_template_path.read_text(encoding="utf-8")
    marker = text.find("\n<SAMPLE_JSON>")
    return text[: marker + 1] if marker != -1 else text


def build_case_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in sample.items() if key not in EXCLUDED_CASE_SAMPLE_KEYS}


def sample_json_block(sample: Dict[str, Any]) -> str:
    return json.dumps(build_case_sample(sample), ensure_ascii=False, indent=2, sort_keys=True)


def render_prompt(base_template: str, sample: Dict[str, Any]) -> str:
    contract = """
==================================================
8. Case Directory Rules
==================================================

You are running inside a prepared case directory with these paths:
- `repo/` : cloned repository checkout already set to the target `head_sha`
- `inputs/sample.json` : the benchmark sample JSON
- `artifacts/` : directory for build outputs, virtual environments, and cache data

Do not edit repository source code.
Do not create files outside the current case directory.
Do not assume any hidden state outside this case directory.
Create any `uv` virtual environment inside the current case directory, for example `.venv` or `artifacts/.venv`.
Use cache directories inside the current case directory as well, for example `artifacts/.uv-cache` and `artifacts/.cache`, instead of relying on `~/.cache`.
Do not give up after the first failed install or build attempt. If the default path fails, try reasonable fallback steps that stay within the case directory and are justified by repository signals.
Only stop when you have either run the most relevant patch-related test path or gathered concrete evidence that hardware, drivers, CUDA, compiler toolchain, or unavailable wheels make meaningful execution impossible.

==================================================
9. Output Contract
==================================================

Your final response must include exactly one JSON object between these marker lines:
%s
<json object here>
%s

The JSON object must contain these keys:
- `status`: one of `reproduced`, `blocked`, `failed`
- `failure_category`: string or null
- `summary`: short human-readable verdict
- `setup_commands`: array of exact shell commands actually used to prepare the environment
- `test_commands`: array of exact shell commands actually used to run tests
- `environment_summary`: short explanation of the chosen environment/build flow
- `evidence`: array of short evidence strings grounded in actual execution

If hardware or toolchain constraints block meaningful validation, set:
- `status` = `blocked`
- `failure_category` = `HARDWARE_OR_SYSTEM_REQUIREMENT_FAILURE`

If the environment cannot be prepared, choose the most specific failure category available.
Never claim success without actual command execution.

For  reproduce result, Do NOT use absolute paths.
setup_commands and test_commands must record only the final successful, directly reproducible command sequence, not the agent’s full trial-and-error history.
Do not include failed commands, exploratory/debugging commands, superseded commands, corrected attempts, or any temporary commands that are not part of the final standard reproduction flow.
If multiple attempts were made, these fields must contain only the last correct runnable path that a user could copy and run directly.

==================================================
10. Sample JSON
==================================================

```json
%s
```
""" % (RESULT_START_MARKER, RESULT_END_MARKER, sample_json_block(sample))
    return base_template.rstrip() + "\n\n" + contract.lstrip()


def prepare_case_directory(
    case_root: Path,
    repo_slug: str,
    pr_number: int,
    template_codex_dir: Optional[Path],
    sample: Dict[str, Any],
    prompt_text: str,
) -> CasePaths:
    case_dir = case_root / build_case_name(repo_slug, pr_number)
    logs_dir = case_dir / "logs"
    paths = CasePaths(
        case_dir=case_dir,
        codex_dir=case_dir / ".codex",
        repo_dir=case_dir / "repo",
        inputs_dir=case_dir / "inputs",
        artifacts_dir=case_dir / "artifacts",
        logs_dir=logs_dir,
        sample_path=case_dir / "inputs" / "sample.json",
        prompt_path=case_dir / "inputs" / "prompt.md",
        stdout_log=logs_dir / "codex.session.jsonl",
        stderr_log=logs_dir / "codex.stderr.log",
        last_message_path=logs_dir / "codex.last-message.txt",
        result_path=logs_dir / "result.json",
    )
    for path in (paths.codex_dir, paths.repo_dir, paths.inputs_dir, paths.artifacts_dir, paths.logs_dir):
        path.mkdir(parents=True, exist_ok=True)

    if template_codex_dir is not None:
        if template_codex_dir.is_dir():
            shutil.copytree(template_codex_dir, paths.codex_dir, dirs_exist_ok=True)
        else:
            print(f"[warn] template Codex directory does not exist: {template_codex_dir}")

    write_text(paths.sample_path, sample_json_block(sample) + "\n")
    write_text(paths.prompt_path, prompt_text)
    return paths


def parse_result_markers(output: str) -> Dict[str, Any]:
    pattern = re.compile(RESULT_START_MARKER + r"\s*(\{.*?\})\s*" + RESULT_END_MARKER, re.DOTALL)
    match = pattern.search(output)
    if not match:
        raise ValueError("Could not find final JSON markers in codex output")
    return json.loads(match.group(1))


def invoke_codex(case_paths: CasePaths, extra_args: Optional[Sequence[str]] = None) -> CodexInvocationResult:
    """Run Codex in the case directory and parse the marker-delimited result JSON it prints last."""
    runner = CodexAgentRunner(extra_args=["--output-last-message", str(case_paths.last_message_path), *(extra_args or [])])
    invocation = runner.invoke(case_paths)
    codex_stats = empty_codex_stats()
    if invocation.exit_code == 127 and invocation.elapsed_seconds is None:
        return CodexInvocationResult(
            command=invocation.command,
            exit_code=127,
            stdout_path=case_paths.stdout_log,
            stderr_path=case_paths.stderr_log,
            parsed_result=None,
            error_message=invocation.error_message,
            codex_stats=codex_stats,
        )

    parsed_result = None
    error_message = None
    last_message = case_paths.last_message_path
    output = last_message.read_text(encoding="utf-8") if last_message.exists() else ""
    if not output.strip():
        output = case_paths.stdout_log.read_text(encoding="utf-8") if case_paths.stdout_log.exists() else ""
    if output.strip():
        try:
            parsed_result = parse_result_markers(output)
        except (ValueError, json.JSONDecodeError) as exc:
            error_message = str(exc)

    codex_stats.update(parse_codex_session_stats(case_paths.stdout_log))
    codex_stats["elapsed_seconds"] = invocation.elapsed_seconds
    return CodexInvocationResult(
        command=invocation.command,
        exit_code=invocation.exit_code,
        stdout_path=case_paths.stdout_log,
        stderr_path=case_paths.stderr_log,
        parsed_result=parsed_result,
        error_message=error_message,
        elapsed_seconds=invocation.elapsed_seconds,
        codex_stats=codex_stats,
    )


def base_result(sample: Dict[str, Any], **overrides: Any) -> Dict[str, Any]:
    """Skeleton ``result.json`` for a failed run; callers override what they know."""
    result = {
        "repo": sample["repo"],
        "pr_number": sample["number"],
        "head_sha": sample.get("head_sha"),
        "status": "failed",
        "failure_category": "AUTOMATION_CONFIGURATION_FAILURE",
        "summary": "",
        "setup_commands": [],
        "test_commands": [],
        "environment_summary": "",
        "evidence": [],
        "codex_exit_code": None,
        "codex_command": [],
        "codex_stats": empty_codex_stats(),
    }
    result.update(overrides)
    return result


def build_result_from_invocation(sample: Dict[str, Any], invocation: CodexInvocationResult) -> Dict[str, Any]:
    codex_stats = empty_codex_stats()
    codex_stats.update(invocation.codex_stats or {})
    if invocation.elapsed_seconds is not None:
        codex_stats["elapsed_seconds"] = invocation.elapsed_seconds
    result = base_result(
        sample,
        summary=invocation.error_message or "Codex invocation failed before a structured result was produced.",
    )
    if invocation.parsed_result is not None:
        result.update(invocation.parsed_result)
    elif invocation.exit_code not in (0, 127):
        result["failure_category"] = "AUTOMATION_EXECUTION_FAILURE"
    result.update(
        {
            "codex_exit_code": invocation.exit_code,
            "codex_command": invocation.command,
            "raw_stdout_log": str(invocation.stdout_path),
            "raw_stderr_log": str(invocation.stderr_path),
            "codex_last_message_path": str(invocation.stdout_path.parent / "codex.last-message.txt"),
            "codex_stats": codex_stats,
        }
    )
    return result


def build_repository_preparation_failure(
    sample: Dict[str, Any],
    case_paths: CasePaths,
    exc: subprocess.CalledProcessError,
) -> Dict[str, Any]:
    return base_result(
        sample,
        failure_category="REPOSITORY_PREPARATION_FAILURE",
        summary="Repository preparation failed while running: " + format_called_process_error(exc),
        evidence=[str(exc)],
        raw_stdout_log=str(case_paths.stdout_log),
        raw_stderr_log=str(case_paths.stderr_log),
    )


def execute_repro(
    data_root: Path,
    repo_slug: str,
    pr_number: int,
    case_root: Path,
    prompt_template_path: Path,
    template_codex_dir: Optional[Path],
    clone_repo_fn=None,
    invoke_codex_fn: Optional[Callable[[CasePaths], CodexInvocationResult]] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    scored_dir: Path = DEFAULT_SCORED_DIR,
) -> Dict[str, Any]:
    emit = progress_callback or (lambda message: None)
    emit("Loading sample metadata")
    sample = load_sample(data_root, repo_slug, pr_number, scored_dir=scored_dir)
    emit("Rendering Codex prompt")
    prompt_text = render_prompt(load_base_prompt(prompt_template_path), sample)
    emit("Preparing case directory")
    case_paths = prepare_case_directory(
        case_root=case_root,
        repo_slug=repo_slug,
        pr_number=pr_number,
        template_codex_dir=template_codex_dir,
        sample=sample,
        prompt_text=prompt_text,
    )
    clone_repo_fn = clone_repo_fn or clone_or_update_repo
    invoke_codex_fn = invoke_codex_fn or invoke_codex
    emit("Preparing repository checkout")
    try:
        clone_repo_fn(case_paths, repo_slug, sample.get("head_sha"), sample["number"])
    except subprocess.CalledProcessError as exc:
        result = build_repository_preparation_failure(sample, case_paths, exc)
    else:
        emit("Running Codex")
        result = build_result_from_invocation(sample, invoke_codex_fn(case_paths))
    emit("Writing normalized result")
    write_json(case_paths.result_path, result)
    return result


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo", required=True)
    parser.add_argument("--pr", required=True, type=int)
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--scored-dir", default=str(DEFAULT_SCORED_DIR), help="fused-review output, relative to --data-root")
    parser.add_argument("--case-root", default="results/repro")
    parser.add_argument("--prompt-template", default="prompts/agent_explore.md")
    parser.add_argument("--template-codex-dir", default=None, help="Optional .codex template directory to copy.")


def run(args: argparse.Namespace) -> int:
    data_root = Path(args.data_root).resolve()
    result = execute_repro(
        data_root=data_root,
        repo_slug=args.repo,
        pr_number=args.pr,
        case_root=(data_root / args.case_root).resolve(),
        prompt_template_path=(data_root / args.prompt_template).resolve(),
        template_codex_dir=(data_root / args.template_codex_dir).resolve() if args.template_codex_dir else None,
        scored_dir=Path(args.scored_dir),
        progress_callback=lambda message: print(f"[repro] {message}", flush=True),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0

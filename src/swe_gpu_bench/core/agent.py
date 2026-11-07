"""Coding-agent runners (Codex, OpenCode) used by evaluation and reproduction."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional, Protocol, Sequence

from swe_gpu_bench.core.config import ModelProfile, profile_from_args
from swe_gpu_bench.core.shell import case_env


class AgentCasePaths(Protocol):
    case_dir: Path
    artifacts_dir: Path
    prompt_path: Path
    agent_stdout_log: Path
    agent_stderr_log: Path




def parse_codex_session_stats(jsonl_path: Path) -> Dict[str, Any]:
    """Parse a codex --json JSONL output and return aggregated stats.

    Aggregates across all turns (codex may emit multiple turn.completed events).
    Returns a dict with:
      input_tokens, cached_input_tokens, output_tokens,
      turns, command_executions, elapsed_seconds (wall time, if recorded externally)
    """
    stats: Dict[str, Any] = {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "turns": 0,
        "command_executions": 0,
    }
    if not jsonl_path.exists():
        return stats
    try:
        with jsonl_path.open(encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                etype = event.get("type", "")
                if etype == "turn.completed":
                    usage = event.get("usage") or {}
                    stats["input_tokens"] += usage.get("input_tokens", 0)
                    stats["cached_input_tokens"] += usage.get("cached_input_tokens", 0)
                    stats["output_tokens"] += usage.get("output_tokens", 0)
                    stats["turns"] += 1
                elif etype == "item.completed":
                    item = event.get("item") or {}
                    if item.get("type") == "command_execution":
                        stats["command_executions"] += 1
    except OSError:
        pass
    return stats


def _extract_error_message(payload: Any) -> Optional[str]:
    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            return None
        if text.startswith("{") or text.startswith("["):
            try:
                nested = json.loads(text)
            except json.JSONDecodeError:
                return text
            return _extract_error_message(nested) or text
        return text

    if isinstance(payload, dict):
        for key in ("message", "error", "data"):
            if key not in payload:
                continue
            message = _extract_error_message(payload[key])
            if message:
                return message
    return None


def parse_codex_error(jsonl_path: Path) -> Optional[str]:
    if not jsonl_path.exists():
        return None
    try:
        with jsonl_path.open(encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if event.get("type") not in {"error", "turn.failed"}:
                    continue
                message = _extract_error_message(event)
                if message:
                    return message
    except OSError:
        return None
    return None


def parse_text_error(log_path: Path) -> Optional[str]:
    if not log_path.exists():
        return None
    try:
        lines = [line.strip() for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError:
        return None
    if not lines:
        return None
    return "\n".join(lines[-2:])


def parse_opencode_error(jsonl_path: Path) -> Optional[str]:
    if not jsonl_path.exists():
        return None
    try:
        with jsonl_path.open(encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if event.get("type") != "error":
                    continue
                error = event.get("error") or {}
                data = error.get("data") or {}
                message = data.get("message") or error.get("message") or error.get("name")
                if message:
                    return str(message)
    except OSError:
        return None
    return None


def _normalize_openai_compatible_base_url(value: str) -> str:
    normalized = value.strip().strip("\"'").rstrip("/")
    if not normalized:
        raise ValueError("empty base URL")
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"invalid base URL: {value!r}")
    return normalized


def _normalize_opencode_env(env: Dict[str, str]) -> None:
    for key in ("MOONSHOT_BASE_URL", "DEEPSEEK_BASE_URL"):
        value = env.get(key)
        if not value:
            continue
        env[key] = _normalize_openai_compatible_base_url(value)


@dataclass
class AgentInvocationResult:
    command: List[str]
    exit_code: int
    stdout_path: Path
    stderr_path: Path
    error_message: Optional[str] = None
    elapsed_seconds: Optional[float] = None
    agent_stats: Dict[str, Any] = field(default_factory=dict)


class AgentRunner:
    """Base for agent runners; subclasses implement ``invoke``."""

    name = "agent"

    def invoke(
        self,
        case_paths: AgentCasePaths,
        extra_env: Optional[Dict[str, str]] = None,
    ) -> AgentInvocationResult:
        raise NotImplementedError

    def _default_env(
        self,
        case_paths: AgentCasePaths,
        extra_env: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        return case_env(case_paths.artifacts_dir, extra_env)


class CodexAgentRunner(AgentRunner):
    """Invokes ``codex exec``; pass/fail is decided afterwards by the harness."""

    name = "codex"

    def __init__(
        self,
        extra_args: Optional[Sequence[str]] = None,
        env_overrides: Optional[Dict[str, str]] = None,
    ) -> None:
        self._extra_args: List[str] = list(extra_args or [])
        self._env_overrides: Dict[str, str] = dict(env_overrides or {})

    def invoke(
        self,
        case_paths: AgentCasePaths,
        extra_env: Optional[Dict[str, str]] = None,
    ) -> AgentInvocationResult:
        prompt_text = case_paths.prompt_path.read_text(encoding="utf-8")
        command: List[str] = [
            "codex",
            "exec",
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
            "--json",
            *self._extra_args,
            prompt_text,
        ]
        env = self._default_env(case_paths, {**self._env_overrides, **(extra_env or {})})

        t0 = time.monotonic()
        try:
            with case_paths.agent_stdout_log.open("w", encoding="utf-8") as out_fh:
                with case_paths.agent_stderr_log.open("w", encoding="utf-8") as err_fh:
                    completed = subprocess.run(
                        command,
                        cwd=str(case_paths.case_dir),
                        text=True,
                        stdout=out_fh,
                        stderr=err_fh,
                        env=env,
                        check=False,
                    )
        except FileNotFoundError:
            return AgentInvocationResult(
                command=command,
                exit_code=127,
                stdout_path=case_paths.agent_stdout_log,
                stderr_path=case_paths.agent_stderr_log,
                error_message="codex executable not found",
            )
        elapsed = round(time.monotonic() - t0, 2)

        stats = parse_codex_session_stats(case_paths.agent_stdout_log)
        stats["elapsed_seconds"] = elapsed
        error_message = parse_codex_error(case_paths.agent_stdout_log)
        if error_message is None and completed.returncode != 0:
            error_message = parse_text_error(case_paths.agent_stderr_log)

        return AgentInvocationResult(
            command=command,
            exit_code=completed.returncode,
            stdout_path=case_paths.agent_stdout_log,
            stderr_path=case_paths.agent_stderr_log,
            error_message=error_message,
            elapsed_seconds=elapsed,
            agent_stats=stats,
        )


class OpenCodeAgentRunner(AgentRunner):
    """Invokes ``opencode run`` with a configured provider/model."""

    name = "opencode"

    def __init__(
        self,
        model: str,
        extra_args: Optional[Sequence[str]] = None,
        config_path: Path | None = None,
        env_overrides: Optional[Dict[str, str]] = None,
    ) -> None:
        self._model = model
        self._extra_args: List[str] = list(extra_args or [])
        self._config_path = config_path
        self._env_overrides: Dict[str, str] = dict(env_overrides or {})

    def invoke(
        self,
        case_paths: AgentCasePaths,
        extra_env: Optional[Dict[str, str]] = None,
    ) -> AgentInvocationResult:
        prompt_text = case_paths.prompt_path.read_text(encoding="utf-8")
        command: List[str] = [
            "opencode",
            "run",
            "--format",
            "json",
            "--model",
            self._model,
            *self._extra_args,
            prompt_text,
        ]
        env = self._default_env(case_paths, {**self._env_overrides, **(extra_env or {})})
        if self._config_path is not None:
            env["OPENCODE_CONFIG"] = str(self._config_path)
        try:
            _normalize_opencode_env(env)
        except ValueError as exc:
            return AgentInvocationResult(
                command=command,
                exit_code=2,
                stdout_path=case_paths.agent_stdout_log,
                stderr_path=case_paths.agent_stderr_log,
                error_message=f"invalid OpenCode provider configuration: {exc}",
            )

        t0 = time.monotonic()
        try:
            with case_paths.agent_stdout_log.open("w", encoding="utf-8") as out_fh:
                with case_paths.agent_stderr_log.open("w", encoding="utf-8") as err_fh:
                    completed = subprocess.run(
                        command,
                        cwd=str(case_paths.case_dir),
                        text=True,
                        stdout=out_fh,
                        stderr=err_fh,
                        env=env,
                        check=False,
                    )
        except FileNotFoundError:
            return AgentInvocationResult(
                command=command,
                exit_code=127,
                stdout_path=case_paths.agent_stdout_log,
                stderr_path=case_paths.agent_stderr_log,
                error_message="opencode executable not found",
            )
        elapsed = round(time.monotonic() - t0, 2)
        session_error = parse_opencode_error(case_paths.agent_stdout_log)
        exit_code = completed.returncode if session_error is None else completed.returncode or 1
        return AgentInvocationResult(
            command=command,
            exit_code=exit_code,
            stdout_path=case_paths.agent_stdout_log,
            stderr_path=case_paths.agent_stderr_log,
            error_message=session_error,
            elapsed_seconds=elapsed,
            agent_stats={"elapsed_seconds": elapsed},
        )


def _profile_value(profile: ModelProfile, env_key: str | None) -> str | None:
    if env_key is None:
        return None
    if env_key in profile.extra_env:
        return profile.extra_env[env_key]
    value = os.getenv(env_key)
    if value:
        return value
    return None


def _require_profile_value(profile: ModelProfile, env_key: str) -> str:
    value = _profile_value(profile, env_key)
    if not value:
        raise ValueError(f"Missing required environment variable {env_key!r} for agent profile {profile.name!r}")
    return value


def _opencode_model_name(profile: ModelProfile) -> str:
    if not profile.model:
        raise ValueError(f"Agent profile {profile.name!r} is missing model")
    if "/" in profile.model:
        return profile.model
    provider = profile.provider or "custom"
    return f"{provider}/{profile.model}"


def _build_opencode_profile(profile: ModelProfile) -> tuple[Path, dict[str, str]]:
    model_name = _opencode_model_name(profile)
    provider_name, model_tail = model_name.split("/", 1)

    env_overrides = dict(profile.extra_env)
    if profile.base_url is not None:
        env_overrides["SWEBENCH_OPENCODE_BASE_URL"] = profile.base_url
    if profile.api_key_env:
        env_overrides["SWEBENCH_OPENCODE_API_KEY"] = _require_profile_value(profile, profile.api_key_env)

    provider_payload = {
        provider_name: {
            "npm": "@ai-sdk/openai-compatible",
            "name": provider_name,
            "options": {
                "baseURL": "{env:SWEBENCH_OPENCODE_BASE_URL}",
                "apiKey": "{env:SWEBENCH_OPENCODE_API_KEY}",
            },
            "models": {
                model_tail: {
                    "name": model_tail,
                }
            },
        }
    }
    config_payload = {
        "$schema": "https://opencode.ai/config.json",
        "permission": {
            "webfetch": "deny",
            "websearch": "deny",
            "question": "deny",
            "task": "allow",
            "lsp": "deny",
            "skill": "allow",
            "external_directory": "allow",
            "read": "allow",
        },
        "provider": provider_payload,
    }

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix="opencode-profile-",
        delete=False,
        encoding="utf-8",
    ) as handle:
        json.dump(config_payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        return Path(handle.name), env_overrides


def build_agent_runner(
    *,
    profile: ModelProfile,
    extra_args: Optional[Sequence[str]] = None,
) -> AgentRunner:
    framework = profile.framework or "codex"
    if framework == "codex":
        runner_extra_args = list(extra_args or [])
        if profile.model:
            runner_extra_args.extend(["--model", profile.model])

        env_overrides = dict(profile.extra_env)
        if profile.base_url is not None:
            normalized = _normalize_openai_compatible_base_url(profile.base_url)
            env_overrides["OPENAI_BASE_URL"] = normalized
            env_overrides["OPENAI_API_BASE"] = normalized
        if profile.api_key_env:
            env_overrides["OPENAI_API_KEY"] = _require_profile_value(profile, profile.api_key_env)
        return CodexAgentRunner(extra_args=runner_extra_args, env_overrides=env_overrides)

    if framework == "opencode":
        config_path, env_overrides = _build_opencode_profile(profile)
        return OpenCodeAgentRunner(
            model=_opencode_model_name(profile),
            extra_args=extra_args,
            config_path=config_path,
            env_overrides=env_overrides,
        )
    raise ValueError(f"Unknown agent framework: {framework!r}")


def agent_runner_from_args(args: argparse.Namespace) -> AgentRunner:
    """Build the runner selected by :func:`core.config.add_profile_arguments` (kind="agent")."""
    return build_agent_runner(profile=profile_from_args(args, kind="agent"))

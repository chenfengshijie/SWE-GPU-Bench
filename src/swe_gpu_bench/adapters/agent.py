from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from swe_gpu_bench.shared.models import ModelProfile

if TYPE_CHECKING:
    from swe_gpu_bench.evaluate.use_cases.eval import EvalCasePaths


REPO_ROOT = Path(__file__).resolve().parents[3]
OPENCODE_CONFIG_PATH = REPO_ROOT / "opencode.json"


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


def parse_opencode_session_stats(jsonl_path: Path) -> Dict[str, Any]:
    """Normalize OpenCode step counters (text/reasoning and cache are separate)."""
    steps = []
    seen = set()
    commands = 0
    subagent_sessions = set()
    if jsonl_path.exists():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            part = event.get("part") or {}
            if event.get('type') == 'tool_use' and part.get('tool') == 'task':
                metadata = part.get('state', {}).get('metadata') or {}
                if metadata.get('sessionId'):
                    subagent_sessions.add(metadata['sessionId'])
            if event.get("type") == "tool_use" and part.get("tool") == "bash":
                commands += part.get("state", {}).get("status") == "completed"
            if event.get("type") != "step_finish":
                continue
            identity = part.get("id")
            if identity and identity in seen:
                continue
            seen.add(identity)
            tokens = part.get("tokens") or {}
            if not all(isinstance(tokens.get(k), int) for k in ("input", "output", "reasoning")):
                steps.append(None)
                continue
            cache = tokens.get("cache") or {}
            if not all(isinstance(cache.get(k), int) for k in ("read", "write")):
                steps.append(None)
                continue
            inp = tokens["input"] + cache["read"] + cache["write"]
            out = tokens["output"] + tokens["reasoning"]
            steps.append({"input_tokens": inp, "output_tokens": out,
                          "cached_input_tokens": cache["read"],
                          "cache_write_input_tokens": cache["write"],
                          "reasoning_tokens": tokens["reasoning"],
                          "total_tokens": tokens.get("total", inp + out)})
    reported = [s for s in steps if s is not None]
    result = {"source": "opencode_step_finish", "turns": len(steps),
              "command_executions": commands,
              "usage_status": "reported" if reported and len(reported) == len(steps)
                              else "partial" if reported else "missing"}
    if subagent_sessions:
        result['subagent_sessions'] = len(subagent_sessions)
        result['usage_scope'] = 'root_session_only; nested session tokens are not present in this stream'
        if reported:
            result['usage_status'] = 'partial'
    for key in ("input_tokens", "output_tokens", "cached_input_tokens",
                "cache_write_input_tokens", "reasoning_tokens", "total_tokens"):
        values = [s[key] for s in reported if isinstance(s.get(key), int)]
        result[key] = sum(values) if values else None
    return result


def parse_codex_error(jsonl_path: Path) -> Optional[str]:
    if not jsonl_path.exists():
        return None
    pending_error = None
    turn_failed = False
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
                item = event.get("item") or {}
                if (event.get("type") == "item.completed" and item.get("type") == "error"
                        and "codex-code-mode-host" in str(item.get("message", ""))):
                    return str(item["message"])
                if (event.get("type") == "turn.completed" and not turn_failed
                        and pending_error and pending_error.startswith("Reconnecting")
                        and ("stream disconnected before completion" in pending_error
                             or ("rate limit exceeded" in pending_error.lower()
                                 and "exceeded token rate limit" in pending_error.lower()))):
                    # A reconnect warning is not the final outcome when the
                    # same stream subsequently reports a completed turn.
                    pending_error = None
                if event.get("type") not in {"error", "turn.failed"}:
                    continue
                turn_failed = turn_failed or event.get("type") == "turn.failed"
                message = _extract_error_message(event)
                if message:
                    pending_error = message
    except OSError:
        return None
    return pending_error


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
    for key in ("MOONSHOT_BASE_URL", "DEEPSEEK_BASE_URL", "OPENAI_BASE_URL"):
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
    """Protocol-style base for agent runners.

    Subclasses implement ``invoke``.  No ABC so new runners can be added
    without touching this module.
    """

    def invoke(
        self,
        case_paths: "EvalCasePaths",
        extra_env: Optional[Dict[str, str]] = None,
    ) -> AgentInvocationResult:
        raise NotImplementedError

    def _default_env(
        self,
        case_paths: "EvalCasePaths",
        extra_env: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        env = dict(os.environ)
        env.pop("SWEBENCH_REPO_CACHE_ROOT", None)
        env["UV_CACHE_DIR"] = str(case_paths.artifacts_dir / ".uv-cache")
        env["XDG_CACHE_HOME"] = str(case_paths.artifacts_dir / ".cache")
        if extra_env:
            env.update(extra_env)
        return env


class CodexAgentRunner(AgentRunner):
    """Invokes ``codex exec`` for the eval task.

    Unlike the repro.py version this runner does **not** request a structured
    JSON output from the agent — pass/fail is determined externally by running
    ``test_commands`` after the agent finishes.
    """

    def __init__(
        self,
        extra_args: Optional[Sequence[str]] = None,
        env_overrides: Optional[Dict[str, str]] = None,
    ) -> None:
        self._extra_args: List[str] = list(extra_args or [])
        self._env_overrides: Dict[str, str] = dict(env_overrides or {})

    def invoke(
        self,
        case_paths: "EvalCasePaths",
    extra_env: Optional[Dict[str, str]] = None,
    ) -> AgentInvocationResult:
        from swe_gpu_bench.shared.model_pause import ensure_model_calls_allowed
        ensure_model_calls_allowed()
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
        env.setdefault("OPENAI_API_KEY", "sk-1234")
        from swe_gpu_bench.shared.solver_guard import guarded_command
        command = guarded_command(command, case_paths.case_dir, env)

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
        case_paths: "EvalCasePaths",
        extra_env: Optional[Dict[str, str]] = None,
    ) -> AgentInvocationResult:
        from swe_gpu_bench.shared.model_pause import ensure_model_calls_allowed
        ensure_model_calls_allowed()
        prompt_text = case_paths.prompt_path.read_text(encoding="utf-8")
        command: List[str] = [
            "opencode",
            "run",
            "--dir",
            str(case_paths.case_dir.resolve()),
            "--format",
            "json",
            "--model",
            self._model,
            *self._extra_args,
            prompt_text,
        ]
        env = self._default_env(case_paths, {**self._env_overrides, **(extra_env or {})})
        env["OPENCODE_CONFIG"] = str(self._config_path or OPENCODE_CONFIG_PATH)
        env["PWD"] = str(case_paths.case_dir.resolve())
        for variable, folder in (("XDG_DATA_HOME", "data"), ("XDG_CONFIG_HOME", "config"),
                                 ("XDG_STATE_HOME", "state")):
            env[variable] = str(case_paths.artifacts_dir / ".opencode" / folder)
        env["OPENCODE_DISABLE_AUTOUPDATE"] = "true"
        from swe_gpu_bench.shared.solver_guard import guarded_command
        command = guarded_command(command, case_paths.case_dir, env)
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
        stats = parse_opencode_session_stats(case_paths.agent_stdout_log)
        stats["elapsed_seconds"] = elapsed
        return AgentInvocationResult(
            command=command,
            exit_code=exit_code,
            stdout_path=case_paths.agent_stdout_log,
            stderr_path=case_paths.agent_stderr_log,
            error_message=session_error,
            elapsed_seconds=elapsed,
            agent_stats=stats,
        )


def get_agent_runner(name: str, **kwargs) -> AgentRunner:
    """Factory that returns an ``AgentRunner`` by name."""
    if name == "codex":
        return CodexAgentRunner(**kwargs)
    if name == "opencode-kimi":
        return OpenCodeAgentRunner(model="moonshot/kimi-k2.5", **kwargs)
    if name == "opencode-deepseek":
        return OpenCodeAgentRunner(model="deepseek/deepseek-reasoner", **kwargs)
    raise ValueError(f"Unknown agent runner: {name!r}")


def _profile_value(profile: ModelProfile, env_key: str | None) -> str | None:
    if env_key is None:
        return None
    if env_key in profile.extra_env:
        return profile.extra_env[env_key]
    value = os.getenv(env_key)
    if value:
        return value
    return None


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
        api_key = _profile_value(profile, profile.api_key_env)
        if api_key is not None:
            env_overrides["SWEBENCH_OPENCODE_API_KEY"] = api_key

    provider_payload = {
        provider_name: {
            "npm": "@ai-sdk/openai-compatible",
            "name": provider_name,
            "options": {
                "baseURL": "{env:SWEBENCH_OPENCODE_BASE_URL}",
                "apiKey": "{env:SWEBENCH_OPENCODE_API_KEY}",
                "includeUsage": True,
            },
            "models": {
                model_tail: {
                    "name": model_tail,
                    **({"options": {"reasoningEffort": profile.reasoning_effort}}
                       if profile.reasoning_effort is not None else {}),
                }
            },
        }
    }
    config_payload = {
        "$schema": "https://opencode.ai/config.json",
        "permission": {
            "webfetch": "allow",
            "websearch": "allow",
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
        if profile.reasoning_effort is not None:
            runner_extra_args.extend(["-c", f'model_reasoning_effort="{profile.reasoning_effort}"'])
        if (profile.provider or "").replace("-", "_") == "openai_responses":
            if not profile.base_url or not profile.api_key_env:
                raise ValueError("Responses agent profile requires base_url and api_key_env")
            for setting in (
                'model_provider="swe_gpu_bench"',
                'model_providers.swe_gpu_bench.name="SWE-GPU-Bench"',
                'model_providers.swe_gpu_bench.wire_api="responses"',
                'model_providers.swe_gpu_bench.base_url=' + json.dumps(profile.base_url),
                'model_providers.swe_gpu_bench.env_key=' + json.dumps(profile.api_key_env),
            ):
                runner_extra_args.extend(["-c", setting])

        env_overrides = dict(profile.extra_env)
        if profile.base_url is not None:
            normalized = _normalize_openai_compatible_base_url(profile.base_url)
            env_overrides["OPENAI_BASE_URL"] = normalized
            env_overrides["OPENAI_API_BASE"] = normalized
        if profile.api_key_env:
            api_key = _profile_value(profile, profile.api_key_env)
            if api_key is not None:
                env_overrides["OPENAI_API_KEY"] = api_key
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

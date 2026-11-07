from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from swe_gpu_bench.shared.models import ModelProfile

DEFAULT_API_KEY = ""  # Credentials must come from the existing provider environment.
DEFAULT_AZURE_ENDPOINT = ""
DEFAULT_API_VERSION = "2024-02-01"
DEFAULT_MODEL = "gpt-5.1-2025-11-13"
DEFAULT_EVAL_ENV_FILE = Path(".env")


@dataclass(frozen=True)
class GenerationTarget:
    provider: str
    model_name: str
    api_key: str
    base_url: str | None = None
    api_version: str | None = None
    temperature: float = 0
    default_max_tokens: int | None = None


class ContextLengthExceededError(RuntimeError):
    pass


def _is_unsupported_temperature_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "temperature" in message and "unsupported parameter" in message


KNOWN_MODEL_SPECS = {
    "gpt-default": {
        "provider": "azure-openai",
        "model_name": os.getenv("OPENAI_MODEL", "gpt-5.2-2025-12-11"),
    },
    "kimi": {
        "provider": "openai-compatible",
        "model_name": "moonshot/kimi-k2.5",
        "base_url_env": "MOONSHOT_BASE_URL",
        "api_key_env": "MOONSHOT_API_KEY",
        "temperature": 1,
        "default_max_tokens": 16384,
    },
    "deepseek": {
        "provider": "openai-compatible",
        "model_name": "deepseek-reasoner",
        "base_url_env": "DEEPSEEK_BASE_URL",
        "api_key_env": "DEEPSEEK_API_KEY",
        "default_max_tokens": 8192,
    },
    "glm": {
        "provider": "openai-compatible",
        "model_name_env": "GLM_MODEL",
        "base_url_env": "GLM_BASE_URL",
        "api_key_env": "GLM_API_KEY",
    },
}

_client = None


def _parse_env_file(env_file: Path | None) -> dict[str, str]:
    if env_file is None or not env_file.exists():
        return {}
    values: dict[str, str] = {}
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


def _merged_env(
    env_file: Path | None = None,
    env_overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    merged = _parse_env_file(env_file)
    merged.update({key: value for key, value in os.environ.items() if value})
    if env_overrides:
        merged.update(dict(env_overrides))
    return merged


def get_client():
    global _client
    if _client is None:
        from openai import AzureOpenAI

        _client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY", DEFAULT_API_KEY),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", DEFAULT_API_VERSION),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", DEFAULT_AZURE_ENDPOINT),
        )
    return _client


def _require_env_values(merged_env: Mapping[str, str], *keys: str) -> list[str]:
    missing = [key for key in keys if not merged_env.get(key)]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
    return [merged_env[key] for key in keys]


def resolve_generation_target(
    model_spec: str,
    *,
    env_file: Path | None = None,
    env_overrides: Mapping[str, str] | None = None,
) -> GenerationTarget:
    merged_env = _merged_env(env_file=env_file, env_overrides=env_overrides)
    spec = KNOWN_MODEL_SPECS.get(model_spec)
    if spec is None:
        return GenerationTarget(
            provider="azure-openai",
            model_name=model_spec,
            api_key=merged_env.get("AZURE_OPENAI_API_KEY", DEFAULT_API_KEY),
            base_url=merged_env.get("AZURE_OPENAI_ENDPOINT", DEFAULT_AZURE_ENDPOINT),
            api_version=merged_env.get("AZURE_OPENAI_API_VERSION", DEFAULT_API_VERSION),
        )

    if spec["provider"] == "azure-openai":
        return GenerationTarget(
            provider="azure-openai",
            model_name=spec["model_name"],
            api_key=merged_env.get("AZURE_OPENAI_API_KEY", DEFAULT_API_KEY),
            base_url=merged_env.get("AZURE_OPENAI_ENDPOINT", DEFAULT_AZURE_ENDPOINT),
            api_version=merged_env.get("AZURE_OPENAI_API_VERSION", DEFAULT_API_VERSION),
        )

    api_key, base_url = _require_env_values(
        merged_env,
        spec["api_key_env"],
        spec["base_url_env"],
    )
    model_name = spec.get("model_name")
    if model_name is None:
        model_name, = _require_env_values(merged_env, spec["model_name_env"])

    return GenerationTarget(
        provider="openai-compatible",
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=spec.get("temperature", 0),
        default_max_tokens=spec.get("default_max_tokens"),
    )


def generate_text(
    prompt: str,
    *,
    model_spec: str = "gpt-default",
    model_name: str | None = None,
    max_tokens: int = 1024,
    env_file: Path | None = None,
    env_overrides: Mapping[str, str] | None = None,
) -> str | None:
    target = resolve_generation_target(
        model_spec,
        env_file=env_file,
        env_overrides=env_overrides,
    )
    if model_name is not None:
        target = GenerationTarget(
            provider=target.provider,
            model_name=model_name,
            api_key=target.api_key,
            base_url=target.base_url,
            api_version=target.api_version,
            temperature=target.temperature,
            default_max_tokens=target.default_max_tokens,
        )
    messages = [{"role": "user", "content": prompt}]

    result = None
    last_exc: Exception | None = None
    for _ in range(3):
        from swe_gpu_bench.shared.model_pause import ensure_model_calls_allowed
        ensure_model_calls_allowed()
        try:
            if target.provider == "azure-openai":
                from openai import AzureOpenAI

                client = AzureOpenAI(
                    api_key=target.api_key,
                    api_version=target.api_version or DEFAULT_API_VERSION,
                    azure_endpoint=target.base_url or DEFAULT_AZURE_ENDPOINT,
                )
            else:
                from openai import OpenAI

                client = OpenAI(api_key=target.api_key, base_url=target.base_url)

            completion = client.chat.completions.create(
                model=target.model_name,
                messages=messages,
                max_tokens=max(max_tokens, target.default_max_tokens or max_tokens),
                temperature=target.temperature,
            )
            result = completion.choices[0].message.content
            break
        except Exception as exc:
            last_exc = exc
            if "context_length_exceeded" in str(exc):
                raise ContextLengthExceededError(str(exc)) from exc
            print(
                f"[llm] {target.provider} request failed for model {target.model_name}: {exc}",
                file=sys.stderr,
            )
            time.sleep(5)
    if result is None and last_exc is not None:
        raise last_exc
    return result


def _profile_env(
    profile: ModelProfile,
    *,
    env_file: Path | None = DEFAULT_EVAL_ENV_FILE,
    env_overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    merged_env = _merged_env(env_file=env_file, env_overrides=env_overrides)
    merged_env.update(profile.extra_env)
    return merged_env


def _profile_api_key(
    profile: ModelProfile,
    *,
    env_file: Path | None = DEFAULT_EVAL_ENV_FILE,
    env_overrides: Mapping[str, str] | None = None,
) -> str:
    merged_env = _profile_env(profile, env_file=env_file, env_overrides=env_overrides)
    if profile.api_key_env:
        value = merged_env.get(profile.api_key_env)
        if value:
            return value
    return DEFAULT_API_KEY


def generate_text_from_profile(
    prompt: str,
    *,
    profile: ModelProfile,
    max_tokens: int = 1024,
    env_file: Path | None = DEFAULT_EVAL_ENV_FILE,
    env_overrides: Mapping[str, str] | None = None,
    request_timeout: float | None = None,
    usage_callback: Callable[[dict], None] | None = None,
    max_attempts: int = 3,
) -> str | None:
    messages = [{"role": "user", "content": prompt}]
    provider = (profile.provider or "openai_compatible").replace("-", "_")
    if profile.reasoning_effort is not None and provider != "openai_responses":
        raise ValueError("Explicit reasoning_effort currently requires the openai_responses provider")

    result = None
    last_exc: Exception | None = None
    client_options = {"timeout": request_timeout, "max_retries": 0} if request_timeout is not None else {}
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    for _ in range(max_attempts):
        from swe_gpu_bench.shared.model_pause import ensure_model_calls_allowed
        ensure_model_calls_allowed()
        request_started = time.monotonic()
        try:
            if provider == "azure_openai":
                from openai import AzureOpenAI

                client = AzureOpenAI(
                    api_key=_profile_api_key(profile, env_file=env_file, env_overrides=env_overrides),
                    api_version=DEFAULT_API_VERSION,
                    azure_endpoint=profile.base_url or DEFAULT_AZURE_ENDPOINT,
                    **client_options,
                )
            else:
                from openai import OpenAI

                client = OpenAI(
                    api_key=_profile_api_key(profile, env_file=env_file, env_overrides=env_overrides),
                    base_url=profile.base_url,
                    **client_options,
                )

            if provider == "openai_responses":
                completion = client.responses.create(
                    model=profile.model or DEFAULT_MODEL,
                    input=prompt,
                    max_output_tokens=max_tokens,
                    store=False,
                    **({"reasoning": {"effort": profile.reasoning_effort}} if profile.reasoning_effort is not None else {}),
                )
                result = completion.output_text
            else:
                request_kwargs = {
                    "model": profile.model or DEFAULT_MODEL,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": 0,
                }
                try:
                    completion = client.chat.completions.create(**request_kwargs)
                except Exception as exc:
                    if not _is_unsupported_temperature_error(exc):
                        raise
                    request_kwargs.pop("temperature", None)
                    completion = client.chat.completions.create(**request_kwargs)
                result = completion.choices[0].message.content
        except Exception as exc:
            last_exc = exc
            if usage_callback is not None:
                usage_callback({'source': provider, 'error_type': type(exc).__name__,
                                'elapsed_seconds': round(time.monotonic() - request_started, 3)})
            if "context_length_exceeded" in str(exc):
                raise ContextLengthExceededError(str(exc)) from exc
            print(
                f"[llm] profile request failed for model {profile.model}: {type(exc).__name__}",
                file=sys.stderr,
            )
            if _ + 1 < max_attempts:
                time.sleep(5)
        else:
            if usage_callback is not None:
                from swe_gpu_bench.shared.usage import response_usage
                usage_callback(response_usage(completion, round(time.monotonic() - request_started, 3)))
            break
    if result is None and last_exc is not None:
        raise last_exc
    return result


def gpt_rsp(prompt: str, model_name: str) -> str | None:
    return generate_text(prompt, model_spec=model_name)

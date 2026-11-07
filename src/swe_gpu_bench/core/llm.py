"""Chat-completion client driven by text profiles, plus response helpers."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

from swe_gpu_bench.core.config import ModelProfile

DEFAULT_MODEL = "gpt-5.2-2025-12-11"
DEFAULT_ENV_FILE = Path(".env")
DEFAULT_AZURE_API_VERSION = "2024-02-01"


class ContextLengthExceededError(RuntimeError):
    pass


def read_env_file(env_file: Path | None) -> dict[str, str]:
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


def _profile_env(profile: ModelProfile, env_file: Path | None) -> dict[str, str]:
    merged = read_env_file(env_file)
    merged.update({key: value for key, value in os.environ.items() if value})
    merged.update(profile.extra_env)
    return merged


def resolve_model(profile: ModelProfile | None) -> str:
    return (profile.model if profile else None) or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL


def profile_label(profile: ModelProfile | None) -> str:
    """Stable directory-friendly name for a profile's outputs."""
    if profile is not None and profile.name and profile.name != "inline":
        return profile.name
    return resolve_model(profile).replace("/", "_")


def _build_client(profile: ModelProfile, env: Mapping[str, str]):
    api_key_env = profile.api_key_env or "OPENAI_API_KEY"
    api_key = env.get(api_key_env)
    if not api_key:
        raise ValueError(f"Missing required environment variable: {api_key_env}")
    provider = (profile.provider or "openai_compatible").replace("-", "_")
    if provider == "azure_openai":
        from openai import AzureOpenAI

        if not profile.base_url:
            raise ValueError(f"Azure profile {profile.name!r} must define base_url")
        return AzureOpenAI(
            api_key=api_key,
            api_version=env.get("AZURE_OPENAI_API_VERSION") or DEFAULT_AZURE_API_VERSION,
            azure_endpoint=profile.base_url,
        )
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=profile.base_url or env.get("OPENAI_BASE_URL") or None)


def _is_unsupported_temperature_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "temperature" in message and "unsupported parameter" in message


def complete(
    prompt: str,
    *,
    profile: ModelProfile | None = None,
    max_tokens: int = 4096,
    env_file: Path | None = DEFAULT_ENV_FILE,
    attempts: int = 3,
    retry_delay_seconds: float = 5.0,
) -> str | None:
    """Send one user message and return the assistant text, retrying transient failures."""
    profile = profile or ModelProfile(name="inline", kind="text")
    env = _profile_env(profile, env_file)
    model = resolve_model(profile)
    last_exc: Exception | None = None
    for _ in range(attempts):
        try:
            client = _build_client(profile, env)
            request: dict[str, Any] = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0,
            }
            try:
                completion = client.chat.completions.create(**request)
            except Exception as exc:
                if not _is_unsupported_temperature_error(exc):
                    raise
                request.pop("temperature")
                completion = client.chat.completions.create(**request)
            return completion.choices[0].message.content
        except ValueError:
            raise
        except Exception as exc:
            if "context_length_exceeded" in str(exc):
                raise ContextLengthExceededError(str(exc)) from exc
            last_exc = exc
            print(f"[llm] request failed for model {model}: {exc}", file=sys.stderr)
            time.sleep(retry_delay_seconds)
    if last_exc is not None:
        raise last_exc
    return None


def trim_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[TRUNCATED]\n"


def parse_json_response(text: str | None) -> dict[str, Any]:
    if not text:
        return {"parse_error": "empty_response", "raw_response": text}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and start < end:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
        return {"parse_error": "invalid_json", "raw_response": text}


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Writing parquet output requires pyarrow.") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)

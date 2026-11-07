"""Model profiles loaded from ``swe_gpu_bench.toml``."""
from __future__ import annotations

import argparse
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = Path("swe_gpu_bench.toml")


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


def _profile_key(profile_kind: str, profile_name: str) -> str:
    return f"{profile_kind}.{profile_name}"


def _build_profile(
    profile_kind: str,
    profile_name: str,
    payload: dict[str, Any],
) -> ModelProfile:
    extra_env = payload.get("extra_env") or {}
    if not isinstance(extra_env, dict):
        raise ValueError(
            f"Profile {profile_kind}.{profile_name} extra_env must be a table/dict, got {type(extra_env).__name__}"
        )
    normalized_env = {str(key): str(value) for key, value in extra_env.items()}
    return ModelProfile(
        name=profile_name,
        kind=profile_kind,
        framework=payload.get("framework"),
        provider=payload.get("provider"),
        model=payload.get("model"),
        base_url=payload.get("base_url"),
        api_key_env=payload.get("api_key_env"),
        extra_env=normalized_env,
    )


def load_profiles(config_path: Path | None = None) -> dict[str, ModelProfile]:
    path = config_path or DEFAULT_CONFIG_PATH
    if not path.exists():
        return {}

    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    profiles_root = raw.get("profiles") or {}
    loaded: dict[str, ModelProfile] = {}

    for profile_kind in ("agent", "text"):
        kind_payload = profiles_root.get(profile_kind) or {}
        if not isinstance(kind_payload, dict):
            raise ValueError(f"profiles.{profile_kind} must be a table")
        for profile_name, profile_payload in kind_payload.items():
            if not isinstance(profile_payload, dict):
                raise ValueError(f"profiles.{profile_kind}.{profile_name} must be a table")
            loaded[_profile_key(profile_kind, profile_name)] = _build_profile(
                profile_kind=profile_kind,
                profile_name=profile_name,
                payload=profile_payload,
            )
    return loaded


def resolve_profile(
    *,
    profile_name: str | None,
    profile_kind: str,
    config_path: Path | None = None,
    framework: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key_env: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> ModelProfile:
    profiles = load_profiles(config_path)
    resolved: ModelProfile | None = None
    if profile_name:
        key = _profile_key(profile_kind, profile_name)
        resolved = profiles.get(key)
        if resolved is None:
            raise KeyError(f"Unknown {profile_kind} profile: {profile_name}")

    merged_env = dict(resolved.extra_env if resolved is not None else {})
    merged_env.update({str(key): str(value) for key, value in (extra_env or {}).items()})

    return ModelProfile(
        name=profile_name or "inline",
        kind=profile_kind,
        framework=framework if framework is not None else (resolved.framework if resolved else None),
        provider=provider if provider is not None else (resolved.provider if resolved else None),
        model=model if model is not None else (resolved.model if resolved else None),
        base_url=base_url if base_url is not None else (resolved.base_url if resolved else None),
        api_key_env=api_key_env if api_key_env is not None else (resolved.api_key_env if resolved else None),
        extra_env=merged_env,
    )


def add_profile_arguments(parser: argparse.ArgumentParser, *, kind: str) -> None:
    """Register the shared ``--profile``/override flags for an agent or text model."""
    parser.add_argument("--profile", default=None, help=f"[profiles.{kind}.<name>] entry in --config")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    if kind == "agent":
        parser.add_argument("--framework", choices=["codex", "opencode"], default=None)
    else:
        parser.add_argument("--provider", choices=["openai_compatible", "azure_openai"], default=None)
        parser.add_argument("--env-file", default=".env", help="Optional dotenv file with API keys")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default=None)


def profile_from_args(args: argparse.Namespace, *, kind: str) -> ModelProfile:
    return resolve_profile(
        profile_name=args.profile,
        profile_kind=kind,
        config_path=Path(args.config),
        framework=getattr(args, "framework", None),
        provider=getattr(args, "provider", None),
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
    )

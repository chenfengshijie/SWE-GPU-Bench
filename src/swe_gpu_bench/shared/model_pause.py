"""Shared campaign cooldowns checked before starting another model invocation."""
import json
import os
from pathlib import Path
import time
import uuid


class ModelCallsPausedError(RuntimeError):
    pass


def rate_limit_delay(manifest: dict, model: str, failure_index: int) -> float:
    """Configured minute-window backoff; defaults preserve older campaigns."""
    policy = manifest.get('model_rate_limit_policies', {}).get(model, {})
    base = float(policy.get('cooldown_base_seconds', 900))
    maximum = float(policy.get('cooldown_max_seconds', 2700))
    if not 60 <= base <= maximum:
        raise ValueError('Invalid model rate-limit cooldown policy')
    return min(maximum, base * max(1, min(int(failure_index), 3)))


def pause_until(path: Path) -> float:
    # Some shared filesystems do not implement flock. Immutable events plus max
    # avoid lost updates without locks, including concurrent publishers.
    paths = [path, *path.with_suffix('.events').glob('*.json')]
    values = []
    for candidate in paths:
        if not candidate.is_file():
            continue
        # A shared-mount index replacement may briefly disappear after stat.
        # Retry that read; never treat unreadable cooldown evidence as expiry.
        for attempt in range(3):
            try:
                values.append(float(json.loads(candidate.read_text()).get('until', 0)))
                break
            except (FileNotFoundError, json.JSONDecodeError) as error:
                if attempt == 2:
                    blocked = ModelCallsPausedError('Cooldown evidence temporarily unavailable')
                    blocked.artifact_path = str(candidate)
                    raise blocked from error
                time.sleep(0.05 * (attempt + 1))
    return max(values, default=0.0)


def ensure_model_calls_allowed() -> None:
    path = os.environ.get('SWEBENCH_MODEL_PAUSE_FILE')
    if path and pause_until(Path(path)) > time.time():
        raise ModelCallsPausedError('Campaign model cooldown is active')


def publish_pause(path: Path, value: dict) -> None:
    from swe_gpu_bench.shared.artifacts import atomic_json
    path.parent.mkdir(parents=True, exist_ok=True)
    event = path.with_suffix('.events') / (uuid.uuid4().hex + '.json')
    atomic_json(event, value)
    # Compatibility/index file; readers always include immutable events, so
    # a late writer replacing this index cannot shorten the effective pause.
    atomic_json(path, {**value, 'until': max(pause_until(path), value['until'])})

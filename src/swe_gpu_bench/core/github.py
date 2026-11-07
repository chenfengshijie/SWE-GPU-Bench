"""Minimal GitHub REST client with token rotation and rate-limit backoff."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from itertools import count
from threading import Lock
from typing import Any
from urllib import error, parse, request

RETRYABLE_STATUS = {500, 502, 503, 504}


class GitHubAPIError(RuntimeError):
    """Raised when the GitHub API returns a non-success response."""


def parse_tokens(token: str | None = None, tokens: str | None = None) -> list[str]:
    """Collect tokens from ``--tokens a,b``, ``--token`` or ``GITHUB_TOKEN``."""
    values = [value.strip() for value in (tokens or "").split(",") if value.strip()]
    if not values:
        single = token or os.environ.get("GITHUB_TOKEN")
        if single:
            values = [single]
    return values


def _int_header(headers: Any, key: str) -> int | None:
    value = headers.get(key) if headers else None
    return int(value) if value and str(value).isdigit() else None


def _read_error_body(exc: error.HTTPError) -> str:
    try:
        body = exc.read()
    except Exception:
        return ""
    if isinstance(body, bytes):
        return body.decode("utf-8", errors="replace")
    return body if isinstance(body, str) else ""


@dataclass
class _TokenState:
    token: str
    available_at: float = 0.0


class GitHubTokenPool:
    """Round-robin over tokens, skipping ones that are cooling down after a rate limit."""

    def __init__(self, tokens: list[str], *, default_wait_seconds: int = 300, reset_padding_seconds: int = 5) -> None:
        active = [token for token in tokens if token]
        if not active:
            raise ValueError("GitHubTokenPool requires at least one token")
        self._states = [_TokenState(token=value) for value in active]
        self._default_wait_seconds = default_wait_seconds
        self._reset_padding_seconds = reset_padding_seconds
        self._lock = Lock()
        self._rr_counter = count()

    def __len__(self) -> int:
        return len(self._states)

    def acquire(self) -> str:
        while True:
            with self._lock:
                now = time.time()
                available = [state for state in self._states if state.available_at <= now]
                if available:
                    return available[next(self._rr_counter) % len(available)].token
                sleep_for = max(min(state.available_at for state in self._states) - now, 1.0)
            time.sleep(sleep_for)

    def wait_seconds(self, *, reset_epoch: int | None = None, retry_after_seconds: int | None = None) -> int:
        if retry_after_seconds is not None:
            return max(retry_after_seconds, 1)
        if reset_epoch is not None:
            return max(reset_epoch - int(time.time()) + self._reset_padding_seconds, 1)
        return self._default_wait_seconds

    def backoff_token(self, token: str, *, reset_epoch: int | None = None, retry_after_seconds: int | None = None) -> int:
        wait = self.wait_seconds(reset_epoch=reset_epoch, retry_after_seconds=retry_after_seconds)
        with self._lock:
            for state in self._states:
                if state.token == token:
                    state.available_at = max(state.available_at, time.time() + wait)
        return wait

    def mark_from_headers(self, token: str, headers: Any) -> None:
        if headers and headers.get("X-RateLimit-Remaining") == "0":
            self.backoff_token(token, reset_epoch=_int_header(headers, "X-RateLimit-Reset"))


@dataclass
class GitHubClient:
    token: str | None = None
    tokens: list[str] | None = None
    base_url: str = "https://api.github.com"
    retry_delay: int = 60
    rate_limit_retry_delay: int = 300
    max_retries: int = 3
    request_timeout: int = 30
    user_agent: str = "SWE-GPU-bench"

    def __post_init__(self) -> None:
        active = [value for value in (self.tokens or []) if value] or ([self.token] if self.token else [])
        self._token_pool = (
            GitHubTokenPool(active, default_wait_seconds=self.rate_limit_retry_delay) if active else None
        )

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        query = "?" + parse.urlencode(params) if params else ""
        body = self._request(f"{self.base_url}{path}{query}", accept="application/vnd.github+json")
        return json.loads(body)

    def get_text(self, url: str, *, accept: str = "text/plain") -> str:
        return self._request(url, accept=accept)

    def _wait_for_retry(self, token: str | None, *, reset_epoch: int | None, retry_after: int | None) -> None:
        pool = self._token_pool
        if pool is not None and token is not None and len(pool) > 1:
            pool.backoff_token(token, reset_epoch=reset_epoch, retry_after_seconds=retry_after)
            return
        if pool is not None:
            time.sleep(pool.wait_seconds(reset_epoch=reset_epoch, retry_after_seconds=retry_after))
        else:
            time.sleep(retry_after or self.retry_delay)

    def _request(self, url: str, *, accept: str) -> str:
        for attempt in range(self.max_retries + 1):
            token = self._token_pool.acquire() if self._token_pool else None
            headers = {"Accept": accept, "User-Agent": self.user_agent, "X-GitHub-Api-Version": "2022-11-28"}
            if token:
                headers["Authorization"] = f"Bearer {token}"
            req = request.Request(url, headers=headers)
            try:
                with request.urlopen(req, timeout=self.request_timeout) as response:
                    if self._token_pool is not None and token:
                        self._token_pool.mark_from_headers(token, getattr(response, "headers", {}))
                    return response.read().decode("utf-8", errors="replace")
            except error.HTTPError as exc:
                body = _read_error_body(exc)
                if attempt < self.max_retries:
                    if exc.code == 403 and "rate limit" in body.lower():
                        self._wait_for_retry(
                            token,
                            reset_epoch=_int_header(exc.headers, "X-RateLimit-Reset"),
                            retry_after=_int_header(exc.headers, "Retry-After"),
                        )
                        continue
                    if exc.code == 429:
                        self._wait_for_retry(
                            token,
                            reset_epoch=None,
                            retry_after=_int_header(exc.headers, "Retry-After") or self.retry_delay,
                        )
                        continue
                    if exc.code in RETRYABLE_STATUS:
                        time.sleep(self.retry_delay)
                        continue
                raise GitHubAPIError(f"GitHub request failed with status {exc.code}: {body}") from exc
            except error.URLError as exc:
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                    continue
                raise GitHubAPIError(f"GitHub request failed: {exc.reason}") from exc
        raise GitHubAPIError("GitHub request failed after retries")

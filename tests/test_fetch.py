import time
from urllib.error import HTTPError

import pytest

from swe_gpu_bench.core.github import GitHubAPIError, GitHubClient
from swe_gpu_bench.collect.fetch import collect_repo_pull_requests


class FakeGitHubClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def get_json(self, path, params=None):
        normalized_params = tuple(sorted((params or {}).items()))
        self.calls.append((path, dict(normalized_params)))
        key = (path, normalized_params)
        if key not in self.responses:
            raise AssertionError(f"Unexpected request: {path} {dict(normalized_params)}")
        return self.responses[key]


def test_collect_repo_pull_requests_writes_paginated_results(tmp_path):
    output_dir = tmp_path / "out"
    client = FakeGitHubClient(
        {
            (
                "/repos/acme/project/pulls",
                (("page", 1), ("per_page", 100), ("state", "closed")),
            ): [
                {
                    "number": 101,
                    "title": "Fix kernel launch",
                    "body": "Resolves launch issue",
                    "state": "closed",
                    "draft": False,
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-02T00:00:00Z",
                    "closed_at": "2024-01-03T00:00:00Z",
                    "merged_at": "2024-01-03T00:00:00Z",
                    "html_url": "https://github.com/acme/project/pull/101",
                    "user": {"login": "alice"},
                    "base": {"ref": "main", "sha": "base-sha", "repo": {"default_branch": "main"}},
                    "head": {"ref": "fix-branch", "sha": "head-sha"},
                    "commits": 2,
                    "comments": 3,
                    "review_comments": 4,
                    "additions": 10,
                    "deletions": 5,
                    "changed_files": 2,
                    "diff_url": "https://github.com/acme/project/pull/101.diff",
                    "patch_url": "https://github.com/acme/project/pull/101.patch",
                }
            ],
            (
                "/repos/acme/project/pulls",
                (("page", 2), ("per_page", 100), ("state", "closed")),
            ): [],
            (
                "/repos/acme/project/pulls/101/commits",
                (("page", 1), ("per_page", 100)),
            ): [
                {"commit": {"message": "Fix kernel launch"}},
                {"commit": {"message": "Add regression test"}},
            ],
            (
                "/repos/acme/project/pulls/101/commits",
                (("page", 2), ("per_page", 100)),
            ): [],
        }
    )

    written = collect_repo_pull_requests(client, "acme/project", output_dir)

    assert written == 1
    pr_file = output_dir / "acme__project" / "101.json"
    assert pr_file.exists()
    payload = pr_file.read_text()
    assert '"repo": "acme/project"' in payload
    assert '"number": 101' in payload
    assert '"commit_messages": [' in payload
    assert '"user_login": "alice"' in payload
    assert '"default_branch": "main"' in payload
    assert '"base_ref": "main"' in payload
    assert '"head_ref": "fix-branch"' in payload
    assert '"base_sha": "base-sha"' in payload
    assert '"head_sha": "head-sha"' in payload
    assert '"closed_at": "2024-01-03T00:00:00Z"' in payload
    assert '"draft": false' in payload
    assert '"commits_count": 2' in payload
    assert '"comments_count": 3' in payload
    assert '"review_comments_count": 4' in payload
    assert '"additions": 10' in payload
    assert '"deletions": 5' in payload
    assert '"changed_files": 2' in payload
    assert '"diff_url": "https://github.com/acme/project/pull/101.diff"' in payload
    assert '"patch_url": "https://github.com/acme/project/pull/101.patch"' in payload
    assert "Fix kernel launch" in payload
    assert "Add regression test" in payload


def test_collect_repo_pull_requests_skips_existing_files(tmp_path):
    output_dir = tmp_path / "out"
    repo_dir = output_dir / "acme__project"
    repo_dir.mkdir(parents=True)
    existing_file = repo_dir / "101.json"
    existing_file.write_text('{"cached": true}\n')

    client = FakeGitHubClient(
        {
            (
                "/repos/acme/project/pulls",
                (("page", 1), ("per_page", 100), ("state", "closed")),
            ): [
                {
                    "number": 101,
                    "title": "Fix kernel launch",
                    "body": "Resolves launch issue",
                    "state": "closed",
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-02T00:00:00Z",
                    "merged_at": "2024-01-03T00:00:00Z",
                    "html_url": "https://github.com/acme/project/pull/101",
                }
            ],
            (
                "/repos/acme/project/pulls",
                (("page", 2), ("per_page", 100), ("state", "closed")),
            ): [],
        }
    )

    written = collect_repo_pull_requests(client, "acme/project", output_dir)

    assert written == 0
    assert existing_file.read_text() == '{"cached": true}\n'
    assert not any(path.endswith("/commits") for path, _ in client.calls)


def test_collect_repo_pull_requests_writes_one_directory_per_repo(tmp_path):
    output_dir = tmp_path / "out"
    client = FakeGitHubClient(
        {
            (
                "/repos/acme/project/pulls",
                (("page", 1), ("per_page", 100), ("state", "closed")),
            ): [
                {
                    "number": 7,
                    "title": "Patch scheduler",
                    "body": "",
                    "state": "closed",
                    "created_at": "2024-02-01T00:00:00Z",
                    "updated_at": "2024-02-02T00:00:00Z",
                    "merged_at": "2024-02-02T00:00:00Z",
                    "html_url": "https://github.com/acme/project/pull/7",
                }
            ],
            (
                "/repos/acme/project/pulls",
                (("page", 2), ("per_page", 100), ("state", "closed")),
            ): [],
            (
                "/repos/acme/project/pulls/7/commits",
                (("page", 1), ("per_page", 100)),
            ): [{"commit": {"message": "Patch scheduler"}}],
            (
                "/repos/acme/project/pulls/7/commits",
                (("page", 2), ("per_page", 100)),
            ): [],
        }
    )

    collect_repo_pull_requests(client, "acme/project", output_dir)

    assert (output_dir / "acme__project").is_dir()
    assert (output_dir / "acme__project" / "7.json").exists()


def test_collect_repo_pull_requests_respects_max_pulls_and_cutoff_date(tmp_path):
    output_dir = tmp_path / "out"
    client = FakeGitHubClient(
        {
            (
                "/repos/acme/project/pulls",
                (("page", 1), ("per_page", 100), ("state", "closed")),
            ): [
                {
                    "number": 11,
                    "title": "Newest",
                    "body": "",
                    "state": "closed",
                    "created_at": "2024-03-10T00:00:00Z",
                    "updated_at": "2024-03-10T00:00:00Z",
                    "merged_at": "2024-03-10T00:00:00Z",
                    "html_url": "https://github.com/acme/project/pull/11",
                },
                {
                    "number": 10,
                    "title": "Old enough",
                    "body": "",
                    "state": "closed",
                    "created_at": "2024-02-01T00:00:00Z",
                    "updated_at": "2024-02-01T00:00:00Z",
                    "merged_at": "2024-02-01T00:00:00Z",
                    "html_url": "https://github.com/acme/project/pull/10",
                },
            ],
            (
                "/repos/acme/project/pulls/11/commits",
                (("page", 1), ("per_page", 100)),
            ): [{"commit": {"message": "Newest commit"}}],
            (
                "/repos/acme/project/pulls/11/commits",
                (("page", 2), ("per_page", 100)),
            ): [],
        }
    )

    written = collect_repo_pull_requests(
        client,
        "acme/project",
        output_dir,
        max_pulls=1,
        cutoff_date="20240301",
    )

    assert written == 1
    assert (output_dir / "acme__project" / "11.json").exists()
    assert not (output_dir / "acme__project" / "10.json").exists()


def test_github_client_retries_429_errors(monkeypatch):
    attempts = {"count": 0}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"ok": true}'

    def fake_urlopen(req, timeout=None):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise HTTPError(
                req.full_url,
                429,
                "too many requests",
                hdrs=None,
                fp=None,
            )
        return FakeResponse()

    sleeps = []
    monkeypatch.setattr("swe_gpu_bench.core.github.request.urlopen", fake_urlopen)
    monkeypatch.setattr("swe_gpu_bench.core.github.time.sleep", sleeps.append)

    client = GitHubClient(token="token", retry_delay=2, max_retries=2)
    payload = client.get_json("/test")

    assert payload == {"ok": True}
    assert attempts["count"] == 2
    assert sleeps == [2]


def test_github_client_raises_after_retry_budget(monkeypatch):
    def fake_urlopen(req, timeout=None):
        raise HTTPError(
            req.full_url,
            429,
            "still rate limited",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr("swe_gpu_bench.core.github.request.urlopen", fake_urlopen)
    monkeypatch.setattr("swe_gpu_bench.core.github.time.sleep", lambda _: None)

    client = GitHubClient(token="token", retry_delay=1, max_retries=1)

    with pytest.raises(GitHubAPIError):
        client.get_json("/test")


def test_github_client_waits_five_minutes_for_github_rate_limit(monkeypatch):
    attempts = {"count": 0}

    class FakeRateLimitResponse:
        def read(self):
            return (
                b'{"message":"API rate limit exceeded for user ID 1","status":"403"}'
            )

        def close(self):
            return None

    class FakeSuccessResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"ok": true}'

    def fake_urlopen(req, timeout=None):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise HTTPError(
                req.full_url,
                403,
                "rate limited",
                hdrs=None,
                fp=FakeRateLimitResponse(),
            )
        return FakeSuccessResponse()

    sleeps = []
    monkeypatch.setattr("swe_gpu_bench.core.github.request.urlopen", fake_urlopen)
    monkeypatch.setattr("swe_gpu_bench.core.github.time.sleep", sleeps.append)

    client = GitHubClient(token="token", max_retries=2)
    payload = client.get_json("/test")

    assert payload == {"ok": True}
    assert attempts["count"] == 2
    assert sleeps == [300]


def test_github_client_does_not_retry_non_rate_limit_403(monkeypatch):
    class FakeForbiddenResponse:
        def read(self):
            return b'{"message":"Resource not accessible by integration","status":"403"}'

        def close(self):
            return None

    def fake_urlopen(req, timeout=None):
        raise HTTPError(
            req.full_url,
            403,
            "forbidden",
            hdrs=None,
            fp=FakeForbiddenResponse(),
        )

    sleeps = []
    monkeypatch.setattr("swe_gpu_bench.core.github.request.urlopen", fake_urlopen)
    monkeypatch.setattr("swe_gpu_bench.core.github.time.sleep", sleeps.append)

    client = GitHubClient(token="token", max_retries=2)

    with pytest.raises(GitHubAPIError):
        client.get_json("/test")

    assert sleeps == []


def test_github_client_rotates_tokens_after_rate_limit(monkeypatch):
    seen_auth = []

    class FakeRateLimitResponse:
        def read(self):
            return (
                b'{"message":"API rate limit exceeded for user ID 1","status":"403"}'
            )

        def close(self):
            return None

    class FakeSuccessResponse:
        def __init__(self, token):
            self.headers = {"X-RateLimit-Remaining": "42"}
            self.token = token

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return ('{"token":"%s"}' % self.token).encode("utf-8")

    def fake_urlopen(req, timeout=None):
        auth = req.headers["Authorization"]
        seen_auth.append(auth)
        if auth == "Bearer token-a":
            raise HTTPError(
                req.full_url,
                403,
                "rate limited",
                hdrs={"X-RateLimit-Reset": str(int(time.time()) + 60)},
                fp=FakeRateLimitResponse(),
            )
        return FakeSuccessResponse(auth)

    monkeypatch.setattr("swe_gpu_bench.core.github.request.urlopen", fake_urlopen)
    monkeypatch.setattr("swe_gpu_bench.core.github.time.sleep", lambda _: None)

    client = GitHubClient(tokens=["token-a", "token-b"], max_retries=2)
    payload = client.get_json("/test")

    assert payload == {"token": "Bearer token-b"}
    assert seen_auth == ["Bearer token-a", "Bearer token-b"]


def test_collect_repo_pull_requests_emits_progress_logs(tmp_path, caplog):
    output_dir = tmp_path / "out"
    client = FakeGitHubClient(
        {
            (
                "/repos/acme/project/pulls",
                (("page", 1), ("per_page", 100), ("state", "closed")),
            ): [
                {
                    "number": 1,
                    "title": "One",
                    "body": "",
                    "state": "closed",
                    "created_at": "2024-03-10T00:00:00Z",
                    "updated_at": "2024-03-10T00:00:00Z",
                    "merged_at": "2024-03-10T00:00:00Z",
                    "html_url": "https://github.com/acme/project/pull/1",
                },
                {
                    "number": 2,
                    "title": "Two",
                    "body": "",
                    "state": "closed",
                    "created_at": "2024-03-09T00:00:00Z",
                    "updated_at": "2024-03-09T00:00:00Z",
                    "merged_at": "2024-03-09T00:00:00Z",
                    "html_url": "https://github.com/acme/project/pull/2",
                },
            ],
            (
                "/repos/acme/project/pulls",
                (("page", 2), ("per_page", 100), ("state", "closed")),
            ): [],
            (
                "/repos/acme/project/pulls/1/commits",
                (("page", 1), ("per_page", 100)),
            ): [{"commit": {"message": "Commit one"}}],
            (
                "/repos/acme/project/pulls/1/commits",
                (("page", 2), ("per_page", 100)),
            ): [],
            (
                "/repos/acme/project/pulls/2/commits",
                (("page", 1), ("per_page", 100)),
            ): [{"commit": {"message": "Commit two"}}],
            (
                "/repos/acme/project/pulls/2/commits",
                (("page", 2), ("per_page", 100)),
            ): [],
        }
    )

    caplog.set_level("INFO")
    collect_repo_pull_requests(client, "acme/project", output_dir, progress_every=1)

    messages = [record.message for record in caplog.records]
    assert any(message.startswith("Collecting acme/project") for message in messages)
    assert "acme/project: inspected=1 written=1 skipped=0 unmerged=0" in messages
    assert "acme/project: inspected=2 written=2 skipped=0 unmerged=0" in messages
    assert "Completed acme/project: inspected=2 written=2 skipped=0 unmerged=0" in messages


def test_collect_repo_pull_requests_skips_unmerged_unless_requested(tmp_path):
    pulls = [
        {
            "number": 5,
            "title": "Abandoned rewrite",
            "state": "closed",
            "created_at": "2024-03-10T00:00:00Z",
            "merged_at": None,
        }
    ]
    responses = {
        ("/repos/acme/project/pulls", (("page", 1), ("per_page", 100), ("state", "closed"))): pulls,
        ("/repos/acme/project/pulls", (("page", 2), ("per_page", 100), ("state", "closed"))): [],
        ("/repos/acme/project/pulls/5/commits", (("page", 1), ("per_page", 100))): [],
    }

    skipping = FakeGitHubClient(responses)
    assert collect_repo_pull_requests(skipping, "acme/project", tmp_path / "default") == 0
    assert not any(path.endswith("/commits") for path, _ in skipping.calls)
    assert not (tmp_path / "default" / "acme__project" / "5.json").exists()

    including = FakeGitHubClient(responses)
    assert collect_repo_pull_requests(including, "acme/project", tmp_path / "all", include_unmerged=True) == 1
    assert (tmp_path / "all" / "acme__project" / "5.json").exists()

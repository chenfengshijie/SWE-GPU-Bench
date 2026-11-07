from pathlib import Path
from urllib.error import HTTPError

from swe_gpu_bench.collect.patches import (
    download_patches,
    metadata_to_diff_api_url,
    patch_output_path,
)
from swe_gpu_bench.core.github import GitHubAPIError, GitHubClient


class FakeClient:
    def __init__(self, failing_urls=()):
        self.calls: list[tuple[str, str]] = []
        self.failing_urls = set(failing_urls)

    def get_text(self, url: str, *, accept: str = "text/plain") -> str:
        self.calls.append((url, accept))
        if url in self.failing_urls:
            raise GitHubAPIError("boom")
        return f"downloaded from {url}\n"


def write_metadata(repo_dir: Path, number: int) -> None:
    repo_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / f"{number}.json").write_text(
        f'{{"repo": "acme/project", "number": {number}, "diff_url": "https://example.com/{number}.diff"}}\n',
        encoding="utf-8",
    )


def test_patch_output_path_uses_repo_directory_and_pr_number():
    assert patch_output_path(Path("patches"), Path("prs/acme__project/101.json")) == Path(
        "patches/acme__project/101.diff"
    )


def test_metadata_to_diff_api_url_uses_repo_and_number():
    assert metadata_to_diff_api_url({"repo": "acme/project", "number": 101}) == (
        "https://api.github.com/repos/acme/project/pulls/101"
    )


def test_download_patches_downloads_and_skips_existing(tmp_path):
    input_dir = tmp_path / "prs"
    write_metadata(input_dir / "acme__project", 101)
    write_metadata(input_dir / "acme__project", 102)
    output_dir = tmp_path / "patches"
    existing = output_dir / "acme__project" / "102.diff"
    existing.parent.mkdir(parents=True)
    existing.write_text("cached\n", encoding="utf-8")
    client = FakeClient()

    downloaded, skipped, failed = download_patches(input_dir, output_dir, client=client)

    assert (downloaded, skipped, failed) == (1, 1, 0)
    assert client.calls == [
        ("https://api.github.com/repos/acme/project/pulls/101", "application/vnd.github.v3.diff")
    ]
    assert (output_dir / "acme__project" / "101.diff").read_text(encoding="utf-8") == (
        "downloaded from https://api.github.com/repos/acme/project/pulls/101\n"
    )
    assert existing.read_text(encoding="utf-8") == "cached\n"


def test_download_patches_counts_failures(tmp_path):
    input_dir = tmp_path / "prs"
    write_metadata(input_dir / "acme__project", 101)
    write_metadata(input_dir / "acme__project", 102)
    client = FakeClient(failing_urls={"https://api.github.com/repos/acme/project/pulls/102"})

    downloaded, skipped, failed = download_patches(input_dir, tmp_path / "patches", client=client, jobs=1)

    assert (downloaded, skipped, failed) == (1, 0, 1)
    assert not (tmp_path / "patches" / "acme__project" / "102.diff").exists()


def test_github_client_rotates_tokens_round_robin_and_after_rate_limit(monkeypatch):
    seen_auth = []

    class FakeRateLimitResponse:
        def read(self):
            return b'{"message":"API rate limit exceeded"}'

        def close(self):
            return None

    class FakeSuccessResponse:
        headers = {"X-RateLimit-Remaining": "42"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b"diff --git a/x b/x\n"

    def fake_urlopen(req, timeout=None):
        auth = req.headers["Authorization"]
        seen_auth.append(auth)
        if auth == "Bearer token-a" and len(seen_auth) > 2:
            raise HTTPError(
                req.full_url,
                403,
                "rate limited",
                hdrs={"X-RateLimit-Reset": "9999999999"},
                fp=FakeRateLimitResponse(),
            )
        return FakeSuccessResponse()

    monkeypatch.setattr("swe_gpu_bench.core.github.request.urlopen", fake_urlopen)
    client = GitHubClient(tokens=["token-a", "token-b"], max_retries=2)

    texts = [client.get_text("https://api.github.com/x", accept="application/vnd.github.v3.diff") for _ in range(3)]

    assert texts == ["diff --git a/x b/x\n"] * 3
    assert seen_auth == ["Bearer token-a", "Bearer token-b", "Bearer token-a", "Bearer token-b"]

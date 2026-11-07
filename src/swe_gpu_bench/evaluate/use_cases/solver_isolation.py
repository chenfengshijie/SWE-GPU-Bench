"""Restrict the solver's Git object database to the pinned base snapshot."""
from pathlib import Path
import shutil
import subprocess
import tempfile


def isolate_git_history(repo: Path, base_commit: str) -> None:
    repo = repo.resolve()
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if actual != base_commit:
        raise ValueError("Solver checkout is not the pinned base commit")
    # Initialized submodules usually store their metadata under the parent's
    # .git/modules. Replacing that directory would break subsequent rebuilds.
    # First give each initialized dependency its own shallow, pinned metadata;
    # recursively discard its other refs/objects rather than preserving future
    # dependency history inside the solver's readable checkout.
    tree = subprocess.check_output(["git", "ls-tree", "-r", "-z", "HEAD"], cwd=repo)
    for entry in tree.split(b"\0"):
        if not entry:
            continue
        header, name = entry.split(b"\t", 1)
        mode, _, commit = header.split()
        child = repo / name.decode()
        if mode == b"160000" and (child / ".git").exists():
            isolate_git_history(child, commit.decode())
    with tempfile.TemporaryDirectory(prefix="swe-gpu-base-git-") as temp:
        fresh = Path(temp)
        def git(*args):
            subprocess.run(["git", *args], cwd=fresh, check=True, capture_output=True)
        git("init", "--quiet")
        git("-c", "protocol.file.allow=always", "fetch", "--depth=1", "--no-tags",
            "--no-write-fetch-head", str(repo), base_commit)
        git("checkout", "--quiet", "--detach", base_commit)
        # Only replace the disposable evaluation checkout's metadata, not its
        # source/build files or the evaluator's shared repository cache.
        old = repo / ".git"
        if old.is_dir():
            shutil.rmtree(old)
        else:
            old.unlink()
        shutil.copytree(fresh / ".git", old)

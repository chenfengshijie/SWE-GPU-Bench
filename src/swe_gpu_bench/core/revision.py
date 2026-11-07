"""Verify the framework checkout before freezing a formal evaluation."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys


def _git(root: Path, *args: str) -> bytes:
    try:
        return subprocess.check_output(
            ['git', *args], cwd=root, stderr=subprocess.DEVNULL, timeout=60,
            env={**os.environ, 'GIT_TERMINAL_PROMPT': '0'},
        )
    except (OSError, subprocess.SubprocessError):
        # Git errors can contain authenticated URLs; never copy them into logs.
        raise ValueError('Cannot verify framework Git state or query origin/master') from None


def verify_master(package: Path) -> tuple[str, dict[str, str]]:
    """Return the live origin/master SHA and committed hashes in frozen-run layout.

    Resolve from the invocation directory, not from an unrelated installed wheel.
    Read committed blobs so ignored files and Git index flags cannot bypass the
    byte comparison when the package is copied into the frozen run.
    """
    root = Path(_git(Path.cwd(), 'rev-parse', '--show-toplevel').decode().strip()).resolve()
    expected_package = root / 'src' / 'swe_gpu_bench'
    if package.resolve() != expected_package or package.is_symlink():
        raise ValueError('--require-master must load the package from this framework source checkout')
    for name, module in tuple(sys.modules.items()):
        if name == 'swe_gpu_bench' or name.startswith('swe_gpu_bench.'):
            location = getattr(module, '__file__', None)
            if location and not Path(location).resolve().is_relative_to(expected_package):
                raise ValueError('--require-master found a package module outside this source checkout')
    if _git(root, 'status', '--porcelain', '--untracked-files=all').strip():
        raise ValueError('--require-master requires a clean working tree, including untracked files')
    head = _git(root, 'rev-parse', 'HEAD').decode().strip()
    remote = _git(root, 'ls-remote', '--exit-code', 'origin', 'refs/heads/master').decode().split()
    if len(remote) != 2 or remote[1] != 'refs/heads/master' or remote[0] != head:
        raise ValueError('--require-master requires HEAD to equal the current origin/master')
    entries = _git(root, 'ls-tree', '-rz', head, '--', 'src/swe_gpu_bench', 'pyproject.toml', 'uv.lock')
    hashes = {}
    for entry in entries.split(b'\0'):
        if not entry:
            continue
        metadata, raw_path = entry.split(b'\t', 1)
        mode, kind, oid = metadata.decode().split()
        path = raw_path.decode()
        if kind != 'blob' or mode not in ('100644', '100755'):
            raise ValueError('--require-master cannot freeze symlinks or submodules in framework sources')
        data = _git(root, 'cat-file', 'blob', oid)
        frozen_name = 'code/' + path.removeprefix('src/') if path.startswith('src/') else path
        hashes[frozen_name] = hashlib.sha256(data).hexdigest()
    if not {'pyproject.toml', 'uv.lock', 'code/swe_gpu_bench/evaluate/campaign.py'} <= hashes.keys():
        raise ValueError('--require-master requires committed framework sources and dependency files')
    return head, hashes


def verify_frozen_source(run: Path, expected: dict[str, str]) -> None:
    paths = [p for p in (run / 'code').rglob('*') if p.is_file()]
    paths += [run / 'pyproject.toml', run / 'uv.lock']
    actual = {p.relative_to(run).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    if actual != expected:
        raise ValueError('Frozen framework files differ from verified origin/master; prepare a new run')

"""Observe candidate-source reads during rebuild/tests, without changing commands."""
from __future__ import annotations
import ctypes
import hashlib
import os
from pathlib import Path
import select
import struct
import subprocess
import threading


class SourceAccess:
    def __init__(self, repo: Path, candidate_names: list[str] | None = None):
        self.repo = repo
        self.candidate_names = candidate_names
        self.fd = -1
        self.watches = {}
        self.files = {}
        self.observed = set()
        self.overflow = False
        self.stop_event = threading.Event()
        self.thread = None
        self.error = None

    def start(self):
        try:
            names = self.candidate_names
            if names is None:
                names = subprocess.check_output(['git', 'diff', '--name-only', 'HEAD', '-z'], cwd=self.repo).decode().split('\0')
            paths = [self.repo / n for n in names if n and not Path(n).is_absolute() and '..' not in Path(n).parts
                     and (self.repo / n).is_file()
                     and not any(p in {'test', 'tests', 'testing'} for p in Path(n).parts)
                     and Path(n).suffix in {'.py', '.cu', '.cuh', '.h', '.hpp', '.cpp', '.cc', '.c', '.cmake', '.txt', '.rs'}]
            # Hash before installing watches: our own reads are not evidence.
            self.files = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
            libc = ctypes.CDLL(None, use_errno=True)
            self.fd = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
            if self.fd < 0:
                raise OSError(ctypes.get_errno(), 'inotify unavailable')
            for path in paths:
                wd = libc.inotify_add_watch(self.fd, os.fsencode(path), 0x1)  # IN_ACCESS, not stat/open
                if wd < 0:
                    raise OSError(ctypes.get_errno(), 'source watch unavailable')
                self.watches[wd] = str(path)
            self.thread = threading.Thread(target=self._read, daemon=True)
            self.thread.start()
        except (OSError, subprocess.CalledProcessError) as error:
            self.error = type(error).__name__

    def _read(self):
        while not self.stop_event.is_set():
            if select.select([self.fd], [], [], .1)[0]:
                self._drain()
        self._drain()

    def _drain(self):
        try:
            data = os.read(self.fd, 1024 * 1024)
        except BlockingIOError:
            return
        offset = 0
        while offset < len(data):
            wd, mask, _, size = struct.unpack_from('iIII', data, offset)
            offset += 16 + size
            if mask & 0x4000:
                self.overflow = True
            if wd in self.watches and mask & 1:
                self.observed.add(self.watches[wd])

    def finish(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5)
        if self.fd >= 0:
            os.close(self.fd)
        changed = [p for p, digest in self.files.items()
                   if not Path(p).is_file() or hashlib.sha256(Path(p).read_bytes()).hexdigest() != digest]
        return {'candidate_files': self.files, 'read_during_validation': sorted(self.observed),
                'changed_during_validation': changed, 'overflow': self.overflow, 'error': self.error,
                'meaning': 'OS-observed source reads during rebuild/test; test outcomes are recorded separately'}

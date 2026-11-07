"""Fail-closed Linux filesystem guard for solver subprocesses only.

User and mount namespaces are inherited by every shell/tool child. Shared datasets, sibling
attempts, evaluator logs, credentials, and the full Git cache are not readable.
This is filesystem isolation, not a network sandbox; upstream retrieval remains
prohibited by the solver prompt and the base Git remote is removed.
"""
from __future__ import annotations
import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def guarded_command(command: list[str], case_dir: Path, env: dict) -> list[str]:
    if env.get('SWEBENCH_SOLVER_ISOLATION') != '1':
        return command
    case_dir = case_dir.resolve()
    writable = [case_dir]
    readable = [Path(p) for p in ('/usr', '/bin', '/sbin', '/lib', '/lib64', '/etc', '/opt', '/sys')]
    # Allow device access for actual CUDA checks; no shared workspace mount.
    writable.append(Path('/dev'))
    # Most runtimes need their own maps and system resource data. Other process
    # environments/cmdlines (which may include evaluator state) are not exposed.
    readable += [Path(p) for p in ('/proc/cpuinfo', '/proc/meminfo', '/proc/stat', '/proc/uptime', '/proc/sys',
                                  '/proc/driver/nvidia', '/proc/devices', '/proc/modules')]
    # The fresh command venv and pinned runtime binaries are siblings of the
    # attempt. Never grant the entire /tmp or worker root (it has the Git cache).
    if env.get('VIRTUAL_ENV'):
        readable.append(Path(env['VIRTUAL_ENV']))
        readable.append(Path(env['VIRTUAL_ENV']).resolve())
    for name in ('codex', 'codex-code-mode-host', 'opencode', 'uv', 'rg', 'python3'):
        if path := shutil.which(name, path=env.get('PATH')):
            readable.append(Path(path).resolve().parent)
            if name == 'python3':
                readable.append(Path(path).resolve().parent.parent)
    if env.get('CODEX_HOME'):
        writable.append(Path(env['CODEX_HOME']))
    if env.get('OPENCODE_CONFIG'):
        readable.append(Path(env['OPENCODE_CONFIG']))
    temp = case_dir / 'artifacts/solver-tmp'
    temp.mkdir(parents=True, exist_ok=True)
    env['TMPDIR'] = str(temp)
    env.pop('PYTHONPATH', None)
    env.pop('SWEBENCH_UV_CACHE_ROOT', None)
    artifacts = case_dir / 'artifacts'
    environments = []
    interpreter_roots = set()
    for parent, directories, files in os.walk(artifacts):
        path = Path(parent)
        if 'pyvenv.cfg' in files or (path / 'bin/python').exists():
            environments.append(str(path))
            # uv environments can point at a managed Python outside the case.
            # Expose only that prepared installation, not its cache parent or
            # siblings. Otherwise a working venv becomes broken in the solver
            # namespace, encouraging replacement with a temporary interpreter.
            executable = (path / 'bin/python').resolve()
            prefix = executable.parent.parent
            if (executable.is_file() and executable.parent.name == 'bin'
                    and prefix != Path('/') and not prefix.is_relative_to(path)
                    and any((prefix / 'lib').glob('python3.*/encodings/__init__.py'))):
                interpreter_roots.add(str(prefix))
            directories[:] = []
        elif len(path.relative_to(artifacts).parts) >= 3:
            directories[:] = []
    hidden_files = []
    repo = case_dir / 'repo'
    if (repo / '.git').exists():
        extra = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard', '-z'], cwd=repo)
        for name in extra.decode().split('\0'):
            path = repo / name
            if name and path.suffix in {'.py', '.cu', '.cuh', '.cpp', '.cc', '.c', '.sh'} and path.is_file():
                content = path.read_text(errors='replace')
                if 'test' in path.name.lower() or 'assert' in content or 'EXPECT_' in content:
                    hidden_files.append(str(path))
    readable.extend(Path(p) for p in sorted(interpreter_roots))
    env['SWEBENCH_GUARD_ROOTS'] = json.dumps({'read': [str(p) for p in readable], 'write': [str(p) for p in writable],
        'artifacts': str(artifacts), 'environments': environments,
        'interpreter_roots': sorted(interpreter_roots), 'hidden_files': hidden_files})
    return ['unshare', '--user', '--map-root-user', '--mount',
            sys.executable, str(Path(__file__).resolve()), *command]


def namespace_root(read_roots: list[Path], write_roots: list[Path], view: dict) -> None:
    """Use Linux user/mount namespaces without depending on Landlock.

    Keep PID numbers consistent with the container's procfs: CUDA creates a
    thread by writing /proc/self/task/<tid>/comm. A synthetic self directory or
    a new PID namespace with the old procfs breaks both CUDA and solver children.
    Each worker container runs one case at a time. Hide all existing processes
    except this solver before exec; its future children inherit the chroot.
    """
    root = Path(tempfile.mkdtemp(prefix='swe-gpu-solver-root-', dir='/tmp'))
    def mount(*args):
        completed = subprocess.run(['mount', *map(str, args)], capture_output=True, text=True)
        if completed.returncode:
            print(completed.stderr.strip(), file=sys.stderr, flush=True)
            completed.check_returncode()
    mount('--make-rprivate', '/')
    mount('-t', 'tmpfs', 'tmpfs', root)
    # Parent directories first. Per-case mounts are deliberately outside the
    # hidden controller mount, so neither datasets nor the Git cache appear.
    roots = {str(p): (p, False) for p in read_roots if p.exists()}
    roots.update({str(p): (p, True) for p in write_roots if p.exists()})
    for path, writable in sorted(roots.values(), key=lambda item: len(item[0].parts)):
        if str(path).startswith('/proc'):
            continue
        dest = root / str(path).lstrip('/')
        dest.parent.mkdir(parents=True, exist_ok=True)
        if path.is_dir():
            dest.mkdir(exist_ok=True)
        else:
            dest.touch(exist_ok=True)
        mount('--rbind' if path.is_dir() else '--bind', path, dest)
        if not writable:
            mount('-o', 'remount,bind,ro', dest)
    if view.get('artifacts'):
        dest = root / view['artifacts'].lstrip('/')
        dest.mkdir(parents=True, exist_ok=True)
        mount('-t', 'tmpfs', 'tmpfs', dest)
        for source in view.get('environments', []):
            target = root / source.lstrip('/')
            target.mkdir(parents=True, exist_ok=True)
            mount('--rbind', source, target)
        # The artifacts overlay above also hides prepared Python installations
        # located inside artifacts but outside the venv itself.
        for source in view.get('interpreter_roots', []):
            if Path(source).is_relative_to(Path(view['artifacts'])):
                target = root / source.lstrip('/')
                target.mkdir(parents=True, exist_ok=True)
                mount('--rbind', source, target)
                mount('-o', 'remount,bind,ro', target)
        (dest / 'solver-tmp').mkdir(exist_ok=True)
    for source in view.get('hidden_files', []):
        mount('--bind', '/dev/null', root / source.lstrip('/'))
        mount('-o', 'remount,bind,ro', root / source.lstrip('/'))
    (root / 'proc').mkdir(exist_ok=True)
    mount('--rbind', '/proc', root / 'proc')
    hidden_processes = []
    for process in Path('/proc').iterdir():
        if process.name.isdigit() and int(process.name) != os.getpid():
            target = root / 'proc' / process.name
            try:
                mount('-t', 'tmpfs', '-o', 'ro,nosuid,nodev,noexec,size=4k', 'tmpfs', target)
            except subprocess.CalledProcessError:
                if process.exists():
                    raise
            hidden_processes.append(int(process.name))
    print('SOLVER_PROC_MODE=masked_container_procfs hidden_processes=' + str(len(hidden_processes)), file=sys.stderr, flush=True)
    cwd = os.getcwd()
    os.chroot(root)
    os.chdir(cwd)
    libc = ctypes.CDLL(None, use_errno=True)
    # Lock root's special exec privilege off and empty the capability bounding
    # set before running model-controlled tools inside the filesystem view.
    if libc.prctl(28, 3, 0, 0, 0) != 0:  # SECBIT_NOROOT | SECBIT_NOROOT_LOCKED
        raise OSError(ctypes.get_errno(), 'securebits failed')
    for capability in range(41):
        libc.prctl(24, capability, 0, 0, 0)
    class Header(ctypes.Structure):
        _fields_ = [('version', ctypes.c_uint32), ('pid', ctypes.c_int)]
    class Cap(ctypes.Structure):
        _fields_ = [('effective', ctypes.c_uint32), ('permitted', ctypes.c_uint32), ('inheritable', ctypes.c_uint32)]
    header = Header(0x20080522, 0)
    caps = (Cap * 2)()
    if libc.capset(ctypes.byref(header), caps) != 0 or libc.prctl(38, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'dropping solver capabilities failed')


def main():
    roots = json.loads(os.environ.pop('SWEBENCH_GUARD_ROOTS'))
    command = sys.argv[1:]
    executable = shutil.which(command[0])
    if not executable:
        raise FileNotFoundError(command[0])
    namespace_root([Path(p) for p in roots['read']], [Path(p) for p in roots['write']], roots)
    os.execve(executable, command, os.environ)


def preflight(case: Path) -> None:
    """Check real tools and CUDA inside the view before any model invocation."""
    case.mkdir(parents=True, exist_ok=True)
    (case / 'artifacts').mkdir(exist_ok=True)
    hidden = case.parent / 'guard-withheld.txt'
    hidden.write_text('evaluator-only preflight marker')
    # Exercise the driver directly: a generic evaluation image need not contain
    # PyTorch or another particular task's framework before setup.
    cuda_probe = '''import ctypes as c
lib = c.CDLL('libcuda.so.1')
def call(name, *args):
    code = getattr(lib, name)(*args)
    assert code == 0, (name, code)
call('cuInit', 0)
device, context, pointer = c.c_int(), c.c_void_p(), c.c_uint64()
call('cuDeviceGet', c.byref(device), 0)
call('cuCtxCreate_v2', c.byref(context), 0, device)
try:
    call('cuMemAlloc_v2', c.byref(pointer), c.c_size_t(4))
    try:
        call('cuMemsetD8_v2', pointer, 42, c.c_size_t(4))
        call('cuCtxSynchronize')
        value = (c.c_ubyte * 4)()
        call('cuMemcpyDtoH_v2', c.byref(value), pointer, c.c_size_t(4))
        assert list(value) == [42] * 4
    finally:
        call('cuMemFree_v2', pointer)
finally:
    call('cuCtxDestroy_v2', context)
print('CUDA_DRIVER_ALLOCATION_COPY_OK')
'''
    commands = [
        ['/bin/sh', '-c', 'test ! -r ' + str(hidden)],
        ['codex', '--version'], ['opencode', '--version'],
        ['python3', '-c', cuda_probe],
        ['python3', '-c', 'import subprocess,sys; subprocess.run([sys.executable,"-c",'
         + repr(cuda_probe) + '],check=True)'],
    ]
    for command in commands:
        env = {**os.environ, 'SWEBENCH_SOLVER_ISOLATION': '1',
               'XDG_CACHE_HOME': str(case / 'artifacts/cache'),
               'XDG_CONFIG_HOME': str(case / 'artifacts/config'),
               'XDG_DATA_HOME': str(case / 'artifacts/data')}
        subprocess.run(guarded_command(command, case, env), cwd=case, env=env, check=True, timeout=120)
    print('SOLVER_FILESYSTEM_AND_CUDA_PREFLIGHT_OK', flush=True)


if __name__ == '__main__':
    main()

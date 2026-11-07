"""Real local Git remotes exercise the release gate without network or GPU calls."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root, stderr=subprocess.DEVNULL, text=True).strip()


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path / 'checkout'
    root.mkdir()
    source = Path(__file__).resolve().parents[1]
    shutil.copytree(source / 'src/swe_gpu_bench', root / 'src/swe_gpu_bench',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('pyproject.toml', 'uv.lock', '.gitignore'):
        shutil.copyfile(source / name, root / name)
    git(root, 'init', '-b', 'master')
    git(root, 'config', 'user.name', 'Fixture')
    git(root, 'config', 'user.email', 'fixture@example.invalid')
    git(root, 'add', '.')
    git(root, 'commit', '-m', 'Fixture')
    remote = tmp_path / 'remote.git'
    git(tmp_path, 'clone', '--bare', str(root), str(remote))
    git(root, 'remote', 'add', 'origin', str(remote))
    (tmp_path / 'profiles.toml').write_text('[profiles.text.fake]\nmodel="fake"\n')
    row = {'instance_id': 'org__repo__1', 'repo': 'org/repo', 'base_commit': 'a' * 40,
           'problem_statement': 'Fixture', 'docker_image': 'public/image:tag'}
    (tmp_path / 'data.jsonl').write_text(json.dumps(row) + '\n')
    cfg = {'output_root': str(tmp_path / 'runs'), 'profiles': 'profiles.toml',
           'hardware': {'gpu_name': 'H100'},
           'datasets': [{'task_type': 'correctness', 'path': 'data.jsonl'}],
           'models': [{'name': 'fake', 'oracle_profile': 'fake', 'modes': ['oracle']}]}
    (tmp_path / 'config.json').write_text(json.dumps(cfg))
    return root


def prepare(root, *, cwd=None, package_root=None, strict=True, script=None):
    args = ['-m', 'swe_gpu_bench', 'evaluate', 'campaign', 'prepare',
            '--config', str(root.parent / 'config.json')]
    if strict:
        args.append('--require-master')
    if script:
        args = ['-c', script, str(root.parent / 'config.json')]
    return subprocess.run([sys.executable, *args], cwd=cwd or root,
                          env={**os.environ, 'PYTHONPATH': str((package_root or root) / 'src'),
                               'PYTHONDONTWRITEBYTECODE': '1'},
                          capture_output=True, text=True, timeout=60)


def test_live_master_is_recorded_and_frozen_code_is_checked(checkout):
    result = prepare(checkout)
    assert result.returncode == 0, result.stderr
    run = Path(result.stdout.strip())
    manifest = json.loads((run / 'manifest.json').read_text())
    assert manifest['code_origin']['verified_master_sha'] == git(checkout, 'rev-parse', 'HEAD')
    # Frozen status still works after the source remote is unavailable.
    git(checkout, 'remote', 'set-url', 'origin', str(checkout.parent / 'missing.git'))
    args = [sys.executable, '-m', 'swe_gpu_bench', 'evaluate', 'campaign', 'status', '--run', str(run)]
    env = {**os.environ, 'PYTHONPATH': str(run / 'code'), 'PYTHONDONTWRITEBYTECODE': '1'}
    assert subprocess.run(args, cwd=checkout, env=env, capture_output=True).returncode == 0
    (run / 'code/swe_gpu_bench/evaluate/campaign_worker.py').write_text('tampered')
    checked = subprocess.run(args, cwd=checkout, env=env, capture_output=True, text=True)
    assert checked.returncode != 0 and 'Frozen input/code changed' in checked.stderr


@pytest.mark.parametrize('change', ['modified', 'staged', 'untracked'])
def test_dirty_checkout_cannot_prepare(checkout, change):
    path = checkout / ('untracked.txt' if change == 'untracked' else 'pyproject.toml')
    path.write_text(path.read_text() + '\n# changed\n' if path.exists() else 'untracked')
    if change == 'staged':
        git(checkout, 'add', '.')
    result = prepare(checkout)
    assert result.returncode != 0 and 'clean working tree' in result.stderr
    assert not (checkout.parent / 'runs').exists()


def test_stale_tracking_ref_cannot_hide_remote_advance(checkout):
    old = git(checkout, 'rev-parse', 'HEAD')
    git(checkout, 'commit', '--allow-empty', '-m', 'Advance')
    git(checkout, 'push', 'origin', 'master')
    git(checkout, 'reset', '--hard', old)
    git(checkout, 'update-ref', 'refs/remotes/origin/master', old)
    result = prepare(checkout)
    assert result.returncode != 0 and 'current origin/master' in result.stderr


def test_unpushed_commit_cannot_prepare(checkout):
    git(checkout, 'commit', '--allow-empty', '-m', 'Unpublished')
    result = prepare(checkout)
    assert result.returncode != 0 and 'current origin/master' in result.stderr


def test_remote_failure_does_not_fall_back_to_cached_ref(checkout):
    git(checkout, 'remote', 'set-url', 'origin', str(checkout.parent / 'missing.git'))
    result = prepare(checkout)
    assert result.returncode != 0 and 'Cannot verify framework' in result.stderr
    assert not (checkout.parent / 'runs').exists()


def test_wrong_installed_package_is_rejected(checkout):
    source = Path(__file__).resolve().parents[1]
    result = prepare(checkout, package_root=source)
    assert result.returncode != 0 and 'this framework source checkout' in result.stderr


def test_invocation_outside_checkout_is_rejected(checkout):
    result = prepare(checkout, cwd=checkout.parent)
    assert result.returncode != 0 and 'Cannot verify framework' in result.stderr


def test_ignored_package_file_cannot_enter_verified_snapshot(checkout):
    (checkout / '.git/info/exclude').write_text('src/swe_gpu_bench/injected.py\n')
    (checkout / 'src/swe_gpu_bench/injected.py').write_text('injected = True\n')
    result = prepare(checkout)
    assert result.returncode != 0 and 'Frozen framework files differ' in result.stderr
    assert list((checkout.parent / 'runs').iterdir()) == []


def test_index_assume_unchanged_cannot_hide_modified_source(checkout):
    path = 'src/swe_gpu_bench/evaluate/campaign_worker.py'
    git(checkout, 'update-index', '--assume-unchanged', path)
    (checkout / path).write_text((checkout / path).read_text() + '\n# hidden edit\n')
    result = prepare(checkout)
    assert result.returncode != 0 and 'Frozen framework files differ' in result.stderr


def test_source_change_during_freeze_is_rejected(checkout):
    script = '''
import sys
from pathlib import Path
from swe_gpu_bench.core import revision
from swe_gpu_bench.evaluate import campaign
original = revision.verify_master
def changed(package):
    result = original(package)
    (package / 'evaluate/campaign_worker.py').write_text('# concurrent edit')
    return result
revision.verify_master = changed
campaign.prepare(Path(sys.argv[1]), require_master=True)
'''
    result = prepare(checkout, script=script)
    assert result.returncode != 0 and 'Frozen framework files differ' in result.stderr


def test_legacy_prepare_remains_available_without_git_remote(checkout):
    git(checkout, 'remote', 'remove', 'origin')
    result = prepare(checkout, strict=False)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((Path(result.stdout.strip()) / 'manifest.json').read_text())
    assert manifest['code_origin']['verified_master_sha'] is None

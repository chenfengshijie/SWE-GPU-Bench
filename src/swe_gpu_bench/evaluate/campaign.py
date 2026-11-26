"""Frozen, resumable Oracle/Agent correctness and performance campaigns."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import tomllib

from swe_gpu_bench.shared.artifacts import (
    atomic_json, atomic_jsonl, directory_lock, new_id, read_json, read_jsonl, sha256,
)
from swe_gpu_bench.shared.config import resolve_profile
from swe_gpu_bench.evaluate.metrics import recipe

TERMINAL = {'TERMINATED', 'DONE', 'FAILED', 'FAILED_TO_LAUNCH', 'STOPPED', 'CLOSED', 'REJECTED', 'CANCELED', 'CANCELLED'}
REAL_OUTCOMES = {'passed', 'test_failed', 'model_output_error', 'performance_not_improved'}


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def safe_name(value: str) -> str:
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value):
        raise ValueError('Unsafe identifier')
    return value


def resolve_path(parent: Path, value: str) -> Path:
    return (parent / Path(value).expanduser()).resolve()


def prepare(config_path: Path, *, require_master: bool = False) -> Path:
    package = Path(__file__).resolve().parents[1]
    verified_master_sha, committed_hashes = None, None
    if require_master:
        from swe_gpu_bench.core.revision import verify_master
        verified_master_sha, committed_hashes = verify_master(package)
    config_path = config_path.resolve()
    config = tomllib.loads(config_path.read_text()) if config_path.suffix == '.toml' else read_json(config_path)
    parent = config_path.parent
    config.setdefault('backend', 'local')
    config.setdefault('concurrency', 1)
    config.setdefault('timeout_seconds', 7200)
    config.setdefault('max_transient_retries', 2)
    config.setdefault('cooldown_seconds', 300)
    config.setdefault('poll_seconds', 15)
    config.setdefault('git_timeout_seconds', 600)
    config.setdefault('scoring', {'nonpass_as_incorrect': True})
    if config['backend'] not in ('local', 'docker', 'external'):
        raise ValueError('backend must be local, docker or external')
    if any(not isinstance(config[k], int) or config[k] <= 0 for k in ('concurrency', 'timeout_seconds', 'poll_seconds', 'git_timeout_seconds')):
        raise ValueError('Concurrency and timeouts must be positive integers')
    if not 0 <= config['max_transient_retries'] <= 20 or config['cooldown_seconds'] < 0:
        raise ValueError('Invalid retry/cooldown policy')
    if not config.get('hardware', {}).get('gpu_name'):
        raise ValueError('hardware.gpu_name is required')
    if config['backend'] == 'docker':
        docker = config.get('docker', {})
        devices = [str(d) for d in docker.get('devices', ['0'])]
        if (not devices or len(set(devices)) != len(devices)
                or any(not re.fullmatch(r'[A-Za-z0-9_.:-]+', d) for d in devices)
                or config['concurrency'] > len(devices)):
            raise ValueError('Docker requires a distinct device per concurrent task')
        if any(not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', n) for n in docker.get('forward_env', [])):
            raise ValueError('forward_env accepts variable names, not credential values')
    profiles = resolve_path(parent, config['profiles'])
    # Only environment variable *names* may be frozen, never credential values.
    profile_data = tomllib.loads(profiles.read_text())
    for kind in profile_data.get('profiles', {}).values():
        for profile in kind.values():
            if profile.get('extra_env') or any(k in profile for k in ('api_key', 'token', 'password')):
                raise ValueError('Profiles must use api_key_env; inline credentials/extra_env are not frozen')
    source_rows, tasks, bindings = {}, [], {}
    for source in config['datasets']:
        task_type = source['task_type']
        if task_type not in ('correctness', 'performance') or task_type in source_rows:
            raise ValueError('One dataset per task_type is required')
        path = resolve_path(parent, source['path'])
        rows = read_jsonl(path)
        bindings[task_type] = {'source': str(path), 'sha256': sha256(path), 'version': source.get('version')}
        if source.get('sha256') and sha256(path) != source['sha256']:
            raise ValueError('Source dataset hash mismatch')
        if source.get('instance_ids'):
            selected = set(source['instance_ids'])
            rows = [r for r in rows if r['instance_id'] in selected]
            if {r['instance_id'] for r in rows} != selected:
                raise ValueError('Requested instance missing')
        if not rows or len(rows) != len({r['instance_id'] for r in rows}):
            raise ValueError('Empty dataset or duplicate instance ID')
        for row in rows:
            safe_name(row['instance_id'])
            if any(not row.get(k) for k in ('docker_image', 'repo', 'base_commit', 'problem_statement')):
                raise ValueError('Missing required task fields')
            if not isinstance(row['base_commit'], str) or not re.fullmatch(r'[0-9a-fA-F]{40}', row['base_commit']):
                raise ValueError('base_commit must be a full immutable commit SHA')
            if task_type == 'performance':
                recipe(row)  # Fail before allocating a GPU or calling a model.
        source_rows[task_type] = rows
    for model in config['models']:
        safe_name(model['name'])
        for task_type, rows in source_rows.items():
            if task_type not in model.get('task_types', source_rows):
                continue
            for mode in model.get('modes', ['oracle', 'agent']):
                if mode not in ('oracle', 'agent'):
                    raise ValueError('Unknown mode')
                profile_name = model[mode + '_profile']
                profile = resolve_profile(profile_name=profile_name, profile_kind='text' if mode == 'oracle' else 'agent', config_path=profiles)
                for row in rows:
                    task_id = '--'.join((task_type, mode, model['name'], row['instance_id']))
                    tasks.append({'task_id': task_id, 'task_type': task_type, 'mode': mode, 'model': model['name'],
                                  'profile': profile_name, 'deployment_model': profile.model,
                                  'reasoning_effort': profile.reasoning_effort,
                                  'instance_id': row['instance_id'], 'record_sha256': digest(row),
                                  'dataset_sha256': bindings[task_type]['sha256'],
                                  'docker_image': row['docker_image'],
                                  'runtime_image': config.get('runtime_image') or row['docker_image']})
    if not tasks or len(tasks) != len({t['task_id'] for t in tasks}):
        raise ValueError('Empty or duplicate matrix positions')
    run_dir = resolve_path(parent, config['output_root']) / new_id()
    run_dir.mkdir(parents=True)
    config['credential_env_file'] = str(resolve_path(parent, config['credential_env_file'])) if config.get('credential_env_file') else None
    config['local_root'] = str(resolve_path(parent, config.get('local_root', '/tmp/swe-gpu-campaign')))
    shutil.copyfile(profiles, run_dir / 'profiles.toml')
    for kind, rows in source_rows.items():
        atomic_jsonl(run_dir / (kind + '.jsonl'), rows)
    # Copy the installed package, including package resources. No workspace data,
    # environments, configuration secrets or history directories are included.
    code = run_dir / 'code' / 'swe_gpu_bench'
    shutil.copytree(package, code, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    source_root = package.parents[1]
    for name in ('pyproject.toml', 'uv.lock'):
        if (source_root / name).is_file():
            shutil.copyfile(source_root / name, run_dir / name)
    if committed_hashes is not None:
        from swe_gpu_bench.core.revision import verify_frozen_source
        try:
            verify_frozen_source(run_dir, committed_hashes)
        except (OSError, ValueError):
            shutil.rmtree(run_dir)
            raise
    frozen = [p for p in (run_dir / 'code').rglob('*') if p.is_file()]
    frozen += [run_dir / 'profiles.toml', *[run_dir / (k + '.jsonl') for k in source_rows]]
    frozen += [run_dir / n for n in ('pyproject.toml', 'uv.lock') if (run_dir / n).is_file()]
    if config['backend'] == 'external':
        external = config['external']
        source = resolve_path(parent, external['script'])
        if external.get('sha256') and sha256(source) != external['sha256']:
            raise ValueError('External adapter hash mismatch')
        destination = run_dir / 'external/adapter.py'
        destination.parent.mkdir()
        shutil.copyfile(source, destination)
        frozen.append(destination)
        config['external'] = {**external, 'script': str(source), 'sha256': sha256(source)}
    exclusions = []
    for exclusion in config.get('exclusions', []):
        kind = exclusion['task_type']
        if exclusion['dataset_sha256'] != bindings[kind]['sha256']:
            raise ValueError('Exclusion belongs to a different dataset version')
        if exclusion['instance_id'] not in {r['instance_id'] for r in source_rows[kind]}:
            raise ValueError('Exclusion does not identify an input instance')
        evidence = resolve_path(parent, exclusion['evidence'])
        if sha256(evidence) != exclusion['evidence_sha256'] or not exclusion.get('reason'):
            raise ValueError('Exclusion evidence missing or changed')
        target = run_dir / 'exclusions' / (digest(exclusion) + '.json')
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(evidence, target)
        frozen.append(target)
        exclusions.append({**exclusion, 'evidence': str(target)})
    repairs = {}
    for repair in config.get('environment_repairs', []):
        kind, iid = repair['task_type'], repair['instance_id']
        row = next(r for r in source_rows[kind] if r['instance_id'] == iid)
        if repair['record_sha256'] != digest(row) or not repair.get('version'):
            raise ValueError('Environment repair binding mismatch')
        evidence = resolve_path(parent, repair['evidence'])
        if sha256(evidence) != repair['evidence_sha256']:
            raise ValueError('Repair evidence mismatch')
        target = run_dir / 'repairs' / (digest(repair) + '.json')
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(evidence, target)
        frozen.append(target)
        repairs[kind + '--' + iid] = {**repair, 'evidence': str(target)}
    config['environment_repairs'] = repairs
    config['exclusions'] = exclusions
    atomic_json(run_dir / 'tasks.json', tasks)
    frozen.append(run_dir / 'tasks.json')
    task_index = {t['task_id']: t for t in tasks}
    for imported in config.get('candidate_imports', []):
        task = task_index[imported['task_id']]
        source = resolve_path(parent, imported['manifest'])
        if sha256(source) != imported['sha256']:
            raise ValueError('Imported candidate manifest changed')
        saved = read_json(source)
        patch = Path(saved['path'])
        if (saved['record_sha256'] != task['record_sha256'] or saved['mode'] != task['mode']
                or saved['model'] != task['model'] or sha256(patch) != saved['sha256']):
            raise ValueError('Imported candidate binding mismatch')
        destination = run_dir / 'candidates' / task['task_id']
        destination.mkdir(parents=True)
        shutil.copyfile(patch, destination / 'candidate.diff')
        atomic_json(destination / 'candidate.json', {**saved, 'path': str(destination / 'candidate.diff'),
                    'imported_from': str(source), 'imported_manifest_sha256': imported['sha256']})
        frozen.extend(destination.iterdir())
    import_history(run_dir, config.get('history', []), parent)
    frozen.extend((run_dir / 'history').glob('*.json'))
    code_commit = verified_master_sha
    if code_commit is None:
        try:
            code_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=package, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            code_commit = None
    manifest = {'schema_version': 1, 'run_id': run_dir.name, 'created_at': time.time(),
                'code_origin': {'git_head': code_commit, 'verified_master_sha': verified_master_sha,
                                'package_path': str(package), 'file_hashes_authoritative': True},
                'config': config, 'datasets': bindings,
                'files': {str(p.relative_to(run_dir)): sha256(p) for p in frozen}}
    atomic_json(run_dir / 'manifest.json', manifest)
    (run_dir / 'manifest.sha256').write_text(sha256(run_dir / 'manifest.json') + '\n')
    atomic_json(run_dir / 'state.json', {'tasks': {}, 'cooldowns': {}})
    summarize(run_dir)
    return run_dir


def verify(run_dir: Path) -> dict:
    if sha256(run_dir / 'manifest.json') != (run_dir / 'manifest.sha256').read_text().strip():
        raise ValueError('Frozen manifest changed')
    manifest = read_json(run_dir / 'manifest.json')
    for name, expected in manifest['files'].items():
        if sha256(run_dir / name) != expected:
            raise ValueError('Frozen input/code changed: ' + name)
    return manifest


def import_history(run_dir: Path, imports: list, parent: Path) -> None:
    tasks = {t['task_id']: t for t in json.loads((run_dir / 'tasks.json').read_text())}
    for item in imports:
        task = tasks[item['task_id']]
        source = resolve_path(parent, item['result'])
        dataset = resolve_path(parent, item['dataset'])
        if sha256(source) != item['result_sha256'] or sha256(dataset) != item['dataset_sha256']:
            raise ValueError('Historical source hash mismatch')
        old = next(r for r in read_jsonl(dataset) if r['instance_id'] == task['instance_id'])
        current = next(r for r in read_jsonl(run_dir / (task['task_type'] + '.jsonl')) if r['instance_id'] == task['instance_id'])
        from swe_gpu_bench.evaluate.use_cases.campaign_audit import classify
        raw = read_json(source)
        if raw.get('instance_id') != task['instance_id']:
            raise ValueError('Historical result instance_id differs from the selected task')
        if raw.get('mode') is not None and raw['mode'] != task['mode']:
            raise ValueError('Historical result mode differs from the selected task')
        decision = classify(raw)
        history = {'task': task, 'source': str(source), 'source_sha256': item['result_sha256'],
                   'dataset_sha256': item['dataset_sha256'], 'historical': True,
                   'data_changed': digest(old) != digest(current),
                   'changed_fields': sorted(k for k in old.keys() | current.keys() if old.get(k) != current.get(k)),
                   'raw_result': raw, 'raw_category': decision['category'], 'reason': decision['reason']}
        history['tests_executed'] = decision['category'] in ('passed', 'test_failed')
        if task['task_type'] == 'performance':
            # Historical LLM judgments cannot be imported as deterministic scores.
            history.update(raw_category='infrastructure_blocked', reason='historical_performance_requires_numeric_revalidation')
        atomic_json(run_dir / 'history' / (task['task_id'] + '.json'), history)


def result_for(run_dir: Path, task_id: str) -> dict | None:
    path = run_dir / 'results' / (task_id + '.json')
    return read_json(path) if path.exists() else None


def history_requires_run(task: dict, history: dict) -> bool:
    return bool(history) and (history['data_changed'] or task['task_type'] == 'performance')


def reusable_history(run_dir: Path, task: dict) -> bool:
    path = run_dir / 'history' / (task['task_id'] + '.json')
    return path.exists() and not history_requires_run(task, read_json(path))


def transient_failures(run_dir: Path, task_id: str) -> int:
    results = [read_json(p) for p in (run_dir / 'attempts' / task_id).glob('attempt-*/result.json')]
    return sum(r.get('retryable', False) and r.get('reason') != 'model_calls_deferred' for r in results)


def summarize(run_dir: Path) -> dict:
    manifest = read_json(run_dir / 'manifest.json')
    config = manifest['config']
    state = read_json(run_dir / 'state.json')
    rows = []
    for task in json.loads((run_dir / 'tasks.json').read_text()):
        result = result_for(run_dir, task['task_id'])
        historical = run_dir / 'history' / (task['task_id'] + '.json')
        has_current_result = result is not None
        result = result if has_current_result else (read_json(historical) if historical.exists() else {})
        needs_run = not has_current_result and history_requires_run(task, result)
        raw = result.get('raw_category', 'missing')
        reason = result.get('reason', 'no_result')
        excluded = any(e['instance_id'] == task['instance_id'] and e['task_type'] == task['task_type']
                       and e['dataset_sha256'] == task['dataset_sha256'] for e in config.get('exclusions', []))
        api_pending = reason in ('rate_limit', 'model_calls_deferred')
        scored = 'excluded' if excluded else 'unresolved' if api_pending or needs_run else 'passed' if raw == 'passed' else 'incorrect'
        if not config['scoring'].get('nonpass_as_incorrect', True) and raw not in REAL_OUTCOMES and not excluded:
            scored = 'unresolved'
        rows.append({**task, 'raw_category': raw, 'reason': reason, 'scored_outcome': scored,
                     'assumed_incorrect': scored == 'incorrect' and raw not in REAL_OUTCOMES,
                     'requires_revalidation': needs_run,
                     'historical': result.get('historical', False), 'data_changed': result.get('data_changed', False),
                     'tests_executed': result.get('tests_executed', False),
                     'current_input_executed': bool(result.get('tests_executed')) and not result.get('historical', False),
                     'attempt': state['tasks'].get(task['task_id'], {}).get('attempt'),
                     'result': str(run_dir / 'results' / (task['task_id'] + '.json')) if has_current_result else None})
    groups = defaultdict(list)
    for row in rows:
        groups[(row['task_type'], row['mode'], row['model'])].append(row)
    totals = []
    for group, items in groups.items():
        eligible = [r for r in items if r['scored_outcome'] != 'excluded']
        denominator = sum(r['scored_outcome'] in ('passed', 'incorrect') for r in eligible)
        passed = sum(r['scored_outcome'] == 'passed' for r in eligible)
        evaluated_rows = [r for r in eligible if not r['requires_revalidation'] and r['raw_category'] in REAL_OUTCOMES]
        evaluated = len(evaluated_rows)
        totals.append(dict(zip(('task_type', 'mode', 'model'), group)) | {
            'positions': len(items), 'eligible': len(eligible), 'scoring_denominator': denominator,
            'passed': passed, 'scored_accuracy': passed / denominator if denominator else None,
            'evaluated': evaluated, 'evaluated_accuracy': sum(r['raw_category'] == 'passed' for r in evaluated_rows) / evaluated if evaluated else None,
            'evaluated_coverage': evaluated / len(eligible) if eligible else None,
            'executed_test_coverage': sum(r['tests_executed'] for r in eligible) / len(eligible) if eligible else None,
            'current_input_coverage': sum(r['current_input_executed'] for r in eligible) / len(eligible) if eligible else None,
            'assumed_incorrect': sum(r['assumed_incorrect'] for r in eligible),
            'unresolved': sum(r['scored_outcome'] == 'unresolved' for r in eligible),
            'raw_categories': dict(Counter(r['raw_category'] for r in items))})
    summary = {'run': str(run_dir), 'positions': len(rows), 'groups': totals,
               'active': sum(s.get('status') not in TERMINAL for s in state['tasks'].values()),
               'note': 'Scoring assumptions do not establish model fault. Historical imports are not current-input executions.'}
    atomic_jsonl(run_dir / 'results-table.jsonl', rows)
    atomic_json(run_dir / 'summary.json', summary)
    return summary


def tick(run_dir: Path, platform) -> dict:
    manifest = verify(run_dir)
    config = manifest['config']
    state = read_json(run_dir / 'state.json')
    tasks = json.loads((run_dir / 'tasks.json').read_text())
    # Recover durable submission intents even if the controller crashed before
    # writing its state index. UNKNOWN always occupies one concurrency slot.
    for task in tasks:
        key = task['task_id']
        attempts = sorted((run_dir / 'attempts' / key).glob('attempt-*'))
        if attempts:
            latest = attempts[-1]
            number = int(latest.name.split('-')[1])
            if state['tasks'].get(key, {}).get('attempt', 0) < number:
                state['tasks'][key] = {'attempt': number, 'status': 'UNKNOWN'}
        entry = state['tasks'].get(key)
        if not entry:
            continue
        attempt = run_dir / 'attempts' / key / f"attempt-{entry['attempt']:04d}"
        current = result_for(run_dir, key)
        if entry['status'] in TERMINAL and current and current.get('attempt_path') == str(attempt):
            continue
        receipt = platform.poll(attempt) if entry['status'] not in TERMINAL else entry
        entry.update(receipt)
        if receipt['status'] in TERMINAL:
            result_path = attempt / 'result.json'
            if result_path.exists():
                result = read_json(result_path)
            else:
                result = {'raw_category': 'infrastructure_blocked', 'reason': 'platform_terminated_without_result',
                          'retryable': True, 'tests_executed': False}
                atomic_json(result_path, result)
            atomic_json(run_dir / 'results' / (key + '.json'), {**result, 'attempt_path': str(attempt)})
            if result.get('reason') == 'rate_limit':
                state['cooldowns'][task['model']] = max(state['cooldowns'].get(task['model'], 0), time.time() + config['cooldown_seconds'])
                atomic_json(run_dir / 'cooldown-events' / (new_id() + '.json'), {'model': task['model'], 'until': state['cooldowns'][task['model']]})
    for event in (run_dir / 'cooldown-events').glob('*.json'):
        item = read_json(event)
        state['cooldowns'][item['model']] = max(state['cooldowns'].get(item['model'], 0), item['until'])
    # Generation may spend time preparing source/environment after its initial
    # check. Adapters check this shared gate immediately before the API call.
    from swe_gpu_bench.shared.model_pause import pause_until, publish_pause
    for model, until in state['cooldowns'].items():
        path = run_dir / 'model-pauses' / (safe_name(model) + '.json')
        if until > pause_until(path):
            publish_pause(path, {'until': until})
    active = sum(e['status'] not in TERMINAL for e in state['tasks'].values())
    for task in tasks:
        if active >= config['concurrency'] or (run_dir / 'STOP').exists():
            break
        key = task['task_id']
        entry = state['tasks'].get(key)
        if entry and entry['status'] not in TERMINAL:
            continue
        if any(e['instance_id'] == task['instance_id'] and e['task_type'] == task['task_type'] for e in config.get('exclusions', [])):
            continue
        if not entry and reusable_history(run_dir, task):
            continue
        previous = result_for(run_dir, key)
        number = entry['attempt'] + 1 if entry else 1
        if previous and (not previous.get('retryable') or transient_failures(run_dir, key) > config['max_transient_retries']):
            continue
        candidate = run_dir / 'candidates' / key / 'candidate.json'
        if not candidate.exists() and state['cooldowns'].get(task['model'], 0) > time.time():
            continue
        attempt = run_dir / 'attempts' / key / f'attempt-{number:04d}'
        attempt.mkdir(parents=True, exist_ok=False)
        atomic_json(attempt / 'task.json', task)
        # Intent is persisted before any remote write. submit is called once.
        atomic_json(attempt / 'intent.json', {'task_id': key, 'attempt': number, 'created_at': time.time(),
                                              'caption': 'swegpu-' + digest([run_dir.name, key, number])[:32]})
        state['tasks'][key] = {'attempt': number, 'status': 'UNKNOWN'}
        atomic_json(run_dir / 'state.json', state)
        try:
            receipt = platform.submit(run_dir, task, attempt, manifest)
        except (OSError, ValueError, subprocess.SubprocessError):
            receipt = {'status': 'UNKNOWN'}
        state['tasks'][key].update(receipt)
        active += 1
        atomic_json(run_dir / 'state.json', state)
    atomic_json(run_dir / 'state.json', state)
    return summarize(run_dir)


def control(run_dir: Path, once=False):
    from swe_gpu_bench.adapters.campaign import Local, Docker, External
    config = verify(run_dir)['config']
    platform = {'local': Local, 'docker': Docker, 'external': External}[config['backend']]()
    with directory_lock(run_dir / 'controller.lock'):
        while True:
            summary = tick(run_dir, platform)
            if once or (run_dir / 'STOP').exists():
                return summary
            state = read_json(run_dir / 'state.json')
            tasks = json.loads((run_dir / 'tasks.json').read_text())
            unfinished = False
            for task in tasks:
                key = task['task_id']
                if reusable_history(run_dir, task):
                    continue
                if any(e['instance_id'] == task['instance_id'] and e['task_type'] == task['task_type'] for e in config.get('exclusions', [])):
                    continue
                entry, result = state['tasks'].get(key, {}), result_for(run_dir, key)
                if not result or entry.get('status') not in TERMINAL or (result.get('retryable') and transient_failures(run_dir, key) <= config['max_transient_retries']):
                    unfinished = True
            if not unfinished:
                return summary
            time.sleep(config['poll_seconds'])


def add_arguments(parser: argparse.ArgumentParser):
    sub = parser.add_subparsers(dest='action', required=True)
    prepare_parser = sub.add_parser('prepare')
    prepare_parser.add_argument('--config', type=Path, required=True)
    prepare_parser.add_argument('--require-master', action='store_true',
                                help='Require a clean source checkout at the live origin/master before freezing')
    for name in ('run', 'resume', 'status', 'summarize'):
        item = sub.add_parser(name)
        item.add_argument('--run', type=Path, required=True, dest='run_dir')
        if name in ('run', 'resume'):
            item.add_argument('--once', action='store_true')


def run(args):
    if args.action == 'prepare':
        print(prepare(args.config, require_master=getattr(args, 'require_master', False)))
    else:
        run_dir = args.run_dir.resolve()
        verify(run_dir)
        frozen = run_dir / 'code'
        if not Path(__file__).resolve().is_relative_to(frozen):
            command = [sys.executable, '-m', 'swe_gpu_bench', 'evaluate', 'campaign', args.action, '--run', str(run_dir)]
            if getattr(args, 'once', False):
                command.append('--once')
            return subprocess.call(command, env={**os.environ, 'PYTHONPATH': str(frozen)})
        value = control(run_dir, args.once) if args.action in ('run', 'resume') else summarize(run_dir)
        print(json.dumps(value, indent=2))
    return 0

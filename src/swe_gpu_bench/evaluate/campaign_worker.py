"""One immutable attempt; candidate generation and fresh validation are separate."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

from swe_gpu_bench.shared.artifacts import atomic_json, atomic_text, read_json, read_jsonl, sha256
from swe_gpu_bench.evaluate.campaign import digest, verify
from swe_gpu_bench.evaluate.metrics import MetricError, compare, parse, recipe


def failure(reason, *, retryable=False, category='infrastructure_blocked'):
    return {'raw_category': category, 'reason': reason, 'retryable': retryable, 'tests_executed': False}


def seal_candidate(run: Path, task: dict, source: Path) -> Path:
    target = run / 'candidates' / task['task_id']
    target.mkdir(parents=True, exist_ok=True)
    marker = target / 'candidate.json'
    if marker.exists():
        saved = read_json(marker)
        if saved['sha256'] != sha256(source):
            raise ValueError('A submitted answer cannot be replaced')
    else:
        atomic_text(target / 'candidate.diff', source.read_text())
        atomic_json(marker, {'path': str(target / 'candidate.diff'), 'sha256': sha256(source),
                             'record_sha256': task['record_sha256'], 'source': str(source),
                             'mode': task['mode'], 'model': task['model'], 'created_at': time.time()})
    return target / 'candidate.diff'


def saved_candidate(run: Path, task: dict) -> Path | None:
    marker = run / 'candidates' / task['task_id'] / 'candidate.json'
    if not marker.exists():
        if (marker.parent / 'candidate.diff').exists():
            raise ValueError('Incomplete candidate seal; regeneration prohibited')
        return None
    saved = read_json(marker)
    path = Path(saved['path'])
    if saved['record_sha256'] != task['record_sha256'] or sha256(path) != saved['sha256']:
        raise ValueError('Saved candidate binding mismatch; regeneration prohibited')
    return path


def generate(run: Path, task: dict, local: Path, config: dict) -> tuple[Path | None, dict]:
    if candidate := saved_candidate(run, task):
        return candidate, {'reused_candidate': True}
    for prior in sorted((run / 'attempts' / task['task_id']).glob('attempt-*'), reverse=True):
        if task['mode'] == 'oracle':
            root = prior / 'oracle' / task['instance_id']
            metadata = root / 'metadata.json'
            patch = root / 'pred.diff'
            complete = metadata.is_file() and read_json(metadata).get('pred_patch_apply_check_status') == 'valid'
        else:
            root = prior / 'generation' / task['instance_id'] / 'artifacts'
            metadata = root / 'generation-result.json'
            patch = root / 'agent.diff'
            saved = read_json(metadata) if metadata.is_file() else {}
            complete = saved.get('status') == 'generated' and saved.get('agent_exit_code') == 0 and not saved.get('agent_error_message')
        if complete and patch.is_file() and patch.read_text().strip():
            return seal_candidate(run, task, patch), {'reused_candidate': True, 'recovered_generation': str(metadata)}
    cooldowns = [read_json(p) for p in (run / 'cooldown-events').glob('*.json')]
    if any(e['model'] == task['model'] and e['until'] > time.time() for e in cooldowns):
        return None, failure('model_calls_deferred', retryable=True)
    from swe_gpu_bench.shared.config import resolve_profile
    from swe_gpu_bench.adapters.llm import _profile_api_key
    from swe_gpu_bench.adapters.agent import build_agent_runner
    from swe_gpu_bench.evaluate.use_cases.eval import execute_eval
    from swe_gpu_bench.evaluate.use_cases.oracle import generate_oracle_patch
    from swe_gpu_bench.evaluate.use_cases.campaign_audit import RATE_LIMIT, TRANSIENT, classify
    profile = resolve_profile(profile_name=task['profile'], profile_kind='text' if task['mode'] == 'oracle' else 'agent',
                              config_path=run / 'profiles.toml')
    env_file = Path(config['credential_env_file']) if config.get('credential_env_file') else None
    resources = Path(__file__).parent / 'resources'
    dataset = run / (task['task_type'] + '.jsonl')
    if task['mode'] == 'oracle':
        metadata = generate_oracle_patch(task['instance_id'], dataset, local / 'oracle', resources / 'oracle_generate.md',
                                         model_spec=task['model'], profile=profile, env_file=env_file, isolate_history=True, request_attempts=1,
                                         max_tokens=config.get('max_output_tokens', 16384), request_timeout=config.get('model_timeout_seconds', 600))
        if metadata.get('skip_reason'):
            return None, {**failure(metadata['skip_reason']), 'generation': metadata}
        if metadata.get('pred_patch_apply_check_status') != 'valid':
            return None, {**failure('completed_response_invalid_patch', category='model_output_error'), 'generation': metadata}
        path = Path(metadata['pred_patch_path'])
    else:
        from swe_gpu_bench.shared.solver_guard import preflight
        # Check tools, withheld files and actual CUDA before incurring model cost.
        preflight(local / 'guard-preflight')
        key = _profile_api_key(profile, env_file=env_file)
        profile = replace(profile, extra_env={**profile.extra_env, profile.api_key_env: key})
        runner = build_agent_runner(profile=profile, extra_args=['--ephemeral'] if profile.framework == 'codex' else [])
        metadata = execute_eval(task['instance_id'], dataset, local / 'generation', resources / 'agent_eval.md',
                                agent_name=task['model'], agent_runner=runner, isolate_history=True,
                                strict_commands=True, generate_only=True)
        if metadata.get('status') != 'generated':
            diagnosis = classify(metadata)
            return None, {**failure(diagnosis['reason'], retryable=diagnosis['retryable']), 'generation': metadata}
        if metadata.get('agent_exit_code') or metadata.get('agent_error_message'):
            message = metadata.get('agent_error_message') or ''
            if RATE_LIMIT.search(message):
                return None, {**failure('rate_limit', retryable=True), 'generation': metadata}
            if TRANSIENT.search(message):
                return None, {**failure('model_service_transient', retryable=True), 'generation': metadata}
            return None, {**failure('agent_invocation_failed'), 'generation': metadata}
        path = Path(metadata['candidate_path'])
        if not path.read_text().strip():
            return None, {**failure('completed_empty_candidate', category='model_output_error'), 'generation': metadata}
    return seal_candidate(run, task, path), {'generation': metadata, 'reused_candidate': False}


def validate(record: dict, task: dict, candidate: Path, local: Path) -> dict:
    from swe_gpu_bench.evaluate.use_cases.eval import (
        prepare_eval_case_dir, run_commands, apply_test_patch, capture_agent_diff,
        untracked_paths, should_rerun_setup_after_agent,
    )
    from swe_gpu_bench.core.campaign_repo import clone_or_update_repo
    from swe_gpu_bench.shared.patch_utils import apply_pred_patch
    from swe_gpu_bench.test_patch_utils import is_test_patch_failure_status
    from swe_gpu_bench.evaluate.use_cases.solver_isolation import isolate_git_history
    from swe_gpu_bench.evaluate.use_cases.campaign_audit import classify

    def prepare_case(name):
        paths = prepare_eval_case_dir(record, local / name, Path(__file__).parent / 'resources/agent_eval.md')
        clone_or_update_repo(paths, record['repo'], record['base_commit'], strict_commit=True,
                             cache_root=Path(os.environ['SWEBENCH_REPO_CACHE_ROOT']))
        isolate_git_history(paths.repo_dir, record['base_commit'])
        return paths

    def succeeded(rows):
        return all(r['exit_code'] == 0 for r in rows)

    def environment(paths, phase):
        interpreters = {str(p.parent.relative_to(paths.artifacts_dir)): p.parent / 'bin/python'
                        for p in paths.artifacts_dir.glob('*/pyvenv.cfg')}
        if os.getenv('VIRTUAL_ENV'):
            interpreters['command-env'] = Path(os.environ['VIRTUAL_ENV']) / 'bin/python'
        versions = {}
        for name, python in sorted(interpreters.items()):
            completed = subprocess.run([str(python), '-m', 'pip', 'list', '--format=json'],
                                       capture_output=True, text=True, timeout=60)
            if completed.returncode:
                raise MetricError('environment_inventory_failed')
            # Editable checkout locations necessarily differ between base and
            # candidate; compare the installed distributions, not scratch paths.
            versions[name] = sorted(
                ({'name': p['name'].lower().replace('_', '-'), 'version': p['version']}
                 for p in json.loads(completed.stdout)), key=lambda p: p['name'])
        atomic_json(paths.command_logs_dir / (phase + '-environment.json'), versions)
        return versions

    def measure(paths, prefix, spec):
        setup = run_commands(record.get('perf_setup_commands', []), prefix + '_setup', paths)
        if not succeeded(setup):
            raise MetricError(prefix + '_setup_failed')
        outputs = run_commands(record['perf_commands'], prefix, paths)
        if not outputs or not succeeded(outputs):
            raise MetricError(prefix + '_commands_failed')
        text = '\n'.join(Path(r['stdout_log']).read_text() for r in outputs)
        if artifact := spec.get('artifact'):
            source = paths.case_dir / artifact
            if not source.is_file():
                raise MetricError('missing_metric_artifact')
            text = source.read_text()
            atomic_text(paths.command_logs_dir / (prefix + '-metric.json'), text)
        measurement = parse(spec, text)
        return measurement, setup + outputs, environment(paths, prefix)

    spec = recipe(record) if task['task_type'] == 'performance' else None
    base_metrics, base_logs = None, []
    if spec:
        base_paths = prepare_case('base')
        base_setup = run_commands(record.get('setup_commands', []), 'setup', base_paths)
        if not succeeded(base_setup):
            return {**failure('base_setup_failed'), 'base_setup_results': base_setup}
        try:
            base_metrics, base_logs, base_environment = measure(base_paths, 'base_perf', spec)
        except MetricError as exc:
            return failure(str(exc))
    paths = prepare_case('validation')
    setup = run_commands(record.get('setup_commands', []), 'setup', paths)
    result = {'instance_id': record['instance_id'], 'mode': task['mode'], 'status': 'error',
              'base_commit': record['base_commit'], 'candidate_sha256': sha256(candidate),
              'setup_commands': record.get('setup_commands', []), 'setup_results': setup,
              'base_perf_results': base_logs, 'test_results': []}
    if not succeeded(setup):
        return {**failure('candidate_setup_failed'), 'execution': result}
    before = untracked_paths(paths.repo_dir)
    applied = apply_pred_patch(paths.repo_dir, candidate.read_text())
    if applied.status != 'applied':
        return {**failure('candidate_patch_apply_failed'), 'execution': result}
    capture_agent_diff(paths, before)
    test_patch_status = apply_test_patch(record.get('test_patch', ''), paths) if record.get('test_patch') else 'no_test_patch'
    result['test_patch_status'] = test_patch_status
    if is_test_patch_failure_status(test_patch_status):
        return {**failure('test_patch_apply_failed'), 'execution': result}
    post = run_commands(record.get('setup_commands', []), 'post_patch_setup', paths) if should_rerun_setup_after_agent(record.get('setup_commands', [])) else []
    result['post_patch_setup_results'] = post
    if succeeded(post):
        result['test_results'] = run_commands(record.get('test_commands', []), 'test', paths)
    result['status'] = 'passed' if result['test_results'] and succeeded(result['test_results']) else 'failed'
    decision = classify(result)
    outcome = {'raw_category': decision['category'], 'reason': decision['reason'], 'retryable': decision['retryable'],
               'tests_executed': decision['category'] in ('passed', 'test_failed'), 'execution': result}
    if spec and decision['category'] == 'passed':
        try:
            measured, logs, candidate_environment = measure(paths, 'candidate_perf', spec)
            result['candidate_perf_results'] = logs
            if base_environment != candidate_environment:
                raise MetricError('base_candidate_environment_mismatch')
            numeric = compare(spec, base_metrics, measured, correctness_passed=True)
            outcome.update(performance=numeric, raw_category='passed' if numeric['improved'] else 'performance_not_improved',
                           reason='deterministic_numeric_comparison')
        except MetricError as exc:
            outcome.update(raw_category='infrastructure_blocked', reason=str(exc), retryable=False)
    return outcome


def execute(run: Path, attempt: Path, local: Path) -> dict:
    manifest = verify(run)
    config = manifest['config']
    task = read_json(attempt / 'task.json')
    record = next(r for r in read_jsonl(run / (task['task_type'] + '.jsonl')) if r['instance_id'] == task['instance_id'])
    if digest(record) != task['record_sha256']:
        raise ValueError('Task record binding mismatch')
    os.environ.update(SWEBENCH_SOLVER_ISOLATION='1', SWEBENCH_VERIFY_SOURCE='1',
                      SWEBENCH_REPO_CACHE_ROOT=str(local / 'repo-cache'),
                      SWEBENCH_GIT_TIMEOUT_SECONDS=str(config.get('git_timeout_seconds', 600)),
                      SWEBENCH_UV_CACHE_ROOT=str(local / 'uv-cache'), UV_LINK_MODE='copy')
    (local / 'repo-cache').mkdir(exist_ok=True)
    (local / 'codex-home').mkdir(exist_ok=True)
    os.environ['CODEX_HOME'] = str(local / 'codex-home')
    os.environ['SWEBENCH_MODEL_PAUSE_FILE'] = str(run / 'model-pauses' / (task['model'] + '.json'))
    candidate, generation = generate(run, task, local, config)
    if candidate is None:
        return generation
    return {**validate(record, task, candidate, local), **generation, 'candidate_sha256': sha256(candidate)}


def copy_evidence(local: Path, attempt: Path):
    names = ('prompt.md', 'pred.diff', 'raw_response.txt', 'edits.json', 'metadata.json', 'usage.jsonl',
             'inputs/instance.json', 'inputs/prompt.md', 'artifacts/agent.diff', 'artifacts/source-candidates.json',
             'artifacts/generation-result.json')
    for phase in ('generation', 'oracle', 'base', 'validation'):
        root = local / phase
        if not root.is_dir():
            continue
        for case in root.iterdir():
            if not case.is_dir() or case.is_symlink() or case.name.startswith('_'):
                continue
            candidates = [case / name for name in names]
            logs = case / 'artifacts/command-logs'
            if logs.is_dir():
                candidates.extend(p for p in logs.iterdir() if p.suffix in ('.log', '.json', '.jsonl'))
            for source in candidates:
                if source.is_file() and not source.is_symlink():
                    destination = attempt / source.relative_to(local)
                    atomic_text(destination, source.read_text(errors='replace'))


def supervise(run: Path, attempt: Path):
    manifest = verify(run)
    config = manifest['config']
    task = read_json(attempt / 'task.json')
    (attempt / 'execution.lock').mkdir(exist_ok=False)
    local = Path(os.getenv('SWEBENCH_NODE_ROOT', config['local_root'])) / run.name / digest(str(attempt))[:16]
    local.mkdir(parents=True, exist_ok=False)
    started = time.time()
    outcome = None
    monitor = None
    monitor_output = None
    try:
        command = ['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total', '--format=csv,noheader']
        hardware = subprocess.run(command, capture_output=True, text=True, timeout=30)
        atomic_json(attempt / 'environment.json', {'command': command, 'exit_code': hardware.returncode,
                    'stdout': hardware.stdout, 'stderr': hardware.stderr, 'runtime_image': os.getenv('SWEBENCH_RUNTIME_IMAGE'),
                    'requested_image': task['runtime_image'], 'python': sys.version, 'local': str(local),
                    'code_sha256': sha256(run / 'manifest.json'), 'job_id': os.getenv('SWEBENCH_JOB_ID')})
        if hardware.returncode or config['hardware']['gpu_name'] not in hardware.stdout:
            raise RuntimeError('requested_gpu_unavailable')
        visible = os.getenv('CUDA_VISIBLE_DEVICES')
        if (visible is None and len(hardware.stdout.strip().splitlines()) != 1) or (visible is not None and len(visible.split(',')) != 1):
            raise RuntimeError('exactly_one_visible_gpu_required')
        if os.getenv('SWEBENCH_RUNTIME_IMAGE') != task['runtime_image']:
            raise RuntimeError('runtime_image_attestation_missing_or_mismatched')
        monitor_output = (attempt / 'gpu-samples.csv').open('w')
        monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,uuid,utilization.gpu,memory.used',
                                    '--format=csv', '--loop=5'], stdout=monitor_output, stderr=subprocess.DEVNULL)
        software = subprocess.run(['bash', '-c', 'python3 --version; nvcc --version; uv --version; codex --version; opencode --version'],
                                  capture_output=True, text=True, timeout=60)
        atomic_json(attempt / 'software.json', {'stdout': software.stdout, 'stderr': software.stderr, 'exit_code': software.returncode})
        repair = config.get('environment_repairs', {}).get(task['task_type'] + '--' + task['instance_id'], {})
        env = dict(os.environ)
        for variable in ('SWEBENCH_SETUP_COMMAND_OVERRIDES', 'SWEBENCH_TEST_COMMAND_OVERRIDES',
                         'SWEBENCH_DEFER_SETUP_COMMAND_HASHES', 'SWEBENCH_DEFER_SETUP_TO_TEST_HASHES'):
            env.pop(variable, None)
        for field, variable in [('setup_command_overrides', 'SWEBENCH_SETUP_COMMAND_OVERRIDES'),
                                ('test_command_overrides', 'SWEBENCH_TEST_COMMAND_OVERRIDES'),
                                ('defer_setup_command_sha256', 'SWEBENCH_DEFER_SETUP_COMMAND_HASHES')]:
            if field in repair:
                env[variable] = json.dumps(repair[field])
        env.update(PYTHONPATH=str(run / 'code'), UV_LINK_MODE='copy', PYTHONDONTWRITEBYTECODE='1')
        # A fresh environment protects the evaluator from recipe pip installs.
        venv = local / 'command-env'
        with (attempt / 'command-environment.log').open('w') as out:
            subprocess.run(['uv', 'venv', '--python', repair.get('python', 'python3'), '--system-site-packages', '--seed', str(venv)],
                           stdout=out, stderr=subprocess.STDOUT, check=True, timeout=300)
        env.update(PATH=str(venv / 'bin') + os.pathsep + env['PATH'], VIRTUAL_ENV=str(venv))
        cmd = [sys.executable, '-m', __name__, '--run', str(run), '--attempt', str(attempt), '--local', str(local), '--execute']
        # __name__ is __main__ under -m; use the stable module path.
        cmd[2] = 'swe_gpu_bench.evaluate.campaign_worker'
        with (attempt / 'execute.stdout.log').open('w') as out, (attempt / 'execute.stderr.log').open('w') as err:
            process = subprocess.Popen(cmd, env=env, stdout=out, stderr=err, start_new_session=True)
            while process.poll() is None:
                copy_evidence(local, attempt)
                atomic_json(attempt / 'heartbeat.json', {'pid': process.pid, 'time': time.time(), 'elapsed_seconds': time.time() - started})
                if time.time() - started >= config['timeout_seconds']:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    outcome = failure('case_timeout')
                    break
                time.sleep(2)
        if outcome is None:
            outcome = read_json(local / 'result.json') if (local / 'result.json').exists() else failure('worker_process_failure', retryable=True)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        # Full traceback stays in the attempt; reports contain no service payload.
        atomic_text(attempt / 'diagnostic.log', traceback.format_exc())
        outcome = failure('worker_preflight_blocked:' + type(exc).__name__)
    finally:
        if monitor is not None:
            monitor.terminate()
            try:
                monitor.wait(timeout=10)
            except subprocess.TimeoutExpired:
                monitor.kill()
                monitor.wait()
        if monitor_output is not None:
            monitor_output.close()
        copy_evidence(local, attempt)
        outcome = outcome or failure('worker_interrupted')
        # References in the report point to retained evidence, not node-local paths.
        outcome = json.loads(json.dumps(outcome).replace(str(local), str(attempt)))
        outcome.update(task=task, elapsed_seconds=time.time() - started, local_scratch=str(local))
        atomic_json(attempt / 'result.json', outcome)
        atomic_json(attempt / 'exit.json', {'finished_at': time.time()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--attempt', type=Path, required=True)
    parser.add_argument('--local', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.execute:
        from swe_gpu_bench.evaluate.use_cases.campaign_audit import RATE_LIMIT, TRANSIENT
        from swe_gpu_bench.shared.model_pause import ModelCallsPausedError
        try:
            result = execute(args.run, args.attempt, args.local)
        except Exception as exc:
            traceback.print_exc()
            text = type(exc).__name__ + ': ' + str(exc)
            if isinstance(exc, ModelCallsPausedError):
                result = failure('model_calls_deferred', retryable=True)
            else:
                result = failure('rate_limit' if RATE_LIMIT.search(text) else 'transient_execution_error' if TRANSIENT.search(text) else 'execution_error:' + type(exc).__name__,
                                 retryable=bool(RATE_LIMIT.search(text) or TRANSIENT.search(text)))
        atomic_json(args.local / 'result.json', result)
    else:
        supervise(args.run, args.attempt)


if __name__ == '__main__':
    main()

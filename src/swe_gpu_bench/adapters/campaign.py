"""Execution adapters. Receipts, not a timeout or absence from a list, free slots."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys

from swe_gpu_bench.shared.artifacts import atomic_json, atomic_text, read_json


def worker_command(run: Path, attempt: Path):
    return [sys.executable, '-m', 'swe_gpu_bench.evaluate.campaign_worker',
            '--run', str(run), '--attempt', str(attempt)]


class Local:
    def submit(self, run, task, attempt, manifest):
        env = {**os.environ, 'PYTHONPATH': str(run / 'code')}
        with (attempt / 'worker.stdout.log').open('w') as out, (attempt / 'worker.stderr.log').open('w') as err:
            process = subprocess.Popen(worker_command(run, attempt), stdout=out, stderr=err, env=env, start_new_session=True)
        receipt = {'status': 'RUNNING', 'pid': process.pid, 'host': socket.gethostname(),
                   'start_ticks': Path(f'/proc/{process.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]}
        atomic_json(attempt / 'receipt.json', receipt)
        return receipt

    def poll(self, attempt):
        if (attempt / 'exit.json').exists():
            return {'status': 'DONE'}
        path = attempt / 'receipt.json'
        if not path.exists():
            # Crash between Popen and recording PID: cannot safely resubmit.
            return {'status': 'UNKNOWN'}
        receipt = read_json(path)
        if receipt.get('host') != socket.gethostname():
            return {'status': 'UNKNOWN'}
        try:
            stat = Path(f"/proc/{receipt['pid']}/stat").read_text().rsplit(')', 1)[1].split()
        except FileNotFoundError:
            return {'status': 'FAILED'}
        if stat[19] != receipt['start_ticks'] or stat[0] == 'Z':
            return {'status': 'FAILED'}
        return {'status': 'RUNNING'}


def bootstrap_script(run: Path, attempt: Path, task: dict) -> str:
    """Portable worker launch contract for containers and external schedulers."""
    resources = run / 'code/swe_gpu_bench/evaluate/resources'
    return '\n'.join([
        'set -euo pipefail', 'umask 077',
        f'exec > >(tee -a {shlex.quote(str(attempt / "bootstrap.log"))}) 2>&1',
        'LOCAL=$(mktemp -d /tmp/swe-gpu-campaign.XXXXXXXX)',
        f'python3 {shlex.quote(str(resources / "bootstrap-tools.py"))} "$LOCAL"',
        'export PATH="$LOCAL/bin:$PATH"', 'export UV_LINK_MODE=copy',
        'export UV_CACHE_DIR="$LOCAL/uv-cache"',
        'uv venv --python 3.12 "$LOCAL/evaluator"',
        f'uv pip install --python "$LOCAL/evaluator/bin/python" -r {shlex.quote(str(resources / "evaluator-requirements.txt"))}',
        f'export PYTHONPATH={shlex.quote(str(run / "code"))}',
        'export SWEBENCH_NODE_ROOT="$LOCAL"',
        f'export SWEBENCH_RUNTIME_IMAGE={shlex.quote(task["runtime_image"])}',
        'export CUDA_VISIBLE_DEVICES=0',
        f'"$LOCAL/evaluator/bin/python" -m swe_gpu_bench.evaluate.campaign_worker --run {shlex.quote(str(run))} --attempt {shlex.quote(str(attempt))}',
    ]) + '\n'


class Docker:
    """One named container and one configured device per attempt, no auto-remove."""
    def submit(self, run, task, attempt, manifest):
        from swe_gpu_bench.evaluate.campaign import TERMINAL
        config = manifest['config']
        cfg = config.get('docker', {})
        occupied = set()
        for key, entry in read_json(run / 'state.json')['tasks'].items():
            if entry['status'] not in TERMINAL:
                receipt = run / 'attempts' / key / f"attempt-{entry['attempt']:04d}" / 'receipt.json'
                if receipt.exists():
                    occupied.add(read_json(receipt).get('device'))
        device = next(str(d) for d in cfg.get('devices', ['0']) if str(d) not in occupied)
        name = read_json(attempt / 'intent.json')['caption']
        # The deterministic name recovers a lost launch response without rerunning.
        atomic_json(attempt / 'receipt.json', {'job_id': name, 'device': device})
        script = attempt / 'worker.sh'
        atomic_text(script, bootstrap_script(run, attempt, task))
        command = ['docker', 'run', '--detach', '--name', name, '--gpus', 'device=' + device,
                   '--shm-size', cfg.get('shm_size', '8g'), '--mount', f'type=bind,src={run},dst={run}',
                   '--env', 'SWEBENCH_JOB_ID=' + name]
        if cfg.get('cpus'):
            command += ['--cpus', str(cfg['cpus'])]
        if cfg.get('memory'):
            command += ['--memory', str(cfg['memory'])]
        if config.get('credential_env_file'):
            path = config['credential_env_file']
            command += ['--mount', f'type=bind,src={path},dst={path},readonly']
        for variable in cfg.get('forward_env', []):
            command += ['--env', variable]  # Value comes from the current environment.
        command += cfg.get('extra_args', [])
        command += ['--entrypoint', 'bash', task['runtime_image'], str(script)]
        atomic_json(attempt / 'submission.json', {'command': command, 'device': device})
        with (attempt / 'submit.log').open('w') as out:
            subprocess.run(command, stdout=out, stderr=subprocess.STDOUT, timeout=180, check=False)
        return self.poll(attempt)

    def poll(self, attempt):
        receipt_path = attempt / 'receipt.json'
        receipt = read_json(receipt_path) if receipt_path.exists() else {}
        name = receipt.get('job_id') or read_json(attempt / 'intent.json')['caption']
        try:
            response = subprocess.run(['docker', 'inspect', '--format', '{{json .State}}', name],
                                      capture_output=True, text=True, timeout=60, check=True)
            state = json.loads(response.stdout)
            status = state['Status']
            normalized = 'TERMINATED' if status in ('exited', 'dead') else 'RUNNING'
            return {**receipt, 'job_id': name, 'status': normalized, 'container_state': status,
                    'exit_code': state.get('ExitCode') if normalized == 'TERMINATED' else None}
        except (OSError, ValueError, KeyError, subprocess.SubprocessError):
            return {**receipt, 'job_id': name, 'status': 'UNKNOWN'}


class External:
    """Frozen, operator-supplied scheduler adapter; only normalized receipts persist."""
    def _invoke(self, operation, run, attempt):
        from swe_gpu_bench.evaluate.campaign import TERMINAL
        script = run / 'external/adapter.py'
        try:
            response = subprocess.run([sys.executable, str(script), operation, '--run', str(run),
                                       '--attempt', str(attempt)], capture_output=True, text=True,
                                      timeout=180, check=True)
            raw = json.loads(response.stdout)
            if not isinstance(raw, dict) or not isinstance(raw.get('status'), str):
                raise ValueError('Malformed adapter receipt')
            status = raw['status']
            if status not in TERMINAL | {'PENDING', 'RUNNING', 'UNKNOWN'}:
                raise ValueError('Unknown adapter status')
            if status in TERMINAL and raw.get('terminal') is not True:
                status = 'UNKNOWN'
            # Do not copy service payloads, tokens or environment variables.
            receipt = {k: raw[k] for k in ('job_id', 'completed_time', 'exit_code') if k in raw}
            receipt['status'] = status
            if not receipt.get('job_id') and status != 'UNKNOWN':
                raise ValueError('A normalized receipt requires a job ID')
            atomic_json(attempt / 'receipt.json', receipt)
            return receipt
        except (OSError, ValueError, KeyError, subprocess.SubprocessError):
            path = attempt / 'receipt.json'
            receipt = read_json(path) if path.exists() else {}
            return {**receipt, 'status': 'UNKNOWN'}

    def submit(self, run, task, attempt, manifest):
        atomic_text(attempt / 'worker.sh', bootstrap_script(run, attempt, task))
        return self._invoke('submit', run, attempt)

    def poll(self, attempt):
        return self._invoke('poll', attempt.parents[2], attempt)

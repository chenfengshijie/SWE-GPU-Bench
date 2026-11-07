import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from swe_gpu_bench.adapters.campaign import Docker, External, bootstrap_script
from swe_gpu_bench.shared.artifacts import atomic_json, read_json
from test_campaign import config


def attempt_at(root, name='task'):
    attempt = root / 'attempts' / name / 'attempt-0001'
    atomic_json(attempt / 'intent.json', {'caption':'fixture-' + name})
    return attempt


@pytest.mark.parametrize('payload,expected', [
    ({'job_id':'123','status':'RUNNING','access_token':'NEVER_PERSIST'}, 'RUNNING'),
    ({'job_id':'123','status':'TERMINATED','terminal':True}, 'TERMINATED'),
    ({'job_id':'123','status':'TERMINATED'}, 'UNKNOWN'),
    ({'job_id':'123','status':4,'completed_time':123}, 'UNKNOWN'),
    ([], 'UNKNOWN'),
])
def test_external_retains_only_normalized_proven_receipts(tmp_path, monkeypatch, payload, expected):
    attempt = attempt_at(tmp_path)
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k:SimpleNamespace(stdout=json.dumps(payload)))
    result = External().poll(attempt)
    assert result['status'] == expected
    assert 'NEVER_PERSIST' not in json.dumps(result)
    receipt = attempt / 'receipt.json'
    if receipt.exists():
        assert 'NEVER_PERSIST' not in receipt.read_text()


def test_external_lookup_failure_retains_known_job(tmp_path, monkeypatch):
    attempt = attempt_at(tmp_path)
    atomic_json(attempt / 'receipt.json', {'job_id':'123','status':'RUNNING'})
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k:(_ for _ in ()).throw(subprocess.TimeoutExpired('adapter', 180)))
    assert External().poll(attempt) == {'job_id':'123','status':'UNKNOWN'}


@pytest.mark.parametrize('state,expected', [('running','RUNNING'), ('exited','TERMINATED'), ('dead','TERMINATED')])
def test_docker_explicit_terminal_state(tmp_path, monkeypatch, state, expected):
    attempt = attempt_at(tmp_path)
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k:SimpleNamespace(stdout=json.dumps({'Status':state,'ExitCode':0})))
    assert Docker().poll(attempt)['status'] == expected


def test_missing_container_stays_unknown(tmp_path, monkeypatch):
    attempt = attempt_at(tmp_path)
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k:(_ for _ in ()).throw(subprocess.CalledProcessError(1,'docker')))
    assert Docker().poll(attempt)['status'] == 'UNKNOWN'


def test_docker_assigns_distinct_devices_and_keeps_secret_values_out_of_spec(tmp_path, monkeypatch):
    first = attempt_at(tmp_path, 'first')
    second = attempt_at(tmp_path, 'second')
    atomic_json(first / 'receipt.json', {'job_id':'first','device':'GPU-one'})
    atomic_json(tmp_path / 'state.json', {'tasks':{'first':{'attempt':1,'status':'UNKNOWN'}}})
    monkeypatch.setenv('API_KEY', 'PRIVATE_VALUE')
    commands = []
    def call(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout=json.dumps({'Status':'running'}), returncode=0)
    monkeypatch.setattr(subprocess, 'run', call)
    manifest={'config':{'docker':{'devices':['GPU-one','GPU-two'],'forward_env':['API_KEY']}}}
    receipt=Docker().submit(tmp_path, {'runtime_image':'public/image:tag'}, second, manifest)
    assert receipt['device']=='GPU-two'
    command=commands[0]
    assert command[command.index('--gpus')+1]=='device=GPU-two'
    assert 'API_KEY' in command and 'PRIVATE_VALUE' not in json.dumps(command)
    assert '--rm' not in command
    assert 'PRIVATE_VALUE' not in (second/'submission.json').read_text()


def test_external_script_frozen_before_submit(config):
    from swe_gpu_bench.evaluate.campaign import prepare, verify
    script = config.parent / 'adapter.py'
    script.write_text('print("adapter")\n')
    cfg = read_json(config);cfg.update(backend='external', external={'script':'adapter.py'})
    atomic_json(config, cfg)
    run = prepare(config)
    script.write_text('changed source')
    verify(run)
    assert (run/'external/adapter.py').read_text() == 'print("adapter")\n'
    (run/'external/adapter.py').write_text('tamper')
    with pytest.raises(ValueError, match='Frozen'):
        verify(run)


def test_docker_rejects_oversubscription(config):
    from swe_gpu_bench.evaluate.campaign import prepare
    cfg=read_json(config);cfg.update(backend='docker',docker={'devices':['0']})
    atomic_json(config,cfg)
    with pytest.raises(ValueError, match='distinct device'):
        prepare(config)


def test_worker_script_is_valid_bash_and_uses_frozen_resources(tmp_path):
    script=bootstrap_script(tmp_path, tmp_path/'attempt', {'runtime_image':'public/image:tag'})
    subprocess.run(['bash','-n'],input=script,text=True,check=True)
    assert 'code/swe_gpu_bench/evaluate/resources' in script
    assert 'CUDA_VISIBLE_DEVICES=0' in script

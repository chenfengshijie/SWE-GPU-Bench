import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import venv

import pytest

from swe_gpu_bench.evaluate import campaign_worker as worker
from swe_gpu_bench.evaluate.campaign import digest
from swe_gpu_bench.shared.artifacts import atomic_jsonl


@pytest.fixture(scope='module')
def command_environment(tmp_path_factory):
    # Task-created virtual environments can contain distributions without pip.
    path = tmp_path_factory.mktemp('command-environment') / 'venv'
    venv.EnvBuilder().create(path)
    purelib = subprocess.check_output(
        [str(path / 'bin/python'), '-c', 'import sysconfig; print(sysconfig.get_path("purelib"))'], text=True
    ).strip()
    metadata = Path(purelib) / 'example-1.0.dist-info'
    metadata.mkdir()
    (metadata / 'METADATA').write_text('Metadata-Version: 2.1\nName: example\nVersion: 1.0\n')
    return path


@pytest.mark.parametrize('task_type', ['correctness', 'performance'])
@pytest.mark.parametrize('mode', ['oracle', 'agent'])
def test_fresh_validation_four_chains(tmp_path, monkeypatch, command_environment, task_type, mode):
    from swe_gpu_bench.evaluate.use_cases import eval as ev, oracle
    from swe_gpu_bench.core import campaign_repo as repro
    from swe_gpu_bench.adapters import agent, llm
    from swe_gpu_bench.shared import solver_guard

    source = tmp_path / 'source'
    source.mkdir()
    subprocess.run(['git','init','-q',str(source)],check=True)
    (source / 'value.py').write_text('VALUE = 1\n')
    subprocess.run(['git','add','.'],cwd=source,check=True)
    subprocess.run(['git','-c','user.name=fixture','-c','user.email=fixture@example.invalid','commit','-qm','base'],cwd=source,check=True)
    base = subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    (source / 'value.py').write_text('VALUE = 2\n')
    patch = subprocess.check_output(['git','diff'],cwd=source,text=True)
    subprocess.run(['git','checkout','--','value.py'],cwd=source,check=True)
    row = dict(instance_id='org__repo__1',repo='org/repo',base_commit=base,problem_statement='Set VALUE to two',
               docker_image='fixture',patch=patch,test_patch='',setup_commands=[],
               test_commands=["python -c 'from repo.value import VALUE; assert VALUE == 2; print(\"1 passed\")'"],
               perf_setup_commands=[],perf_commands=["python -c 'from repo.value import VALUE; print(10 / VALUE)'"],
               command_contract='portable_unified_v1',performance_metric={'name':'latency','unit':'ms','workload':'one','better':'lower'})
    run = tmp_path / 'run'
    run.mkdir()
    atomic_jsonl(run / (task_type + '.jsonl'),[row])
    (run / 'profiles.toml').write_text('[profiles.text.fake]\nprovider="openai_responses"\nmodel="fake"\n[profiles.agent.fake]\nframework="codex"\nmodel="fake"\napi_key_env="FAKE_API_KEY"\n')
    local = tmp_path / 'local'
    local.mkdir()
    (local / 'repo-cache').mkdir()
    monkeypatch.setenv('SWEBENCH_REPO_CACHE_ROOT',str(local / 'repo-cache'))
    monkeypatch.setenv('VIRTUAL_ENV', str(command_environment))
    monkeypatch.setenv('PATH', str(command_environment / 'bin') + os.pathsep + os.environ['PATH'])
    # This CPU host may forbid inotify; source-observer failures are tested separately.
    monkeypatch.delenv('SWEBENCH_VERIFY_SOURCE', raising=False)

    def clone(paths, repo, base_commit, **kwargs):
        subprocess.run(['git','clone','-q',str(source),str(paths.repo_dir)],check=True)
        subprocess.run(['git','checkout','-q',base_commit],cwd=paths.repo_dir,check=True)

    monkeypatch.setattr(repro,'clone_or_update_repo',clone)
    monkeypatch.setattr(ev,'clone_or_update_repo',clone)
    monkeypatch.setattr(oracle,'clone_or_update_repo',clone)
    monkeypatch.setattr(solver_guard,'preflight',lambda _:None)  # CPU fixture, not an isolation/GPU smoke.
    monkeypatch.setattr(llm,'_profile_api_key',lambda *a,**k:'fixture')

    def invoke(paths):
        visible = json.loads(paths.instance_path.read_text())
        assert 'patch' not in visible and 'test_commands' not in visible
        assert 'assert VALUE' not in paths.prompt_path.read_text()
        assert subprocess.check_output(['git','rev-list','--count','HEAD'],cwd=paths.repo_dir,text=True).strip() == '1'
        (paths.repo_dir / 'value.py').write_text('VALUE = 2\n')
        return agent.AgentInvocationResult(command=['fixture'],exit_code=0,stdout_path=paths.agent_stdout_log,stderr_path=paths.agent_stderr_log)

    monkeypatch.setattr(agent,'build_agent_runner',lambda **kwargs:SimpleNamespace(invoke=invoke))
    monkeypatch.setattr(oracle,'generate_text_from_profile',lambda prompt, **kwargs:json.dumps({'edits':[{'path':'value.py','old':'VALUE = 1','new':'VALUE = 2'}]}))
    task = dict(task_id=task_type+'--'+mode,task_type=task_type,mode=mode,model='fake',profile='fake',instance_id=row['instance_id'],record_sha256=digest(row))
    candidate, metadata = worker.generate(run,task,local,{})
    assert candidate is not None
    result = worker.validate(row,task,candidate,local)
    assert result['raw_category'] == 'passed', result
    assert result['tests_executed']
    if task_type == 'performance':
        assert result['performance']['mean_improvement_percent'] == 50
        inventory = json.loads((local / 'base/org__repo__1/artifacts/command-logs/base_perf-environment.json').read_text())
        assert {'name': 'example', 'version': '1.0'} in inventory['command-env']
    again, reused = worker.generate(run,task,tmp_path / 'does-not-exist',{})
    assert again == candidate and reused['reused_candidate']


@pytest.mark.parametrize('output', ['collected 0 items', '1 skipped', 'Ran 0 tests'])
def test_zero_tests_not_passed(tmp_path, output):
    from swe_gpu_bench.evaluate.use_cases.campaign_audit import classify
    log = tmp_path / 'stdout.log';log.write_text(output)
    err = tmp_path / 'stderr.log';err.write_text('')
    decision = classify({'instance_id':'one','status':'passed','test_results':[{'command':'pytest','exit_code':0,'stdout_log':str(log),'stderr_log':str(err)}]})
    assert decision['category'] == 'infrastructure_blocked'
    assert decision['reason'] == 'no_tests_executed'


def test_unknown_evaluator_fields_are_not_solver_input(tmp_path, monkeypatch):
    from swe_gpu_bench.evaluate.use_cases.eval import prepare_eval_case_dir
    monkeypatch.setenv('SWEBENCH_SOLVER_ISOLATION', '1')
    template = tmp_path / 'prompt'; template.write_text('Solve it.')
    row = {'instance_id':'one', 'problem_statement':'task', 'baseline_measurements':{'value':123},
           'reference_answer':'secret', 'performance_metric':{'unit':'ms', 'name':'latency', 'baseline':123}}
    paths = prepare_eval_case_dir(row, tmp_path / 'cases', template)
    visible = json.loads(paths.instance_path.read_text())
    assert 'baseline_measurements' not in visible and 'reference_answer' not in visible
    assert visible['performance_metric'] == {'unit':'ms', 'name':'latency'}


def test_guard_preflight_fails_closed_without_namespace_support(tmp_path, monkeypatch):
    from swe_gpu_bench.shared import solver_guard
    calls = []
    def unavailable(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command, stderr='unshare: Operation not permitted')
    monkeypatch.setattr(solver_guard.subprocess, 'run', unavailable)
    with pytest.raises(subprocess.CalledProcessError):
        solver_guard.preflight(tmp_path / 'guard')
    assert len(calls) == 1 and calls[0][0] == 'unshare'


@pytest.mark.parametrize('deferred', [False, True])
def test_execute_classifies_transport_timeout_and_cooldown_without_model_failure(tmp_path, monkeypatch, deferred):
    from swe_gpu_bench.shared.model_pause import ModelCallsPausedError
    class APITimeoutError(RuntimeError):
        pass
    error = ModelCallsPausedError('cooldown') if deferred else APITimeoutError('Request timed out.')
    def fail(*args):
        raise error
    monkeypatch.setattr(worker, 'execute', fail)
    monkeypatch.setattr(worker.sys, 'argv', ['worker', '--run', str(tmp_path), '--attempt', str(tmp_path),
                                           '--local', str(tmp_path), '--execute'])
    worker.main()
    result=json.loads((tmp_path/'result.json').read_text())
    assert result['raw_category']=='infrastructure_blocked' and result['retryable']
    assert result['reason']==('model_calls_deferred' if deferred else 'transient_execution_error')

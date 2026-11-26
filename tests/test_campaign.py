import json
from pathlib import Path
import subprocess

import pytest

from swe_gpu_bench.evaluate import campaign as c
from swe_gpu_bench.evaluate.campaign_worker import seal_candidate, saved_candidate
from swe_gpu_bench.shared.artifacts import atomic_json, atomic_jsonl, read_json, sha256


@pytest.fixture
def config(tmp_path):
    row = dict(instance_id='org__repo__1', repo='org/repo', base_commit='a' * 40,
               problem_statement='Fix it', docker_image='public/image:tag', patch='gold', test_patch='hidden',
               setup_commands=[], test_commands=['pytest test.py'], perf_commands=['echo 1'],
               command_contract='portable_unified_v1', performance_metric=dict(name='latency', unit='ms', workload='one', better='lower'))
    atomic_jsonl(tmp_path / 'data.jsonl', [row])
    (tmp_path / 'profiles.toml').write_text('[profiles.text.fake]\nmodel="fake"\n[profiles.agent.fake]\nmodel="fake"\nframework="codex"\n')
    value = dict(output_root=str(tmp_path / 'runs'), profiles='profiles.toml', backend='local',
                 concurrency=2, hardware={'gpu_name': 'H100'}, max_transient_retries=2, cooldown_seconds=100,
                 datasets=[dict(task_type=k, path='data.jsonl') for k in ('correctness', 'performance')],
                 models=[dict(name='fake', oracle_profile='fake', agent_profile='fake')])
    atomic_json(tmp_path / 'config.json', value)
    return tmp_path / 'config.json'


class Platform:
    def __init__(self):
        self.submissions = []
        self.status = 'RUNNING'

    def submit(self, run, task, attempt, manifest):
        self.submissions.append(attempt)
        assert (attempt / 'intent.json').exists()
        return {'status': self.status}

    def poll(self, attempt):
        return {'status': self.status}


def test_context_limit_is_not_a_completed_model_answer(config, monkeypatch):
    from swe_gpu_bench.evaluate import campaign_worker
    from swe_gpu_bench.evaluate.use_cases import oracle
    run = c.prepare(config)
    task = json.loads((run / 'tasks.json').read_text())[0]
    monkeypatch.setattr(oracle, 'generate_oracle_patch', lambda *a, **k: {
        'skip_reason': 'context_length_exceeded', 'pred_patch_apply_check_status':'skipped'})
    candidate, result = campaign_worker.generate(run, task, run / 'scratch', {})
    assert candidate is None and result['raw_category'] == 'infrastructure_blocked'
    assert result['reason'] == 'context_length_exceeded' and not result['retryable']


@pytest.mark.parametrize('persistent', [False, True])
def test_shared_index_replacement_is_retried_without_empty_state(tmp_path, monkeypatch, persistent):
    path = tmp_path / 'state.json'
    original = Path.read_text
    calls = []
    def read(self, **kwargs):
        if self == path:
            calls.append(1)
            if persistent or len(calls) == 1:
                raise FileNotFoundError(str(path))
            return '{"tasks": {"one": {"status": "UNKNOWN"}}}'
        return original(self, **kwargs)
    monkeypatch.setattr(Path, 'read_text', read)
    if persistent:
        with pytest.raises(FileNotFoundError):
            read_json(path)
        assert len(calls) == 3
    else:
        assert read_json(path)['tasks']['one']['status'] == 'UNKNOWN'
        assert len(calls) == 2


def test_freeze_four_combinations_and_tamper(config):
    run = c.prepare(config)
    tasks = json.loads((run / 'tasks.json').read_text())
    assert len(tasks) == 4
    assert {(t['task_type'], t['mode']) for t in tasks} == {(kind, mode) for kind in ('correctness', 'performance') for mode in ('oracle', 'agent')}
    (config.parent / 'data.jsonl').write_text('changed source')
    c.verify(run)
    with (run / 'correctness.jsonl').open('a') as out:
        out.write('tamper')
    with pytest.raises(ValueError, match='Frozen input'):
        c.verify(run)


def test_mutable_base_ref_is_not_a_frozen_input(config):
    dataset=config.parent/'data.jsonl'
    row=json.loads(dataset.read_text());row['base_commit']='main'
    atomic_jsonl(dataset,[row])
    with pytest.raises(ValueError,match='immutable commit'):
        c.prepare(config)


def test_unknown_receipt_survives_lost_state_without_resubmit(config):
    run = c.prepare(config)
    platform = Platform()
    platform.status = 'UNKNOWN'
    c.tick(run, platform)
    assert len(platform.submissions) == 2
    atomic_json(run / 'state.json', {'tasks': {}, 'cooldowns': {}})
    c.tick(run, platform)
    assert len(platform.submissions) == 2
    assert read_json(run / 'summary.json')['active'] == 2


def test_rate_limit_cooldown_and_exhaustion_stay_unresolved(config):
    run = c.prepare(config)
    platform = Platform()
    c.tick(run, platform)
    for attempt in platform.submissions:
        atomic_json(attempt / 'result.json', dict(raw_category='infrastructure_blocked', reason='rate_limit', retryable=True))
    platform.status = 'DONE'
    c.tick(run, platform)
    assert len(platform.submissions) == 2  # both modes share the same model cooldown
    from swe_gpu_bench.shared.model_pause import pause_until
    assert pause_until(run / 'model-pauses/fake.json') == read_json(run / 'state.json')['cooldowns']['fake']
    rows = [json.loads(l) for l in (run / 'results-table.jsonl').read_text().splitlines()]
    assert all(r['scored_outcome'] == 'unresolved' for r in rows[:2])


def test_candidate_survives_validation_retry_and_tamper_fails(config):
    run = c.prepare(config)
    task = json.loads((run / 'tasks.json').read_text())[0]
    source = config.parent / 'candidate.diff'
    source.write_text('candidate')
    saved = seal_candidate(run, task, source)
    assert saved_candidate(run, task) == saved
    source.write_text('new model answer')
    with pytest.raises(ValueError, match='cannot be replaced'):
        seal_candidate(run, task, source)
    saved.write_text('tamper')
    with pytest.raises(ValueError, match='regeneration prohibited'):
        saved_candidate(run, task)


def test_model_failure_not_regenerated(config):
    run = c.prepare(config)
    platform = Platform()
    c.tick(run, platform)
    original = platform.submissions[:]
    for attempt in original:
        atomic_json(attempt / 'result.json', dict(raw_category='test_failed', reason='assertion', retryable=False, tests_executed=True))
    platform.status = 'DONE'
    c.tick(run, platform)
    assert len(platform.submissions) == 4
    c.tick(run, platform)
    assert all(len(list(p.parent.glob('attempt-*'))) == 1 for p in original)


def test_exclusion_requires_exact_dataset_evidence(config):
    cfg = read_json(config)
    evidence = config.parent / 'evidence.json'
    evidence.write_text('{"reason":"data error"}')
    cfg['exclusions'] = [dict(task_type='correctness', instance_id='org__repo__1', dataset_sha256='wrong',
                             evidence='evidence.json', evidence_sha256=sha256(evidence), reason='data error')]
    atomic_json(config, cfg)
    with pytest.raises(ValueError, match='different dataset'):
        c.prepare(config)


def test_summary_preserves_assumptions_and_actual_coverage(config):
    run = c.prepare(config)
    summary = c.summarize(run)
    assert all(g['assumed_incorrect'] == 1 and g['executed_test_coverage'] == 0 for g in summary['groups'])
    assert all(g['scoring_denominator'] == 1 for g in summary['groups'])


@pytest.mark.parametrize('kind,changed,rerun', [
    ('correctness', True, True), ('correctness', False, False), ('performance', False, True),
])
def test_history_scoring_and_scheduling(config, kind, changed, rerun):
    cfg = read_json(config)
    row = json.loads((config.parent / 'data.jsonl').read_text())
    cfg['models'][0].update(task_types=[kind], modes=['oracle'])
    if changed:
        row['problem_statement'] = 'old'
    atomic_jsonl(config.parent / 'old.jsonl', [row])
    stdout = config.parent / 'test.out'
    stderr = config.parent / 'test.err'
    stdout.write_text('1 passed')
    stderr.write_text('')
    atomic_json(config.parent / 'old-result.json', {
        'instance_id': row['instance_id'], 'mode': 'oracle', 'status': 'passed',
        'test_results': [{'command': 'pytest', 'stdout_log': str(stdout), 'stderr_log': str(stderr), 'exit_code': 0}],
    })
    cfg['history'] = [dict(task_id=kind + '--oracle--fake--org__repo__1', result='old-result.json', dataset='old.jsonl',
                          result_sha256=sha256(config.parent / 'old-result.json'), dataset_sha256=sha256(config.parent / 'old.jsonl'))]
    atomic_json(config, cfg)
    run = c.prepare(config)
    rows = [json.loads(l) for l in (run / 'results-table.jsonl').read_text().splitlines()]
    assert rows[0]['historical'] and not rows[0]['current_input_executed']
    assert rows[0]['scored_outcome'] == ('unresolved' if rerun else 'passed')
    group = c.summarize(run)['groups'][0]
    assert group['evaluated'] == group['scoring_denominator'] == int(not rerun)
    platform = Platform()
    c.tick(run, platform)
    assert len(platform.submissions) == int(rerun)

    raw = read_json(config.parent / 'old-result.json')
    raw['instance_id'] = 'different-task'
    atomic_json(config.parent / 'old-result.json', raw)
    cfg['history'][0]['result_sha256'] = sha256(config.parent / 'old-result.json')
    atomic_json(config, cfg)
    with pytest.raises(ValueError, match='Historical result instance_id'):
        c.prepare(config)


def test_local_without_gpu_records_blocked(config, monkeypatch):
    from swe_gpu_bench.evaluate import campaign_worker as w
    run = c.prepare(config)
    task = json.loads((run / 'tasks.json').read_text())[0]
    attempt = run / 'attempts' / task['task_id'] / 'attempt-0001'
    atomic_json(attempt / 'task.json', task)
    monkeypatch.setenv('SWEBENCH_NODE_ROOT', str(config.parent / 'local'))
    monkeypatch.setattr(w.subprocess, 'run', lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError('nvidia-smi')))
    w.supervise(run, attempt)
    assert read_json(attempt / 'result.json')['raw_category'] == 'infrastructure_blocked'
    assert (attempt / 'exit.json').exists()


def test_immediate_terminal_receipt_is_processed(config):
    run = c.prepare(config)
    platform = Platform()
    platform.status = 'DONE'
    c.tick(run,platform)
    for attempt in platform.submissions:
        atomic_json(attempt/'result.json', {'raw_category':'passed','reason':'tests','retryable':False,'tests_executed':True})
    c.tick(run,platform)
    assert len(list((run/'results').glob('*.json'))) == 2


def test_rate_limit_retry_budget_is_bounded_but_remains_unscored(config, monkeypatch):
    cfg = read_json(config);cfg['cooldown_seconds']=0;atomic_json(config,cfg)
    run=c.prepare(config);platform=Platform()
    for _ in range(8):
        c.tick(run,platform)
        for attempt in platform.submissions:
            if not (attempt/'result.json').exists():
                atomic_json(attempt/'result.json',{'raw_category':'infrastructure_blocked','reason':'rate_limit','retryable':True})
        platform.status='DONE'
    assert len(platform.submissions)==12  # four tasks, one initial plus two retries
    assert all(g['scoring_denominator']==0 and g['unresolved']==1 for g in c.summarize(run)['groups'])


def test_recover_saved_generation_after_worker_crash(config, monkeypatch):
    from swe_gpu_bench.evaluate.campaign_worker import generate
    run=c.prepare(config);task=json.loads((run/'tasks.json').read_text())[0]
    root=run/'attempts'/task['task_id']/'attempt-0001'/'oracle'/task['instance_id']
    atomic_json(root/'metadata.json',{'pred_patch_apply_check_status':'valid'})
    (root/'pred.diff').write_text('completed-candidate')
    patch,meta=generate(run,task,config.parent/'unused',{})
    assert patch.read_text()=='completed-candidate'
    assert meta['reused_candidate']

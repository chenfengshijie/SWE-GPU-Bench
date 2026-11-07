import json
from pathlib import Path

import pytest

from swe_gpu_bench.evaluate.metrics import MetricError, compare, parse, recipe, registry

FIXTURES = sorted((Path(__file__).parent / 'fixtures/performance').glob('*.json'))


@pytest.mark.parametrize('path', FIXTURES, ids=lambda p: p.stem)
def test_every_pinned_recipe_parses_selected_metric_and_rejects_missing(path):
    fixture = json.loads(path.read_text())
    spec = registry()[fixture['instance_id']]
    assert fixture['command_sha256'] == spec['command_sha256']
    base, candidate = parse(spec, fixture['base']), parse(spec, fixture['candidate'])
    result = compare(spec, base, candidate, correctness_passed=True)
    assert result['mean_improvement_percent'] == pytest.approx(20)
    assert result['improved']
    with pytest.raises(MetricError):
        parse(spec, '')
    with pytest.raises(MetricError):
        parse(spec, fixture['base'] + '\n' + fixture['base'])
    with pytest.raises(MetricError, match='correctness'):
        compare(spec, base, candidate, correctness_passed=False)


def test_exact_39_recipes_and_fixtures():
    assert len(registry()) == len(FIXTURES) == 39


@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf'), True, None])
def test_nonpositive_nonfinite_or_missing_rejected(value):
    with pytest.raises(MetricError):
        compare({'better': 'lower', 'unit': 'ms'}, {'one': 10}, {'one': value}, correctness_passed=True)


def test_workloads_and_units_are_not_silently_discarded():
    spec = next(s for s in registry().values() if s['parser'] == 'tagged')
    with pytest.raises(MetricError, match='unit_mismatch'):
        parse(spec, 'SWE_GPU_METRIC ' + json.dumps({'name':'latency','value':2,'unit':'us','workload':spec['workloads'][0]}))
    with pytest.raises(MetricError, match='workload_mismatch'):
        compare({'better':'lower'}, {'one':10, 'two':20}, {'one':5}, correctness_passed=True)


def test_percentage_mean_not_ratio_of_averages():
    result = compare({'better':'lower','unit':'ms'}, {'a':1,'b':100}, {'a':0.5,'b':110}, correctness_passed=True)
    assert result['mean_improvement_percent'] == pytest.approx(20)


def test_hash_mismatch_and_unsupported_never_fall_back_to_judge():
    with pytest.raises(MetricError, match='hash_mismatch'):
        recipe({'instance_id':next(iter(registry())), 'perf_commands':['echo 1']})
    with pytest.raises(MetricError, match='unsupported_recipe'):
        recipe({'instance_id':'unknown'})


def test_general_scalar_protocol():
    record = {'command_contract':'portable_unified_v1', 'performance_metric':{'name':'latency','unit':'ms','workload':'x','better':'lower'},'perf_commands':['bench']}
    spec = recipe(record)
    assert parse(spec, '1.2e-3\n') == {'x': 0.0012}
    with pytest.raises(MetricError):
        parse(spec, 'warmup 1\n0.12\n')


@pytest.mark.parametrize('instance,old,new', [
    ('flashinfer-ai__flashinfer__1483', 'batch_size: 32', 'batch_size: 16'),
    ('ggml-org__llama.cpp__20391', 'Qwen_Qwen3.5-0.8B-Q8_0.gguf', 'different.gguf'),
])
def test_fixed_workload_cannot_silently_change(instance, old, new):
    fixture = json.loads((FIXTURES[0].parent / (instance + '.json')).read_text())
    with pytest.raises(MetricError, match='unexpected'):
        parse(registry()[instance], fixture['base'].replace(old, new))


def test_legacy_perf_entry_has_no_llm_fallback(tmp_path, monkeypatch):
    from swe_gpu_bench.evaluate.perf_validate import call_perf_judge
    from swe_gpu_bench.core import llm
    monkeypatch.setattr(llm,'complete',lambda *a,**k:pytest.fail('numeric evaluation called a model'))
    result=call_perf_judge({'instance_id':'unsupported'},'',[],[],tmp_path)
    assert result['failure_reason']=='unsupported_recipe'
    record={'command_contract':'portable_unified_v1','performance_metric':{'name':'latency','unit':'ms','workload':'x','better':'lower'},'perf_commands':['bench']}
    base=tmp_path/'base';base.write_text('10\n')
    candidate=tmp_path/'candidate';candidate.write_text('8\n')
    result=call_perf_judge(record,'',[{'exit_code':0,'stdout_log':str(base)}],[{'exit_code':0,'stdout_log':str(candidate)}],tmp_path)
    assert result['verdict']=='improved' and result['mean_improvement_percent']==20

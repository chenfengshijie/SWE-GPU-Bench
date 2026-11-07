"""Versioned, deterministic performance parsing. No generated code or model judge."""
from __future__ import annotations

import csv
import hashlib
from importlib.resources import files
import io
import json
import math
from pathlib import PurePath
import re

from swe_gpu_bench.shared.scalar_metric import parse_scalar, validate_performance_metric


class MetricError(ValueError):
    pass


def command_hash(record: dict) -> str:
    commands = {k: record.get(k, []) for k in
                ('setup_commands', 'test_commands', 'perf_setup_commands', 'perf_commands')}
    return hashlib.sha256(json.dumps(commands, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def registry() -> dict:
    return json.loads(files('swe_gpu_bench.evaluate').joinpath(
        'resources/performance-20261103.json').read_text())['recipes']


def recipe(record: dict) -> dict:
    if record.get('command_contract') == 'portable_unified_v1':
        metric = validate_performance_metric(record.get('performance_metric'))
        if len(record.get('perf_commands', [])) != 1:
            raise MetricError('scalar_requires_one_command')
        return {**metric, 'parser': 'scalar'}
    spec = registry().get(record['instance_id'])
    if not spec:
        raise MetricError('unsupported_recipe')
    if command_hash(record) != spec['command_sha256']:
        raise MetricError('recipe_command_hash_mismatch')
    if record.get('better') != spec['better']:
        raise MetricError('recipe_direction_mismatch')
    return spec


def positive(value) -> float:
    try:
        number = float(value)
    except (ValueError, TypeError):
        raise MetricError('invalid_measurement') from None
    if isinstance(value, bool) or not math.isfinite(number) or number <= 0:
        raise MetricError('nonpositive_or_nonfinite_measurement')
    return number


def json_payload(text: str, kind):
    decoder = json.JSONDecoder()
    found = []
    index = 0
    while index < len(text):
        if text[index] not in '[{':
            index += 1
            continue
        try:
            obj, end = decoder.raw_decode(text[index:])
        except ValueError:
            index += 1
            continue
        if kind(obj):
            found.append(obj)
        index += end
    if len(found) != 1:
        raise MetricError('missing_or_duplicate_json_measurement')
    return found[0]


def parse(spec: dict, text: str) -> dict[str, float]:
    result = {}

    def add(key, value):
        if key in result:
            raise MetricError('duplicate_workload:' + key)
        result[key] = positive(value)

    parser = spec['parser']
    if parser == 'scalar':
        try:
            add(spec.get('workload', 'scalar'), parse_scalar(text))
        except ValueError as exc:
            raise MetricError('invalid_scalar_measurement') from exc
    elif parser == 'tagged':
        for line in text.splitlines():
            if line.startswith('SWE_GPU_METRIC '):
                try:
                    row = json.loads(line[len('SWE_GPU_METRIC '):])
                except ValueError as exc:
                    raise MetricError('invalid_tagged_json') from exc
                if row.get('name') != spec['metric']:
                    continue
                if row.get('unit') != spec['unit']:
                    raise MetricError('unit_mismatch')
                add(row.get('workload'), row.get('value'))
        if set(result) != set(spec['workloads']):
            raise MetricError('missing_or_unexpected_workload')
    elif parser == 'named':
        for key in spec['keys']:
            values = re.findall(r'(?m)^' + re.escape(key) + r'=([^\s]+)', text)
            if len(values) != 1:
                raise MetricError('missing_or_duplicate_metric:' + key)
            add(key, values[0])
    elif parser == 'xformers':
        for dtype, value in re.findall(r'(?m)^dtype=(\w+)\s+torch_fw_ms=\S+\s+xf_fw_ms=(\S+)\s*$', text):
            add(dtype, value)
        if set(result) != set(spec['workloads']):
            raise MetricError('missing_or_unexpected_workload')
    elif parser == 'duration':
        for line in text.splitlines():
            match = re.fullmatch(r'(vocab_size: .+?, batch_size: .+?, distrib: .+?, p: .+?), duration: (\S+) us, effective bandwidth: .+', line)
            if match:
                add(match[1], match[2])
        if set(result) != set(spec['workloads']):
            raise MetricError('missing_or_unexpected_workload')
    elif parser == 'llama':
        rows = json_payload(text, lambda x: isinstance(x, list) and bool(x) and isinstance(x[0], dict) and 'avg_ts' in x[0])
        for row in rows:
            if row.get('n_prompt') != 0 or row.get('n_gen') != 128:
                raise MetricError('unexpected_llama_workload')
            row = {**row, 'model_filename': PurePath(row.get('model_filename', '')).name}
            if row['model_filename'] != spec['model_filename']:
                raise MetricError('unexpected_llama_model')
            key = json.dumps({k: row.get(k) for k in ('model_filename', 'model_type', 'model_size', 'model_n_params', 'n_prompt', 'n_gen', 'n_batch', 'n_ubatch', 'type_k', 'type_v', 'flash_attn', 'split_mode', 'n_gpu_layers')}, sort_keys=True)
            add(key, row['avg_ts'])
        if len(result) != 1:
            raise MetricError('unexpected_llama_workload_count')
    elif parser == 'nvbench':
        obj = json_payload(text, lambda x: isinstance(x, dict) and 'benchmarks' in x)
        for bench in obj['benchmarks']:
            if bench.get('name') != spec['benchmark']:
                continue
            for state in bench.get('states', []):
                if state.get('is_skipped') or state.get('skip_reason'):
                    raise MetricError('skipped_workload')
                axes = state.get('axis_values')
                if axes is None:
                    raise MetricError('missing_nvbench_axes')
                axis_map = {re.sub(r'\{[^}]*\}', '', a['name']): str(a['value']) for a in axes}
                if not all(axis_map.get(k) == str(v) for k, v in spec['axes'].items()) or state.get('device') != 0:
                    raise MetricError('unexpected_nvbench_workload')
                key = json.dumps({'benchmark': bench['name'], 'axes': axis_map}, sort_keys=True)
                summaries = [s for s in state.get('summaries', []) if s.get('tag') == 'nv/cold/time/gpu/mean']
                if len(summaries) != 1:
                    raise MetricError('missing_or_duplicate_gpu_timing')
                values = [v['value'] for v in summaries[0].get('data', []) if v.get('name') == 'value']
                if len(values) != 1:
                    raise MetricError('missing_or_duplicate_gpu_value')
                add(key, values[0])
    elif parser == 'cutlass':
        # CUTLASS 0.x profiler CSV: provider/kernel/problem axes/runtime/GFLOPs.
        lines = text.splitlines()
        header = next((i for i, line in enumerate(lines) if 'Kernel' in line and 'Runtime' in line and ',' in line), None)
        if header is None:
            raise MetricError('missing_cutlass_header')
        for row in csv.DictReader(io.StringIO('\n'.join(lines[header:]))):
            row = {k.strip(): v.strip() for k, v in row.items() if k and isinstance(v, str)}
            kernel = row.get('Kernel', '')
            if 's884gemm_tn' not in kernel:
                continue
            timing = [k for k in row if k.startswith('Runtime')]
            if len(timing) != 1 or timing[0] not in ('Runtime', 'Runtime (ms)'):
                raise MetricError('cutlass_unit_mismatch')
            if row.get('Disposition', '').lower() != 'passed':
                raise MetricError('cutlass_correctness_failed')
            if any(row.get(k) != v for k, v in {'M':'480', 'N':'280', 'K':'224'}.items()):
                raise MetricError('unexpected_cutlass_workload')
            key = json.dumps({k: v for k, v in row.items() if k in ('Provider', 'Kernel', 'M', 'N', 'K', 'Layout_A', 'Layout_B', 'Beta', 'batch_count')}, sort_keys=True)
            add(key, row[timing[0]])
    else:
        raise MetricError('unsupported_parser')
    if not result:
        raise MetricError('missing_measurement')
    return result


def compare(spec: dict, base: dict, candidate: dict, *, correctness_passed: bool) -> dict:
    if not correctness_passed:
        raise MetricError('correctness_not_passed')
    if not base or set(base) != set(candidate):
        raise MetricError('workload_mismatch')
    direction = spec['better']
    if direction not in ('lower', 'higher'):
        raise MetricError('invalid_direction')
    improvements = {}
    for key, raw in base.items():
        b, c = positive(raw), positive(candidate[key])
        value = 100 * ((b - c) if direction == 'lower' else (c - b)) / b
        if not math.isfinite(value):
            raise MetricError('nonfinite_improvement')
        improvements[key] = value
    mean = math.fsum(improvements.values()) / len(improvements)
    return {'base': base, 'candidate': candidate, 'unit': spec['unit'], 'better': direction,
            'improvements_percent': improvements, 'mean_improvement_percent': mean,
            'improved': mean > 0, 'statistical_significance': 'not_claimed'}

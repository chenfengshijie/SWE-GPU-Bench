"""Selected historical classification, repair and input-isolation regressions."""
import json
import os
from pathlib import Path
from unittest.mock import patch
import pytest
from swe_gpu_bench.evaluate.use_cases.campaign_audit import classify
from swe_gpu_bench.shared.artifacts import atomic_json
from swe_gpu_bench.shared.solver_guard import guarded_command


def result(tmp_path, text='2 passed', status='passed', exit_code=0):
    logs = tmp_path / 'eval/one/artifacts/command-logs'
    logs.mkdir(parents=True, exist_ok=True)
    (logs / 'test-01.stdout.log').write_text(text)
    (logs / 'test-01.stderr.log').write_text('')
    return {'instance_id': 'one', 'status': status, 'mode': 'agent', 'evidence_path': str(tmp_path),
            'test_results': [{'exit_code': exit_code, 'command': 'pytest tests/test.py',
            'stdout_log': '/gone/test-01.stdout.log', 'stderr_log': '/gone/test-01.stderr.log'}]}


@pytest.mark.parametrize('text,status,code,expected', [
    ('2 passed', 'passed', 0, 'passed'), ('1 failed, 1 passed', 'failed', 1, 'test_failed'),
    ('1 failed, 2 skipped', 'failed', 1, 'test_failed'),
    ('1 failed, 2 passed in 0.05s', 'passed', 0, 'infrastructure_blocked'),
    ('==== 2 passed, 1 failed in 0.05s ====', 'passed', 0, 'infrastructure_blocked'),
    ('10 skipped', 'passed', 0, 'infrastructure_blocked'),
    ('collected 0 items', 'passed', 0, 'infrastructure_blocked'),
    ('No tests were found', 'passed', 0, 'infrastructure_blocked'),
    ('ModuleNotFoundError: missing', 'failed', 1, 'infrastructure_blocked'),
    ('error: template invalid', 'failed', 1, 'infrastructure_blocked'),
    ('pair.h(164): error: function cannot be referenced -- it is a deleted function', 'failed', 1, 'model_output_error'),
    ('tuple_cat.pass.cpp:364: error: the value is not usable in a constant expression', 'failed', 1, 'model_output_error'),
    ('add_min_max.h(125): error: expression must be a modifiable lvalue', 'failed', 1, 'model_output_error'),
    ('activation.h(374): error: no suitable constructor exists to convert from "float" to "cutlass::bfloat16_t"', 'failed', 1, 'model_output_error'),
    ('movm.cu(137): error: identifier "SM75_U32x1_MOVM_T" is undefined', 'failed', 1, 'model_output_error'),
    ('movm.cu(137): error: no instance of function template "cute::make_tiled_copy" matches the argument list', 'failed', 1, 'model_output_error'),
    ('nvcc fatal: Unsupported gpu architecture\nerror: identifier "x" is undefined', 'failed', 1, 'infrastructure_blocked'),
    ('download HTTP 503', 'failed', 1, 'infrastructure_blocked'),
    ('node:4065:4321 [0] NCCL INFO TUNER/Plugin: libnccl-net.so: cannot open shared object file\nnode:4065:4321 [0] NCCL INFO Using internal tuner plugin.\nAssertionError: discarded step changed momentum', 'failed', 1, 'test_failed'),
    ('node:4065:4321 [0] NCCL INFO Using internal tuner plugin.\nImportError: required_extension.so: cannot open shared object file', 'failed', 1, 'infrastructure_blocked'),
    ('echo success', 'passed', 0, 'infrastructure_blocked'),
])
def test_log_based_classification(tmp_path, text, status, code, expected):
    r = result(tmp_path, text, status, code)
    decision = classify(r)
    assert decision['category'] == expected
    assert len(decision['logs']) == 2
    assert all(len(x['sha256']) == 64 for x in decision['logs'])


def test_install_and_patch_failure_never_pass(tmp_path):
    r = result(tmp_path)
    r['setup_results'] = [{**r['test_results'][0], 'exit_code': 1}]
    assert classify(r)['category'] == 'infrastructure_blocked'
    r.pop('setup_results')
    r.update(status='error', failure_category='PRED_PATCH_APPLY_FAILURE', test_results=[])
    assert classify(r)['category'] != 'passed'


@pytest.mark.parametrize('detail,expected', [
    ('', 'test_failed'),
    ('sdy_opt: command not found\n', 'infrastructure_blocked'),
    ('Connection reset by peer\n', 'infrastructure_blocked'),
])
def test_lit_assertion_failure_survives_later_zip_failure(tmp_path, detail, expected):
    text = ("FAIL: XLA :: example.mlir (1 of 1)\n"
            "******************** TEST 'XLA :: example.mlir' FAILED ********************\n"
            "Script:\n--\nsdy_opt example.mlir | FileCheck example.mlir\n--\n"
            "Exit Code: 1\nCommand Output (stderr):\n--\n" + detail +
            "example.mlir:9:17: error: CHECK-NEXT: expected string not found in input\n"
            "<stdin>:5:88: note: scanning from here\n %1 = call @baz()\n"
            "********************\nFailed Tests (1):\n  XLA :: example.mlir\n"
            "Failed: 1\nexternal/bazel_tools/tools/test/test-setup.sh: line 428: zip: command not found\n")
    r = result(tmp_path, text, status='failed', exit_code=3)
    decision = classify(r)
    assert decision['category'] == expected
    if expected == 'test_failed':
        assert decision['partial_infrastructure_reasons'] == ['bazel_test_output_zip_missing']
        r['test_results'][0]['source_access'] = {'candidate_files': ['pass.cc'], 'read_during_validation': []}
        assert classify(r)['reason'] == 'candidate_source_not_read_during_validation'


def test_bazel_wrapper_fail_labels_without_lit_failure_are_not_test_failures(tmp_path):
    text = ('Passed: 1\nexternal/bazel_tools/tools/test/test-setup.sh: line 428: zip: command not found\n'
            'FAIL: //xla/test:one.mlir.test\nINFO: Build completed, 30 tests FAILED\n')
    decision = classify(result(tmp_path, text, status='failed', exit_code=3))
    assert decision['category'] == 'infrastructure_blocked'
    assert decision['reason'] == 'test_environment_failure'


def test_missing_evidence_not_accepted(tmp_path):
    r = result(tmp_path)
    Path(r['evidence_path'], 'eval/one/artifacts/command-logs/test-01.stdout.log').unlink()
    assert classify(r)['reason'] == 'test_log_missing'


def test_compile_diagnosis_is_bound_to_result_and_evidence_and_cannot_award_pass(tmp_path):
    from swe_gpu_bench.evaluate.use_cases.campaign_audit import reviewed_decision, sha256
    result_path = tmp_path / 'result.json'
    evidence = tmp_path / 'compiler.log'
    evidence.write_text('fatal error: required_project_header.hpp: No such file')
    atomic_json(result_path, {'status': 'error'})
    decision = {'category': 'infrastructure_blocked', 'reason': 'needs_diagnosis', 'retryable': False}
    review = {'task_id': 'one', 'result_sha256': sha256(result_path), 'category': 'model_output_error',
              'reason': 'missing_required_project_header', 'rationale': 'Patch omits the required project header.',
              'evidence': [{'path': str(evidence), 'sha256': sha256(evidence)}]}
    path = tmp_path / 'adjudications/one.json'
    atomic_json(path, review)
    assert reviewed_decision(tmp_path, 'one', result_path, decision)['category'] == 'model_output_error'
    evidence.write_text('changed')
    assert reviewed_decision(tmp_path, 'one', result_path, decision)['adjudication_status'] == 'invalid_evidence'
    atomic_json(result_path, {'status': 'error', 'attempt': 2})
    assert reviewed_decision(tmp_path, 'one', result_path, decision)['adjudication_status'] == 'stale_result'
    atomic_json(path, {**review, 'result_sha256': sha256(result_path), 'category': 'passed'})
    with pytest.raises(ValueError):
        reviewed_decision(tmp_path, 'one', result_path, decision)


def test_completed_invalid_model_patch_preserved(tmp_path):
    old = tmp_path / 'oracle/one'
    old.mkdir(parents=True)
    (old / 'raw_response.txt').write_text('a completed malformed response')
    r = {'instance_id': 'one', 'mode': 'oracle', 'status': 'error', 'evidence_path': str(tmp_path),
         'failure_category': 'PRED_PATCH_INVALID', 'generation': {'pred_patch_apply_check_status': 'invalid'}}
    assert classify(r)['category'] == 'model_output_error'


@pytest.mark.parametrize('before_claim', [True, False])
def test_old_worker_cooldown_traceback_only_defers_proven_preclaim_failure(tmp_path, before_claim):
    log = tmp_path / 'bootstrap.log'
    source = "if not tasks[0].get('resume_patch') and pause_until(run / 'model-pauses'" if before_claim else 'publish_pause(path, value)'
    log.write_text('Traceback\n' + source + '\nModelCallsPausedError: Cooldown evidence temporarily unavailable')
    decision = classify({'failure_category': 'PLATFORM_INTERRUPTED', 'bootstrap_log_path': str(log)})
    assert decision['reason'] == ('model_calls_deferred' if before_claim else 'bootstrap_environment_failure')


def test_scoring_fields_and_setup_drivers_not_in_agent_input(tmp_path, monkeypatch):
    from swe_gpu_bench.evaluate.use_cases.eval import prepare_eval_case_dir
    monkeypatch.setenv('SWEBENCH_SOLVER_ISOLATION', '1')
    template = tmp_path / 'template'
    template.write_text('Solve the task.')
    record = {'instance_id': 'one', 'repo': 'r', 'problem_statement': 'fix code', 'base_commit': 'base',
              'patch': 'gold', 'test_patch': 'testgold', 'test_commands': ['secret_test'],
              'setup_commands': ['secret_setup_driver'], 'perf_commands': ['secret_perf']}
    paths = prepare_eval_case_dir(record, tmp_path / 'eval', template)
    safe = paths.instance_path.read_text() + paths.prompt_path.read_text()
    assert 'fix code' in safe
    for value in ('gold', 'secret_test', 'secret_setup_driver', 'secret_perf'):
        assert value not in safe


@pytest.mark.parametrize('text', ['\x1b[32m4 passed\x1b[0m', 'Passed: 1 (100.00%)',
    'Executed 1 out of 1 test: 1 test passes.', 'PASS backend=cuda elements=192',
    'validated_gpu_records=37', '========= ERROR SUMMARY: 0 errors'])
def test_native_and_ansi_summaries_retained(tmp_path, text):
    assert classify(result(tmp_path, text))['category'] == 'passed'


def test_completed_empty_response_is_model_output_not_api_retry(tmp_path):
    old = tmp_path / 'oracle/one'
    old.mkdir(parents=True)
    (old / 'raw_response.txt').write_text('')
    r = {'instance_id': 'one', 'mode': 'oracle', 'status': 'error', 'evidence_path': str(tmp_path),
         'failure_category': 'PRED_PATCH_INVALID', 'generation': {'pred_patch_apply_check_status': 'invalid',
             'usage': {'status': 'reported', 'reported_calls': 1}}}
    assert classify(r)['category'] == 'model_output_error'
    r['generation']['usage'] = {'status': 'missing'}
    assert classify(r)['category'] == 'infrastructure_blocked'


def test_rate_limit_pauses_and_unread_candidate_cannot_pass(tmp_path):
    r = result(tmp_path)
    r['test_results'][0]['source_access'] = {'candidate_files': {'kernel.py': 'hash'}, 'read_during_validation': []}
    assert classify(r)['reason'] == 'candidate_source_not_read_during_validation'
    r.update(status='error', test_results=[], failure_category='INFRASTRUCTURE_ERROR', error_type='RateLimitError')
    assert classify(r)['reason'] == 'rate_limit'


def test_rate_limit_requires_error_event_and_git_tls_eof_is_retryable(tmp_path):
    r = result(tmp_path)
    r.update(status='error', test_results=[], failure_category='AGENT_EXECUTION_FAILURE')
    log = tmp_path / 'eval/one/artifacts/command-logs/agent.stdout.jsonl'
    message = 'Requests have exceeded the throughput limit on your configured deployment.'
    log.write_text(json.dumps({'type': 'item.completed', 'item': {'error': message}}) + '\n')
    assert classify(r)['reason'] != 'rate_limit'
    log.write_text(json.dumps({'type': 'error', 'message': message}) + '\n')
    assert classify(r)['reason'] == 'rate_limit'
    log.write_text('')
    (tmp_path / 'worker.stderr.log').write_text('RPC failed; curl 56 GnuTLS recv error (-110): The TLS connection was non-properly terminated.\nfatal: early EOF')
    assert classify(r)['reason'] == 'transient_infrastructure' and classify(r)['retryable']


def test_silent_native_assertion_driver_retained(tmp_path):
    r = result(tmp_path, '')
    r['setup_commands'] = ["cat > artifacts/native_test.cu <<'EOF'\nassert(value == expected);\nEOF"]
    r['test_results'][0]['command'] = 'nvcc artifacts/native_test.cu -o artifacts/native_test && artifacts/native_test'
    assert classify(r)['category'] == 'passed'
    r['setup_commands'] = []
    assert classify(r)['category'] == 'infrastructure_blocked'


def test_rebuild_preserves_existing_venv_and_records_original(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from swe_gpu_bench.evaluate.use_cases.eval import run_commands
    monkeypatch.setenv('SWEBENCH_VERIFY_SOURCE', '1')
    paths = SimpleNamespace(repo_dir=tmp_path, case_dir=tmp_path, artifacts_dir=tmp_path, command_logs_dir=tmp_path)
    original = 'UV_CACHE_DIR=artifacts/cache artifacts/uvbin/uv venv --python 3.12 artifacts/venv && uv pip install -e repo'
    with patch('swe_gpu_bench.shared.source_access.SourceAccess') as observer, \
         patch('swe_gpu_bench.evaluate.use_cases.eval.run_logged_shell_commands',
               side_effect=lambda commands, **kwargs: [{'command': c, 'exit_code': 0} for c in commands]):
        observer.return_value.finish.return_value = {}
        rows = run_commands([original], 'post_patch_setup', paths)
        assert 'uv venv --allow-existing --python' in rows[0]['command']
        assert rows[0]['original_command'] == original
        assert run_commands([original], 'setup', paths)[0]['command'] == original


def test_bootstrap_deterministic_errors_do_not_auto_retry(tmp_path):
    log = tmp_path / 'bootstrap.log'
    log.write_text('unshare: unshare failed: Operation not permitted\n')
    r = {'instance_id': 'one', 'status': 'error', 'mode': 'agent',
         'failure_category': 'PLATFORM_INTERRUPTED', 'bootstrap_log_path': str(log), 'platform_status': 'FAILED'}
    assert classify(r)['reason'] == 'bootstrap_environment_failure'
    assert classify(r)['retryable'] is False
    log.write_text('ConnectionResetError\n')
    assert classify(r)['retryable'] is True


def test_missing_compiler_is_infrastructure_even_with_failed_test_summary(tmp_path):
    r = result(tmp_path, "FileNotFoundError: [Errno 2] No such file or directory: 'clang'\n11 failed, 6 passed")
    r['status'] = 'failed'
    r['test_results'][0]['exit_code'] = 1
    assert classify(r)['category'] == 'infrastructure_blocked'


@pytest.mark.parametrize('failure,expected', [
    ('E AssertionError: assert tensor(False)', 'test_failed'),
    ("E ModuleNotFoundError: No module named 'numpy'", 'infrastructure_blocked'),
])
def test_optional_dependency_warning_does_not_replace_actual_failure(tmp_path, failure, expected):
    text = (failure + '\n================ warnings summary ================\n'
            "torch/foo.py: UserWarning: Failed to initialize NumPy: No module named 'numpy'\n"
            '-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html\n'
            '================ 1 failed, 1 passed, 1 warning in 4.36s ================\n')
    assert classify(result(tmp_path, text, 'failed', 1))['category'] == expected


def test_pytest_startup_diagnostic_does_not_hide_numerical_failure(tmp_path):
    r = result(tmp_path, "Optional download: Permission denied\n.F\n==== FAILURES ====\n"
               "E AssertionError: Tensor-likes are not close!\n1 failed, 1 passed", 'failed', 1)
    assert classify(r)['category'] == 'test_failed'
    # A different failing command or stderr stream must not be discarded.
    Path(tmp_path, 'eval/one/artifacts/command-logs/test-01.stderr.log').write_text('ModuleNotFoundError: missing dependency')
    assert classify(r)['category'] == 'infrastructure_blocked'


def test_invalid_generated_cuda_requires_actual_test_execution(tmp_path):
    r = result(tmp_path, 'tinygrad.device.CompileError: Nvrtc Error 6\n<null>(15): error: identifier "alu1" is undefined')
    r['status'] = 'failed'
    r['test_results'][0]['exit_code'] = 1
    assert classify(r)['category'] == 'infrastructure_blocked'
    from swe_gpu_bench.evaluate.use_cases.campaign_audit import log_path
    stdout = log_path(r, r['test_results'][0]['stdout_log'])
    stdout.write_text(stdout.read_text() + '\nRan 1 test in 0.33s\nFAILED (errors=1)\n')
    assert classify(r)['reason'] == 'executed_test_generated_kernel_compile_failure'


def test_native_crash_requires_successful_gpu_control_for_failure_classification(tmp_path):
    r = result(tmp_path, 'Fatal Python error: Segmentation fault', status='failed', exit_code=-11)
    assert classify(r)['category'] == 'infrastructure_blocked'
    control = tmp_path / 'control.log'
    control.write_text('17 passed\nvalidated asynchronous CUDA access to CPU-owned memory 67108864\n')
    r['test_results'].append({'command': 'trusted_cuda_control', 'exit_code': 0,
                             'stdout_log': str(control), 'stderr_log': r['test_results'][0]['stderr_log']})
    assert classify(r)['reason'] == 'native_test_crash_with_successful_cuda_control'
    r['test_results'][1]['exit_code'] = 1
    assert classify(r)['category'] == 'infrastructure_blocked'


def test_deferred_command_repair_rejects_changed_recipe(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from swe_gpu_bench.evaluate.use_cases.eval import run_commands
    monkeypatch.setenv('SWEBENCH_VERIFY_SOURCE', '1')
    monkeypatch.setenv('SWEBENCH_DEFER_SETUP_COMMAND_HASHES', json.dumps(['0' * 64]))
    with pytest.raises(ValueError, match='does not match'):
        run_commands(['echo modified'], 'setup', SimpleNamespace())


def test_setup_environment_override_is_hashed_and_cannot_change_test_commands(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import hashlib
    from swe_gpu_bench.evaluate.use_cases.eval import run_commands
    original = "printf 'base'"
    monkeypatch.setenv('SWEBENCH_VERIFY_SOURCE', '1')
    monkeypatch.setenv('SWEBENCH_SETUP_COMMAND_OVERRIDES', json.dumps({hashlib.sha256(original.encode()).hexdigest(): "printf 'runtime'"}))
    paths = SimpleNamespace(repo_dir=tmp_path, case_dir=tmp_path, artifacts_dir=tmp_path, command_logs_dir=tmp_path)
    setup = run_commands([original], 'setup', paths)[0]
    assert Path(setup['stdout_log']).read_text() == 'runtime'
    assert setup['original_command'] == original
    assert setup['environment_repair'] == 'documented_setup_command_override'
    test = run_commands([original], 'test', paths)[0]
    assert Path(test['stdout_log']).read_text() == 'base'
    with pytest.raises(ValueError, match='does not match'):
        run_commands(['echo changed'], 'setup', paths)


def test_test_environment_repair_preserves_original_command_evidence(tmp_path, monkeypatch):
    from swe_gpu_bench.evaluate.use_cases.eval import run_commands
    from types import SimpleNamespace
    import hashlib
    paths = SimpleNamespace(repo_dir=tmp_path / 'repo', case_dir=tmp_path, artifacts_dir=tmp_path, command_logs_dir=tmp_path)
    paths.repo_dir.mkdir(parents=True, exist_ok=True)
    original = "printf 'unchanged assertion'"
    repaired = "cd repo && " + original
    monkeypatch.setenv('SWEBENCH_VERIFY_SOURCE', '1')
    monkeypatch.setenv('SWEBENCH_TEST_COMMAND_OVERRIDES', json.dumps({hashlib.sha256(original.encode()).hexdigest(): repaired}))
    rows = run_commands([original], 'test', paths)
    assert rows[0]['command'] == repaired
    assert rows[0]['original_command'] == original
    assert rows[0]['exit_code'] == 0
    with pytest.raises(ValueError, match='frozen commands'):
        run_commands(['different command'], 'test', paths)


@pytest.mark.parametrize('diagnostic', [
    'kernel.cu(42): error: expression must have a constant value',
    'kernel.cu(42): error: no instance of overloaded function "reduce" matches the argument list',
])
def test_static_candidate_compile_diagnostics(tmp_path, diagnostic):
    outcome = result(tmp_path, text=diagnostic, status='error', exit_code=1)
    assert classify(outcome)['category'] == 'model_output_error'


def test_custom_gpu_driver_failed_comparison_needs_executable_evidence(tmp_path):
    outcome = result(tmp_path, text='uint8_t mismatch at case 11, width 0', status='failed', exit_code=2)
    assert classify(outcome)['category'] == 'infrastructure_blocked'
    outcome['setup_commands'] = ['cat > artifacts/bit_trunc_gpu_test.cu <<EOF\nif (result != expected) { std::exit(2); }\nEOF']
    outcome['test_results'][0]['command'] = 'nvcc artifacts/bit_trunc_gpu_test.cu -o artifacts/bit_trunc_gpu_test && artifacts/bit_trunc_gpu_test'
    assert classify(outcome)['category'] == 'test_failed'


def test_distributed_port_collision_is_infrastructure_even_with_failed_count(tmp_path):
    outcome = result(tmp_path, text='RuntimeError: server socket failed: EADDRINUSE, address already in use\n1 failed', status='failed', exit_code=1)
    audit = classify(outcome)
    assert audit['category'] == 'infrastructure_blocked'
    assert audit['reason'] == 'test_transient' and audit['retryable']


def test_genuine_failed_command_is_retained_when_another_command_hits_infrastructure(tmp_path):
    first = result(tmp_path / 'first', text='AssertionError: incorrect GPU value\n1 failed', status='failed', exit_code=1)
    other = result(tmp_path / 'other', text='RuntimeError: EADDRINUSE\n1 failed', status='failed', exit_code=1)
    for key in ('stdout_log', 'stderr_log'):
        from swe_gpu_bench.evaluate.use_cases.campaign_audit import log_path
        old = log_path(other, other['test_results'][0][key])
        new = old.with_name('other-' + old.name)
        old.rename(new)
        other['test_results'][0][key] = str(new)
    first['test_results'].extend(other['test_results'])
    audit = classify(first)
    assert audit['category'] == 'test_failed'
    assert audit['reason'] == 'executed_failure_with_partial_infrastructure'
    assert audit['partial_infrastructure_reasons'] == ['test_transient']


def test_cuda_binary_architecture_mismatch_is_not_a_model_test_failure(tmp_path):
    outcome = result(tmp_path, text='RuntimeError: CUDA error: no kernel image is available for execution on the device\n150 failed', status='failed', exit_code=1)
    audit = classify(outcome)
    assert audit['category'] == 'infrastructure_blocked'
    assert audit['reason'] == 'cuda_binary_architecture_mismatch'
    assert not audit['retryable']


def test_guard_does_not_widen_access_for_non_python_venv_symlink(tmp_path):
    case = tmp_path / 'case'
    venv = case / 'artifacts/.venv'
    (venv / 'bin').mkdir(parents=True)
    (venv / 'pyvenv.cfg').write_text('venv')
    outside = tmp_path / 'unrelated/bin'
    outside.mkdir(parents=True)
    binary = outside / 'python'
    binary.write_text('not a prepared Python installation')
    (venv / 'bin/python').symlink_to(binary)
    env = {**os.environ, 'SWEBENCH_SOLVER_ISOLATION': '1'}
    guarded_command(['/bin/true'], case, env)
    assert json.loads(env['SWEBENCH_GUARD_ROOTS'])['interpreter_roots'] == []

"""Evidence-based accounting for the frozen evaluation tasks.

Historical files are read only. Decisions retain their source and log digests;
unknown execution is never turned into a model failure or a pass.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re

COMPLETED = {'passed', 'test_failed', 'model_output_error'}
TERMINAL_JOBS = {'DONE', 'FAILED', 'FAILED_TO_LAUNCH', 'STOPPED', 'CLOSED', 'REJECTED', 'CANCELED', 'CANCELLED'}
RATE_LIMIT = re.compile(r'(?i)rate.?limit|too many requests|HTTP[ :]+429|ModelCallsPausedError|exceeded the throughput limit|exceed your limit.*provisioned throughput')
TRANSIENT = re.compile(r'(?i)(?:HTTP(?: status)?[ :]+(?:429|502|503|504)|rate.?limit|too many requests|ConnectionResetError|ConnectionError|APITimeoutError|TimeoutExpired|APIConnectionError|connection (?:reset|refused|timed out)|temporary failure in name resolution|network is unreachable|EADDRINUSE|Address already in use|remote end closed|could not resolve host|TLS handshake timeout|TimeoutError: The read operation timed out|TLS connection was non-properly terminated|curl (?:18|28|52|56)\b|fatal: early EOF|ReadTimeout|ConnectTimeout|502 Bad Gateway|503 Service Unavailable)')
ENVIRONMENT = re.compile(r'(?i)(?:ModuleNotFoundError|ImportError:|No module named|cannot open shared object|no matching distribution found|could not find a version that satisfies|No space left on device|CUDA driver version is insufficient|no CUDA-capable device|CUDA is not available|Torch not compiled with CUDA|command not found|Permission denied|unsupported gpu architecture|no kernel image is available for execution on the device|fatal error: [^\n]+: No such file|nvcc fatal|CMake Error|ERROR collecting|errors during collection)')
MISSING_TOOL = re.compile(r'''FileNotFoundError:[^\n]*No such file or directory:\s*['"](?:clang(?:\+\+)?(?:-\d+)?|nvcc|gcc|g\+\+|cc|c\+\+|ninja|cmake|make|git|ld|ptxas)['"]''')
ASSERTION = re.compile(r'(?:AssertionError|torch\.testing\.assert_close|Arrays are not|\[\s*FAILED\s*\]|\b[1-9][0-9]* failed\b|FAILED \([^\n]*failures=|\bFAIL: test|Numerical mismatch)')
PASS = re.compile(r'(?i)(?:\b([1-9][0-9]*) passed\b|\[\s*PASSED\s*\]\s*([1-9][0-9]*) tests?|\bRan ([1-9][0-9]*) tests?\b|\b100% tests passed\b|\bAll tests passed\b)')
NATIVE_PASS = re.compile(r'(?im)(?:\bPassed:\s*[1-9][0-9]*|Executed [1-9][0-9]* out of [1-9][0-9]* tests?: [1-9][0-9]* tests? passes|^PASS(?:\s|:)|^passed [a-z_]|validated_gpu_records=[1-9]|ERROR SUMMARY: 0 errors|overall relative L2 error:)')
CUSTOM_PASS = re.compile(r'(?im)(?:\b(?:correctness|validation|regression|checks|cases|workload) passed\b|: PASS\s*$|[1-9][0-9]*/[1-9][0-9]* tests passed|Executed [1-9][0-9]* out of [1-9][0-9]* tests?: [1-9][0-9]* tests? pass\.|matched independent host expectations)')
COMPILE_ERROR = re.compile(r'(?i)(?:error[^\n]*(?:no member|no matching function|not declared|undeclared identifier|cannot convert|undefined reference|not a member|too (?:many|few) arguments|static assertion failed|deleted function|use of deleted|cannot be referenced|not usable in a constant expression|expression must have a constant value|expression must be a modifiable lvalue|no suitable constructor exists to convert|no instance of (?:function template|overloaded function)[^\n]*matches the argument list|identifier[^\n]*is undefined|name followed by[^\n]*must be a class or namespace name)|has no member named|SyntaxError:)')
ANSI = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')


def failure_diagnostic(text: str) -> str:
    """Exclude pytest's warning appendix, retaining errors and test counts.

    Optional dependency warnings (for example torch's NumPy bridge) do not
    explain a numerical assertion failure. Only remove a delimited appendix;
    actual import exceptions in the traceback must remain infrastructure errors.
    """
    text = re.sub(r'(?ms)^=+ warnings summary =+\n.*?^-- Docs: [^\n]*\n', '', text)
    # NCCL reports absent optional network/tuner plugins at INFO and uses its
    # built-in implementation. These diagnostics are not import exceptions.
    text = re.sub(r'(?m)^\S+:\d+:\d+\s+\[\d+\]\s+NCCL INFO[^\n]*\n?', '', text)
    failures = re.search(r'(?m)^=+ FAILURES =+\s*$', text)
    if failures and not re.search(r'(?m)^=+ ERRORS =+|^Traceback|^E\s+|^ERROR', text[:failures.start()]):
        # Pytest has delimited the actual failing test tracebacks. Optional
        # plugin startup diagnostics before that section are not test failures.
        text = text[failures.start():]
    return text


def custom_driver_evidence(result: dict) -> bool:
    """Native assertion executables often succeed without printing a summary."""
    commands = '\n'.join(r.get('command', '') for r in result.get('test_results', []))
    for setup in result.get('setup_commands', []):
        if not (re.search(r'\bassert\b|EXPECT_|CHECK\(|raise AssertionError|EXIT_FAILURE|std::abort|std::exit\([1-9][0-9]*\)', setup)
                or ('expected' in setup and '!=' in setup and re.search(r'\breturn\s+1\s*;', setup))):
            continue
        for source in re.findall(r'artifacts/[^\s\"\'<>;]+\.(?:py|cu|cpp|cc|c)\b', setup):
            if source in commands or Path(source).stem in commands:
                return True
    # libcudacxx .pass.cpp cases are assertion tests, not compile-only probes.
    # Require the produced executable to actually occur after its output flag.
    for row in result.get('test_results', []):
        command = row.get('command', '')
        if '.pass.cpp' in command:
            match = re.search(r'-o\s+(artifacts/[\w./-]+)', command)
            if match and match.group(1) in command[match.end():]:
                return True
    return False


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as error:
        error.artifact_path = str(path)
        raise


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reviewed_decision(run: Path, task_id: str, result_path: Path, decision: dict) -> dict:
    """Apply an evidence-bound diagnosis without rewriting raw results.

    Manual diagnosis can resolve compile/import ambiguity, but cannot create a
    pass or claim tests executed. Stale diagnoses never transfer to a successor.
    """
    path = run / 'adjudications' / (task_id + '.json')
    if not path.is_file():
        return decision
    review = read_json(path)
    if review.get('task_id') != task_id or review.get('result_sha256') != sha256(result_path):
        return {**decision, 'adjudication_status': 'stale_result'}
    if review.get('category') not in {'model_output_error', 'infrastructure_blocked'}:
        raise ValueError('A diagnosis cannot fabricate passed or executed-test outcomes')
    evidence = review.get('evidence', [])
    if not review.get('rationale') or not evidence or any(
            not Path(ref['path']).is_file() or sha256(Path(ref['path'])) != ref['sha256'] for ref in evidence):
        return {**decision, 'adjudication_status': 'invalid_evidence'}
    return {**decision, 'automated_category': decision['category'],
            'category': review['category'], 'reason': review['reason'], 'retryable': False,
            'adjudication_path': str(path), 'adjudication_sha256': sha256(path),
            'adjudication_status': 'verified'}


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def log_path(result: dict, value: str) -> Path:
    # Recorded /tmp paths no longer exist. The durable allowlist has the same
    # command-log basename. Prefer durable evidence even on a live worker host.
    durable = Path(result.get('evidence_path', '/nonexistent'))
    candidate = durable / 'eval' / result['instance_id'] / 'artifacts/command-logs' / Path(value).name
    return candidate if candidate.is_file() else Path(value)


def read_logs(result: dict, phase: str) -> tuple[str, list[dict], bool]:
    texts, evidence = [], []
    complete = True
    for row in result.get(phase + '_results', []):
        for field in ('stdout_log', 'stderr_log'):
            if not row.get(field):
                complete = False
                continue
            path = log_path(result, row[field])
            if not path.is_file():
                complete = False
                continue
            data = path.read_bytes()
            texts.append(ANSI.sub('', data.decode(errors='replace')))
            evidence.append({'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)})
    return '\n'.join(texts), evidence, complete


def classify(result: dict | None) -> dict:
    if result is None:
        return {'category': 'missing', 'reason': 'no_result', 'retryable': True, 'logs': []}
    from swe_gpu_bench.shared.artifacts import skipped_test_output
    logs = []
    phase_text = {}
    complete = {}
    for phase in ('setup', 'post_patch_setup', 'test'):
        phase_text[phase], refs, complete[phase] = read_logs(result, phase)
        logs.extend(refs)
    tests = result.get('test_results') or []
    text = phase_text['test']
    category = result.get('failure_category') or ''
    decision = {'category': 'infrastructure_blocked', 'reason': category or 'unverified_execution',
                'retryable': False, 'logs': logs, 'test_commands': len(tests),
                'test_exit_codes': [r.get('exit_code') for r in tests]}

    def finish(kind, reason, retry=False):
        return {**decision, 'category': kind, 'reason': reason, 'retryable': retry}

    if category == 'PLATFORM_INTERRUPTED' and result.get('bootstrap_log_path'):
        bootstrap = Path(result['bootstrap_log_path'])
        if bootstrap.is_file():
            text = bootstrap.read_text(errors='replace')
            logs.append({'path': str(bootstrap), 'sha256': sha256(bootstrap), 'bytes': bootstrap.stat().st_size})
            if 'MODEL_CALLS_PAUSED' in text:
                return finish('infrastructure_blocked', 'model_calls_deferred', True)
            if ('ModelCallsPausedError: Cooldown evidence temporarily unavailable' in text
                    and "if not tasks[0].get('resume_patch') and pause_until(run / 'model-pauses'" in text):
                # Older workers let the fail-closed pre-claim check escape.
                # No case was claimed/model invoked at this failing check.
                return finish('infrastructure_blocked', 'model_calls_deferred', True)
            if re.search(r'OSError: \[Errno (?:38|95)\] (?:Function not implemented|Operation not supported)', text):
                return finish('infrastructure_blocked', 'bootstrap_environment_failure')
            if TRANSIENT.search(text):
                return finish('infrastructure_blocked', 'bootstrap_transient', True)
            if re.search(r'Traceback|unshare:|mount:|AssertionError|command not found|Permission denied|SOLVER_ISOLATION_UNAVAILABLE', text):
                return finish('infrastructure_blocked', 'bootstrap_environment_failure')
        if result.get('platform_status') in {'REJECTED', 'FAILED_TO_LAUNCH'}:
            return finish('infrastructure_blocked', 'platform_launch_rejected')

    # Check the failing phase only. Warnings during a successful install or
    # assertion failure must not erase a genuine test result.
    for phase in ('setup', 'post_patch_setup'):
        if any(r.get('exit_code') != 0 for r in result.get(phase + '_results', [])):
            if phase == 'post_patch_setup' and COMPILE_ERROR.search(phase_text[phase]) and not ENVIRONMENT.search(phase_text[phase]):
                return finish('model_output_error', 'candidate_compile_error')
            transient = bool(TRANSIENT.search(phase_text[phase]))
            return finish('infrastructure_blocked', phase + ('_transient' if transient else '_failure_needs_diagnosis'), transient)
    if tests:
        if not complete['test']:
            return finish('infrastructure_blocked', 'test_log_missing')
        # failed+skipped still represents executed tests; no pass+all skipped does not.
        if skipped_test_output(text) and not ASSERTION.search(text):
            return finish('infrastructure_blocked', 'no_tests_executed')
        observations = [r['source_access'] for phase in ('post_patch_setup', 'test')
                        for r in result.get(phase + '_results', []) if 'source_access' in r]
        if observations:
            if any(o.get('error') or o.get('overflow') or o.get('changed_during_validation') for o in observations):
                return finish('infrastructure_blocked', 'candidate_source_evidence_incomplete')
            if any(o.get('candidate_files') for o in observations) and not any(o.get('read_during_validation') for o in observations):
                return finish('infrastructure_blocked', 'candidate_source_not_read_during_validation')
        if any(r.get('exit_code') != 0 for r in tests):
            failing_text = '\n'.join(failure_diagnostic(ANSI.sub('', log_path(result, r[k]).read_text(errors='replace')))
                for r in tests if r.get('exit_code') != 0 for k in ('stdout_log', 'stderr_log')
                if r.get(k) and log_path(result, r[k]).is_file())
            if len(tests) > 1:
                per_command = [classify({**result, 'test_results': [row]})
                               for row in tests if row.get('exit_code') != 0]
                genuine = [item for item in per_command if item['category'] == 'test_failed']
                infrastructure = [item['reason'] for item in per_command if item['category'] == 'infrastructure_blocked']
                if genuine and infrastructure:
                    return {**finish('test_failed', 'executed_failure_with_partial_infrastructure'),
                            'partial_infrastructure_reasons': sorted(set(infrastructure))}
            # Bazel can report both a genuine lit/FileCheck assertion failure
            # and a later failure to zip undeclared outputs in one command.
            # Inspect the bounded lit failure block, not Bazel's aggregate FAIL
            # labels (which also appear when only the zip wrapper failed).
            if 'zip: command not found' in failing_text:
                lit_blocks = re.findall(
                    r"^\*+ TEST '[^\n]+' FAILED \*+\s*\n(.*?)^\*{4,}\s*$",
                    failing_text, re.MULTILINE | re.DOTALL)
                for block in lit_blocks:
                    if (re.search(r'^Exit Code: [1-9][0-9]*$', block, re.MULTILINE)
                            and re.search(r'error: CHECK(?:-[A-Z]+)?: (?:expected string not found in input|excluded string found in input)', block)
                            and 'note: scanning from here' in block
                            and not ENVIRONMENT.search(block)
                            and not MISSING_TOOL.search(block)
                            and not TRANSIENT.search(block)):
                        return {**finish('test_failed', 'executed_filecheck_failure_with_packaging_error'),
                                'partial_infrastructure_reasons': ['bazel_test_output_zip_missing']}
            if TRANSIENT.search(failing_text):
                return finish('infrastructure_blocked', 'test_transient', True)
            if 'no kernel image is available for execution on the device' in failing_text:
                return finish('infrastructure_blocked', 'cuda_binary_architecture_mismatch')
            if ENVIRONMENT.search(failing_text) or MISSING_TOOL.search(failing_text):
                return finish('infrastructure_blocked', 'test_environment_failure')
            if (re.search(r'Nvrtc Error[\s\S]*error[^\n]*identifier [^\n]* is undefined', failing_text)
                    and re.search(r'Ran [1-9][0-9]* tests?\b|\b[1-9][0-9]* failed\b', failing_text)):
                # The repository itself imported successfully and a test ran.
                # A compiler task emitting an invalid kernel is a functional
                # test failure, distinct from failing to build the repository.
                return finish('test_failed', 'executed_test_generated_kernel_compile_failure')
            if 'Nvrtc Error' in failing_text:
                # Runtime kernel generation without an executed-test count
                # needs diagnosis; static C++ compilation patterns below must
                # not turn an unverified runtime probe into a model outcome.
                return finish('infrastructure_blocked', 'compile_or_runtime_failure_needs_diagnosis')
            if COMPILE_ERROR.search(failing_text):
                return finish('model_output_error', 'candidate_compile_error')
            if 'Fatal Python error: Segmentation fault' in failing_text:
                # This trusted control checks 64 MiB of actual GPU transfers
                # through the candidate implementation, independently of its
                # graph tests. Its success establishes a working CUDA runtime;
                # a crash in the separate graph test is an executed failure.
                healthy_cuda_control = any(r.get('exit_code') == 0 and r.get('stdout_log')
                    and re.search(r'validated asynchronous CUDA access to CPU-owned memory [1-9][0-9]*',
                                  log_path(result, r['stdout_log']).read_text(errors='replace')) for r in tests)
                if healthy_cuda_control and PASS.search(text):
                    return finish('test_failed', 'native_test_crash_with_successful_cuda_control')
            if (custom_driver_evidence(result)
                    and re.search(r'(?m)^\w+ mismatch at case [0-9]+, width [0-9]+$', failing_text)):
                return finish('test_failed', 'executed_custom_gpu_comparison_failure')
            if ASSERTION.search(failing_text):
                return finish('test_failed', 'executed_test_failure')
            if re.search(r'(?:[A-Z_]*ASSERT\([^\n]*failed|Assertion [^\n]* failed)', failing_text):
                return finish('test_failed', 'executed_native_assertion_failure')
            if re.search(r'Executed [1-9][0-9]* out of [1-9][0-9]* tests?: [1-9][0-9]* fails locally|Failed Tests \([1-9][0-9]*\)|device-side assert|CUDA error: an illegal memory access', failing_text):
                return finish('test_failed', 'executed_native_test_failure')
            return finish('infrastructure_blocked', 'compile_or_runtime_failure_needs_diagnosis')
        if result.get('status') == 'passed' and not category:
            # Custom assertion drivers are tests too; exit 0 is accepted only
            # with an explicit assertion in the executed command, not echo/true.
            assertions = any(re.search(r'\bassert\b|assert_close|assert_allclose', r.get('command', '')) for r in tests)
            if PASS.search(text) or NATIVE_PASS.search(text) or CUSTOM_PASS.search(text) or assertions or custom_driver_evidence(result):
                return finish('passed', 'test_summary_or_executed_assertion_driver')
            return finish('infrastructure_blocked', 'positive_test_evidence_missing')

    generation = result.get('generation') or {}
    durable = Path(result.get('evidence_path', '/nonexistent'))
    old = durable / 'oracle' / result['instance_id']
    if not generation and (old / 'metadata.json').is_file():
        generation = read_json(old / 'metadata.json')
    if category in {'PRED_PATCH_INVALID', 'PRED_PATCH_APPLY_FAILURE'}:
        response = old / 'raw_response.txt'
        usage = generation.get('usage') or {}
        response_present = response.is_file() and (response.stat().st_size > 0 or
                            (usage.get('status') == 'reported' and usage.get('reported_calls', 0) > 0))
        if response_present and generation.get('pred_patch_apply_check_status') == 'invalid':
            return finish('model_output_error', 'completed_response_invalid_patch')
        if response_present and generation.get('structured_edit_status') in {'invalid', 'error'}:
            return finish('model_output_error', 'completed_response_invalid_edit')
    diagnostic = result.get('error_type', '')
    for path in (durable / 'worker.stderr.log', durable / 'eval' / result['instance_id'] / 'artifacts/command-logs/agent.stderr.log',
                 durable / 'eval' / result['instance_id'] / 'artifacts/command-logs/agent.stdout.jsonl'):
        if path.is_file():
            data = path.read_bytes()
            # Only error events, not successful solver command output, establish
            # API errors. The diagnostic is never included in the report.
            value = data.decode(errors='replace')
            if path.name == 'agent.stdout.jsonl':
                events = []
                for line in value.splitlines():
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(event, dict) and event.get('type') in {'error', 'turn.failed'}:
                        events.append(json.dumps(event))
                value = '\n'.join(events)
            diagnostic += '\n' + value
            logs.append({'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)})
    if 'ModelCallsPausedError' in diagnostic:
        return finish('infrastructure_blocked', 'model_calls_deferred', True)
    if RATE_LIMIT.search(diagnostic):
        return finish('infrastructure_blocked', 'rate_limit', True)
    if TRANSIENT.search(diagnostic) or category in {'WORKER_PROCESS_FAILURE', 'PLATFORM_INTERRUPTED'}:
        return finish('infrastructure_blocked', 'transient_infrastructure', True)
    if 'RateLimit' in diagnostic:
        return finish('infrastructure_blocked', 'rate_limit', True)
    return decision


def patch_candidate(result: dict | None, old_record: dict, current: dict) -> dict | None:
    """Reuse only completed generation for the identical solver problem/base."""
    if not result or any(old_record.get(k) != current.get(k) for k in ('repo', 'base_commit', 'problem_statement')):
        return None
    root = Path(result.get('evidence_path', '/nonexistent'))
    iid = current['instance_id']
    if result['mode'] == 'oracle':
        old = root / 'oracle' / iid
        metadata = old / 'metadata.json'
        if metadata.is_file() and read_json(metadata).get('pred_patch_apply_check_status') == 'valid':
            patch = old / 'pred.diff'
            if patch.is_file():
                return {'kind': 'oracle', 'path': str(patch), 'sha256': sha256(patch), 'generation_dir': str(old)}
    else:
        patch = root / 'eval' / iid / 'artifacts/agent.diff'
        # An interrupted solver's partial edits are not a submitted answer.
        if result.get('failure_category') not in {'CASE_TIMEOUT', 'AGENT_EXECUTION_FAILURE', 'WORKER_PROCESS_FAILURE', 'INFRASTRUCTURE_ERROR'} and patch.is_file() and patch.stat().st_size:
            return {'kind': 'agent', 'path': str(patch), 'sha256': sha256(patch)}
    return None


def partition_batches(tasks: list[dict]) -> list[list[dict]]:
    """Stable model/mode/runtime grouping, repository locality, 3 then 2+2."""
    groups = {}
    for task in tasks:
        groups.setdefault((task['model'], task['mode'], task.get('runtime_docker_image', task['docker_image'])), []).append(task)
    buckets = []
    for key in sorted(groups):
        rows = sorted(groups[key], key=lambda t: (t.get('pilot_priority', 1), t['repo'], t['instance_id']))
        batches = []
        while rows:
            size = 2 if len(rows) in (2, 4) else min(3, len(rows))
            batches.append(rows[:size]); rows = rows[size:]
        buckets.append(batches)
    # Round robin model/mode, including the initial ten pilots.
    return [bucket[i] for i in range(max(map(len, buckets), default=0)) for bucket in buckets if i < len(bucket)]

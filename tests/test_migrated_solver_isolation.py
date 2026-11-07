import subprocess
import pytest
from swe_gpu_bench.evaluate.use_cases.solver_isolation import isolate_git_history


def test_solver_cannot_read_future_commit_or_remote(tmp_path):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()
    git("init", "--quiet")
    git("config", "user.name", "Fixture")
    git("config", "user.email", "fixture@example.invalid")
    source = tmp_path / "source.py"
    source.write_text("broken\n")
    git("add", ".")
    git("commit", "-qm", "base")
    base = git("rev-parse", "HEAD")
    source.write_text("gold solution\n")
    git("commit", "-qam", "solution")
    head = git("rev-parse", "HEAD")
    git("remote", "add", "origin", "https://example.invalid/repo")
    git("checkout", "--quiet", "--detach", base)
    isolate_git_history(tmp_path, base)
    assert git("rev-parse", "HEAD") == base
    assert git("rev-list", "--all", "--count") == "1"
    assert git("remote") == ""
    assert source.read_text() == "broken\n"
    assert subprocess.run(["git", "cat-file", "-e", head], cwd=tmp_path, capture_output=True).returncode != 0
    assert not (tmp_path / ".git/objects/info/alternates").exists()


def test_history_isolation_preserves_pinned_submodule_for_rebuild(tmp_path):
    def git(root, *args):
        return subprocess.check_output(['git', '-c', 'protocol.file.allow=always', *args], cwd=root, text=True).strip()
    dependency = tmp_path / 'dependency'
    parent = tmp_path / 'parent'
    for root in [dependency, parent]:
        root.mkdir()
        git(root, 'init', '--quiet')
        git(root, 'config', 'user.name', 'Fixture')
        git(root, 'config', 'user.email', 'fixture@example.invalid')
    (dependency / 'header.h').write_text('base\n')
    git(dependency, 'add', '.')
    git(dependency, 'commit', '-qm', 'base')
    pinned = git(dependency, 'rev-parse', 'HEAD')
    (dependency / 'header.h').write_text('future\n')
    git(dependency, 'commit', '-qam', 'future')
    future = git(dependency, 'rev-parse', 'HEAD')
    git(parent, 'submodule', 'add', '--quiet', str(dependency), 'thirdparty/dependency')
    child = parent / 'thirdparty/dependency'
    git(child, 'checkout', '--quiet', '--detach', pinned)
    git(parent, 'add', '.')
    git(parent, 'commit', '-qm', 'base with pinned dependency')
    base = git(parent, 'rev-parse', 'HEAD')
    assert (child / '.git').is_file()
    isolate_git_history(parent, base)
    # Post-patch setup must be able to initialize/update the same pinned source.
    git(parent, 'submodule', 'update', '--init', '--recursive')
    assert git(child, 'rev-parse', 'HEAD') == pinned
    assert git(child, 'rev-list', '--all', '--count') == '1'
    assert git(child, 'remote') == ''
    assert (child / 'header.h').read_text() == 'base\n'
    assert subprocess.run(['git', 'cat-file', '-e', future], cwd=child, capture_output=True).returncode != 0


@pytest.mark.parametrize("phase,category", [("setup", "SETUP_FAILURE"), ("rebuild", "POST_PATCH_SETUP_FAILURE")])
def test_strict_evaluation_rejects_partial_setup_even_if_tests_pass(phase, category):
    from swe_gpu_bench.evaluate.use_cases.eval import _determine_status
    failed = [{"exit_code": 1}, {"exit_code": 0}]
    status = _determine_status(failed if phase == "setup" else [],
                               failed if phase == "rebuild" else [],
                               [{"exit_code": 0}], "no_test_patch", strict_commands=True)
    assert status == ("error", category)


def test_strict_evaluation_does_not_count_all_skipped_as_pass(tmp_path):
    from swe_gpu_bench.evaluate.use_cases.eval import _determine_status
    log = tmp_path / "stdout.log"
    log.write_text("3 skipped in 0.1s\n")
    assert _determine_status([], [], [{"exit_code": 0, "stdout_log": str(log)}],
                             "no_test_patch", strict_commands=True) == ("error", "NO_TESTS_EXECUTED")


def test_responses_agent_profile_pins_provider_and_effort():
    from swe_gpu_bench.shared.models import ModelProfile
    from swe_gpu_bench.adapters.agent import build_agent_runner
    profile = ModelProfile(name="fixture", kind="agent", provider="openai_responses", model="gpt-6.1-sol",
                           base_url="https://example.invalid/v1", api_key_env="BENCH_KEY", reasoning_effort="high")
    runner = build_agent_runner(profile=profile)
    assert 'model_provider="swe_gpu_bench"' in runner._extra_args
    assert 'model_providers.swe_gpu_bench.wire_api="responses"' in runner._extra_args
    assert 'model_reasoning_effort="high"' in runner._extra_args

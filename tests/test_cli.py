import os
import subprocess
import sys
from pathlib import Path

import pytest

from swe_gpu_bench import cli


@pytest.mark.parametrize("command", cli.COMMANDS, ids=lambda command: f"{command.stage}-{command.name}")
def test_every_registered_command_builds_help(command, capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main([command.stage, command.name, "--help"])

    assert excinfo.value.code == 0
    assert command.help in capsys.readouterr().out


def test_command_names_are_unique_per_stage() -> None:
    keys = [(command.stage, command.name) for command in cli.COMMANDS]
    assert len(keys) == len(set(keys))
    assert {command.stage for command in cli.COMMANDS} == set(cli.STAGE_HELP)


def test_main_strips_separator_and_dispatches(monkeypatch) -> None:
    seen = {}

    def fake_run(args):
        seen["eval_root"] = args.eval_root
        return 3

    monkeypatch.setattr("swe_gpu_bench.evaluate.summary.run", fake_run)

    assert cli.main(["evaluate", "summary", "--", "--eval-root", "results/x"]) == 3
    assert seen == {"eval_root": "results/x"}


def test_unknown_command_exits_with_usage_error(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["generate", "does-not-exist"])

    assert excinfo.value.code == 2


def test_stage_help_imports_no_command_modules() -> None:
    code = (
        "import sys\n"
        "from swe_gpu_bench import cli\n"
        "cli.build_parser(['collect'])\n"
        "loaded = [name for name in sys.modules if name.startswith('swe_gpu_bench.') and name != 'swe_gpu_bench.cli']\n"
        "print(','.join(sorted(loaded)))\n"
    )
    src = Path(cli.__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(src), os.environ.get("PYTHONPATH")]))}
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env)

    assert completed.stdout.strip() == ""

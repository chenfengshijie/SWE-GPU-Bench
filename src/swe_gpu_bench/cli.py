"""``swe-gpu-bench <stage> <command> [options]``.

Every command lives in one module exposing an ``add_*arguments(parser)`` / ``run*(args) -> int``
pair; register new commands in ``COMMANDS``.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from typing import NamedTuple, Optional, Sequence


class Command(NamedTuple):
    stage: str
    name: str
    help: str
    module: str
    add_arguments: str = "add_arguments"
    run: str = "run"


COMMANDS: tuple[Command, ...] = (
    Command("evaluate", "campaign", "Prepare, run, resume and report frozen evaluations", "evaluate.campaign"),
    Command("collect", "fetch", "Fetch merged PR metadata from GitHub", "collect.fetch"),
    Command("collect", "patches", "Download PR diffs", "collect.patches"),
    Command("collect", "annotate", "Annotate PRs with deterministic signals", "collect.filter", "add_annotate_arguments", "run_annotate"),
    Command("collect", "filter", "Keep candidate PRs and count rejection reasons", "collect.filter", "add_filter_arguments", "run_filter"),
    Command("collect", "fused-review", "LLM review for benchmark fit", "collect.review", "add_fused_arguments", "run_fused"),
    Command("collect", "perf-review", "LLM review of perf PRs for benchmarkability", "collect.review", "add_perf_arguments", "run_perf"),
    Command("generate", "repro", "Let Codex reproduce a PR's environment and tests", "generate.repro"),
    Command("generate", "verify", "Replay recorded setup/test commands", "generate.verify"),
    Command("generate", "query-generate", "Write problem statements for verified PRs", "generate.query"),
    Command("generate", "split-patch", "Split PR diffs into patch and test patch", "generate.split_patch"),
    Command("generate", "dataset-build", "Assemble the dataset JSONL", "generate.dataset", "add_build_arguments", "run_build"),
    Command("generate", "dataset-finalize", "Re-split patches / regenerate statements", "generate.dataset", "add_finalize_arguments", "run_finalize"),
    Command("generate", "dataset-check", "Offline quality checks for dataset rows", "generate.check"),
    Command("generate", "perf-repro", "Discover and replay perf commands on base/head", "generate.perf_repro"),
    Command("generate", "perf-dataset-build", "Assemble the perf dataset JSONL", "generate.dataset", "add_perf_build_arguments", "run_perf_build"),
    Command("generate", "export", "Export rows in the published Hugging Face schema", "generate.dataset", "add_export_arguments", "run_export"),
    Command("evaluate", "agent", "Evaluate an agent on one instance", "evaluate.agent_eval"),
    Command("evaluate", "perf-agent", "Let an agent write a perf patch for one instance", "evaluate.perf_eval"),
    Command("evaluate", "oracle-generate", "Generate oracle-context LLM patches", "evaluate.oracle", "add_generate_arguments", "run_generate"),
    Command("evaluate", "oracle-validate", "Grade an oracle patch with the hidden tests", "evaluate.oracle", "add_validate_arguments", "run_validate"),
    Command("evaluate", "perf-validate", "Grade a perf patch and compare numeric base/candidate metrics", "evaluate.perf_validate"),
    Command("evaluate", "summary", "Summarize result.json files under an eval root", "evaluate.summary"),
)
STAGE_HELP = {
    "collect": "Collect PRs, patches, and candidate reviews",
    "generate": "Reproduce environments and build datasets",
    "evaluate": "Run agent, oracle, and perf evaluations",
}


def _load(command: Command, attr: str):
    return getattr(importlib.import_module(f"swe_gpu_bench.{command.module}"), attr)


def build_parser(argv: Sequence[str] = ()) -> argparse.ArgumentParser:
    """Build the parser; only the module of the selected command (if any) is imported."""
    parser = argparse.ArgumentParser(prog="swe-gpu-bench")
    stages = parser.add_subparsers(dest="stage", required=True, metavar="stage")
    stage_parsers = {
        stage: stages.add_parser(stage, help=help_text).add_subparsers(dest="command", required=True, metavar="command")
        for stage, help_text in STAGE_HELP.items()
    }
    selected = tuple(argv[:2])
    for command in COMMANDS:
        sub = stage_parsers[command.stage].add_parser(command.name, help=command.help, description=command.help)
        sub.set_defaults(_command=command)
        if (command.stage, command.name) == selected:
            _load(command, command.add_arguments)(sub)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    if tokens[2:3] == ["--"]:
        del tokens[2]
    args = build_parser(tokens).parse_args(tokens)
    return int(_load(args._command, args._command.run)(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())

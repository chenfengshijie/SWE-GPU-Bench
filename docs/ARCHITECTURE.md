# Architecture

SWE-GPU-Bench has three stages, each exposed as `swe-gpu-bench <stage> <command>`:

1. `collect`: fetch PR metadata and diffs from GitHub, annotate and filter candidates, and run LLM reviews.
2. `generate`: reproduce and verify PR environments, write problem statements, split patches, and build/export the correctness and performance datasets.
3. `evaluate`: run agents and the oracle baseline against dataset rows, grade correctness and performance, and summarize results.

## Package Layout

```text
src/swe_gpu_bench/
├── cli.py        # COMMANDS registry and argument dispatch
├── core/         # shared helpers, no stage imports
│   ├── agent.py      # codex / opencode CLI runners
│   ├── config.py     # swe_gpu_bench.toml profiles and --profile/--model overrides
│   ├── diff.py       # diff parsing, test-path detection, diff summaries (GPU signal, size), new-interface names
│   ├── git.py        # clone cache, checkout, diff capture, hiding history after base_commit
│   ├── github.py     # GitHub REST client with token rotation
│   ├── io.py         # JSON/JSONL IO, data layout, verified-record loading
│   ├── llm.py        # OpenAI-compatible chat completion
│   ├── patch.py      # apply / check model patches
│   ├── selection.py  # --instance-id / --all dataset selection
│   └── shell.py      # run setup/test commands in a case directory
├── collect/      # fetch, patches, filter (annotate + filter), review (fused + perf)
├── generate/     # repro, verify, query, split_patch, check, dataset (build/finalize/export/perf), perf_repro
└── evaluate/     # agent_eval, perf_eval, oracle, perf_validate, summary
```

`prompts/` holds every prompt template the commands load, `configs/repos.toml` lists the default repositories for `collect fetch`, and `swe_gpu_bench.toml` defines the model profiles.

## Dependency Direction

`core` never imports a stage. Stage modules import `core` and, where a workflow is genuinely shared, a sibling module in the same stage (for example `evaluate.oracle` and `evaluate.perf_validate` grade patches through `evaluate.agent_eval`, and `generate.dataset` reuses `generate.query`, `generate.split_patch`, and `generate.check`). Stages do not import each other; they communicate through files under `data/` and `results/`.

## Quality Gates

Each stage drops what the next one cannot trust, using only local, deterministic checks:

- `collect filter`: merged into the default branch, resolved issue, test change, GPU signal, bounded size, no binary diffs; the rejection counts per reason are written next to the output.
- `generate verify --dataset`: `base_commit` + `test_patch` must fail the tests, and adding the reference patch must make them pass.
- `generate dataset-check` (also run by `dataset-build` and `export`): row-level errors and warnings, see the README for the codes.
- `evaluate agent` / `evaluate perf-agent`: the agent's case directory holds an allowlist of task fields, and the checkout keeps only `base_commit`'s ancestry.

## Commands

`cli.py` registers each command as `Command(stage, name, help, module, add_arguments, run)`. The module exposes `add_arguments(parser)` and `run(args) -> int` (or the named alternatives when one module serves several commands). `build_parser` imports only the module of the selected command, so `--help` and unrelated commands never pay for heavy imports.

To add a command, write those two functions in the stage module and append one `Command(...)` entry. Commands that call a model use `core.config.add_profile_arguments(parser, kind="agent" | "text")` and `resolve_*_profile(args)` so `--profile`, `--model`, `--base-url`, and `--api-key-env` behave the same everywhere.

## Data Layout

| Path | Written by |
|---|---|
| `data/prs/<owner>__<repo>/<pr>.json` | `collect fetch`, `collect annotate` |
| `data/patches/<owner>__<repo>/<pr>.diff` | `collect patches` |
| `data/filtered_prs_with_tests.json`, `data/filtered_prs_with_tests.summary.json` | `collect filter` |
| `data/fused_scored/`, `data/perf_reviewed/` | `collect fused-review`, `collect perf-review` |
| `results/repro/<case>/`, `results/verify/<case>/` | `generate repro`, `generate verify --case-dir` |
| `results/verify_dataset/<instance_id>/result.json` | `generate verify --dataset` |
| `data/queries/`, `data/split_patch/` | `generate query-generate`, `generate split-patch` |
| `data/dataset/swe_gpu_bench.jsonl`, `data/perf_dataset/swe_gpu_bench_perf.jsonl` | `generate dataset-build`, `generate perf-dataset-build` |
| `data/dataset/swe_gpu_bench.check.json` | `generate dataset-build`, `generate dataset-check` |
| `data/export/<split>.jsonl`, `data/export/<split>.check.json` | `generate export` |
| `results/<eval>/<instance_id>/result.json` | `evaluate *` |

Scratch checkouts go under `tmp/`; `data/`, `results/`, and `tmp/` are git-ignored.

## Tests

`tests/test_<module>.py` mirrors the source modules. Tests never touch the network or a GPU: they mock `subprocess.run` in `core.shell`, `core.git`, and `core.agent`, use fake GitHub clients, and patch `core.llm.complete`.

## Campaign evaluations

`evaluate/campaign.py` owns preparation, scheduling and reporting;
`evaluate/campaign_worker.py` separates generation from fresh validation;
`evaluate/metrics.py` selects versioned numeric parsers. `adapters/campaign.py`
implements local/container/external submission and reconciliation. Packaged `evaluate/resources/`
holds templates, pinned tools/dependencies and the 39-recipe registry.

The proven Oracle/Agent helpers remain under `evaluate/use_cases/` with their
migration regression tests. Their dependencies are explicitly selected under
`shared/` and `adapters/`; pinned checkout code is in `core/campaign_repo.py`.
No stage imports another stage. These modules do not depend on historical
workspace files, and collection/generation keep their existing implementation.
See [campaign contracts](EVALUATION_CAMPAIGNS.md).

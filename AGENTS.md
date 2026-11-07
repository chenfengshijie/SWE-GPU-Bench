# Repository Guidelines

## Project Structure & Module Organization

SWE-GPU-Bench builds and evaluates a benchmark of GPU-library pull requests (published as the Hugging Face dataset `Froze/SWE-GPU-Bench`). Everything ships as the `swe_gpu_bench` package with one CLI, `swe-gpu-bench <stage> <command>`.

- `src/swe_gpu_bench/cli.py`: the `COMMANDS` registry; every command is one `Command(...)` entry.
- `src/swe_gpu_bench/core/`: shared helpers (GitHub, git, shell, LLM, agent runners, patches, profiles, IO). `core` never imports a stage.
- `src/swe_gpu_bench/collect/`, `generate/`, `evaluate/`: the three pipeline stages; see `docs/ARCHITECTURE.md`.
- `tests/`: pytest tests, one `test_<module>.py` per source module.
- `prompts/`: prompt templates loaded by the commands. `configs/repos.toml`: default repositories. `swe_gpu_bench.toml`: model profiles.
- Generated data goes to `data/`, results to `results/`, scratch checkouts to `tmp/`; all are git-ignored.

## Build, Test, and Development Commands

Use Python 3.12 or newer. Run commands from the repository root:

```sh
uv venv --python 3.12 && uv pip install -e '.[test]'   # or: python3.12 -m venv .venv && pip install -e '.[test]'
.venv/bin/python -m pytest
.venv/bin/python -m pytest tests/test_split_patch.py -q
swe-gpu-bench --help
swe-gpu-bench evaluate agent --help                    # python -m swe_gpu_bench ... works too
```

See `README.md` for the full collect → generate → evaluate pipeline.

## Coding Style & Naming Conventions

Follow existing Python style: four-space indentation, `snake_case` functions and modules, `PascalCase` classes, and uppercase constants. Use type annotations, `pathlib.Path` for filesystem operations, and explicit error handling at CLI boundaries. Dataset directories encode `owner/repo` as `owner__repo`. No formatter or linter is configured.

A new command is an `add_arguments(parser)` / `run(args) -> int` pair in its stage module plus one `Command(...)` entry in `cli.py`. Commands that call a model use `core.config.add_profile_arguments` instead of defining their own model flags.

## Testing Guidelines

Name files `test_<feature>.py` and functions `test_<behavior>`. Use pytest fixtures such as `tmp_path` and `monkeypatch`, fake GitHub clients, and mocked `subprocess.run` (in `core.shell`, `core.git`, `core.agent`) and `core.llm.complete` to avoid live services. Cover pagination, retries, filtering decisions, persisted output, and reproduction/evaluation failures when changing those behaviors. No coverage threshold is configured.

## Commit & Pull Request Guidelines

Use short, imperative subjects describing one focused change. PR descriptions should explain the behavior change, relevant issues, and test commands/results; include example output for dataset or CLI changes.

## Security & Configuration

Supply credentials through `GITHUB_TOKEN`, `OPENAI_API_KEY`, and `OPENAI_BASE_URL` where supported. Never embed credentials in code, documentation or committed configuration. Keep generated data, logs, virtual environments, and bytecode out of new commits.

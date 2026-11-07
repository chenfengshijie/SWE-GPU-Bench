# Building the Dataset

This repo also ships the pipeline used to build the benchmark. Every step reads and writes the default `data/` and `results/` paths shown below.

```text
collect fetch -> collect patches -> collect annotate -> collect filter -> collect fused-review
  -> generate repro -> generate verify -> generate split-patch / query-generate -> generate dataset-build
  -> generate verify --dataset -> generate dataset-finalize (optional) -> generate dataset-check -> generate export
collect fused-review -> collect perf-review -> generate perf-repro -> generate perf-dataset-build
```

### Collect

```bash
export GITHUB_TOKEN=...   # or --tokens a,b,c to rotate through several
swe-gpu-bench collect fetch --repos NVIDIA/cutlass Dao-AILab/flash-attention   # default list: configs/repos.toml
swe-gpu-bench collect patches
swe-gpu-bench collect annotate
swe-gpu-bench collect filter                                                   # -> data/filtered_prs_with_tests.json
swe-gpu-bench collect fused-review --profile openai_default                    # -> data/fused_scored/
swe-gpu-bench collect perf-review --profile openai_default                     # -> data/perf_reviewed/
```

`collect fetch` keeps merged PRs only (`--include-unmerged` keeps the rest) and records the repository's default branch. `collect annotate` writes deterministic signals into each PR JSON: resolved issues, test changes, merged, default-branch base, GPU signal (a changed path or line that looks GPU-specific), and the size of the non-test, non-doc change. `collect filter` keeps PRs that reference a resolved issue, change tests, were merged into the default branch (not a release, backport, or stacked `gh/<user>/<n>/base` branch), touch GPU code, change at most 1000 lines in at most 30 code files, and carry no binary diffs. It logs how many PRs each criterion rejected and writes the counts to `data/filtered_prs_with_tests.summary.json`. `--allow-no-issue`, `--include-unmerged`, `--include-non-default-base`, `--allow-no-gpu-signal`, `--max-code-lines`, and `--max-code-files` relax the criteria.

### Generate

```bash
# Let Codex discover setup/test commands for one PR, then replay them in a fresh checkout
swe-gpu-bench generate repro --repo Dao-AILab/flash-attention --pr 1365        # -> results/repro/flash-attention-pr1365/
swe-gpu-bench generate verify --case-dir results/repro/flash-attention-pr1365  # -> results/verify/flash-attention-pr1365/

# Assemble the dataset from verified cases (any directory of verify results works as --verified-dir)
swe-gpu-bench generate split-patch    --verified-dir results/verify
swe-gpu-bench generate query-generate --verified-dir results/verify --profile openai_default
swe-gpu-bench generate dataset-build  --verified-dir results/verify            # -> data/dataset/swe_gpu_bench.jsonl

# Fail-to-pass check: tests must fail on base_commit + test_patch and pass once the reference patch is applied too
swe-gpu-bench generate verify --dataset data/dataset/swe_gpu_bench.jsonl --all # -> results/verify_dataset/

# Offline quality checks (no GPU, network, or model needed)
swe-gpu-bench generate dataset-check --verify-root results/verify_dataset \
  --output data/dataset/swe_gpu_bench.clean.jsonl                              # -> data/dataset/swe_gpu_bench.check.json

# Export rows in the published Hugging Face schema; rows with check errors are dropped
swe-gpu-bench generate export --docker-image <image> --verify-root results/verify_dataset  # -> data/export/test.jsonl

# Performance dataset
swe-gpu-bench generate perf-repro --repo deepspeedai/DeepSpeed --pr 7886       # -> results/perf_repro/
swe-gpu-bench generate perf-dataset-build                                      # -> data/perf_dataset/swe_gpu_bench_perf.jsonl
```

`task_type` is derived from the fused review's `pr_category` (`bugfix` -> `bug`, `perf` -> `perf`, otherwise `feature`). `docker_image` is not produced by the pipeline; pass it to `export` with `--docker-image`. Use `export --instance-id-file ids.txt --split <name>` to write a subset split.

`split-patch` puts a diff section in `test_patch` when its path holds a whole test token (`tests/`, `test_x.py`, `x_test.cc`, `XTest.cpp`, but not `warpspecialized.hpp`), when the test commands name or execute it (including the sources of example binaries and new files next to them), or when it is a build file whose changed lines register a test target. `dataset-finalize` re-splits `patch` + `test_patch` together with the current rules and keeps every other field; with `--regenerate-queries` the prompt also lists the names the reference patch introduces and the hidden tests call, since the problem statement has to mention them.

`dataset-check` reports, per row, errors that make it unusable as a task and warnings that need a human look:

| Severity | Codes |
|---|---|
| error | `missing_commit`, `empty_patch`, `no_test_commands`, `unverified` (with `--verify-root`, a row without a dataset verification result counts as unverified), `unmerged_pr`, `split_mismatch` (fix with `dataset-finalize`), `unappliable_binary`, `cpu_only_tests` (`NO_CUDA=1`, `JAX_PLATFORMS=cpu`, `TRITON_INTERPRET=1`, an empty `CUDA_VISIBLE_DEVICES`, llvm-only `TVM_TEST_TARGETS`, `bazel test` without `--config=cuda`), `host_path` (`/home/<user>`, ...), `duplicate` (same instance id, or the same changed lines as an earlier usable row) |
| warning | `no_test_patch`, `large_patch`, `cuda_path`, `network_in_tests`, `unpinned_torch`, `blackwell_signal`, `underspecified_interface`, `harness_entangled`, `no_gpu_signal` |

`--ignore CODE ...` skips codes. `dataset-build` writes the same report next to its output, and `export` drops rows with errors unless `--keep-invalid`, writing the report to `data/export/<split>.check.json`. `perf-repro` marks a case `ready` only when setup, correctness, and perf commands all pass on both commits.

---

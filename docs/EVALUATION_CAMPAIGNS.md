# Oracle/Agent evaluation campaigns

See the [validation report](EVALUATION_VALIDATION.md) for CPU/package checks and
the four recorded H100 smoke outcomes, including their limitations.

The campaign interface evaluates correctness and performance records using the
same task, attempt, candidate and result contracts. It preserves the collection
and generation commands already on master. The former top-level `collector/`
was replaced upstream by `src/swe_gpu_bench/collect/` before this change.

## Install and configure

For formal evaluations, clone the repository, check out `master` and run
`git pull --ff-only origin master`. Check that `origin` points to the intended
framework repository. Use Python 3.12 and the checked-in lock:

```bash
uv sync --frozen --extra test --python 3.12
uv run swe-gpu-bench evaluate campaign --help
mkdir -p data
cp configs/evaluation/campaign.toml data/my-campaign.toml
```

Edit paths in the copy; all relative paths resolve from the config's directory.
Select datasets explicitly and set the model profiles. Profiles support
`reasoning_effort`, Codex or OpenCode, and provider endpoints. Secrets come from
`api_key_env` or the existing file named by `credential_env_file`. Preparation
copies profile definitions, never that credential file. Inline secret values and
profile `extra_env` are rejected. Input records must have a nonempty
`docker_image`. Optional `runtime_image` explicitly overrides execution images;
both requested and actual images remain in evidence.

```bash
uv run swe-gpu-bench evaluate campaign prepare --require-master --config data/my-campaign.toml
uv run swe-gpu-bench evaluate campaign run --run /path/printed/by/prepare
uv run swe-gpu-bench evaluate campaign status --run /path/to/run
uv run swe-gpu-bench evaluate campaign resume --run /path/to/run
uv run swe-gpu-bench evaluate campaign summarize --run /path/to/run
```

`prepare` freezes selected records, complete source-file hashes, profiles, package
code/resources, dependency files, evidence-bound exclusions and repairs. `run`
and `resume` use the frozen package. Hash verification precedes scheduling.
With `--require-master`, preparation also requires a clean working tree and a
source-loaded package matching the live `origin/master`. It rejects unavailable
remotes, stale or unpublished commits, and frozen bytes that differ from that
commit. `code_origin.verified_master_sha` records the verified framework SHA;
without the flag it is null. The sample repository's `base_commit` is independent.
Existing runs never update themselves to a newer master.
Changing inputs, models, repairs or code requires a new run; historical runs stay
unchanged. `status` reads local evidence; `run --once` / `resume --once` perform one
platform reconciliation and scheduling tick.

## Four execution paths

Each dataset has `task_type="correctness"` or `"performance"`; each model can
select `modes=["oracle", "agent"]` and a subset of `task_types`.

| task_type | mode | Generation | Independent validation |
|---|---|---|---|
| correctness | oracle | Text model receives the problem and base files identified by reference-patch paths | Fresh base checkout, candidate, trusted test patch, rebuild, tests |
| correctness | agent | Codex/OpenCode modifies an isolated base environment | Fresh base checkout and the submitted candidate, then hidden tests |
| performance | oracle | Same Oracle input contract | Measure base, validate candidate correctness, measure candidate, compare fixed metrics |
| performance | agent | Same Agent input contract | Same deterministic performance validation |

For a single path, set one dataset and one mode in the example config. For four
paths, retain both datasets and both modes. A model may use different Oracle and
Agent profiles. The original single-instance CLI commands remain available;
`perf-validate` retains its argument signature but uses numeric parsing. Its
legacy `llm_judgment` result key is a compatibility alias with `method=deterministic`.
Use campaigns for strict isolation, recovery, freezing and coverage reports.

The reference patch is used only to locate Oracle input files. Neither reference
edits nor evaluation test/performance scripts are model input. Agent generation
runs environment setup before solver invocation, with scoring assets masked by
the filesystem guard. Both modes restrict Git objects to the pinned base snapshot,
including initialized submodules. Candidate capture precedes any trusted test
patch. Validation uses a fresh checkout so solver-written scoring files are not
carried forward.

Strict campaign checkouts fetch only the pinned base commit with depth one.
Once that commit exists in the private per-task cache, validation checkouts reuse
it without refreshing the remote. This avoids downloading an entire repository's
history and never substitutes a pull request's solved head for a missing base.

The Linux solver guard requires user/mount namespaces and checks actual CUDA in
the solver and a child process before Agent model calls. It hides shared datasets,
sibling attempts, evaluator scripts/logs, other process environments and the full
Git cache. It fails closed. Network access remains available to model endpoints
and dependency installation; this is not a network sandbox. Prompts forbid
retrieving upstream answers. Dedicated one-case worker containers are the intended
deployment. CPU fixtures simulate model calls and do not establish GPU isolation.

## Execution adapters

Local mode runs inside an already provisioned compatible container. Set
`SWEBENCH_RUNTIME_IMAGE` to its actual image, assign exactly one GPU via
`CUDA_VISIBLE_DEVICES`, and install `uv`, Codex and OpenCode. The image declaration
is an operator attestation; local mode does not create Docker containers. An
absent/mismatched image, unavailable GPU, unsupported namespaces or missing tools
produces an infrastructure result, never a model failure. Set local concurrency
to one unless separate processes/containers have independent GPU assignments.

Docker mode launches a named container for each attempt, mounts only the frozen
run and an optional credential file, and assigns one configured GPU device.
Configure `[docker].devices` (indices or UUIDs), `cpus`, `memory`, `shm_size`,
`forward_env` and `extra_args`; concurrency cannot exceed the number of devices.
A lost launch response is recovered by the deterministic container name. Missing
containers stay unknown; only an inspected `exited`/`dead` state frees the slot.
Containers are retained for diagnosis instead of automatically removed.

Use a task-compatible Linux x86-64 image with Python 3, Git, Bash, build tools and
CUDA libraries/toolchain appropriate to the record. The optional
`containers/evaluator.Dockerfile` builds a generic CUDA development image:

```bash
docker build -f containers/evaluator.Dockerfile -t swe-gpu-evaluator:local .
```

Set `runtime_image` to that tag to explicitly override record images. A single
image is not compatible with every task; record-specific setup and evidence-bound
repairs remain necessary. The host needs a compatible NVIDIA driver and
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html).
Device selection follows [Docker GPU configuration](https://docs.docker.com/engine/containers/gpu/).
The solver guard requires user/mount namespaces. The example config shows explicit
container security options for hosts whose default policy blocks those operations;
choose options appropriate to the host. Unsupported isolation fails closed.
The CUDA preflight allocates, fills and copies device memory using the driver API,
including a child process, without requiring a particular ML framework.

External mode runs an operator-supplied Python adapter, copied into the frozen run
and checked by hash. This supports batch schedulers and cluster services without
embedding deployment names, endpoints, queues or accounts in the package. See
[the adapter protocol](EVALUATION_ADAPTERS.md). Keep deployment-specific code and
configuration outside the repository. All three backends use the same frozen
tasks, worker, candidates, result schema and scoring rules.

The packaged bootstrap downloads tools with pinned hashes and installs locked
evaluator dependencies. Each task has a separate command environment so recipe
installs cannot replace evaluator dependencies. The default per-case deadline is
7,200 seconds; each Git operation is bounded by `git_timeout_seconds` (default 600). Adapter calls have bounded timeouts; uncertainty never authorizes a
second submission.

## Recovery and evidence

The controller uses an OS lock. Each remote submit has an intent written before
the request and a separately recovered receipt. Unknown submissions consume a
slot, including after a controller crash, and are looked up by unique caption.
A missing job in a recent listing is not terminal evidence. Known job IDs require
an explicit terminal detail response. Never delete an uncertain intent to retry.
Local recovery verifies host, PID and process start time; a lost launch receipt
also remains unknown.

Attempts have separate directories and a worker claim. Completed results are not
rewritten. Mutable `state.json`, `results/` and summaries are indexes; attempts
retain their original evidence. `candidates/<task>/candidate.json` binds a submitted
patch to its record hash. Retries use that answer and reject changed patches.
Actual test failures and completed invalid/empty answers are not regenerated.
To revalidate an existing answer with repaired evaluator code, prepare a new run
with `candidate_imports=[{task_id="...", manifest=".../candidate.json",
sha256="<manifest SHA256>"}]`. The record hash, model, mode and patch hash must
match. The imported patch and provenance become frozen inputs. This reuses the
answer while producing new validation evidence; it does not import a pass.
Transient infrastructure errors have a configurable retry limit (default two
additional attempts). Model cooldowns are shared across modes; validation with a
saved candidate can continue during cooldown. Deferral itself consumes no
transient-retry budget.

Create `STOP` in a run to stop new scheduling. It leaves accepted jobs running.
After diagnosing the cause, remove `STOP` and use `resume`. Never remove a worker
claim until its platform job has a confirmed terminal state. New attempts use
new directories; scratch locations and preserved logs are recorded in results.

`attempts/<task>/attempt-NNNN/` holds task/config bindings, receipts, environment,
stdout/stderr, command exit codes/timing, model response or Agent logs, candidate
metadata, source-read evidence and the final raw result. Missing results and
timeouts are explicit. Zero-test/all-skipped execution does not count as passed.

## Performance metrics

The registry in `evaluate/resources/performance-20261103.json` contains all 39
recipes in the revised 2026-11-03 EVALUATION.md. Each parser is bound to instance ID,
direction and a canonical SHA256 of setup/test/perf-setup/perf command arrays.
Changed or unknown recipes fail explicitly; there is no LLM fallback. The generic
`portable_unified_v1` protocol accepts one positive finite stdout scalar and exact
`performance_metric={name,unit,workload,better}` metadata.

Base and candidate use the same worker, GPU, image, command recipe and workloads.
The candidate must pass correctness before a valid performance conclusion.
Original warmups/repeats stay in the frozen recipe. For each workload:

* `lower`: `100 * (base - candidate) / base`
* `higher`: `100 * (candidate - base) / base`

The task score is the arithmetic mean of these percentages; a mean strictly above
zero is improved. This does not claim statistical significance. Units, duplicate
metrics, non-finite/nonpositive values, missing metrics, skipped NVBench states,
failed commands and mismatched workloads are rejected. Throughput reciprocal to
latency is not double-counted: DeepSpeed #8732 uses latency, xformers #299 uses only
`xf_fw_ms`, cuda-python #709 uses the two medians, and NVBench selects GPU time for
the requested axes. CUTLASS uses Runtime from matching profiler CSV rows.

All 39 parser fixtures label synthetic values explicitly and identify their
format source. They are parser regressions, not replay measurements or historical
baselines. Baselines are always measured during evaluation and hidden from solvers.

## Scoring, history and repairs

Reports retain `raw_category`, `reason`, `scored_outcome`, `assumed_incorrect`,
test execution coverage, evaluated accuracy/coverage, current-input coverage and
the scoring denominator. Default policy excludes evidenced data errors, leaves
API rate-limit cases unresolved even after retries exhaust, and counts other
nonpasses as incorrect by convention. Infrastructure failures do not become proven
model failures. `nonpass_as_incorrect=false` leaves them unscored.

Explicit `history` entries require task ID, result path/hash and dataset path/hash.
Imports retain raw results, original source hashes and changed-field lists. Their
current-input execution coverage is zero even when an old result is retained.
Historical LLM performance judgments require numeric revalidation. No implicit
search of old workspaces or best-of selection occurs.

Exclusions require `task_type`, `instance_id`, exact `dataset_sha256`, `reason`,
`evidence` and `evidence_sha256`. They never apply automatically to other data.

Environment repairs are a list of versioned objects with `task_type`,
`instance_id`, `record_sha256`, `version`, `evidence`, `evidence_sha256`, and optional
`python`, `setup_command_overrides`, `test_command_overrides` or
`defer_setup_command_sha256`. Overrides map exact original command SHA256 to an
environment-corrected command. Review them as trusted evaluator code; hashes
establish provenance, not semantic equivalence. Preserve workloads and assertions.
Repairs are frozen in a new run rather than silently applied to historical output.

## Offline validation

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q
uv build
```

The suite includes master collection tests, migrated generation/isolation/classifier
tests, four CPU execution chains, freezing/tamper checks, ambiguous receipts,
cooldown/retry policy, immutable candidates, historical imports and 39 parser
fixtures. CI runs without private datasets, GPU access or model service calls.

# External evaluation adapter contract

Set `backend="external"` and `[external].script` to an operator-maintained Python
file outside the repository. Optional `sha256` verifies the source before copying
it to `run/external/adapter.py`; the frozen manifest always includes its hash.
Scheduler settings may live in `[external.options]` in the private campaign
configuration. Use environment variables or credential files for secrets.

The controller invokes the frozen script with its Python interpreter:

```text
adapter.py submit --run /absolute/run --attempt /absolute/attempt
adapter.py poll   --run /absolute/run --attempt /absolute/attempt
```

`submit` runs exactly once per durable `intent.json`. It must launch the already
written `attempt/worker.sh` with Bash, one compatible GPU and the task's
`runtime_image`. Mount the run read/write at the same path on the worker; mount
the configured credential file read-only if needed. The script bootstraps the
pinned tools and dependencies and invokes the public campaign worker.

`poll` must recover an ambiguous submission from the intent's unique `caption`,
or query a known receipt's job ID. It must never submit work. Keep pagination and
lookup failures explicit. An absent job in a listing is not proof of completion.

Print exactly one JSON object to stdout and exit zero:

```json
{"job_id":"scheduler-job-123","status":"RUNNING"}
```

Statuses are `PENDING`, `RUNNING`, `UNKNOWN`, or a terminal status from
`evaluate.campaign.TERMINAL`. Terminal responses require explicit terminal proof:

```json
{"job_id":"scheduler-job-123","status":"TERMINATED","terminal":true,"exit_code":0}
```

The adapter must obtain this proof from the scheduler's job detail API. Do not
infer it from a timeout, a numeric code without documented meaning, or a missing
list entry. Optional `completed_time` may be included. The public adapter retains
only normalized receipt fields; never print access tokens or raw service payloads.
Write useful, sanitized diagnostics to the attempt directory. Exceptions, malformed
responses and missing terminal proof produce `UNKNOWN`, retaining the slot.

The worker writes `result.json` and `exit.json`. A result file alone does not free
an external slot while the scheduler still reports an active job. If a confirmed
terminal job has no result, the controller records an infrastructure failure and
applies its bounded retry policy. Model test failures are not automatically retried.

The same interface supports Slurm, Kubernetes and hosted job services. Their
deployment-specific adapters, URLs, queue names, images and account identifiers
belong in local configuration, not in this repository.

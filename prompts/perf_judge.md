You are judging whether a code patch improved the performance of a GPU software workload.

You are given the same benchmark commands run twice in the same environment: once on the base
commit (BASELINE) and once after applying the patch (PATCHED). Correctness tests already passed on
the patched code. Compare only what the benchmark output actually shows.

Rules:
- Identify the primary metric the benchmark reports (latency, throughput, memory, time, ...) and
  whether lower or higher is better for it.
- Extract one representative baseline value and one patched value for that metric. If several runs
  or sizes are reported, use the aggregate the benchmark itself emphasizes, or the median.
- `delta_pct` is the relative change in the "better" direction: positive means the patch is faster /
  leaner, negative means it regressed. Use `(base - patched) / base * 100` when lower is better and
  `(patched - base) / base * 100` when higher is better.
- `improved` requires a consistent improvement larger than the run-to-run noise visible in the logs
  (treat changes under ~3% as noise unless the output shows tight variance).
- `no_improvement` when the metric is flat or worse.
- `unclear` when the output is missing, unparsable, not comparable between the two runs, or too noisy.
- Do not speculate about performance from the patch text alone.

Respond with exactly one JSON object and nothing else:

```json
{
  "verdict": "improved | no_improvement | unclear",
  "confidence": "high | medium | low",
  "base_value": 0.0,
  "patched_value": 0.0,
  "metric_unit": "ms",
  "delta_pct": 0.0,
  "reasoning": "one or two sentences grounded in the logged numbers",
  "data_quality": "good | noisy | insufficient"
}
```

# Licensing boundaries and third-party material

## Original work

The root MIT license covers original framework code, documentation, dataset
organization, annotations and problem descriptions only to the extent the
contributors own or may license those rights. Copyrighted source material
embedded in a description remains subject to its original terms.

## Benchmark source material

Benchmark records can contain upstream code patches, test patches, issue and pull
request text, and references to third-party projects. Those materials retain
their original licenses, copyright notices and attribution requirements. The
MIT license here does not replace them or grant rights the contributors do not
hold. When exporting records, preserve the source repository and commit/PR
identifiers, existing notices, and the applicable upstream license alongside the
materials. Check the license at the source revision, not only the current branch.

This code release does not bundle the benchmark dataset or upstream repository
checkouts. The Hugging Face dataset is a separate distribution; its existing
license metadata is not changed by this code release.

## Files shipped here

- `tests/fixtures/performance/`: 39 synthetic numeric-output fixtures, not GPU
  measurements or upstream source patches. Repository identifiers and output
  syntax identify the workloads being parsed.
- `src/swe_gpu_bench/evaluate/resources/performance-20261103.json`: parser recipes,
  workload selectors and content hashes; not upstream implementation source.
- Small patches and repository histories constructed in tests are test fixtures.
- Third-party project names identify benchmark sources and dependencies.
  Trademark rights remain with their owners; no endorsement is implied.

## Dependencies and downloaded tools

Dependencies and tools are installed or downloaded separately; their own licenses
continue to apply. Runtime Python dependencies are recorded in `uv.lock` and the
packaged evaluator requirements. The bootstrap additionally downloads pinned uv, Codex (including its bundled
helper tools) and OpenCode distributions.
Container base images and CUDA components retain their suppliers' terms. Retain
upstream notices when redistributing any of those artifacts.

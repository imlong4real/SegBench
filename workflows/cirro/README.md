# SegBench for Cirro

This adapter runs the frozen TSU-20 Xenium or kidney VisiumHD benchmark in
Cirro. Algorithm code remains in `src/segbench`; `workflows/cirro/` contains
only dataset staging, orchestration, immutable environment recipes, and tidy
report assembly.

## Release gate

Run `run_scope=smoke` for both datasets first. The full run is permitted only
after every method, the held-out evaluation, and the output-contract checks
complete and their canonical local/Cirro summaries agree. Parameters in
`frozen_manifest.json` are fixed before the held-out metrics are inspected.

The lung graph passes `reference_train` only to SPLIT/common-input preparation.
`evaluation_holdout` is delivered only to the final evaluator. The kidney
graph uses the same Space Ranger output for all applicable methods and the
harmonized reference (`obs.lineage`) only for evaluation.

## Outputs

Cirro publishes `benchmark_results/` containing each method's native and
standardized outputs, logs, resolved config receipt, runtime record and input
hashes. `benchmark_results/evaluation/` contains the six plot-ready TSVs,
comparison plots, RCTD/reference results, and `benchmark_manifest.json`.
Host RSS and GPU VRAM are recorded separately.

## Reproducibility

`nextflow.config` contains OCI digest pins; method source commits and frozen
parameters are recorded in `frozen_manifest.json`. The effective transcript
population, PMI path/SHA-256, reference split, method receipts, and resource
records are copied into every final result. No credential, token, dependency
installer, or mutable production image tag is present in this directory.

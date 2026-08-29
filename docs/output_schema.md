# Output schema

Every method writes the same directory layout, so downstream code never needs
a per-method special case.

```
<outdir>/
├── outputs/
│   ├── <method>_transcripts_standardized.parquet   the refined assignment
│   ├── <method>_cell_by_gene.h5ad                  cells x genes counts
│   └── <method>_raw_output/                        the tool's native output
├── benchmark_stats.json      ← the comparable performance statistics
├── benchmark_stats.tsv       ← the same, flattened to one row
├── schema_validation_report.json
├── runtime_memory.json
├── runtime_by_stage.tsv
├── config_receipt.json       exact args, input hashes, versions, git commit
├── run_summary.md            human-readable summary
├── external_time.txt         /usr/bin/time capture for the tool proper
└── run.log                   full log, including the tool's stdout/stderr
```

## 1. The transcript contract

`outputs/<method>_transcripts_standardized.parquet` — one row per transcript.

| column | type | required | meaning |
|---|---|:--:|---|
| `x`, `y` | float32 | ✓ | spatial coordinates, input units |
| `feature_name` | str | ✓ | gene / target name |
| `cell_id` | str | ✓ | assigned cell, or the literal `UNASSIGNED` |
| `method` | str | ✓ | method label |
| `z` | float32 | | axial coordinate |
| `transcript_id` | int/str | | stable id, for joining back to the input |
| `qv` | float32 | | vendor quality value |
| `overlaps_nucleus` | uint8 | | 1 if the molecule falls in a nucleus |
| `original_cell_id` | str | | the pre-refinement assignment |
| `assignment_confidence` | float32 | | method-specific score |
| `cleaned_status` | str | | `retained` / `cleaned_to_unassigned` |

Unassigned molecules are normalised to the single token `UNASSIGNED`; the
per-method spellings (`""`, `NA`, `0`, `-1`, `background`) are all collapsed
into it, so `frac_assigned` is comparable across tools.

### Two deliberate exceptions

**SPLIT** does not emit this table. `SPLIT::purify` returns a cell-by-gene
matrix of *fractional expected counts*, so which individual molecule was
removed is not recoverable. It is scored at cell level instead, and writes
`outputs/split_pruning_summary.json` with count-level pruning estimates.
`benchmark_stats.json` records `qc.transcript_level = false`.

**Bin2Cell** emits `outputs/bin2cell_bin_assignments.parquet`, where each row
is a 2 µm *bin* rather than a transcript — Visium HD has no molecule
resolution. `feature_name` is the placeholder `__bin__` because a bin carries a
whole expression vector. `entity_kind` is `bin` in the stats file, and
"assigned" means the bin fell inside a called cell.

## 2. The statistics contract

`benchmark_stats.json` is the only file the aggregator reads.

```jsonc
{
  "schema_version": "1.0",
  "method": "proseg", "modality": "imaging", "entity_kind": "cell",
  "status": "ok", "sample_name": "TSU20", "dataset": "tsu20_xenium",

  "runtime": {
    "total_seconds": 1.91,        // whole wrapper, including format conversion
    "method_seconds": 0.71,       // the external tool ONLY  <- compare this
    "by_stage_seconds": { "load_inputs": 0.10, "run_method": 0.71, ... }
  },
  "memory": {
    "peak_rss_gb": 0.036,         // best available estimate
    "method_peak_rss_gb": 0.036,  // the tool only, via /usr/bin/time
    "inprocess_peak_rss_gb": 0.19,
    "source": "external_time"     // or "psutil_inprocess"
  },
  "entities": {
    "entity_kind": "cell", "n_entities": 282, "n_genes": 60,
    "median_transcripts_per_entity": 45.0
  },
  "transcripts": {
    "n_total": 20000, "n_assigned": 16159, "n_unassigned": 3841,
    "frac_assigned": 0.808, "n_input": 20000, "delta_vs_input": 0
  },
  "qc": { /* method-relevant, see below */ },
  "provenance": { "command": "...", "hostname": "...", "method_version": "...",
                  "outputs": [...], "generated_utc": "..." }
}
```

### Which numbers to compare

- **Runtime** — use `runtime.method_seconds`, not `total_seconds`. The total
  includes our own parquet↔CSV shims, which differ per method and would
  penalise tools that need more format conversion.
- **Memory** — use `memory.method_peak_rss_gb`. `source` tells you whether it
  came from `/usr/bin/time` around the real tool (`external_time`, trustworthy)
  or from sampling our own process (`psutil_inprocess`, an underestimate for
  subprocess-based tools). Always check `source` before comparing.
- **Cells** — `entities.n_entities`. `entity_kind` says whether those are cells
  or bins; never compare across different `entity_kind` values.
- **Assignment** — `transcripts.frac_assigned`. `delta_vs_input` catches methods
  that emit more rows than they consumed (proseg can, when it splits molecules
  across voxel layers).

### Method-relevant QC (`qc`)

| method | notable keys |
|---|---|
| baysor | `scale`, `min_molecules_per_cell`, `runtime_valid_for_benchmark` |
| proseg | `n_proseg_cells`, `voxel_layers` |
| segger | `max_epochs`, `accelerator`, `n_bound_transcripts`, `mean_assignment_score` |
| split | `transcript_level: false`, `estimated_removed_counts`, `frac_counts_removed` |
| celladmix | `n_removed_transcripts`, `n_retained_transcripts`, `num_factors` |
| tracer | `platform`, `pmi_threshold`, `tau` |
| bin2cell | `n_bins_input`, `n_bins_assigned_to_cell`, `median_bins_per_cell`, `mpp` |

Every scalar `qc` key is flattened into the summary table as `qc_<key>`, so a
new method's QC appears in the aggregate without touching the aggregator.

## 3. Aggregation

```bash
segbench collect benchmark_output/tsu20 --out summary.tsv
```

writes one row per run with the shared columns plus every `qc_*` column found.
`segbench suite` runs this automatically and leaves
`benchmark_summary.tsv` next to the runs.

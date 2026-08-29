# segmentation_benchmark_pipeline

A reproducible benchmarking suite for spatial-transcriptomics **segmentation**
and **transcript-refinement** methods.

Every method runs through one interface and produces one output contract, so
runtime, peak memory, cell counts and transcript assignment are directly
comparable across tools:

```bash
segbench run proseg --dataset tsu20_xenium --outdir benchmark_output/tsu20/proseg
segbench suite imaging_full --dataset tsu20_xenium
```

| modality | methods |
|---|---|
| **Imaging** (molecule-resolved: Xenium, CosMx, MERFISH) | Baysor, ProSeg, Segger, SPLIT, CellAdmix, TRACER |
| **Sequencing** (array / binned: Visium HD) | Bin2Cell, TRACER |

---

## Install

The suite itself is pure Python and needs no installation — `bin/segbench`
puts `src/` on `PYTHONPATH` and runs.

```bash
git clone https://github.com/imlong4real/segmentation_benchmark_pipeline.git
cd segmentation_benchmark_pipeline

# Core dependencies (Python >= 3.10)
pip install numpy pandas pyarrow psutil pyyaml anndata scipy scanpy

# Check the wiring — no external tool required
./bin/segbench selftest

# Optional: put segbench on PATH
export PATH="$PWD/bin:$PATH"
```

Point `SEGBENCH_DATA` at wherever your datasets live so configs stay portable:

```bash
export SEGBENCH_DATA=/scratch/$USER/spatial_data
```

### Method environments

Methods need genuinely different runtimes (a Julia binary, a Rust binary, R, a
CUDA Python). `configs/environments.yaml` declares one per method and is the
only file naming machine-specific paths; everything resolves through `${VAR}`:

```bash
cp configs/environments.local.example.sh configs/environments.local.sh
$EDITOR configs/environments.local.sh     # set SEGBENCH_ENV_ROOT, TRACER_VENV
source configs/environments.local.sh
```

`segbench run <method>` then **re-execs into that method's own interpreter**,
so nothing needs activating by hand. Install only the methods you want, then
ask which are ready:

```bash
segbench doctor
```

```
baysor       NOT READY
             missing binary:baysor
             needs   baysor >= 0.7 (Julia binary)
proseg       READY
```

| method | install |
|---|---|
| baysor | `julia -e 'using Pkg; Pkg.add(PackageSpec(url="https://github.com/kharchenkolab/Baysor.git", rev="v0.7.0")); Pkg.build("Baysor")'` |
| proseg | `cargo install proseg --version 3.0.10 --locked` |
| segger | container (`reproducibility/segger/segger.def`) or pip + torch-geometric with CUDA |
| split, celladmix | R ≥ 4.3, then `Rscript workflow/scripts/_count_correction/install_split_celladmix_deps.R` |
| tracer | the TRACER python package |
| bin2cell | `pip install bin2cell` (+ `stardist`/`tensorflow`, or pass `--labels-npz`) |

Container definitions for a reproducible Linux/Apptainer setup live in
`reproducibility/*.def`.

---

## Input schema

Imaging methods consume a **standardized transcripts parquet** — one row per
molecule:

| column | type | required | meaning |
|---|---|:--:|---|
| `x`, `y` | float | ✓ | spatial coordinates |
| `feature_name` | str | ✓ | gene / target name |
| `cell_id` | str | ✓ | current assignment, or `UNASSIGNED` |
| `z` | float | | axial coordinate |
| `transcript_id` | int/str | | stable id |
| `qv` | float | | vendor quality value |
| `overlaps_nucleus` | 0/1 | | molecule falls inside a nucleus |

`x_location`/`y_location`/`gene` are accepted and renamed automatically.

Build one from a raw Xenium bundle:

```bash
python workflow/scripts/_benchmark/standardize_method_output.py \
  --method xenium_default --xenium-dir <bundle> --out-dir <dir>
```

**Segger** additionally wants the raw Xenium bundle (`--xenium-dir`) for real
nucleus boundaries. **Bin2Cell** takes a Visium HD 2 µm bin matrix
(`--input-h5ad`) plus the H&E image instead of a transcript table. Details in
[docs/methods.md](docs/methods.md).

---

## Running

### One method

```bash
segbench run baysor \
  --transcripts data/TSU-20/filtered_df_standardized.parquet \
  --outdir benchmark_output/tsu20/baysor \
  --sample-name TSU20 --threads 8 --overwrite
```

Or drive it from a dataset config so the paths live in one place:

```bash
segbench run baysor --dataset tsu20_xenium --outdir benchmark_output/tsu20/baysor
```

Per-method commands and flags: [docs/methods.md](docs/methods.md), or
`segbench run <method> --help`.

### The whole suite

```bash
segbench suite imaging_full     --dataset tsu20_xenium
segbench suite sequencing_full  --dataset visium_hd_demo
segbench suite smoke            --dataset tsu20_xenium   # 50k-transcript subsample
```

Methods whose tools are not installed are **skipped**, not failed, and a
failing method does not abort the rest — a partial benchmark is more useful
than none. The suite writes `benchmark_summary.tsv` and `suite_result.json`
into the output root.

### Scoring and comparing

`collect` stacks the per-run stats; `evaluate` additionally computes the
cross-method biological metrics and draws the comparison figures:

```bash
segbench evaluate /scratch4/$USER/segbench_runs/nsclc_xenium/methods \
  --dataset nsclc_xenium --min-reference-cells 50
```

writes `comparison_table.csv`, `comparison.{png,pdf}`, `cost_scatter.{png,pdf}`
and `comparison.md`. Metric definitions are held identical across methods by
routing every method through the same code (RCTD via `run_rctd.R`, Kendall and
marker log2FC via `get_metric.py`). Quantities that genuinely differ in meaning
are reported as `n/a` with a reason rather than coerced — see
[docs/audit.md](docs/audit.md#5-metric-comparability).

`--min-reference-cells` drops sparsely-represented reference cell types from
RCTD and the marker/Kendall metrics so rare populations cannot dominate a
median.

### Aggregating

```bash
segbench collect benchmark_output/tsu20 --out summary.tsv
```

```
method status  total_seconds  method_seconds  peak_rss_gb  n_entities  n_transcripts_assigned  frac_assigned
proseg     ok       1.906432        0.708646     0.035896         282                   16159        0.80795
baysor     ok      41.203100       38.902000     2.140000         311                   17402        0.87010
```

### Other commands

```bash
segbench list [-v]        # methods, and whether they can run here
segbench doctor [method]  # resolved env paths + dependency report (--json)
segbench selftest         # dependency-free wiring check
```

### On a cluster

```bash
sbatch --export=ALL,METHOD=proseg,DATASET=nsclc_xenium,SEGBENCH_REPO=$PWD \
       scripts/slurm/run_method.sbatch
```

---

## Configuration

Four layers, lowest precedence first:

1. `configs/methods/<method>.yaml` — method defaults
2. `configs/datasets/<name>.yaml` — sample paths (`--dataset`)
3. a user config (`--config`)
4. explicit CLI flags

Paths may be repo-relative or use `${SEGBENCH_DATA}` / any `${ENV_VAR}`, which
is how machine-specific locations stay out of tracked files.

```yaml
# configs/datasets/tsu20_xenium.yaml
dataset: {name: tsu20_xenium, platform: xenium, modality: imaging}
sample_name: TSU20
inputs:
  transcripts:    ${SEGBENCH_DATA}/lung_cancer_xenium_10x/TSU-20/filtered_df_standardized.parquet
  xenium_dir:     ${SEGBENCH_DATA}/lung_cancer_xenium_10x/TSU-20
  reference_h5ad: ${SEGBENCH_DATA}/lung_cancer_scrna_10x/lung_cancer_scrna_split.h5ad
```

The dataset configs in this repo are **templates** — copy one and point it at
your data.

---

## Output schema

Every run writes the same layout:

```
<outdir>/
├── outputs/
│   ├── <method>_transcripts_standardized.parquet   the refined assignment
│   ├── <method>_cell_by_gene.h5ad                  cells x genes counts
│   └── <method>_raw_output/                        the tool's native output
├── benchmark_stats.json      comparable performance statistics
├── benchmark_stats.tsv       the same, flattened to one row
├── schema_validation_report.json
├── runtime_memory.json  runtime_by_stage.tsv  external_time.txt
├── config_receipt.json       exact args, input hashes, versions, git commit
├── run_summary.md
└── run.log
```

`benchmark_stats.json` carries runtime (total **and** the external tool alone),
peak RSS (with `source` recording how it was measured), entity counts,
assigned/unassigned transcripts, and method-relevant QC.

**Compare `runtime.method_seconds`, not `total_seconds`** — the total includes
format conversion that differs per method. Full field-by-field reference:
[docs/output_schema.md](docs/output_schema.md).

Two methods deviate deliberately: **SPLIT** is cell-level (fractional expected
counts make per-molecule attribution unrecoverable) and **Bin2Cell** reports
2 µm *bins* rather than transcripts. Both are flagged in the stats via
`entity_kind` and `qc.transcript_level`.

---

## Adding another method

Two files: a wrapper module and one registry entry. The CLI, suite runner,
`doctor` and aggregator all read the registry, so nothing else changes.

See [docs/adding_a_method.md](docs/adding_a_method.md) for the skeleton and the
five rules that keep a new method comparable with the existing ones.

---

## Repository layout

```
bin/segbench                 one-command entry point (no install needed)
src/segbench/
├── cli.py                   run / suite / list / doctor / collect / selftest
├── registry.py              the method registry — the only place methods are declared
├── common.py                shared runner library: staging, timing, peak RSS,
│                            standardization, schema validation, provenance
├── stats.py                 the benchmark_stats.json contract + aggregation
├── config.py                layered YAML config + ${ENV} path resolution
└── methods/                 one wrapper per method
    ├── baysor.py  proseg.py  segger.py  split.py
    ├── celladmix.py  tracer.py  bin2cell.py
    └── _base.py             shared CLI flags + config resolution

configs/
├── methods/                 per-method defaults
├── datasets/                per-sample input paths (templates)
└── suites/                  which methods a suite runs

docs/                        method reference, output schema, HPC notes
tests/                       dependency-free smoke test + fixture generator
reproducibility/             Apptainer/Singularity defs, conda env, renv lock
scripts/slurm/               SLURM job scripts for GPU Segger runs
workflow/                    Snakemake pipeline + metrics/figure scripts
├── Snakefile_benchmark      benchmark DAG
├── rules/                   Snakemake rules
└── scripts/
    ├── get_metric.py                 transcript-level benchmark metrics
    ├── get_cell_level_metric.py      cell-level metrics (use for SPLIT)
    ├── _benchmark/                   standardization + metric collection
    ├── _segmentation/                per-tool output adapters
    └── _v2_*, _v3_*, _roi_*          publication figure / ROI analysis scripts
benchmark_output/            run outputs (gitignored)
```

The `workflow/` Snakemake layer is the older pipeline and remains functional;
`segbench` is the supported entry point for benchmarking. `get_metric.py`
computes the downstream biological metrics (label transfer, reference
consistency, marker specificity, cPMI coherence) on top of any method's
standardized parquet.

---

## Reproducibility

Every run records the exact command, resolved arguments, input SHA-1 hashes,
tool and package versions, hostname and git commit in `config_receipt.json`.
Seeds are explicit (`--seed`, default 1). Container definitions for a pinned
Linux environment are in `reproducibility/`.

Known limitation: `memory.source` is `psutil_inprocess` rather than
`external_time` for in-process (pure Python) methods, where there is no
subprocess for `/usr/bin/time` to measure. Check that field before comparing
memory across methods.

# Run-tools plan for TSU-20 (non-TRACER)

This document is the honest mapping between what the original SPLIT Xenium
pipeline expects, what the host machine can and cannot do, and what command
will be used per tool. It is the source of truth referenced by
`workflow/configs/tsu20_tools.yml` and by `README.md` §"Per-tool status".

> **TRACER is out of scope for this run.** All `tracer_*` rules and config
> keys are disabled in `tsu20_tools.yml`.

## 0. Host environment (probed)

| Tool / runtime | Status | Notes |
|----------------|--------|-------|
| OS / arch | macOS 14, **arm64 (Apple Silicon)** | Major constraint: most vendor Linux x86_64 binaries are blocked. |
| `python3` | 3.13.5 (anaconda3) | OK. |
| `snakemake` | 9.20.0 (pip, in `~/.local/bin`) | Installed in Phase 1. |
| `R` | **NOT INSTALLED** | Required by SPLIT and cellAdmix. Installable via conda (`r-base=4.4`) — adds ~600 MB. |
| `mamba` | not installed | Optional. |
| `conda` | 25.7.0 at `~/anaconda3/bin/conda` | Usable directly with absolute path. |
| `singularity` / `apptainer` | **NOT INSTALLED** | Neither is available natively on macOS. The SPLIT template assumes Singularity. |
| `docker` | not installed | Cannot substitute for Singularity here. |
| `julia` | 1.10.11 | OK — Baysor 0.7 is Julia-based. |
| `cargo` | 1.94.0 | OK — Proseg can be built from source. |
| `nvidia-smi` / GPU | none | Segger requires CUDA; not runnable on this host. |

## 1. SPLIT template version pins (READ-ONLY reference)

From `README_SPLIT_TEMPLATE.md`:

| Tool | Version expected by template | Container `.def` |
|------|------------------------------|------------------|
| Xenium Ranger | 4.0.0 | `reproducibility/10x.def` |
| Baysor | 0.7.0 | `reproducibility/baysor.def` |
| Proseg | 3.0.10 | `reproducibility/proseg.def` |
| Segger | fork `senbaikang/segger_dev@96e531d` | `reproducibility/segger/segger.def` (sub-dir, may not exist locally) |
| R | 4.4.2 + renv | `reproducibility/r/r.def` |
| Snakemake | ≥ 9.0 | conda env via `reproducibility/environment.yml` |

The original rules (under `workflow/rules/_segmentation/*.smk` and
`workflow/rules/_count_correction/*.smk`) drive these versions via the
`config["containers"]["..."]` mapping in `config_split_original/config.yml`.

## 2. Per-tool execution plan

### 2.1 Xenium default

- **What "running" means**: the TSU-20 bundle already contains the
  output of Xenium's on-instrument segmentation. There is no resegmentation
  to perform unless we run `xeniumranger resegment` (which is the SPLIT
  `run10x` rule and requires Xenium Ranger). For benchmarking purposes we
  consume the existing `transcripts.parquet` / `cells.parquet` directly.
- **Execution path**: `workflow/scripts/_benchmark/standardize_method_output.py --method xenium_default`.
- **Status**: ✅ runnable natively on macOS (already exercised end-to-end in the previous session).
- **Output**: `results/tsu20_tools/xenium_default/standardized/{transcripts,cells,cell_by_gene*,method_info}`.

### 2.2 Baysor

- **Template plan**: rule `runBaysor` runs `baysor run -c {toml} {tx.parquet} :cell_id` inside the Baysor 0.7 container, then `adjustBaysorResults` post-processes the CSV, then `normaliseBaysor` re-imports via `xeniumranger import-segmentation`.
- **macOS plan**: install Baysor 0.7 natively via Julia (Julia 1.10 is present). Run `baysor run` directly against `dataset/lung_cancer_xenium_10x/TSU-20/transcripts.parquet`. Skip `xeniumranger import-segmentation` (impossible without Xenium Ranger); instead feed the resulting `segmentation.csv` straight into `standardize_method_output.py --method baysor` so the benchmark contract is still produced.
- **Status**: 🟡 **expected runnable** (subject to a slow Julia install of the Baysor package). Worst case: install succeeds but downstream `xeniumranger`-dependent rules in the *original* SPLIT pipeline cannot complete on this host.
- **Output**: `results/tsu20_tools/baysor/raw/segmentation.csv` (real Baysor output) → `results/tsu20_tools/baysor/standardized/...`.

### 2.3 Proseg

- **Template plan**: rule `runProseg` invokes `proseg --xenium {bundle}`; `runProseg2Baysor` converts; `normaliseProseg` requires Xenium Ranger.
- **macOS plan**: `cargo install proseg --version 3.0.10` (or build from source). Run `proseg --xenium dataset/lung_cancer_xenium_10x/TSU-20` directly. Standardize the resulting `transcript-metadata.csv.gz`.
- **Status**: 🟡 **expected runnable** (depends on Rust crate availability and macOS arm64 compatibility).
- **Output**: `results/tsu20_tools/proseg/raw/...` → `results/tsu20_tools/proseg/standardized/...`.

### 2.4 Segger

- **Template plan**: requires CUDA, the `python_cuda` Singularity container, a GPU, and tile-based training.
- **macOS plan**: **NOT RUNNABLE on this host.** No NVIDIA GPU, no CUDA, no Singularity.
- **Status**: ❌ blocked. Documented explicitly in `README.md` and `tsu20_tools.yml`.

### 2.5 ovrlpy

- **Template plan**: rule `runOvrlpy` runs `workflow/scripts/_count_correction/ovrlpy_sample.py` inside `python_cuda`.
- **macOS plan**: ovrlpy is pip-installable Python — no CUDA required. Use the existing benchmark wrapper script and a per-tool venv to keep deps isolated.
- **Status**: 🟢 expected runnable. Risk: ovrlpy may have pinned scientific-Python deps that fight Python 3.13; in that case we install into an explicit Python 3.11 venv.
- **Output**: `results/tsu20_tools/ovrlpy/{signal_integrity.parquet, signal_strength.parquet, transcript_info.parquet, method_info.json}`.

### 2.6 cellAdmix

- **Template plan**: not present in the original SPLIT pipeline — added by this benchmark. Requires R, `kharchenkolab/cellAdmix`, and cluster labels.
- **macOS plan**: install `r-base=4.4` + `r-arrow` + `r-matrix` + `r-jsonlite` + `r-optparse` + `r-devtools` via conda; install cellAdmix from GitHub once into the conda env. Use the Xenium graphclust labels at `dataset/.../analysis/clustering/gene_expression_graphclust/clusters.csv` as **temporary cluster labels** (recorded in `method_info.json`).
- **Status**: 🟡 conditional on the conda install completing. If it fails or takes too long in this session, the wrapper will fail loudly with the exact missing package.
- **Output**: `results/tsu20_tools/celladmix_xenium_default/{corrected_counts.mtx, cell_metadata.parquet, method_info.json}` (+ optional `admixture_metrics.csv`).

### 2.7 SPLIT

- **Template plan**: rule `runSplitFullyPurified` consumes (a) a Seurat object with a counts assay, and (b) a **post-processed RCTD doublet-mode object** (`post_processed_output.rds`) produced by the original pipeline's `cell_type_annotation/.../rctd_*` chain. The R wrapper calls `SPLIT::purify(counts, rctd, DO_purify_singlets = TRUE)`.
- **macOS plan**: SPLIT itself is a fast R wrapper; the heavy dependency is `spacexr` (RCTD). To produce a real RCTD output we would need (i) the Seurat object built by the original pipeline (`std_seurat_analysis/.../preprocessed_seurat.rds`), and (ii) a reference single-cell dataset. The reference is not provided in this repo. Without it RCTD cannot run, and without RCTD SPLIT cannot run.
- **Status**: ❌ blocked unless either the RCTD object is provided externally or a single-cell reference is supplied. Per the user's instruction we **do not stub**: the wrapper script will exit non-zero with a clear, actionable error naming:
  - the config key (`split.rctd_post_processed_rds`),
  - the expected file (`post_processed_output.rds`, RCTD doublet-mode),
  - the rule that consumes it (`runSplitFullyPurified` in `workflow/rules/_count_correction/_split/split_fully_purified.smk`),
  - how to generate it from the original SPLIT pipeline if a reference is available.

### 2.8 Original SPLIT Snakefile vs. benchmark Snakefile

Because the original `workflow/Snakefile` requires:

- a populated `config_split_original/experiments.yml` (multi-condition / gene panel / donor / sample wildcards),
- Singularity (for every container-backed rule),
- Xenium Ranger 4.0.0 (`run10x`, `normaliseBaysor`, `normaliseProseg`, `normaliseSegger`),
- the full Seurat → RCTD → SPLIT chain (for `split_*` correction).

…it **cannot be executed end-to-end on this host**. We use
`workflow/Snakefile_benchmark` as the active driver, which reuses the
existing SPLIT-derived scripts (Baysor adjustments, Proseg conversions,
ovrlpy script) and standardizes outputs into the benchmark contract.
This was the explicit guidance in the previous task ("Prefer adding
benchmark-level wrapper rules over modifying original rules heavily").

## 3. Stub policy

Per the user's instruction:

- **Default**: every wrapper script runs the real tool. If the tool or a
  required dependency is missing, the script exits non-zero with an
  actionable message naming the file/package/config key.
- **Opt-in only**: a stub path remains available via `--allow-stub` for
  the smoke-test config (`benchmark_lung_tiny.yml`). It is **not enabled**
  in `tsu20_tools.yml`.
- `method_info.json` carries `extra.stub = true` when stubs run, so
  validation can refuse to declare success.

## 4. Order of operations in this session

1. Install Snakemake (done, 9.20.0).
2. Standardize Xenium default output (no install required).
3. Install ovrlpy in a Python 3.11 venv (lightweight) and run it.
4. Install Baysor 0.7 (Julia) → run it on TSU-20 → standardize.
5. Install Proseg 3.0.10 (`cargo install`) → run it on TSU-20 → standardize.
6. Conda-install `r-base` + cellAdmix deps → run cellAdmix.
7. SPLIT and Segger: produce documented failure outputs (no stubs).
8. Validate everything with `validate_tool_outputs.py`.
9. Update README with what worked / what didn't.

Each step is committed independently so a partial-session result is still
useful.

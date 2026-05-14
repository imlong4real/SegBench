# Segmentation benchmark pipeline (TSU-20 non-TRACER run)

This repository runs spatial-transcriptomics segmentation and
count-correction tools on a single Xenium sample
(`dataset/lung_cancer_xenium_10x/TSU-20`).

> **TRACER is intentionally disabled for this run.** No `tracer_*` rules
> are scheduled and no TRACER stub outputs are produced. Every wrapper
> either runs the real tool or fails loudly with an actionable error.
> A method is marked `PASS` only when non-empty real outputs AND a
> non-stub `method_info.json` exist.

---

## Current Status

Authoritative validation table:
`results/tsu20_tools/summary/tool_output_validation.csv`
(regenerate with `workflow/scripts/_benchmark/validate_real_tool_outputs.py`).

| Method | Status | Notes |
|---|:---:|---|
| `xenium_default` | PASS | 1,436,900 transcripts, 58,648 cells, 794,804 cell×gene entries (Xenium bundle standardized) |
| `baysor` | PASS | Baysor 0.7.0 native install (Julia 1.10); 1,436,887 transcripts, 39,833 cells, 701,130 entries |
| `proseg` | PASS | Proseg 3.0.10 native install (cargo); 1,551,332 transcripts, 53,340 cells, 662,588 entries |
| `ovrlpy_xenium_default` | PASS | ovrlpy 1.1.0; signal_integrity / signal_strength / per-cell VSI / pseudocell summary parquets |
| `split_xenium_default` | PASS | spacexr 2.2.1 RCTD doublet mode + SPLIT 0.1.3 `purify(DO_purify_singlets = TRUE)`; scRNA reference = `dataset/lung_cancer_scrna_10/lung_cancer_scrna_split.h5ad`, 35,954 ref cells, 58,648 spatial cells, 292 shared genes |
| `celladmix_xenium_default` | DEBUG_PASS | cellAdmix 0.1.0 molecule-level workflow (samp_ct_equal → run_knn_nmf → run_crf_all → get_mol_knn → run_bridge_test → check_fp); 1,635,024 input transcripts → 1,519,449 retained, 57,136 cells, 22 temporary graphclust labels. Real run, biological labels not yet curated → `DEBUG_PASS` not `PASS`. |
| `segger` | IN PROGRESS | Container building on DSAI HPC cluster (nvl partition, H100). See [docs/hpc_segger_smoke.md](docs/hpc_segger_smoke.md). |
| `tracer_*` | DISABLED | Intentionally disabled in `workflow/configs/tsu20_real_tools.yml`. |

---

## Host & dependencies

Developed on **macOS 14, arm64 (Apple Silicon)** without Singularity/Apptainer
and without a GPU. The original SPLIT template (`reproducibility/*.def`)
assumes Linux + Singularity; the section below documents the native
equivalents used here.

```bash
# Snakemake (pip into anaconda3, lands in ~/.local/bin)
/Users/lyuan13/anaconda3/bin/pip install --user "snakemake>=9.0,<10"

# Proseg 3.0.10 (Rust)
cargo install proseg --version 3.0.10 --locked        # → ~/.cargo/bin/proseg

# Baysor 0.7.0 (Julia source — no macOS prebuilt binary)
julia -e 'using Pkg; Pkg.add(PackageSpec(url="https://github.com/kharchenkolab/Baysor.git", rev="v0.7.0")); Pkg.build("Baysor")'
# Produces an executable shim at ~/.julia/bin/baysor

# ovrlpy 1.1.0 in an isolated Python 3.11 env
/Users/lyuan13/anaconda3/bin/conda create -n tracer_benchmark_ovrlpy -y -c conda-forge python=3.11 numpy pandas pyarrow
conda run -n tracer_benchmark_ovrlpy pip install ovrlpy

# R 4.4 conda env with cairo/Seurat system libs (needed by cellAdmix/SPLIT)
/Users/lyuan13/anaconda3/bin/conda create -n tracer_benchmark_r -y -c conda-forge \
  r-base=4.4 r-matrix r-arrow r-jsonlite r-optparse r-remotes r-devtools \
  cairo pango freetype harfbuzz fribidi libtiff pkg-config r-cairo r-seurat
/Users/lyuan13/anaconda3/bin/conda install -n tracer_benchmark_r -y \
  -c conda-forge -c bioconda \
  bioconductor-biocparallel bioconductor-sparsematrixstats r-hdf5r hdf5

# spacexr (RCTD), SPLIT, cellAdmix from GitHub
conda run -n tracer_benchmark_r \
  Rscript workflow/scripts/_count_correction/install_split_celladmix_deps.R
```

The canonical (Linux + Singularity) reproducibility recipe lives in
`reproducibility/*.def`; on this macOS host we substitute native installs
+ conda envs as shown above.

---

## Quick start

```bash
# Validate inputs are present
ls dataset/lung_cancer_xenium_10x/TSU-20/transcripts.parquet
ls dataset/lung_cancer_scrna_10/lung_cancer_scrna_split.h5ad

# Snakemake dry-run for SPLIT + cellAdmix paths
~/.local/bin/snakemake -n \
  --configfile workflow/configs/tsu20_real_tools.yml \
  --cores 8

# Run validation (any tool, idempotent)
python workflow/scripts/_benchmark/validate_real_tool_outputs.py \
  --results results/tsu20_tools \
  --out results/tsu20_tools/summary/tool_output_validation.csv
```

---

## Per-tool tutorials

### 1. Xenium default (bundle standardization)

**Purpose.** Convert the existing Xenium bundle output into the
benchmark's standardized contract (`transcripts.parquet`, `cells.parquet`,
`cell_by_gene.mtx`, …). Xenium Ranger itself (`run10x` in the inherited
pipeline) is **not** run — Xenium Ranger is Linux-only and not available
here.

**Requirements.** Python with `pandas`, `pyarrow`, `scipy` (the anaconda3
base env on this host suffices).

**Input files**
- `dataset/lung_cancer_xenium_10x/TSU-20/transcripts.parquet`
- `dataset/lung_cancer_xenium_10x/TSU-20/cells.parquet`

**Command**

```bash
/Users/lyuan13/anaconda3/bin/python3 workflow/scripts/_benchmark/standardize_method_output.py \
  --method xenium_default \
  --xenium-dir dataset/lung_cancer_xenium_10x/TSU-20 \
  --out-dir results/tsu20_tools/standardized/xenium_default \
  --qv-threshold 30 --threads 8 \
  --log results/tsu20_tools/logs/standardize_xenium_default.log
```

**Outputs.** `results/tsu20_tools/standardized/xenium_default/` —
`transcripts.parquet`, `cells.parquet`, `cell_by_gene.mtx` (+ barcodes /
features TSV), `cell_metadata.parquet`, `method_info.json`.

**Known limitations.** Does not actually re-run Xenium segmentation; that
requires Xenium Ranger 4.0.0 from 10x (Linux only).

**Current TSU-20 status.** PASS, 1,436,900 transcripts, 58,648 cells.

---

### 2. Baysor 0.7.0

**Purpose.** Probabilistic transcript-to-cell segmentation that uses a
prior (the Xenium `cell_id` column) and refines it via Markov-random-field
clustering.

**Requirements.** Julia 1.10.x. SPLIT template's `reproducibility/baysor.def`
expects the Linux x86_64 prebuilt zip; on macOS arm64 we install from
source via the Julia `Pkg.add(rev="v0.7.0")` path.

**Input files**
- `dataset/lung_cancer_xenium_10x/TSU-20/transcripts.parquet`
- `workflow/configs/baysor_xenium.toml` (Baysor parameter file inherited
  from the SPLIT template)

**Commands**

```bash
mkdir -p results/tsu20_tools/baysor/raw && cd results/tsu20_tools/baysor/raw && \
  JULIA_NUM_THREADS=4 ~/.julia/bin/baysor run \
    -c $REPO/workflow/configs/baysor_xenium.toml \
    --polygon-format=GeometryCollection \
    $REPO/dataset/lung_cancer_xenium_10x/TSU-20/transcripts.parquet \
    :cell_id

# Standardize to benchmark contract:
/Users/lyuan13/anaconda3/bin/python3 workflow/scripts/_benchmark/standardize_method_output.py \
  --method baysor \
  --baysor-segmentation-csv results/tsu20_tools/baysor/raw/segmentation.csv \
  --xenium-dir dataset/lung_cancer_xenium_10x/TSU-20 \
  --out-dir results/tsu20_tools/standardized/baysor \
  --qv-threshold 30 --threads 8 \
  --log results/tsu20_tools/logs/standardize_baysor.log
```

**Outputs.** Raw: `results/tsu20_tools/baysor/raw/segmentation.csv`,
`segmentation_polygons_2d.json`, `segmentation_polygons_3d.json`,
`segmentation_log.log`, `segmentation_params.dump.toml`.
Standardized: `results/tsu20_tools/standardized/baysor/`.

**Known limitations.** The original SPLIT pipeline's `normaliseBaysor`
rule re-imports the Baysor output via `xeniumranger import-segmentation`;
that step requires Xenium Ranger and is skipped here. The benchmark
standardizer reads `segmentation.csv` directly and produces the contract
without that intermediate.

**Current TSU-20 status.** PASS. 1,436,887 transcripts assigned;
39,833 cells; 701,130 cell×gene entries.

---

### 3. Proseg 3.0.10

**Purpose.** Fast probabilistic segmentation that produces both raw and
expected (posterior-mean) counts, transcript-to-cell assignments, and
cell polygons.

**Requirements.** Rust (cargo) on the host, or the Linux Singularity
container `reproducibility/proseg.def`.

**Input files**
- `dataset/lung_cancer_xenium_10x/TSU-20/transcripts.parquet` (full Xenium
  transcripts; Proseg 3.0.10 with `--xenium` expects the *parquet file*
  path, not the bundle directory).

**Commands**

```bash
mkdir -p results/tsu20_tools/proseg/raw && cd results/tsu20_tools/proseg/raw && \
  ~/.cargo/bin/proseg --nthreads 4 \
    --output-counts counts.mtx.gz \
    --output-expected-counts expected-counts.mtx.gz \
    --output-cell-metadata cell-metadata.csv.gz \
    --output-transcript-metadata transcript-metadata.csv.gz \
    --output-gene-metadata gene-metadata.csv.gz \
    --output-cell-polygons cell-polygons.geojson.gz \
    --output-cell-polygon-layers cell-polygons-layers.geojson.gz \
    --xenium $REPO/dataset/lung_cancer_xenium_10x/TSU-20/transcripts.parquet

# Standardize:
/Users/lyuan13/anaconda3/bin/python3 workflow/scripts/_benchmark/standardize_method_output.py \
  --method proseg \
  --proseg-transcript-metadata results/tsu20_tools/proseg/raw/transcript-metadata.csv.gz \
  --out-dir results/tsu20_tools/standardized/proseg \
  --qv-threshold 30 --threads 8 \
  --log results/tsu20_tools/logs/standardize_proseg.log
```

**Outputs.** Raw: `counts.mtx.gz`, `expected-counts.mtx.gz`,
`cell-metadata.csv.gz`, `transcript-metadata.csv.gz`, `gene-metadata.csv.gz`,
`cell-polygons.geojson.gz`, `cell-polygons-layers.geojson.gz`,
`proseg-output.zarr/`. Standardized: `results/tsu20_tools/standardized/proseg/`.

**Known limitations.**
- The SPLIT template README says 3.0.10 but the upstream `proseg.def`
  pins `v3.1.0`; we follow the README and use 3.0.10.
- Unassigned transcripts are coded as the maximum integer cell-id in
  Proseg's output; the standardizer relabels them as `"UNASSIGNED"`
  (string) before writing parquet — a mixed-type column otherwise crashes
  pyarrow.
- Cross-check: Proseg "cell counts" differ from Xenium's because Proseg
  re-segments rather than refining the prior.

**Current TSU-20 status.** PASS. 1,551,332 transcripts, 53,340 cells,
662,588 entries.

---

### 4. ovrlpy 1.1.0

**Purpose.** Vertical signal-integrity QC. Identifies spatial regions
where transcripts likely belong to overlapping cells in 3D.

**Requirements.** Python 3.11 in an isolated env (the anaconda3 base
Python 3.13 has an anndata/dask/xarray circular import that ovrlpy 1.1
hits). ovrlpy is pip-installable; no GPU required.

**Input files.** Standardized transcripts (any segmentation method's
`transcripts.parquet`).

**Command**

```bash
conda run -n tracer_benchmark_ovrlpy python workflow/scripts/_benchmark/run_ovrlpy_benchmark.py \
  --standardized-dir results/tsu20_tools/standardized/xenium_default \
  --out-dir          results/tsu20_tools/ovrlpy_xenium_default \
  --cell-diameter 10 --n-expected-celltypes 30 \
  --log results/tsu20_tools/logs/ovrlpy_xenium_default.log
```

**Outputs.** `signal_integrity.parquet`, `signal_strength.parquet`,
`cell_signal_integrity.parquet`, `pseudocell_summary.parquet`,
`cell_id_code_map.parquet`, `transcript_info.parquet`, `method_info.json`.

**Known limitations.**
- ovrlpy 1.x renamed the high-level API; the original SPLIT pipeline
  script called `ovrlpy.run(df, cell_diameter, n_expected_celltypes)`
  which no longer exists. The wrapper at
  `workflow/scripts/_benchmark/run_ovrlpy_benchmark.py` now uses the
  `Ovrlp(...).analyse()` flow and `cell_integrity_from_transcripts(...)`.
- `cell_integrity_from_transcripts` requires integer cell IDs with an
  integer unassigned sentinel (-1); the wrapper factorises the
  standardized string cell IDs and persists the lookup at
  `cell_id_code_map.parquet`.

**Current TSU-20 status.** PASS. Full signal-integrity grid (~27 MB),
signal-strength grid (~33 MB), per-cell VSI (~17 MB), 5,755 pseudocells.

---

### 5. cellAdmix (kharchenkolab/cellAdmix, 0.1.0)

**Purpose.** Molecule-level admixture detection and removal.

**Requirements.** R 4.4 + Seurat 5.x + Cairo + a chain of CRAN/Bioc
packages installable via
`workflow/scripts/_count_correction/install_split_celladmix_deps.R`. The
unexported helpers `get_mol_knn` and `check_fp` are accessed via
`asNamespace("cellAdmix")`.

**Input files**
- Standardized Xenium transcripts
  (`results/tsu20_tools/standardized/xenium_default/transcripts.parquet`)
- A cluster-label CSV. **Caveat:** curated cell-type labels are not yet
  available for this run, so the script uses the Xenium graphclust
  output at `dataset/lung_cancer_xenium_10x/TSU-20/analysis/clustering/gene_expression_graphclust/clusters.csv`
  as a **temporary technical placeholder**, prefixed `graphclust_<id>`.
  This is sufficient to drive the workflow but should not be reported
  as a biological result — hence the `DEBUG_PASS` status, not `PASS`.

**Command**

```bash
# (1) Install all deps once
conda run -n tracer_benchmark_r \
  Rscript workflow/scripts/_count_correction/install_split_celladmix_deps.R

# (2) Prepare shared inputs (also used by SPLIT)
python workflow/scripts/_count_correction/prepare_tsu20_common_inputs.py \
  --xenium-dir dataset/lung_cancer_xenium_10x/TSU-20 \
  --scrna-h5ad dataset/lung_cancer_scrna_10/lung_cancer_scrna_split.h5ad \
  --clusters dataset/lung_cancer_xenium_10x/TSU-20/analysis/clustering/gene_expression_graphclust/clusters.csv \
  --outdir results/tsu20_tools/common_inputs

# (3) Run cellAdmix
conda run -n tracer_benchmark_r \
  Rscript workflow/scripts/_count_correction/run_celladmix_tsu20_real.R \
    --xenium-dir dataset/lung_cancer_xenium_10x/TSU-20 \
    --clusters dataset/lung_cancer_xenium_10x/TSU-20/analysis/clustering/gene_expression_graphclust/clusters.csv \
    --outdir results/tsu20_tools/celladmix_xenium_default \
    --num-factors 10 --nmol-dsamp 10000 --n-cells-nmf 2000 --cores 1
```

**Tutorial workflow executed (per `dataset/NSCLC_tutorial_fulldata.ipynb`):**

1. `samp_ct_equal` — class-balanced sampling
2. `run_knn_nmf` — kNN factorisation
3. `run_crf_all` — CRF refinement
4. `get_mol_knn` (from package namespace) — molecule-level kNN
5. `run_bridge_test`
6. `extract_bridge_res`
7. `check_fp` (from package namespace) — false-positive screen
8. Molecule removal + corrected-count matrix construction

**Outputs.** `cleaned_transcripts.parquet`, `removed_transcripts.parquet`,
`factor_assignments.csv`, `corrected_counts.mtx`, `cell_metadata.parquet`,
`celladmix_result.rds`, `nmf_result.rds`, `crf_result.rds`,
`knn_result.rds`, `method_info.json`, plus `logs/`.

**Known limitations.**
- Cells with ≤2 molecules are dropped before the CRF/bridge step (the
  CRF needs >2 molecules per cell); 1,313 such cells were removed in
  this run.
- Cluster labels are placeholders (graphclust IDs), not real cell types.
  Replace with finetuned labels (RCTD, scNym, etc.) before reporting
  biological conclusions.
- `cellAdmix` exports vary across versions; the runner uses
  `asNamespace("cellAdmix")` to call internal helpers that newer
  versions may or may not expose.

**Current TSU-20 status.** DEBUG_PASS. 1,635,024 input transcripts →
1,519,449 retained (115,575 removed), 57,136 cells, 22 temporary
graphclust labels.

---

### 6. SPLIT (bdsc-tds/SPLIT, 0.1.3)

**Purpose.** Profile purification of segmented counts. SPLIT consumes a
post-processed RCTD doublet-mode object plus the spatial cell-by-gene
matrix and returns purified counts.

**Requirements.** R 4.4 + Seurat 5.x + spacexr 2.2.1 (RCTD) + SPLIT 0.1.3.
Plus a real single-cell reference with cell-type labels.

**Input files**
- `dataset/lung_cancer_xenium_10x/TSU-20/` (Xenium bundle)
- `dataset/lung_cancer_scrna_10/lung_cancer_scrna_split.h5ad`
  (scRNA-seq reference; the `cell_type` column is used for labels)

**Commands**

```bash
# (1) Inspect the reference h5ad once to confirm cell-type column
python workflow/scripts/_count_correction/inspect_h5ad_reference.py \
  --h5ad dataset/lung_cancer_scrna_10/lung_cancer_scrna_split.h5ad \
  --xenium-gene-panel dataset/lung_cancer_xenium_10x/TSU-20/gene_panel.json \
  --out results/tsu20_tools/reference_lung_scrna_h5ad_summary.txt

# (2) Prepare shared Xenium/scRNA matrices (writes results/tsu20_tools/common_inputs/)
python workflow/scripts/_count_correction/prepare_tsu20_common_inputs.py \
  --xenium-dir dataset/lung_cancer_xenium_10x/TSU-20 \
  --scrna-h5ad dataset/lung_cancer_scrna_10/lung_cancer_scrna_split.h5ad \
  --clusters dataset/lung_cancer_xenium_10x/TSU-20/analysis/clustering/gene_expression_graphclust/clusters.csv \
  --outdir results/tsu20_tools/common_inputs

# (3) Install R deps if not already present
conda run -n tracer_benchmark_r \
  Rscript workflow/scripts/_count_correction/install_split_celladmix_deps.R

# (4) Run SPLIT
conda run -n tracer_benchmark_r \
  Rscript workflow/scripts/_count_correction/run_split_tsu20_real.R \
    --xenium-dir dataset/lung_cancer_xenium_10x/TSU-20 \
    --scrna-h5ad dataset/lung_cancer_scrna_10/lung_cancer_scrna_split.h5ad \
    --celltype-column auto \
    --outdir results/tsu20_tools/split_xenium_default
```

**Workflow executed.**

1. Build `spacexr::Reference` from scRNA counts + `cell_type` labels.
2. Build `spacexr::SpatialRNA` from Xenium counts + cell centroids.
3. `spacexr::create.RCTD(..., doublet_mode = TRUE)` then `run.RCTD(...)`.
4. Save `RCTD_raw.rds`.
5. `SPLIT::run_post_process_RCTD(...)` → `post_processed_RCTD.rds`.
6. `SPLIT::purify(counts, rctd, DO_purify_singlets = TRUE)`.

**Outputs**
- `RCTD_raw.rds`
- `post_processed_RCTD.rds`
- `split_result.rds`
- `purified_counts.mtx`
- `cell_meta.csv` / `cell_meta.parquet`
- `xe_purified.rds`
- `method_info.json`
- `logs/run_split_tsu20_real.log`

**Known limitations.**
- spacexr / RCTD is installed from GitHub (Bioconductor lags). Version
  pin: `spacexr` 2.2.1, `SPLIT` 0.1.3.
- Seurat 5 stores counts in layers; the wrapper tries `layer="counts"`
  and falls back to `slot="counts"` for older Seurat versions.
- RCTD doublet mode runtime: ~30 minutes on M1 with `cores=2`. Scales
  with `n_spatial_cells * n_reference_cells * n_shared_genes`.
- Reference + spatial must share enough genes: 292 shared in this run.
  See `results/tsu20_tools/common_inputs/gene_overlap_report.csv`.

**Current TSU-20 status.** PASS. spacexr 2.2.1 + SPLIT 0.1.3,
35,954 reference cells, 58,648 spatial cells, 292 shared genes,
runtime ≈ 1,821 s.

---

### 7. Segger — SKIPPED on this host

**Purpose.** Graph-neural-network transcript-to-cell assignment.

**Requirements.** CUDA + NVIDIA GPU + the `python_cuda` Singularity
container from the original SPLIT pipeline. Version pin (per template):
`senbaikang/segger_dev@96e531dd7313dfe9c19111b029b49e582531044f`.

**Current TSU-20 status.** SKIPPED. The macOS arm64 host has no GPU,
no CUDA, and no Singularity. A non-stub `method_info.json` at
`results/tsu20_tools/segger/raw/method_info.json` records the blocker
and the unblock procedure (move to a Linux+CUDA host, build
`segger.def`, wire `containers.python_cuda` in the config, re-enable
`methods.segger`).

---

## Repository layout

```
.
├── README.md
├── .gitignore
├── config_split_original/             ← original SPLIT pipeline configs (read-only)
├── dataset/                           ← *not tracked* — Xenium bundle(s) + scRNA reference
├── reproducibility/                   ← original SPLIT Singularity .def files (Linux build target)
├── resources/marker_sets/
├── results/                           ← *not tracked* — every run output
└── workflow/
    ├── Snakefile                      ← original SPLIT pipeline (unchanged)
    ├── Snakefile_benchmark            ← benchmark driver (single-sample)
    ├── configs/
    │   ├── tsu20_real_tools.yml       ← THIS run's config (TRACER off, real tools on)
    │   ├── tsu20_tools.yml            ← earlier benchmark-only config (kept for reference)
    │   ├── benchmark_lung_tiny.yml    ← smoke test only
    │   └── benchmark_lung_small.yml
    ├── envs/
    ├── rules/
    │   ├── _benchmark/                ← uniform standardized-contract rules
    │   └── _count_correction/
    │       ├── split_real_tsu20.smk
    │       └── celladmix_real_tsu20.smk
    └── scripts/
        ├── _benchmark/
        │   ├── standardize_method_output.py
        │   ├── run_ovrlpy_benchmark.py
        │   ├── validate_real_tool_outputs.py
        │   └── …
        └── _count_correction/
            ├── inspect_h5ad_reference.py
            ├── prepare_tsu20_common_inputs.py
            ├── install_split_celladmix_deps.R
            ├── run_split_tsu20_real.R
            └── run_celladmix_tsu20_real.R
```

---

## Snakemake entry points

```bash
# Dry-run the SPLIT + cellAdmix scheduled paths
~/.local/bin/snakemake -n \
  --configfile workflow/configs/tsu20_real_tools.yml \
  --cores 8

# Run SPLIT and cellAdmix together
~/.local/bin/snakemake \
  --configfile workflow/configs/tsu20_real_tools.yml \
  --cores 8 \
  results/tsu20_tools/split_xenium_default/method_info.json \
  results/tsu20_tools/celladmix_xenium_default/method_info.json
```

The driving config:

```yaml
methods:
  tracer: false
  segger: false
  split_xenium_default: true
  celladmix_xenium_default: true
```

---

## Reproducibility notes

- Versions actually used in this run: Snakemake 9.20.0, Baysor 0.7.0,
  Proseg 3.0.10, ovrlpy 1.1.0, R 4.4.3, Seurat 5.5.0, spacexr 2.2.1,
  SPLIT 0.1.3, cellAdmix 0.1.0.
- All version pins from the upstream SPLIT template are documented in
  the per-tool sections above. The canonical Singularity recipe lives in
  `reproducibility/{10x,baysor,proseg,r}.def` (Linux + sudo + Singularity
  required to build); the install procedure above is the macOS-native
  equivalent.
- TSU-20 dataset, scRNA reference, all `results/`, all `logs/`, all
  `.snakemake/` state, all `*.sif`/`*.sqsh`, all `__pycache__/`, all
  `.DS_Store` are ignored by `.gitignore` and are NOT tracked.

## Troubleshooting

- **`snakemake: command not found`** — install with
  `pip install --user "snakemake>=9.0,<10"`; ensure `~/.local/bin` is on `$PATH`.
- **No Singularity on macOS** — build the `.def` files on a Linux host, OR
  use the native installs above and bypass the original SPLIT pipeline's
  `xeniumranger`-dependent rules.
- **ovrlpy import error / dask circular import** — happens in mixed
  anaconda envs with anndata + dask. Use the dedicated
  `tracer_benchmark_ovrlpy` conda env (Python 3.11).
- **Cairo / NMF / Seurat fails to compile in R** — make sure the conda
  env has the cairo system lib (`conda install -n tracer_benchmark_r -c
  conda-forge cairo pango freetype harfbuzz fribidi libtiff pkg-config
  r-cairo r-seurat`) before retrying `install_split_celladmix_deps.R`.
- **h5ad cell-type column not detected** — rerun
  `inspect_h5ad_reference.py`, then pass an explicit
  `--celltype-column <name>` to `run_split_tsu20_real.R`.
- **Low gene overlap between Xenium and scRNA reference** — check
  `results/tsu20_tools/common_inputs/gene_overlap_report.csv`. This run had
  292 shared genes.
- **Seurat v4 vs v5 count access** — `run_split_tsu20_real.R` tries
  `layer="counts"` first and falls back to `slot="counts"`.
- **cellAdmix internal helpers** — `get_mol_knn` and `check_fp` are not
  exported in the installed GitHub build; the runner uses
  `asNamespace("cellAdmix")` to call them.
- **cellAdmix low-molecule cells** — cells with ≤2 molecules are removed
  before the CRF/bridge step (1,313 cells dropped in this run).
- **Xenium Ranger download link changed** — `reproducibility/10x.def`
  pins a 10x download URL; if 10x rotates the link, patch the `.def`.
- **Segger requires CUDA + GPU** — previously skipped on macOS. Now
  configured for the DSAI HPC cluster (dsailogin). See
  [docs/hpc_segger_smoke.md](docs/hpc_segger_smoke.md) for the full
  HPC smoke-test procedure. Apptainer 1.4.4 is on compute nodes only;
  build via `sbatch scripts/slurm/build_python_cuda_container.sbatch`.
- **SPLIT requires a single-cell reference** — provide a labelled scRNA
  h5ad via `--scrna-h5ad` and ensure the cell-type column is detectable
  by `inspect_h5ad_reference.py`. Without it RCTD cannot run, and
  without RCTD SPLIT cannot run.

---

## Attribution

Benchmark adapted from the SPLIT Xenium analysis pipeline. Cite the
original methods when reporting results:

- SPLIT — https://github.com/bdsc-tds/SPLIT (0.1.3)
- Baysor — https://github.com/kharchenkolab/Baysor (0.7.0)
- Proseg — https://github.com/dcjones/proseg (3.0.10)
- Segger — https://github.com/EliHei2/segger_dev (fork senbaikang/segger_dev@96e531d)
- cellAdmix — https://github.com/kharchenkolab/cellAdmix (0.1.0)
- ovrlpy — https://github.com/HiDiHlabs/ovrl.py (1.1.0)
- spacexr / RCTD — https://github.com/dmcable/spacexr (2.2.1)
- Xenium Ranger — 10x Genomics (4.0.0 expected by template)

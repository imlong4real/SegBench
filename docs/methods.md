# Method reference

Inputs, configuration and environment for each wrapper. All of them share the
common flags (`--outdir`, `--config`, `--dataset`, `--sample-name`, `--seed`,
`--threads`, `--overwrite`, `--max-transcripts`, `--dry-run`) and all of them
produce the output contract in [output_schema.md](output_schema.md).

`--max-transcripts N` subsamples the input before the method runs and is the
supported way to smoke-test any imaging method quickly.

---

## Imaging (molecule-resolved)

### baysor — de-novo segmentation

| | |
|---|---|
| **Needs** | `baysor` ≥ 0.7 binary (Julia) |
| **Input** | `--transcripts` standardized parquet |
| **Config** | `configs/methods/baysor.yaml` |
| **Entity** | cell |

```bash
segbench run baysor \
  --transcripts data/TSU-20/filtered_df_standardized.parquet \
  --outdir benchmark_output/tsu20/baysor \
  --sample-name TSU20 --threads 8 --overwrite
```

Install: `julia -e 'using Pkg; Pkg.add(PackageSpec(url="https://github.com/kharchenkolab/Baysor.git", rev="v0.7.0")); Pkg.build("Baysor")'` → shim at `~/.julia/bin/baysor`. Or build `reproducibility/baysor.def`.

Key flags: `--scale` (cell radius; estimated from nucleus spacing when unset — **set it explicitly for reproducible runs**), `--min-molecules-per-cell`, `--prior-segmentation-confidence`, `--use-prior`, `--baysor-config` (supply your own TOML).

`--mode wrap` standardizes a pre-existing `segmentation.csv` without running Baysor; its runtime is then flagged `runtime_valid_for_benchmark: false`.

### proseg — de-novo segmentation

| | |
|---|---|
| **Needs** | `proseg` ≥ 3.0 binary (Rust) |
| **Input** | `--transcripts` standardized parquet |
| **Config** | `configs/methods/proseg.yaml` |
| **Entity** | cell |

```bash
segbench run proseg \
  --transcripts data/TSU-20/filtered_df_standardized.parquet \
  --outdir benchmark_output/tsu20/proseg \
  --sample-name TSU20 --threads 8 --overwrite
```

Install: `cargo install proseg --version 3.0.10 --locked`.

Key flags: `--voxel-layers` (z-axis voxels), `--extra-proseg-args`. proseg can emit *more* rows than it consumed when it splits molecules across voxel layers — `transcripts.delta_vs_input` in the stats file makes that visible.

### segger — GNN segmentation (GPU)

| | |
|---|---|
| **Needs** | `segger`, `torch` + `torch-geometric` with CUDA |
| **Input** | `--xenium-dir` (preferred) or `--transcripts` |
| **Config** | `configs/methods/segger.yaml` |
| **Entity** | cell |

```bash
segbench run segger \
  --xenium-dir data/TSU-20 \
  --outdir benchmark_output/tsu20/segger \
  --container containers/python_cuda.sif \
  --segger-cli-dir /opt/segger_dev/src/segger/cli \
  --sample-name TSU20 --max-epochs 3 --threads 8 --overwrite
```

The wrapper drives all three Segger stages (`create_dataset_fast` → `train_model` → `predict_fast`) and standardizes the result.

**Input note.** Segger needs a Xenium-style bundle with nucleus boundaries. Pass `--xenium-dir` whenever you have the real bundle. If you pass only `--transcripts`, the wrapper *derives* boundaries from `overlaps_nucleus` molecules via convex hulls — usable, but the accuracy is not comparable to real boundaries, and the wrapper warns.

**Execution.** Add `--container <sif>` to run every step through `apptainer exec --nv`, with `--bind src:dst` as needed; otherwise it runs natively. `--accelerator cpu` works but is far too slow for anything beyond a wiring check.

`--mode wrap --segger-transcripts <parquet>` standardizes an existing run.

### celladmix — admixture correction

| | |
|---|---|
| **Needs** | R ≥ 4.3 with `cellAdmix` |
| **Input** | `--transcripts`, plus a cluster/cell-type label source |
| **Config** | `configs/methods/celladmix.yaml` |
| **Entity** | cell (original IDs retained) |

```bash
segbench run celladmix \
  --transcripts data/TSU-20/filtered_df_standardized.parquet \
  --outdir benchmark_output/tsu20/celladmix \
  --sample-name TSU20 --threads 4 --overwrite
```

Operates on the *existing* cell assignment and flags contaminating molecules; it does not redraw boundaries. Retained molecules keep their `cell_id` (`cleaned_status=retained`), removed ones become `UNASSIGNED` (`cleaned_status=cleaned_to_unassigned`), and `original_cell_id` is preserved throughout — so pre/post comparison is exact.

Install R deps once: `Rscript workflow/scripts/_count_correction/install_split_celladmix_deps.R`.

### split — RCTD purification (cell-level)

| | |
|---|---|
| **Needs** | R ≥ 4.3 with `spacexr` + `SPLIT` |
| **Input** | `--transcripts`, `--reference-h5ad` |
| **Config** | `configs/methods/split.yaml` |
| **Entity** | cell |

```bash
segbench run split \
  --transcripts data/TSU-20/filtered_df_standardized.parquet \
  --reference-h5ad data/scrna/lung_cancer_scrna_split.h5ad \
  --outdir benchmark_output/tsu20/split \
  --sample-name TSU20 --threads 4 --overwrite
```

**SPLIT is scored at cell level, not transcript level.** `SPLIT::purify` returns fractional expected counts, so which individual molecule was removed cannot be recovered. The wrapper therefore writes no transcript parquet; it emits purified/original cell-by-gene matrices plus count-level pruning estimates in `outputs/split_pruning_summary.json`, and sets `qc.transcript_level = false`. Score it with `workflow/scripts/get_cell_level_metric.py`, **not** the transcript-level `get_metric.py`.

### tracer — cPMI-guided refinement

| | |
|---|---|
| **Needs** | the `tracer` python package |
| **Input** | `--transcripts`, `--pmi` panel |
| **Config** | `configs/methods/tracer.yaml` |
| **Entity** | cell |

```bash
segbench run tracer \
  --transcripts data/TSU-20/filtered_df_standardized.parquet \
  --pmi results/reference_pmi/lung_cancer_pmi.csv.gz \
  --platform xenium --pmi-threshold 0.2 \
  --outdir benchmark_output/tsu20/tracer \
  --sample-name TSU20 --overwrite
```

Build the cPMI panel first with TRACER's `build_pmi_from_scrna.py`. Point `TRACER_HOME` at your TRACER checkout if a config references it.

---

## Sequencing (array / binned)

### bin2cell — Visium HD bins → cells

| | |
|---|---|
| **Needs** | `bin2cell`; `stardist` + `tensorflow` unless `--labels-npz` is given |
| **Input** | `--input-h5ad` (2 µm bins) + `--source-image` (H&E) |
| **Config** | `configs/methods/bin2cell.yaml` |
| **Entity** | **bin** in the assignment table, **cell** after calling |

```bash
segbench run bin2cell \
  --input-h5ad data/visium_hd/square_002um/filtered_feature_bc_matrix.h5 \
  --spaceranger-dir data/visium_hd/square_002um \
  --source-image data/visium_hd/Visium_HD_He.tif \
  --outdir benchmark_output/visium_hd/bin2cell \
  --sample-name VisiumHD --mpp 0.5 --prob-thresh 0.01 --overwrite
```

Install: `pip install bin2cell` (plus `stardist`/`tensorflow` for segmentation).

Rows in `outputs/bin2cell_bin_assignments.parquet` are 2 µm **bins**, not transcripts, and `feature_name` is the placeholder `__bin__` — a bin carries a whole expression vector. "Assigned" means the bin fell inside a called cell. `entities.n_entities` is the number of **cells** called.

Key flags: `--mpp`, `--prob-thresh`, `--expand-microns`, `--use-gex` (also segment a gene-expression density image and salvage secondary labels), `--labels-npz` (skip StarDist entirely by supplying a precomputed label image — the way to run this without TensorFlow).

### tracer_seq — TRACER on binned data

Same wrapper as `tracer`, registered separately with `modality: sequencing`,
`entity_kind: bin`, and a lower `min_tx_per_cell_for_scores` (a 2 µm bin holds
far fewer counts than a segmented cell). Config: `configs/methods/tracer_seq.yaml`.

```bash
segbench run tracer_seq --dataset visium_hd_demo \
  --pmi results/reference_pmi/panel.csv.gz \
  --outdir benchmark_output/visium_hd/tracer_seq
```

---

## Environment summary

| method | runtime | install |
|---|---|---|
| baysor | Julia binary | `Pkg.add(Baysor)` or `reproducibility/baysor.def` |
| proseg | Rust binary | `cargo install proseg --locked` |
| segger | Python + CUDA | container `reproducibility/segger/segger.def`, or pip + torch-geometric |
| split | R | conda env + `install_split_celladmix_deps.R` |
| celladmix | R | conda env + `install_split_celladmix_deps.R` |
| tracer / tracer_seq | Python | the TRACER package |
| bin2cell | Python | `pip install bin2cell` (+ stardist/tensorflow) |

`segbench doctor` reports which of these are present on the current host.

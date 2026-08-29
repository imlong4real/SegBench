# HPC Segger Smoke Test

**Cluster:** DSAI (dsailogin login node, Rocky Linux 8)  
**Date:** 2026-05-14  
**Status:** BLOCKED — GPU queue wait (container built successfully)

---

## 1. Repository

**Path:** `/weka/home/lyuan13/segmentation_benchmark_pipeline`  
**Input parquet:** `/weka/home/lyuan13/TRACER/tutorials/lung_cancer/data/lung_cancer_df.parquet`

> **Note:** Instructions referenced `filtered_df.parquet`; actual file is `lung_cancer_df.parquet`.
> Both `/home/lyuan13` and `/weka/home/lyuan13` resolve to the same location on this cluster.

---

## 2. What Segger Expects

### Input

The Segger pipeline (`workflow/rules/_segmentation/segger.smk`) expects a **Xenium bundle directory** as input, not a raw parquet. The pipeline calls:

```
runSeggerPreprocess  →  create_dataset_fast.py  --base_dir <xenium_bundle>  --sample_type xenium
runSeggerTrain       →  train_model.py
runSeggerPredict     →  predict_fast.py
```

### Xenium Bundle Format

| File | Columns required |
|------|-----------------|
| `transcripts.parquet` | `x_location`, `y_location`, `z_location`, `feature_name`, `transcript_id`, `qv`, `overlaps_nucleus`, `cell_id` |
| `nucleus_boundaries.parquet` | `cell_id`, `vertex_x`, `vertex_y` |
| `experiment.xenium` | JSON metadata stub |

The input `lung_cancer_df.parquet` has columns: `x, y, z, feature_name, cell_id, nucleus_distance, transcript_id, fov_name, qv, overlaps_nucleus`. These are renamed when building the Xenium bundle.

---

## 3. Container

### Definition

**Path:** `reproducibility/python_cuda/python_cuda.def`

Multi-stage Singularity/Apptainer build:
- **Base:** `nvidia/cuda:12.1.0-runtime-ubuntu22.04`
- **Package manager:** Micromamba
- **Segger repo:** `bdsc-tds/segger_dev` @ `4bf56dec2a364de8eee4fcab663e798eb106e21a`
- **Conda environments installed inside:**
  - `segger_cuda` — Python 3.11, PyTorch 2.1.2 + CUDA 12.1, PyG, dask-cuda, Segger
  - `general_cuda` — general analysis environment

**Segger install location inside container:**
`/opt/segger_dev/src/segger/cli/`

**CLI entry points:**
```bash
mamba run -n segger_cuda python3 /opt/segger_dev/src/segger/cli/create_dataset_fast.py
mamba run -n segger_cuda python3 /opt/segger_dev/src/segger/cli/train_model.py
mamba run -n segger_cuda python3 /opt/segger_dev/src/segger/cli/predict_fast.py
```

### Alternative definition

`reproducibility/segger/segger.def` — simpler pip-based build from `EliHei2/segger_dev` (untracked, not used by the pipeline config).

### How to build

**Important:** Apptainer is NOT on the login node. Build via sbatch:

```bash
sbatch scripts/slurm/build_python_cuda_container.sbatch
```

This submits to the `nvl` partition (H100 GPUs, account `adeshpa6`) and runs:
```bash
apptainer build --fakeroot --force \
    containers/python_cuda.sif \
    reproducibility/python_cuda/python_cuda.def
```

Expected build time: **~30–60 minutes** (pulls Docker layers + installs conda envs).

Container output: `containers/python_cuda.sif` (expected ~3–5 GB)

---

## 4. GPU Allocation

**Apptainer availability:**
- Login node (`dsailogin`): NOT available
- Compute nodes: `/usr/bin/apptainer` version 1.4.4-1.el9

**Available GPU partitions:**

| Partition | GPU | Status |
|-----------|-----|--------|
| `nvl` | H100 x4/node | **Healthy — recommended** |
| `h100` | H100 x4/node | Healthy |
| `l40s` | L40S x8/node | Some nodes draining |
| `a100` | A100 x8/node | **Mostly drained — avoid** |

**Slurm account:** `adeshpa6`

**Safe GPU allocation (no node pinning):**
```bash
salloc -p nvl -A adeshpa6 --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=01:00:00
```

---

## 5. Build the Container

```bash
# From the repo root (login node):
sbatch scripts/slurm/build_python_cuda_container.sbatch

# Monitor:
squeue -u $USER
tail -f logs/slurm/build_python_cuda_*.out

# Verify:
ls -lh containers/python_cuda.sif
```

---

## 6. CUDA Verification

After building, verify GPU passthrough:

```bash
sbatch scripts/slurm/run_segger_cuda_check.sbatch

# Expected output:
# cuda_available True
# cuda_device    NVIDIA H100 80GB HBM3
# segger import OK ...
```

---

## 7. Create Smoke Input

The smoke subset script reads the lung cancer parquet and writes:

| Output | Description |
|--------|-------------|
| `results/segger_smoke/input/transcripts_100k.parquet` | ≤100k transcripts from spatial ROI |
| `results/segger_smoke/input/input_summary.json` | Schema summary |
| `results/segger_smoke/input/xenium_bundle/` | Minimal Xenium bundle for Segger preprocess |

The script is run inside the smoke sbatch. It requires `pandas` + `numpy` (available in `anaconda3/2024.02-1`).

---

## 8. Run the Smoke Test

```bash
# Requires containers/python_cuda.sif to exist first
sbatch scripts/slurm/run_segger_smoke.sbatch

# Monitor:
squeue -u $USER
tail -f logs/slurm/segger_smoke_<jobid>.out
tail -f logs/slurm/segger_smoke_<jobid>.err
```

**Steps inside the job:**
1. `nvidia-smi` — GPU visibility
2. PyTorch CUDA check inside container
3. Segger import check inside container
4. `create_segger_smoke_subset.py` — creates 100k-row Xenium bundle
5. `create_dataset_fast.py` — Segger preprocess (200×200 tiles)
6. `train_model.py` — 3-epoch training
7. `predict_fast.py` — prediction

**Expected outputs:**

| Path | Description |
|------|-------------|
| `results/segger_smoke/preprocessed_data/` | Graph tiles from preprocess |
| `results/segger_smoke/trained_model/` | Lightning model checkpoint |
| `results/segger_smoke/segger_output/` | Segger predictions (parquet + h5ad) |

---

## 9. Validate

```bash
# After smoke test completes:
module load anaconda3/2024.02-1
conda run python workflow/scripts/_benchmark/validate_segger_smoke.py

# Report:
cat results/segger_smoke/segger_smoke_validation.json
```

PASS criteria:
- `containers/python_cuda.sif` non-empty
- Input subset ≤ 100,000 rows
- Preprocessed data dir non-empty
- Trained model dir non-empty
- Segger output dir non-empty with non-zero files

---

## 10. Current Status (2026-05-14) — COMPLETE ✓

| Item | Status | Notes |
|------|--------|-------|
| Repo pulled | **DONE** | At latest commit |
| `.gitignore` | **DONE** | `*.parquet`, `containers/`, `*.sif`, `results/`, `logs/` |
| Input parquet | **DONE** | `lung_cancer_df.parquet` — 1,436,900 rows, Xenium format |
| Container def | **DONE** | `reproducibility/python_cuda/python_cuda.def` |
| Container SIF | **DONE** | `containers/python_cuda.sif` — 18 GB |
| TSU-20 real data | **DONE** | `results/segger_smoke/input/xenium_bundle/TSU-20/` |
| Preprocessing (create_dataset_fast.py) | **DONE** | 271 tiles: train=137, val=15, test=119 (job 1436236) |
| Training (3 epochs, NVIDIA L40S) | **DONE** | Checkpoint: `results/segger_TSU20/trained_model/lightning_logs/version_0/checkpoints/epoch=2-step=105.ckpt` |
| Prediction (predict_fast.py) | **DONE** | All 3 splits processed after edge_label patch (job 1436928) |
| Validation | **PASS** | `results/segger_TSU20/segger_validation.json` |

**Key output files:**

| File | Size | Description |
|------|------|-------------|
| `results/segger_TSU20/segger_output/.../segger_adata.h5ad` | 50 MB | AnnData object |
| `results/segger_TSU20/segger_output/.../segger_transcripts.parquet` | 49 MB | Transcript-to-cell assignments |
| `results/segger_TSU20/segger_output/.../transcripts_df.parquet/` | ~17 MB | 64-part Dask parquet |

**Rerun command (from scratch):**
```bash
sbatch scripts/slurm/run_segger_smoke.sbatch
```

**Rerun prediction only (tiles + checkpoint already exist):**
```bash
sbatch scripts/slurm/run_segger_predict_retry.sbatch
```

**Validate:**
```bash
module load anaconda3/2024.02-1
python workflow/scripts/_benchmark/validate_segger_smoke.py
cat results/segger_TSU20/segger_validation.json
```

---

## 11. Blockers Log

| Blocker | Resolution |
|---------|-----------|
| `apptainer` not on login node | Submit all container operations as sbatch jobs |
| Input file named `lung_cancer_df.parquet` not `filtered_df.parquet` | Scripts updated to use actual file name |
| `a100` partition mostly drained | Use `nvl` or `h100` partition |
| Build `%files` needs build CWD = `reproducibility/python_cuda/` | Fixed: sbatch `cd`s into def directory before building |
| GPU queue full — all partitions busy | Resolved: ran on L40S (l03, job 1436144) |
| `Trainer(devices=0)` MisconfigurationException — job 1436144 | Fixed: `--devices 1` (see §12) |
| **`KeyError: 'edge_label'` in predict_fast.py — job 1436236** | **FIXED** — see §13 |

---

## 12. Job 1436144 Failure — Diagnosis and Fix

**Job:** 1436144  
**Partition:** l40s (node l03)  
**Duration:** 00:01:11  
**Exit code:** 1:0

### What succeeded

- GPU visible: NVIDIA L40S, CUDA 13.0 driver
- `torch.cuda.is_available()` → True inside container
- Segger import OK
- Preprocessing: `create_dataset_fast.py` ran successfully, wrote 533 tiles
  - `test_tiles/processed/`: 131 tiles
  - `train_tiles/processed/`: 124 tiles *(Segger auto-splits — do NOT manually cp)*
  - `val_tiles/processed/`: 16 tiles *(as above)*
- Tiles preserved at `results/segger_TSU20/preprocessed_data/` for reuse

### Root cause

`train_model.py` called with `--devices 0`. PyTorch Lightning's `Trainer` under
the `cuda` accelerator interprets `devices=0` as "zero GPUs", which is invalid:

```
lightning_fabric.utilities.exceptions.MisconfigurationException:
  `Trainer(devices=0)` value is not a valid input using cuda accelerator.
```

**Source:** `--devices 0` was the literal flag passed from the sbatch. In Lightning,
`devices=N` (integer) with `accelerator=cuda` means "use N GPUs". `0` means zero
devices, not "GPU index 0". The correct value is `1`.

### Secondary issue found during diagnosis

The original script did `cp -r test_tiles/. train_tiles/` and `cp -r test_tiles/. val_tiles/`
based on the assumption that Segger only writes to `test_tiles`. This is wrong:
`create_dataset_fast.py --sample_type xenium` already creates all three splits.
The copies were polluting train/val with test-tile duplicates.

### Fixes applied

| Change | Before | After |
|--------|--------|-------|
| Lightning device count | `--devices 0` | `--devices 1` |
| Smoke epochs | `--max_epochs 200` | `--max_epochs 3` |
| Split copy | `cp test_tiles → train_tiles, val_tiles` | **Removed** (Segger auto-splits) |
| Preprocessing skip | Always re-runs | Skips if `.pt` tiles already exist |
| CUDA preflight assertion | None | Added before `train_model.py` |
| Command echo | None | `printf '%s\n' "${TRAIN_ARGS[@]}"` |
| `train_model.py --help` dump | None | Saved to `docs/segger_train_model_help.txt` |
| Tile count awk field | `$(NF-1)` → `processed` | `$(NF-2)` → `test_tiles` etc. |

### New job

**Job 1436236** — submitted 2026-05-14, queued on l40s/nvl/h100.

Logs: `logs/slurm/segger_TSU20_1436236.{out,err}`

---

## 13. Job 1436236 Failure — KeyError: 'edge_label' in predict_fast.py

**Job:** 1436236 | FAILED ExitCode 1:0 | 00:02:22 on l40s (l03)

### What succeeded

- Preprocessing (271 tiles: train=137, val=15, test=119), Training (3 epochs, checkpoint saved)
- predict_fast: Processing Train batches 35/35 ✓, Validation 4/4 ✓
- predict_fast: **Processing Test batches 0/30 FAILED**

### Root cause

PyG's `RandomLinkSplit` transform does not consistently add `edge_label` to
sparse/small tile graphs. Of 119 test tiles, **79 were missing `edge_label`**
(only 40 had it). Train and val tiles were consistent (all had it). When
`predict_fast.py`'s DataLoader collates a batch that mixes tiles with and without
`edge_label`, PyG's `Batch.from_data_list` raises `KeyError: 'edge_label'`.

Schema inspection result:
```
train_tiles: total=137 has_edge_label=137 missing=0
val_tiles:   total=15  has_edge_label=15  missing=0
test_tiles:  total=119 has_edge_label=40  missing=79
```

### Fix

Two new utility scripts:

| Script | Purpose |
|--------|---------|
| `workflow/scripts/_segmentation/inspect_segger_pyg_schema.py` | Inspect all .pt tiles; report edge_label presence; write CSV+JSON |
| `workflow/scripts/_segmentation/patch_segger_edge_label.py` | Backup test_tiles/processed/, then add `edge_label=zeros(N)` where missing |

And a lean retry job:
- `scripts/slurm/run_segger_predict_retry.sbatch` — skip preprocess/train, run inspect→patch→predict

Prediction `--num_workers 0` used in retry to surface errors in main thread.

### Result: Job 1436928 — COMPLETED (exit 0) in 00:01:23

- All 119 test tiles patched (backup at `test_tiles/processed_backup/`)
- Test batches 30/30 ✓
- Output: `results/segger_TSU20/segger_output/` — 116 MB total
- **Validation: PASS** → `results/segger_TSU20/segger_validation.json`

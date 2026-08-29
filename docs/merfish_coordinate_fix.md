# MERFISH Coordinate System Fix

## Problem

The first Segger run on `merfish_mouse_ileum` used `roi_transcripts.parquet` as input,
which stores spatial coordinates in **pixel/voxel units**, not physical µm:

| File                         | x range      | y range      | z values (9 planes)       |
|------------------------------|-------------|-------------|---------------------------|
| `roi_transcripts.parquet`    | 112 – 5720  | 0 – 9391    | 0, 13.8, 27.5, … 110.1   |
| `input_transcripts_um.parquet` | −3109 – −2498 | −1357 – −334 | 2.5, 4.0, … 14.5 µm     |

Scale factor: **~9.18 pixels/µm** (consistent across x, y, and z).

All other ROI datasets (`atera_cervical`, `cosmx_nsclc`, `xenium5k_cervical`) store
coordinates in µm in `roi_transcripts.parquet` and are unaffected.

## Effect on Segger training

Segger's `create_dataset_fast.py` interprets `tile_width`/`tile_height` in the same
units as the input coordinates. With pixel coordinates:

- `tile_size=100` covered only **100 px / 9.18 ≈ 10.9 µm** of actual tissue per tile
- Each tile captured far too few cells, starving the graph of training signal
- The model barely trained: `best_val_auroc = 0.577` (near-random baseline 0.5)
- Prediction produced `score = NaN` for 99.99% of transcripts → only 79 assigned

## Fix

Use `tracer_seg/input_transcripts_um.parquet` instead of `roi_transcripts.parquet`.
This file has identical schema (`x, y, z, feature_name, cell_id, transcript_id,
overlaps_nucleus, platform, sample, roi_id`) with correct physical µm coordinates.

Corrected run parameters:

| Parameter     | Bogus run (px) | Fixed run (µm) |
|---------------|----------------|----------------|
| input file    | `roi_transcripts.parquet` | `tracer_seg/input_transcripts_um.parquet` |
| tile_size     | 100 (= 10.9 µm actual) | **25 µm** |
| max_epochs    | 50 | **200** |
| num_tx_tokens | 250 | 250 (unchanged) |
| x/y extent    | 5608 × 9391 px | **611 × 1023 µm** |
| n_tiles       | 660 | ~1000 (estimated) |

## Pipeline change

`run_segger_roi_single.sbatch` now accepts an optional `ROI_TX_OVERRIDE` export
variable. When set, it overrides `roi_transcripts.parquet` as the bundle input.

`submit_segger_roi_all.sh` automatically passes:
```bash
ROI_TX_OVERRIDE=/home/lyuan13/scratchadeshpa6/benchmark_data/merfish_mouse_ileum/tracer_seg/input_transcripts_um.parquet
```
for the `merfish_mouse_ileum` dataset.

## Reproduction

```bash
sbatch \
  --export="DATASET=merfish_mouse_ileum,NUM_TX_TOKENS=250,TILE_SIZE=25,MAX_EPOCHS=200,\
ROI_TX_OVERRIDE=/home/lyuan13/scratchadeshpa6/benchmark_data/merfish_mouse_ileum/tracer_seg/input_transcripts_um.parquet" \
  scripts/slurm/run_segger_roi_single.sbatch
```

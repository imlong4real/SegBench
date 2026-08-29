# Segger predict_fast.py — edge_label KeyError Debug Log

## Job 1436236 — Diagnosis

**Status:** FAILED ExitCode 1:0 after 00:02:22 on l40s (l03)

### What succeeded

| Stage | Result |
|-------|--------|
| GPU check | NVIDIA L40S, CUDA 13.0 — OK |
| Container CUDA | `cuda_available True` — OK |
| Segger import | OK |
| `create_dataset_fast.py` preprocessing | **OK** — 271 tiles (train=137, val=15, test=119) |
| `train_model.py` 3 epochs | **OK** — checkpoint saved at `results/segger_TSU20/trained_model/lightning_logs/version_0/checkpoints/epoch=2-step=105.ckpt` |
| predict: Processing Train batches | **OK** — 35/35 |
| predict: Processing Validation batches | **OK** — 4/4 |
| predict: Processing Test batches | **FAILED** — 0/30 |

### Exact error

```
KeyError: Caught KeyError in DataLoader worker process 0.
Original Traceback (most recent call last):
  ...
  File ".../torch_geometric/data/collate.py", line 95, in collate
    values = [store[attr] for store in stores]
  ...
KeyError: 'edge_label'
```

Callstack: `predict_fast.py → predict_parquet.py:segment() → DataLoader → Batch.from_data_list → collate`

### Interpretation

`predict_fast.py` iterates all three dataset splits (train, val, test) to assign
cell labels to every transcript. Train/Val tiles have consistent `edge_label` on
every tile (added by PyG's `RandomLinkSplit` during preprocessing). Some test
tiles **lack `edge_label`** — likely small/sparse graphs where `RandomLinkSplit`
found too few negative edges and skipped the supervision label tensor.

When PyG's `Batch.from_data_list` collates a batch that mixes tiles with and
without `edge_label`, it finds the attribute in the first tile and tries to
retrieve it from all tiles → `KeyError` on tiles that don't have it.

### Secondary warning (not the crash cause)

```
UserWarning: An output with one or more elements was resized since it had
shape [N], which does not match the required output shape [2, N].
```

This is a pre-existing PyG/PyTorch bug in `collate.py:204` where `cat_dim`
resolves to 0 when concatenating `edge_label_index`-like tensors. It's a
warning, not a crash, and train/val process despite it.

---

## Fix Applied — Job 1436928

### Scripts written

| Script | Purpose |
|--------|---------|
| `workflow/scripts/_segmentation/inspect_segger_pyg_schema.py` | Load all .pt tiles and report edge_label presence per split. Writes `debug/pyg_schema_summary.{csv,json}` |
| `workflow/scripts/_segmentation/patch_segger_edge_label.py` | Back up test_tiles/processed/ then add `edge_label=zeros(N)` (and `edge_label_index=zeros(2,0)` if also missing) to all tiles that lack it |
| `scripts/slurm/run_segger_predict_retry.sbatch` | GPU job: inspect → patch → predict (skips preprocess/train, reuses existing checkpoint) |

### Patch logic

For each `.pt` tile in `test_tiles/processed/`:
- If HeteroData: iterate all edge stores
- If edge store has `edge_label_index` but no `edge_label`: set `edge_label = zeros(edge_label_index.shape[1])`
- If edge store has neither: set `edge_label_index = zeros(2,0)`, `edge_label = zeros(0)`
- If both already present: no change

Backup saved to `test_tiles/processed_backup/` before patching.

### Predict command changes for retry

- `--num_workers 0` (main thread collation — avoids worker process masking the error if patch is incomplete)
- `--batch_size 4` (unchanged)
- Output cleared (`rm -rf segger_output`) then re-created

---

## Job 1436928 — Outcome

**Status: COMPLETED (exit 0) in 00:01:23 on l40s**

| Stage | Result |
|-------|--------|
| Schema inspection | train 137/137 ✓, val 15/15 ✓, **test 40/119 had edge_label, 79 missing** |
| Patch test_tiles | **119/119 patched** (0 errors), backup at `test_tiles/processed_backup/` |
| Processing Train batches | 35/35 ✓ |
| Processing Validation batches | 4/4 ✓ |
| Processing Test batches | **30/30 ✓** |
| Prediction output written | 116 MB total |
| Validation | **PASS** (no blockers) |

**Tile patch report:** `results/segger_TSU20/debug/edge_label_patch_report.csv`  
**Schema summary:** `results/segger_TSU20/debug/pyg_schema_summary.json`  
**Prediction output:** `results/segger_TSU20/segger_output/segger_embedding_1001_0.5_0.5_False_4_12.0_5_5.0_20260514/`

| Output file | Size |
|-------------|------|
| `segger_adata.h5ad` | 50 MB |
| `segger_transcripts.parquet` | 49 MB |
| `transcripts_df.parquet/` (64 parts) | ~17 MB |
| `segmentation_log.json` | 369 B |

**Validation JSON:** `results/segger_TSU20/segger_validation.json` — Status: PASS

---

## Next Steps If 1436928 Fails

1. Read new logs: `tail -200 logs/slurm/segger_predict_retry_1436928.err`
2. Check schema JSON: `cat results/segger_TSU20/debug/pyg_schema_summary.json`
3. Check patch report: `cat results/segger_TSU20/debug/edge_label_patch_report.csv`
4. Common next failure modes:
   - `edge_label_index` also missing and zero-tensor shape rejected → patch with `edge_label_index = zeros(2,0)`
   - Wrong `edge_label` shape (2D expected) → reshape
   - OOM during test batch collation → reduce `--batch_size`
   - Missing output files → check if predict_fast requires all three splits

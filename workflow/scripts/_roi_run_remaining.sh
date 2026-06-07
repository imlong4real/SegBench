#!/usr/bin/env bash
# Sequentially run the remaining SPLIT + cellAdmix ROI jobs (memory-safe: one at
# a time). atera Baysor is handled separately. SPLIT uses full-panel
# _common_inputs; cellAdmix uses _common_inputs_celladmix (HVG) for the dense
# cervical panels and full _common_inputs for cosmx/merfish.
# NOTE: NOT set -e — we want every job attempted even if one fails.
export PATH="$HOME/.julia/bin:$HOME/.cargo/bin:$PATH"
ROOT=/Users/lyuan13/Desktop/segmentation_benchmark_pipeline
PY=/Users/lyuan13/anaconda3/envs/spatial/bin/python
cd "$ROOT"
LOGD=/tmp/roi_logs; mkdir -p "$LOGD"

ref_for () { case "$1" in
  atera_cervical|xenium5k_cervical) echo "dataset/cervical_scrna/h5ad/cervical_scrna_adc_scc_marker_annotated.h5ad cell_type";;
  cosmx_nsclc) echo "dataset/lung_cancer_scrna_10x/lung_cancer_50k.h5ad Cell_Cluster_level1";;
  merfish_mouse_ileum) echo "dataset/mouse_ileum_scrna/h5ad/gut_scrna_gse92332_ileum_annotated.h5ad cell_type";;
esac; }

run_split () {
  local ds=$1; read ref col <<< "$(ref_for "$ds")"
  local CI="results/fig3_cross_platform_roi_benchmark/$ds/_common_inputs"
  local OUT="results/fig3_cross_platform_roi_benchmark/$ds/SPLIT_seg"
  echo "[$(date +%H:%M:%S)] SPLIT $ds START"
  $PY workflow/scripts/run_split.py \
    --transcripts dataset/$ds/roi_transcripts.parquet \
    --reference-h5ad "$ref" --reference-celltype-col "$col" \
    --common-inputs "$CI" --features-tsv "$CI/xenium_features.tsv" \
    --outdir "$OUT" --sample-name "$ds" --seed 1 --overwrite \
    --cores 4 --umi-min 10 --counts-min 10 > "$LOGD/split_$ds.log" 2>&1
  echo "[$(date +%H:%M:%S)] SPLIT $ds EXIT=$?"
}

run_celladmix () {
  local ds=$1; local cidir=$2; read ref col <<< "$(ref_for "$ds")"
  local CI="results/fig3_cross_platform_roi_benchmark/$ds/$cidir"
  local OUT="results/fig3_cross_platform_roi_benchmark/$ds/cellAdmix_seg"
  echo "[$(date +%H:%M:%S)] cellAdmix $ds START (ci=$cidir)"
  $PY workflow/scripts/run_celladmix.py \
    --transcripts dataset/$ds/roi_transcripts.parquet \
    --reference-h5ad "$ref" --reference-celltype-col "$col" \
    --common-inputs "$CI" --xenium-dir "$CI" \
    --clusters "$CI/xenium_cell_metadata_with_clusters.parquet" \
    --outdir "$OUT" --sample-name "$ds" --seed 1 --overwrite \
    --num-factors 8 --nmol-dsamp 3000 --n-cells-nmf 1000 --bridge-cells 150 --cores 4 \
    > "$LOGD/celladmix_$ds.log" 2>&1
  echo "[$(date +%H:%M:%S)] cellAdmix $ds EXIT=$?"
}

# Smaller datasets first so the concurrently-running atera Baysor finishes
# before atera's heavy RCTD/cellAdmix start (17 GB RAM ceiling).
# --- SPLIT for the 3 remaining datasets (x5k already done) ---
run_split cosmx_nsclc
run_split merfish_mouse_ileum
run_split atera_cervical

# --- cellAdmix for the 3 remaining datasets (x5k already done) ---
run_celladmix cosmx_nsclc    _common_inputs             # full 958 genes
run_celladmix merfish_mouse_ileum _common_inputs        # full 236 genes
run_celladmix atera_cervical _common_inputs_celladmix   # HVG ~1953 genes

echo "[$(date +%H:%M:%S)] ALL REMAINING DONE"

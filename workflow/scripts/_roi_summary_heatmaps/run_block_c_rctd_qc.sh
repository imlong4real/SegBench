#!/usr/bin/env bash
# Stage 2 (Block C): RCTD entropy / max-weight for every non-SPLIT entity.
# SPLIT reuses its internal RCTD; combined TRACER is intentionally skipped
# (RCTD is reported for TRACER-refined / TRACER-reconstructed separately).
#
# Reads the per-entity matrices written by build_matrices_and_metrics.py
# (_work/<ds>/<entity>.h5ad) and the dataset scRNA reference.
# Run with the tracer_benchmark_r env (spacexr available).
set -o pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RSCRIPT="$HOME/anaconda3/envs/tracer_benchmark_r/bin/Rscript"
RUN_RCTD="$REPO/workflow/scripts/run_rctd.R"
SUMMARY_NAME="${ROI_SUMMARY_NAME:-summary_heatmaps}"
WORK="$REPO/results/fig3_cross_platform_roi_benchmark/$SUMMARY_NAME/_work_qc"
METRICS="$REPO/results/fig3_cross_platform_roi_benchmark/$SUMMARY_NAME/metrics_qc"

# dataset -> reference (case statement; macOS system bash 3.2 has no assoc arrays)
ref_for() {  # echoes "<h5ad>::<celltype_col>"
  case "$1" in
    atera_cervical|xenium5k_cervical)
      echo "dataset/cervical_scrna/h5ad/cervical_scrna_adc_scc_marker_annotated.h5ad::cell_type" ;;
    cosmx_nsclc)
      echo "dataset/lung_cancer_scrna_10x/lung_cancer_50k.h5ad::Cell_Cluster_level1" ;;
    merfish_mouse_ileum)
      echo "dataset/mouse_ileum_scrna/h5ad/gut_scrna_gse92332_ileum_annotated.h5ad::cell_type" ;;
  esac
}

ENTITIES="${RCTD_ENTITIES:-original TRACER_refined TRACER_reconstructed baysor proseg segger celladmix}"
MAX_CORES="${RCTD_MAX_CORES:-1}"

for ds in atera_cervical xenium5k_cervical cosmx_nsclc merfish_mouse_ileum; do
  ref_spec="$(ref_for "$ds")"; ref_h5ad="${ref_spec%%::*}"; ref_col="${ref_spec##*::}"
  for ent in $ENTITIES; do
    spatial="$WORK/$ds/$ent.h5ad"
    [ -f "$spatial" ] || { echo "[skip] $ds/$ent (no matrix)"; continue; }
    outdir="$METRICS/$ds/$ent/rctd"
    if [ -f "$outdir/rctd_entropy_metrics.tsv" ]; then
      echo "[done] $ds/$ent already has RCTD"; continue
    fi
    mkdir -p "$outdir"
    # Sparse panels (MERFISH ~236 genes) need a lower reference min_UMI, else the
    # whole-transcriptome reference cells fall below spacexr's default 100 over
    # the shared panel and cell types drop under the 25-cell minimum.
    ref_min_umi=100
    case "$ds" in merfish_mouse_ileum) ref_min_umi=10 ;; esac
    exclude_arg=()
    case "$ds" in atera_cervical|xenium5k_cervical) exclude_arg=(--exclude-celltypes Unannotated) ;; esac
    echo "==== RCTD $ds / $ent (ref_min_umi=$ref_min_umi ${exclude_arg[*]-}; max_cores=$MAX_CORES) ===="
    "$RSCRIPT" "$RUN_RCTD" \
      --spatial-h5ad "$spatial" \
      --reference-h5ad "$REPO/$ref_h5ad" \
      --reference-celltype-col "$ref_col" \
      --outdir "$outdir" \
      --doublet-mode doublet \
      --umi-min 10 \
      --umi-min-sigma 20 \
      --reference-min-umi "$ref_min_umi" \
      "${exclude_arg[@]}" \
      --max-cores "$MAX_CORES" \
      > "$outdir/rctd.log" 2>&1
    if [ $? -eq 0 ]; then echo "  OK"; else echo "  FAILED (see $outdir/rctd.log)"; tail -3 "$outdir/rctd.log"; fi
  done
done
echo "ALL RCTD DONE"

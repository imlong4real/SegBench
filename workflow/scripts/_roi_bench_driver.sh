#!/usr/bin/env bash
# Driver for the cross-platform ROI segmentation benchmark (Baysor + proseg).
# cellAdmix + SPLIT are handled separately (need a generalized common_inputs).
#
# Usage: _roi_bench_driver.sh <tool> <dataset>
#   tool    = baysor | proseg
#   dataset = atera_cervical | xenium5k_cervical | cosmx_nsclc | merfish_mouse_ileum
set -euo pipefail
export PATH="$HOME/.julia/bin:$HOME/.cargo/bin:$PATH"
ROOT="${SEGBENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${SEGBENCH_PYTHON:-python3}"
cd "$ROOT"

tool=$1; ds=$2
TX="dataset/$ds/roi_transcripts.parquet"
OUT="results/fig3_cross_platform_roi_benchmark/$ds/${tool}_seg"

# Per-dataset scRNA reference + cell-type column.
case "$ds" in
  atera_cervical|xenium5k_cervical)
    REF="dataset/cervical_scrna/h5ad/cervical_scrna_adc_scc_marker_annotated.h5ad"; COL="cell_type";;
  cosmx_nsclc)
    REF="dataset/lung_cancer_scrna_10x/lung_cancer_50k.h5ad"; COL="Cell_Cluster_level1";;
  merfish_mouse_ileum)
    REF="dataset/mouse_ileum_scrna/h5ad/gut_scrna_gse92332_ileum_annotated.h5ad"; COL="cell_type";;
  *) echo "unknown dataset $ds"; exit 2;;
esac

# Per-platform Baysor cell radius (microns) + min molecules/cell, sized to the
# panel/density observed in the ROI (median tx/cell: atera 694, x5k 64, cosmx
# 272, merfish 67). Cell radius ~ sqrt(area_per_cell/pi); use a conservative 6µm.
case "$ds" in
  atera_cervical)      SCALE=6.0; MINMOL=20;;
  xenium5k_cervical)   SCALE=6.0; MINMOL=15;;
  cosmx_nsclc)         SCALE=7.0; MINMOL=20;;
  merfish_mouse_ileum) SCALE=8.0; MINMOL=15;;
esac

mkdir -p "$OUT"
if [ "$tool" = "proseg" ]; then
  $PY workflow/scripts/run_proseg.py \
    --transcripts "$TX" --reference-h5ad "$REF" --reference-celltype-col "$COL" \
    --outdir "$OUT" --sample-name "$ds" --seed 1 --overwrite --nthreads 4
elif [ "$tool" = "baysor" ]; then
  $PY workflow/scripts/run_baysor.py --mode run \
    --transcripts "$TX" --reference-h5ad "$REF" --reference-celltype-col "$COL" \
    --outdir "$OUT" --sample-name "$ds" --seed 1 --overwrite \
    --scale "$SCALE" --min-molecules-per-cell "$MINMOL" --n-threads 4
else
  echo "unknown tool $tool"; exit 2
fi

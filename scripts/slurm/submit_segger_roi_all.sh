#!/usr/bin/env bash
# Submit one Segger ROI job per dataset.
#
# Parameters chosen per dataset:
#   NUM_TX_TOKENS : gene embedding vocab — must be >= n_unique_genes per dataset
#   TILE_SIZE     : spatial tile in µm (all ROIs ~300–1000 µm extent)
#   MAX_EPOCHS    : 200 for all datasets
#
# MERFISH note: roi_transcripts.parquet stores pixel coordinates (~9.18 px/µm).
# tracer_seg/input_transcripts_um.parquet provides correct physical µm coords
# (611×1023 µm ROI, z=2.5–14.5 µm). Use ROI_TX_OVERRIDE for MERFISH.
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

BENCH_DATA="/home/lyuan13/scratchadeshpa6/benchmark_data"
MERFISH_UM="${BENCH_DATA}/merfish_mouse_ileum/tracer_seg/input_transcripts_um.parquet"

declare -A TOK=(   [atera_cervical]=17500 [cosmx_nsclc]=1000 [merfish_mouse_ileum]=250  [xenium5k_cervical]=5000 )
declare -A TILE=(  [atera_cervical]=25    [cosmx_nsclc]=25   [merfish_mouse_ileum]=25    [xenium5k_cervical]=25   )
declare -A EPOCHS=([atera_cervical]=200   [cosmx_nsclc]=200  [merfish_mouse_ileum]=200   [xenium5k_cervical]=200  )

for DS in atera_cervical cosmx_nsclc merfish_mouse_ileum xenium5k_cervical; do
    EXTRA_EXPORT=""
    if [[ "${DS}" == "merfish_mouse_ileum" ]]; then
        EXTRA_EXPORT=",ROI_TX_OVERRIDE=${MERFISH_UM}"
    fi
    JOBID=$(sbatch \
        --job-name="segger_${DS}" \
        --export="DATASET=${DS},NUM_TX_TOKENS=${TOK[$DS]},TILE_SIZE=${TILE[$DS]},MAX_EPOCHS=${EPOCHS[$DS]}${EXTRA_EXPORT}" \
        --output="logs/slurm/segger_roi_${DS}_%j.out" \
        --error="logs/slurm/segger_roi_${DS}_%j.err" \
        scripts/slurm/run_segger_roi_single.sbatch \
        | awk '{print $NF}')
    echo "Submitted ${DS}: job ${JOBID}  (tokens=${TOK[$DS]} tile=${TILE[$DS]}µm epochs=${EPOCHS[$DS]}${EXTRA_EXPORT:+ [µm override]})"
done
echo ""
echo "Monitor with: squeue -u lyuan13 | grep segger"
echo "Logs:         logs/slurm/segger_roi_<dataset>_<jobid>.out"

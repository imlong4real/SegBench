#!/bin/bash
# ============================================================================
# run_segger_roi_single.sbatch
# Run full Segger pipeline (bundle prep → preprocess → train → predict →
# standardize) on a single cross-platform ROI benchmark dataset.
#
# Submit via submit_segger_roi_all.sh, or manually:
#   sbatch --export=DATASET=atera_cervical,NUM_TX_TOKENS=1000,TILE_SIZE=25,MAX_EPOCHS=200 \
#          scripts/slurm/run_segger_roi_single.sbatch
#
# Required --export variables:
#   DATASET        atera_cervical | cosmx_nsclc | merfish_mouse_ileum | xenium5k_cervical
#   NUM_TX_TOKENS  gene embedding vocabulary size
#   TILE_SIZE      spatial tile size in µm (e.g. 25 for small ROIs, 100 for MERFISH)
#   MAX_EPOCHS     training epochs (200 for small, 50 for MERFISH)
#
# Outputs:
#   results/segger_roi/<DATASET>/xenium_bundle/
#   results/segger_roi/<DATASET>/preprocessed_data/
#   results/segger_roi/<DATASET>/trained_model/
#   results/segger_roi/<DATASET>/segger_output/
#   results/segger_roi/<DATASET>/final/
#   results/segger_roi/<DATASET>/benchmark/
# ============================================================================
#SBATCH --job-name=segger_roi
#SBATCH --partition=nvl,h100
#SBATCH --account=adeshpa6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/segger_roi_%x_%j.out
#SBATCH --error=logs/slurm/segger_roi_%x_%j.err

set -euo pipefail

# ── Validate required exports ─────────────────────────────────────────────────
: "${DATASET:?DATASET must be set via --export}"
: "${NUM_TX_TOKENS:?NUM_TX_TOKENS must be set via --export}"
: "${TILE_SIZE:?TILE_SIZE must be set via --export}"
: "${MAX_EPOCHS:?MAX_EPOCHS must be set via --export}"

# ── Environment diagnostics ───────────────────────────────────────────────────
echo "=== Environment diagnostics ==="
echo "  DATASET:       ${DATASET}"
echo "  NUM_TX_TOKENS: ${NUM_TX_TOKENS}"
echo "  TILE_SIZE:     ${TILE_SIZE}"
echo "  MAX_EPOCHS:    ${MAX_EPOCHS}"
echo "  ROI_TX_OVERRIDE: ${ROI_TX_OVERRIDE:-(none, using roi_transcripts.parquet)}"
echo "  hostname:      $(hostname)"
echo "  SLURM_JOB_ID:  ${SLURM_JOB_ID:-unset}"
echo "  date:          $(date)"
echo "  apptainer:     $(apptainer --version 2>&1 || echo NOT FOUND)"
echo "  nvidia-smi:    $(which nvidia-smi 2>&1 || echo NOT FOUND)"
echo "  /usr/bin/time: $([[ -x /usr/bin/time ]] && echo found || echo 'NOT FOUND (wall-clock fallback)')"
echo "================================================"

# ── Portable timing helper ────────────────────────────────────────────────────
run_timed() {
    local logfile="$1"; shift
    if [[ -x /usr/bin/time ]]; then
        /usr/bin/time -v -o "${logfile}" "$@"
    else
        echo "(NOTE: /usr/bin/time not found; peak RSS unavailable)" \
            >> "${logfile}" 2>/dev/null || true
        "$@"
    fi
}

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_DIR="/weka/home/lyuan13/segmentation_benchmark_pipeline"
BENCH_DATA="/home/lyuan13/scratchadeshpa6/benchmark_data"
EXTRA_PKGS="${REPO_DIR}/results/segger_smoke/.site_packages"

REPO_CONT="/home/lyuan13/segmentation_benchmark_pipeline"
EXTRA_PKGS_CONT="${REPO_CONT}/results/segger_smoke/.site_packages"

# Dataset-specific paths (host)
# ROI_TX_OVERRIDE allows passing an alternative coordinate file (e.g.
# tracer_seg/input_transcripts_um.parquet for MERFISH, where
# roi_transcripts.parquet stores pixel rather than µm coordinates).
if [[ -n "${ROI_TX_OVERRIDE:-}" ]]; then
    ROI_TX="${ROI_TX_OVERRIDE}"
    echo "  ROI_TX (override): ${ROI_TX}"
else
    ROI_TX="${BENCH_DATA}/${DATASET}/roi_transcripts.parquet"
fi
OUT_ROOT="${REPO_DIR}/results/segger_roi/${DATASET}"
BUNDLE_DIR="${OUT_ROOT}/xenium_bundle"
PREPROC_DIR="${OUT_ROOT}/preprocessed_data"
MODEL_DIR="${OUT_ROOT}/trained_model"
PREDICT_DIR="${OUT_ROOT}/segger_output"
FINAL_DIR="${OUT_ROOT}/final"
BENCHMARK_DIR="${OUT_ROOT}/benchmark"
TIMING_DIR="${BENCHMARK_DIR}/timing_raw"

# Container-visible counterparts (Apptainer maps /weka/home/lyuan13 → /home/lyuan13)
BUNDLE_DIR_CONT="${REPO_CONT}/results/segger_roi/${DATASET}/xenium_bundle"
PREPROC_DIR_CONT="${REPO_CONT}/results/segger_roi/${DATASET}/preprocessed_data"
MODEL_DIR_CONT="${REPO_CONT}/results/segger_roi/${DATASET}/trained_model"
PREDICT_DIR_CONT="${REPO_CONT}/results/segger_roi/${DATASET}/segger_output"

cd "${REPO_DIR}"
mkdir -p "${BUNDLE_DIR}" "${PREPROC_DIR}" "${MODEL_DIR}" "${PREDICT_DIR}" \
         "${FINAL_DIR}" "${BENCHMARK_DIR}" "${TIMING_DIR}" logs/slurm

# ── Guards ────────────────────────────────────────────────────────────────────
if [[ ! -f "${ROI_TX}" ]]; then
    echo "ERROR: roi_transcripts.parquet not found: ${ROI_TX}"
    exit 1
fi
if [[ ! -s "${REPO_DIR}/containers/python_cuda.sif" ]]; then
    echo "ERROR: containers/python_cuda.sif not found."
    exit 1
fi

# ── Apptainer wrapper ─────────────────────────────────────────────────────────
WRAP_DIR="/tmp/segger_roi_${DATASET}_$$"
mkdir -p "${WRAP_DIR}"
trap "rm -rf ${WRAP_DIR}" EXIT
cat > "${WRAP_DIR}/run.sh" << WRAP
#!/bin/bash
export LD_LIBRARY_PATH="/opt/conda/envs/segger_cuda/lib:\${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${EXTRA_PKGS_CONT}:\${PYTHONPATH:-}"
exec /opt/conda/envs/segger_cuda/bin/python3 "\$@"
WRAP
chmod +x "${WRAP_DIR}/run.sh"

segger_exec() {
    apptainer exec --nv \
        --pwd "${REPO_CONT}" \
        containers/python_cuda.sif "${WRAP_DIR}/run.sh" "$@"
}

# Host Python (for non-GPU pre/post steps)
PYBIN_HOST="/apps/software/spack/gcc/8.5.0/anaconda3/2024.02-1-jbzsrx3q6jyqoh2f3wy6n7oxhdav64ml/bin/python3"

# ── GPU info ──────────────────────────────────────────────────────────────────
echo ""; echo "=== nvidia-smi ==="; nvidia-smi
GPU_MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
GPU_MEM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")
GPU_MEM_GB=$(awk "BEGIN{printf \"%.1f\", ${GPU_MEM_MIB}/1024}" 2>/dev/null || echo "unknown")
echo "  GPU: ${GPU_MODEL}  VRAM: ${GPU_MEM_MIB} MiB (${GPU_MEM_GB} GB)"

# ── CUDA preflight ────────────────────────────────────────────────────────────
echo ""; echo "=== CUDA preflight ==="
segger_exec -c "
import torch, sys
assert torch.cuda.is_available(), 'CUDA not available'
print('CUDA OK — device_count =', torch.cuda.device_count())
print('Device 0:', torch.cuda.get_device_name(0))
"

# ── Start GPU memory monitor ──────────────────────────────────────────────────
GPU_DMON_LOG="${TIMING_DIR}/gpu_dmon.log"
echo "# nvidia-smi dmon (started: $(date))" > "${GPU_DMON_LOG}"
nvidia-smi dmon -d 5 -s mu -o T >> "${GPU_DMON_LOG}" &
DMON_PID=$!
cleanup() { kill "${DMON_PID}" 2>/dev/null || true; }
trap 'cleanup; rm -rf ${WRAP_DIR}' EXIT

# ============================================================
# Step 1: Build Xenium-like input bundle from roi_transcripts
# ============================================================
echo ""; echo "=== Step 1: Prepare Segger input bundle (${DATASET}) ==="
# Determine platform string from roi_summary.json if present
PLATFORM=$(python3 -c "
import json, pathlib
p = pathlib.Path('${BENCH_DATA}/${DATASET}/roi_summary.json')
d = json.loads(p.read_text()) if p.exists() else {}
print(d.get('platform','unknown'))
" 2>/dev/null || echo "unknown")
echo "  Platform: ${PLATFORM}"

"${PYBIN_HOST}" workflow/scripts/_segmentation/prepare_roi_segger_bundle.py \
    --input   "${ROI_TX}" \
    --outdir  "${BUNDLE_DIR}" \
    --dataset "${DATASET}" \
    --platform "${PLATFORM}"

echo "[$(date)] Bundle prepared: ${BUNDLE_DIR}"
ls -lh "${BUNDLE_DIR}/"

# ============================================================
# Step 1b: Clear stale preprocessed tiles and model logs
# (Segger's create_dataset_fast.py asserts dirs are empty)
# ============================================================
echo ""; echo "=== Step 1b: Clearing stale preprocessed_data and model version ==="
rm -rf "${PREPROC_DIR}"
mkdir -p "${PREPROC_DIR}"
# Remove only empty or failed PL version dirs (leave any with checkpoints)
for vdir in "${MODEL_DIR}"/lightning_logs/version_*/; do
    if [[ ! -d "${vdir}/checkpoints" ]] || [[ -z "$(ls "${vdir}/checkpoints/"*.ckpt 2>/dev/null)" ]]; then
        echo "  Removing empty version dir: ${vdir}"
        rm -rf "${vdir}"
    fi
done
echo "  Done"

# ============================================================
# Step 2: Segger preprocessing (create tiles)
# ============================================================
echo ""; echo "=== Step 2: Segger preprocess (tile_size=${TILE_SIZE}µm) ==="
PREPROC_TIME_LOG="${TIMING_DIR}/preprocess_time_v.txt"
PREPROC_WALL_START=$(date +%s)
run_timed "${PREPROC_TIME_LOG}" \
    apptainer exec --nv \
        --pwd "${REPO_CONT}" \
        containers/python_cuda.sif "${WRAP_DIR}/run.sh" \
        /opt/segger_dev/src/segger/cli/create_dataset_fast.py \
        --base_dir    "${BUNDLE_DIR_CONT}" \
        --data_dir    "${PREPROC_DIR_CONT}" \
        --sample_type xenium \
        --tile_width  "${TILE_SIZE}" \
        --tile_height "${TILE_SIZE}" \
        --n_workers   8 \
        2>&1
PREPROC_WALL_END=$(date +%s)
PREPROC_WALL_SEC=$(( PREPROC_WALL_END - PREPROC_WALL_START ))
echo "[$(date)] Preprocess done in ${PREPROC_WALL_SEC}s"
echo "PREPROC_WALL_SECONDS=${PREPROC_WALL_SEC}" >> "${BENCHMARK_DIR}/wall_times.env"

# Count tiles
N_TILES=$(find "${PREPROC_DIR}" -name "*.pt" 2>/dev/null | wc -l)
echo "[$(date)] Tiles created: ${N_TILES}"
echo "N_TILES=${N_TILES}" >> "${BENCHMARK_DIR}/wall_times.env"

if [[ "${N_TILES}" -lt 5 ]]; then
    echo "ERROR: Too few tiles (${N_TILES}). Check dataset spatial extent vs TILE_SIZE=${TILE_SIZE}."
    echo "  ROI extent:"
    "${PYBIN_HOST}" -c "
import pandas as pd
df = pd.read_parquet('${ROI_TX}')
print(f'  x: {df.x.min():.1f} - {df.x.max():.1f}  range={df.x.max()-df.x.min():.1f}')
print(f'  y: {df.y.min():.1f} - {df.y.max():.1f}  range={df.y.max()-df.y.min():.1f}')
print(f'  n_transcripts: {len(df):,}')
"
    exit 1
fi

echo "[$(date)] Tile split summary:"
find "${PREPROC_DIR}" -name "*.pt" | awk -F/ '{print $(NF-2)}' | sort | uniq -c

# ============================================================
# Step 3: Train model
# ============================================================
echo ""; echo "=== Step 3: Segger train (max_epochs=${MAX_EPOCHS}, num_tx_tokens=${NUM_TX_TOKENS}) ==="
TRAIN_ARGS=(
    /opt/segger_dev/src/segger/cli/train_model.py
    --dataset_dir   "${PREPROC_DIR_CONT}"
    --models_dir    "${MODEL_DIR_CONT}"
    --sample_tag    "${DATASET}"
    --num_tx_tokens "${NUM_TX_TOKENS}"
    --accelerator   cuda
    --devices       1
    --max_epochs    "${MAX_EPOCHS}"
    --batch_size    4
    --num_workers   8
    --save_best_model True
    --precision     16-mixed
)
echo "Training command:"; printf '  segger_exec %s\n' "${TRAIN_ARGS[@]}"; echo ""

TRAIN_TIME_LOG="${TIMING_DIR}/train_time_v.txt"
TRAIN_WALL_START=$(date +%s)
run_timed "${TRAIN_TIME_LOG}" \
    apptainer exec --nv \
        --pwd "${REPO_CONT}" \
        containers/python_cuda.sif "${WRAP_DIR}/run.sh" \
        "${TRAIN_ARGS[@]}" 2>&1
TRAIN_WALL_END=$(date +%s)
TRAIN_WALL_SEC=$(( TRAIN_WALL_END - TRAIN_WALL_START ))
echo "[$(date)] Training done in ${TRAIN_WALL_SEC}s"
echo "TRAIN_WALL_SECONDS=${TRAIN_WALL_SEC}" >> "${BENCHMARK_DIR}/wall_times.env"

# Verify checkpoint
CKPT=$(find "${MODEL_DIR}" -name "*.ckpt" 2>/dev/null | sort | tail -1)
if [[ -z "${CKPT}" ]]; then
    echo "ERROR: No checkpoint found under ${MODEL_DIR}"
    exit 1
fi
echo "BEST_CKPT=${CKPT}" >> "${BENCHMARK_DIR}/wall_times.env"
echo "[$(date)] Best checkpoint: ${CKPT}"

# ============================================================
# Step 3b: Patch edge_label on all tile splits
# ============================================================
echo ""; echo "=== Step 3b: Patch edge_label on preprocessed tiles ==="
PATCH_REPORT="${BENCHMARK_DIR}/edge_label_patch_report.csv"
apptainer exec --nv \
    --pwd "${REPO_CONT}" \
    containers/python_cuda.sif "${WRAP_DIR}/run.sh" \
    workflow/scripts/_segmentation/patch_segger_edge_label.py \
    --dataset-dir "${PREPROC_DIR_CONT}" \
    --splits train_tiles val_tiles test_tiles \
    --report-csv "${PATCH_REPORT}" 2>&1

# ============================================================
# Step 4: Predict
# ============================================================
echo ""; echo "=== Step 4: Segger predict ==="
# Stop GPU monitor before prediction to avoid interference
kill "${DMON_PID}" 2>/dev/null || true
trap 'rm -rf ${WRAP_DIR}' EXIT

# Restart monitor for prediction
nvidia-smi dmon -d 5 -s mu -o T >> "${GPU_DMON_LOG}" &
DMON_PID=$!
trap 'kill "${DMON_PID}" 2>/dev/null || true; rm -rf ${WRAP_DIR}' EXIT

# Use ROI bundle transcripts as the target for assignment
TX_FILE_CONT="${BUNDLE_DIR_CONT}/transcripts.parquet"

PREDICT_ARGS=(
    /opt/segger_dev/src/segger/cli/predict_fast.py
    --segger_data_dir  "${PREPROC_DIR_CONT}"
    --models_dir       "${MODEL_DIR_CONT}"
    --benchmarks_dir   "${PREDICT_DIR_CONT}"
    --transcripts_file "${TX_FILE_CONT}"
    --batch_size       4
    --num_workers      0
    --model_version    0
    --knn_method       kd_tree
    --cell_id_col      segger_cell_id
    --use_cc           False
)
echo "Predict command:"; printf '  segger_exec %s\n' "${PREDICT_ARGS[@]}"; echo ""

PREDICT_TIME_LOG="${TIMING_DIR}/predict_time_v.txt"
PREDICT_WALL_START=$(date +%s)
run_timed "${PREDICT_TIME_LOG}" \
    apptainer exec --nv \
        --pwd "${REPO_CONT}" \
        containers/python_cuda.sif "${WRAP_DIR}/run.sh" \
        "${PREDICT_ARGS[@]}" 2>&1
PREDICT_WALL_END=$(date +%s)
PREDICT_WALL_SEC=$(( PREDICT_WALL_END - PREDICT_WALL_START ))
echo "[$(date)] Prediction done in ${PREDICT_WALL_SEC}s"
echo "PREDICT_WALL_SECONDS=${PREDICT_WALL_SEC}" >> "${BENCHMARK_DIR}/wall_times.env"

kill "${DMON_PID}" 2>/dev/null || true
trap 'rm -rf ${WRAP_DIR}' EXIT

# ============================================================
# Step 5: Flatten prediction output and assemble final/
# ============================================================
echo ""; echo "=== Step 5: Assemble final/ directory ==="
PRED_SUBDIR=$(find "${PREDICT_DIR}" -maxdepth 1 -name "segger_embedding_*" -type d 2>/dev/null | sort | tail -1)
if [[ -z "${PRED_SUBDIR}" ]]; then
    echo "ERROR: No segger_embedding_* subdir found under ${PREDICT_DIR}"
    ls -lh "${PREDICT_DIR}/" 2>/dev/null || true
    exit 1
fi
echo "[$(date)] Prediction subdir: ${PRED_SUBDIR}"

for f in segger_transcripts.parquet segger_adata.h5ad segmentation_log.json; do
    if [[ -f "${PRED_SUBDIR}/${f}" ]]; then
        cp "${PRED_SUBDIR}/${f}" "${FINAL_DIR}/${f}"
        echo "  Copied: ${f}"
    else
        echo "  WARNING: ${f} not found in ${PRED_SUBDIR}"
    fi
done

# Concatenate partitioned transcripts_df.parquet if present
if [[ -d "${PRED_SUBDIR}/transcripts_df.parquet" ]]; then
    "${PYBIN_HOST}" -c "
import pyarrow as pa, pyarrow.parquet as pq, pathlib
src = pathlib.Path('${PRED_SUBDIR}/transcripts_df.parquet')
parts = sorted(src.glob('*.parquet'))
print(f'Concatenating {len(parts)} transcripts_df parts')
pq.write_table(pa.concat_tables([pq.read_table(p) for p in parts]),
               '${FINAL_DIR}/transcripts_df.parquet')
print('Done: transcripts_df.parquet')
"
elif [[ -f "${PRED_SUBDIR}/transcripts_df.parquet" ]]; then
    cp "${PRED_SUBDIR}/transcripts_df.parquet" "${FINAL_DIR}/transcripts_df.parquet"
fi

echo ""; echo "=== Final directory contents ==="; ls -lh "${FINAL_DIR}/"

# ============================================================
# Step 6: Standardize for get_metric.py
# ============================================================
echo ""; echo "=== Step 6: Standardize transcript output ==="
"${PYBIN_HOST}" workflow/scripts/_segmentation/standardize_segger_output.py \
    --input  "${FINAL_DIR}/segger_transcripts.parquet" \
    --output "${FINAL_DIR}/segger_transcripts_standardized.parquet" \
    --report "${BENCHMARK_DIR}/schema_validation_report.json"

# ============================================================
# Step 7: Collect benchmark metadata
# ============================================================
echo ""; echo "=== Step 7: Collect runtime / convergence metadata ==="
"${PYBIN_HOST}" workflow/scripts/_segmentation/collect_segger_runtime_metadata.py \
    --models-dir             "${MODEL_DIR}" \
    --outdir                 "${BENCHMARK_DIR}" \
    --model-version          0 \
    --time-v-log             "${TRAIN_TIME_LOG}" \
    --predict-time-v-log     "${PREDICT_TIME_LOG}" \
    --preprocess-time-v-log  "${PREPROC_TIME_LOG}" \
    --gpu-dmon-log           "${GPU_DMON_LOG}" \
    --gpu-model              "${GPU_MODEL}" \
    --slurm-job-id           "${SLURM_JOB_ID:-local}" \
    --container-sif          "${REPO_DIR}/containers/python_cuda.sif" \
    --dataset                "${DATASET}" \
    --run-label              "final_convergence"

# Patch wall-clock times from env into runtime_memory.json
"${PYBIN_HOST}" -c "
import json, pathlib
p = pathlib.Path('${BENCHMARK_DIR}/runtime_memory.json')
d = json.loads(p.read_text())
# Read wall_times.env
wt = {}
we = pathlib.Path('${BENCHMARK_DIR}/wall_times.env')
if we.exists():
    for line in we.read_text().splitlines():
        if '=' in line:
            k, _, v = line.partition('=')
            wt[k.strip()] = v.strip()
d['stages']['preprocess']['wall_clock_seconds'] = int(wt.get('PREPROC_WALL_SECONDS', 0)) or None
d['stages']['train']['wall_clock_seconds']      = int(wt.get('TRAIN_WALL_SECONDS', 0)) or None
d['stages']['predict']['wall_clock_seconds']    = int(wt.get('PREDICT_WALL_SECONDS', 0)) or None
total = sum(v for v in [d['stages'][s]['wall_clock_seconds'] for s in ('preprocess','train','predict')] if v)
d['runtime_seconds'] = total or None
d['n_tiles'] = int(wt.get('N_TILES', 0)) or None
p.write_text(json.dumps(d, indent=2))
print(f'Patched runtime_memory.json: total={total}s')
"

# ============================================================
# Step 8: Write validation JSON
# ============================================================
"${PYBIN_HOST}" -c "
import json, pathlib, datetime
final_dir = pathlib.Path('${FINAL_DIR}')
val = {
    'timestamp': datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
    'dataset': '${DATASET}',
    'run_label': 'segger_roi_final',
    'slurm_job_id': '${SLURM_JOB_ID:-local}',
    'gpu_model': '${GPU_MODEL}',
    'checks': {f: {'ok': (final_dir/f).exists()}
               for f in ['segger_transcripts.parquet',
                         'segger_transcripts_standardized.parquet',
                         'segger_adata.h5ad','segmentation_log.json']},
}
val['status'] = 'PASS' if all(v['ok'] for v in val['checks'].values()) else 'FAIL'
val['blockers'] = [k for k, v in val['checks'].items() if not v['ok']]
with open(final_dir / 'segger_validation.json', 'w') as f:
    json.dump(val, f, indent=2)
print('segger_validation.json status:', val['status'])
if val['blockers']:
    print('  BLOCKERS:', val['blockers'])
"

# ============================================================
# Step 8b: Write markdown parameter / config summary
# ============================================================
echo ""; echo "=== Step 8b: Write benchmark config summary ==="
"${PYBIN_HOST}" - << PYSUMMARY
import json, pathlib, datetime

bench   = pathlib.Path('${BENCHMARK_DIR}')
final   = pathlib.Path('${FINAL_DIR}')
dataset = '${DATASET}'
gpu     = '${GPU_MODEL}'
job_id  = '${SLURM_JOB_ID:-local}'

# --- runtime_memory.json (may not exist if metadata step was skipped) ---
rt_path = bench / 'runtime_memory.json'
rt = json.loads(rt_path.read_text()) if rt_path.exists() else {}
stages  = rt.get('stages', {})
pre_s   = stages.get('preprocess', {}).get('wall_clock_seconds')
trn_s   = stages.get('train',      {}).get('wall_clock_seconds')
prd_s   = stages.get('predict',    {}).get('wall_clock_seconds')
total_s = rt.get('runtime_seconds') or (
    sum(x for x in [pre_s, trn_s, prd_s] if x) or None)

def _fmt(s):
    if s is None: return 'N/A'
    m, sec = divmod(int(s), 60)
    return f'{m}m {sec}s ({s}s)'

# --- convergence ---
conv_path = bench / 'training_convergence.tsv'
best_epoch = rt.get('training', {}).get('best_epoch', 'N/A')
best_loss  = rt.get('training', {}).get('best_val_loss', 'N/A')
best_auroc = rt.get('training', {}).get('best_val_auroc', 'N/A')
best_f1    = rt.get('training', {}).get('best_val_f1', 'N/A')

# --- assignment stats ---
std_pq = final / 'segger_transcripts_standardized.parquet'
try:
    import pandas as pd
    df = pd.read_parquet(std_pq, columns=['cell_id'])
    n_total = len(df)
    UNASSIGNED = {'UNASSIGNED','Unassigned','unassigned','DROP','nan','None','','0','-1','NA'}
    n_assigned = int((~df['cell_id'].astype(str).isin(UNASSIGNED)).sum())
    n_cells    = int(df['cell_id'].astype(str)[~df['cell_id'].astype(str).isin(UNASSIGNED)].nunique())
    pct_asgn   = f'{100*n_assigned/n_total:.1f}%' if n_total else 'N/A'
except Exception as e:
    n_total = n_assigned = n_cells = 'N/A'; pct_asgn = f'(error: {e})'

# --- GPU peak memory ---
gpu_mem_mib = rt.get('gpu_peak_memory_mib')
gpu_mem_str = f'{gpu_mem_mib} MiB ({gpu_mem_mib/1024:.2f} GB)' if isinstance(gpu_mem_mib, (int,float)) else 'N/A'

# --- peak RAM (not tracked without /usr/bin/time) ---
rss_gb = rt.get('peak_rss_gb_observed')
rss_str = f'{rss_gb:.2f} GB' if isinstance(rss_gb, float) else 'not measured (no /usr/bin/time)'

n_tiles = rt.get('n_tiles', '${N_TILES:-N/A}')

md = f"""# Segger ROI Benchmark — {dataset}

Generated: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
SLURM job:  {job_id}
GPU:        {gpu}

## Segger Parameters

| Parameter         | Value                  |
|-------------------|------------------------|
| num_tx_tokens     | ${NUM_TX_TOKENS}       |
| tile_width/height | ${TILE_SIZE} µm        |
| max_epochs        | ${MAX_EPOCHS}          |
| batch_size        | 4                      |
| num_workers       | 8                      |
| precision         | 16-mixed               |
| knn_method        | kd_tree                |
| use_cc            | False                  |
| accelerator       | cuda                   |

## Dataset

| Item              | Value                  |
|-------------------|------------------------|
| dataset           | {dataset}              |
| n_tiles           | {n_tiles}              |
| n_transcripts     | {n_total:,} |
| n_assigned        | {n_assigned:,} ({pct_asgn}) |
| n_unique_cells    | {n_cells:,} |

## Runtime

| Stage       | Wall-clock        |
|-------------|-------------------|
| Preprocess  | {_fmt(pre_s)}     |
| Train       | {_fmt(trn_s)}     |
| Predict     | {_fmt(prd_s)}     |
| **Total**   | **{_fmt(total_s)}** |

## Memory

| Resource          | Peak                   |
|-------------------|------------------------|
| GPU memory        | {gpu_mem_str}          |
| CPU RAM           | {rss_str}              |

## Convergence (best epoch)

| Metric            | Value                  |
|-------------------|------------------------|
| best_epoch        | {best_epoch}           |
| best_val_loss     | {best_loss}            |
| best_val_auroc    | {best_auroc}           |
| best_val_f1       | {best_f1}              |

## Output Files

| File | Path |
|------|------|
| Standardized transcripts | \`{final}/segger_transcripts_standardized.parquet\` |
| Runtime metadata         | \`{bench}/runtime_memory.json\` |
| Convergence table        | \`{bench}/training_convergence.tsv\` |
| Schema validation        | \`{bench}/schema_validation_report.json\` |
| Edge-label patch report  | \`{bench}/edge_label_patch_report.csv\` |
"""

out = bench / 'segger_benchmark_summary.md'
out.write_text(md)
print(f'Wrote: {out}')
print(md)
PYSUMMARY

# ============================================================
# Done
# ============================================================
echo ""
echo "============================================================"
echo "[$(date)] Segger ROI run COMPLETE: ${DATASET}"
echo "============================================================"
echo ""
echo "Final outputs:"
ls -lh "${FINAL_DIR}/"
echo ""
echo "Benchmark metadata:"
ls -lh "${BENCHMARK_DIR}/"
echo ""
WALL_TOTAL=$(( ${PREPROC_WALL_SEC:-0} + ${TRAIN_WALL_SEC:-0} + ${PREDICT_WALL_SEC:-0} ))
echo "Wall-clock summary:"
echo "  Preprocess: ${PREPROC_WALL_SEC:-N/A}s"
echo "  Train:      ${TRAIN_WALL_SEC:-N/A}s"
echo "  Predict:    ${PREDICT_WALL_SEC:-N/A}s"
echo "  Total:      ${WALL_TOTAL}s"
echo ""
echo "Standardized transcript file (ready for get_metric.py):"
echo "  ${FINAL_DIR}/segger_transcripts_standardized.parquet"

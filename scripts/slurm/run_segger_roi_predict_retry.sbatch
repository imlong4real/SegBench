#!/bin/bash
# ============================================================================
# run_segger_roi_predict_retry.sbatch
# Re-run only the prediction + post-processing stages for a completed ROI
# training run. Used when training succeeded but prediction failed (e.g.
# KeyError: 'edge_label' on test tiles).
#
# Applies patch_segger_edge_label.py before predict_fast.py.
#
# Submit:
#   sbatch --export=DATASET=merfish_mouse_ileum,NUM_TX_TOKENS=250,TILE_SIZE=100,MAX_EPOCHS=50 \
#          scripts/slurm/run_segger_roi_predict_retry.sbatch
# ============================================================================
#SBATCH --job-name=segger_retry
#SBATCH --partition=nvl,h100
#SBATCH --account=adeshpa6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm/segger_retry_%x_%j.out
#SBATCH --error=logs/slurm/segger_retry_%x_%j.err

set -euo pipefail

: "${DATASET:?DATASET must be set via --export}"
: "${NUM_TX_TOKENS:?NUM_TX_TOKENS must be set via --export}"
: "${TILE_SIZE:?TILE_SIZE must be set via --export}"
: "${MAX_EPOCHS:?MAX_EPOCHS must be set via --export}"

echo "=== predict-retry diagnostics ==="
echo "  DATASET:       ${DATASET}"
echo "  NUM_TX_TOKENS: ${NUM_TX_TOKENS}"
echo "  TILE_SIZE:     ${TILE_SIZE}"
echo "  MAX_EPOCHS:    ${MAX_EPOCHS}"
echo "  hostname:      $(hostname)"
echo "  SLURM_JOB_ID:  ${SLURM_JOB_ID:-unset}"
echo "  date:          $(date)"
echo "================================================"

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

REPO_DIR="${SEGBENCH_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}"
EXTRA_PKGS="${REPO_DIR}/results/segger_smoke/.site_packages"
REPO_CONT="${SEGBENCH_REPO_CONT:-$REPO_DIR}"
EXTRA_PKGS_CONT="${REPO_CONT}/results/segger_smoke/.site_packages"

OUT_ROOT="${REPO_DIR}/results/segger_roi/${DATASET}"
BUNDLE_DIR="${OUT_ROOT}/xenium_bundle"
PREPROC_DIR="${OUT_ROOT}/preprocessed_data"
MODEL_DIR="${OUT_ROOT}/trained_model"
PREDICT_DIR="${OUT_ROOT}/segger_output"
FINAL_DIR="${OUT_ROOT}/final"
BENCHMARK_DIR="${OUT_ROOT}/benchmark"
TIMING_DIR="${BENCHMARK_DIR}/timing_raw"

BUNDLE_DIR_CONT="${REPO_CONT}/results/segger_roi/${DATASET}/xenium_bundle"
PREPROC_DIR_CONT="${REPO_CONT}/results/segger_roi/${DATASET}/preprocessed_data"
MODEL_DIR_CONT="${REPO_CONT}/results/segger_roi/${DATASET}/trained_model"
PREDICT_DIR_CONT="${REPO_CONT}/results/segger_roi/${DATASET}/segger_output"

cd "${REPO_DIR}"
mkdir -p "${PREDICT_DIR}" "${FINAL_DIR}" "${BENCHMARK_DIR}" "${TIMING_DIR}" logs/slurm

# ── Verify training checkpoint exists ────────────────────────────────────────
CKPT=$(find "${MODEL_DIR}" -name "*.ckpt" 2>/dev/null | sort | tail -1)
if [[ -z "${CKPT}" ]]; then
    echo "ERROR: No checkpoint found under ${MODEL_DIR}. Run training first."
    exit 1
fi
echo "[$(date)] Checkpoint: ${CKPT}"

# ── Apptainer wrapper ─────────────────────────────────────────────────────────
WRAP_DIR="/tmp/segger_retry_${DATASET}_$$"
mkdir -p "${WRAP_DIR}"
trap "rm -rf ${WRAP_DIR}" EXIT
cat > "${WRAP_DIR}/run.sh" << WRAP
#!/bin/bash
export LD_LIBRARY_PATH="/opt/conda/envs/segger_cuda/lib:\${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${EXTRA_PKGS_CONT}:\${PYTHONPATH:-}"
exec /opt/conda/envs/segger_cuda/bin/python3 "\$@"
WRAP
chmod +x "${WRAP_DIR}/run.sh"

PYBIN_HOST="/apps/software/spack/gcc/8.5.0/anaconda3/2024.02-1-jbzsrx3q6jyqoh2f3wy6n7oxhdav64ml/bin/python3"

# ── GPU info ──────────────────────────────────────────────────────────────────
nvidia-smi
GPU_MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
GPU_MEM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")
GPU_MEM_GB=$(awk "BEGIN{printf \"%.1f\", ${GPU_MEM_MIB}/1024}" 2>/dev/null || echo "unknown")
echo "  GPU: ${GPU_MODEL}  VRAM: ${GPU_MEM_MIB} MiB (${GPU_MEM_GB} GB)"

# ── GPU monitor ───────────────────────────────────────────────────────────────
GPU_DMON_LOG="${TIMING_DIR}/gpu_dmon_retry.log"
echo "# nvidia-smi dmon retry (started: $(date))" > "${GPU_DMON_LOG}"
nvidia-smi dmon -d 5 -s mu -o T >> "${GPU_DMON_LOG}" &
DMON_PID=$!
trap 'kill "${DMON_PID}" 2>/dev/null || true; rm -rf ${WRAP_DIR}' EXIT

# ============================================================
# Step A: Patch edge_label on preprocessed tiles
# ============================================================
echo ""; echo "=== Step A: Patch edge_label on preprocessed tiles ==="
PATCH_REPORT="${BENCHMARK_DIR}/edge_label_patch_report.csv"
apptainer exec --nv \
    --pwd "${REPO_CONT}" \
    containers/python_cuda.sif "${WRAP_DIR}/run.sh" \
    workflow/scripts/_segmentation/patch_segger_edge_label.py \
    --dataset-dir "${PREPROC_DIR_CONT}" \
    --splits train_tiles val_tiles test_tiles \
    --report-csv "${PATCH_REPORT}" 2>&1

# ============================================================
# Step B: Predict
# ============================================================
echo ""; echo "=== Step B: Segger predict ==="
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
echo "Predict command:"; printf '  %s\n' "${PREDICT_ARGS[@]}"; echo ""

PREDICT_TIME_LOG="${TIMING_DIR}/predict_time_v_retry.txt"
PREDICT_WALL_START=$(date +%s)
run_timed "${PREDICT_TIME_LOG}" \
    apptainer exec --nv \
        --pwd "${REPO_CONT}" \
        containers/python_cuda.sif "${WRAP_DIR}/run.sh" \
        "${PREDICT_ARGS[@]}" 2>&1
PREDICT_WALL_END=$(date +%s)
PREDICT_WALL_SEC=$(( PREDICT_WALL_END - PREDICT_WALL_START ))
echo "[$(date)] Prediction done in ${PREDICT_WALL_SEC}s"

kill "${DMON_PID}" 2>/dev/null || true
trap 'rm -rf ${WRAP_DIR}' EXIT

# ============================================================
# Step C: Assemble final/
# ============================================================
echo ""; echo "=== Step C: Assemble final/ directory ==="
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
# Step D: Standardize
# ============================================================
echo ""; echo "=== Step D: Standardize transcript output ==="
"${PYBIN_HOST}" workflow/scripts/_segmentation/standardize_segger_output.py \
    --input  "${FINAL_DIR}/segger_transcripts.parquet" \
    --output "${FINAL_DIR}/segger_transcripts_standardized.parquet" \
    --report "${BENCHMARK_DIR}/schema_validation_report.json"

# ============================================================
# Step E: Collect runtime / convergence metadata
# ============================================================
echo ""; echo "=== Step E: Collect runtime / convergence metadata ==="
TRAIN_TIME_LOG="${TIMING_DIR}/train_time_v.txt"
PREPROC_TIME_LOG="${TIMING_DIR}/preprocess_time_v.txt"

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

# Patch predict wall-clock into runtime_memory.json
"${PYBIN_HOST}" -c "
import json, pathlib
p  = pathlib.Path('${BENCHMARK_DIR}/runtime_memory.json')
d  = json.loads(p.read_text())
we = pathlib.Path('${BENCHMARK_DIR}/wall_times.env')
wt = {}
if we.exists():
    for line in we.read_text().splitlines():
        if '=' in line:
            k, _, v = line.partition('=')
            wt[k.strip()] = v.strip()
# Preserve existing preprocess/train times; update predict
d['stages']['preprocess']['wall_clock_seconds'] = int(wt.get('PREPROC_WALL_SECONDS', 0)) or None
d['stages']['train']['wall_clock_seconds']      = int(wt.get('TRAIN_WALL_SECONDS', 0)) or None
d['stages']['predict']['wall_clock_seconds']    = ${PREDICT_WALL_SEC}
total = sum(x for x in [d['stages'][s]['wall_clock_seconds']
                         for s in ('preprocess','train','predict')] if x)
d['runtime_seconds'] = total or None
d['n_tiles'] = int(wt.get('N_TILES', 0)) or None
p.write_text(json.dumps(d, indent=2))
print(f'Patched runtime_memory.json: total={total}s, predict=${PREDICT_WALL_SEC}s')
"

# ============================================================
# Step F: Validation JSON
# ============================================================
"${PYBIN_HOST}" -c "
import json, pathlib, datetime
final_dir = pathlib.Path('${FINAL_DIR}')
val = {
    'timestamp': datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
    'dataset': '${DATASET}',
    'run_label': 'segger_roi_predict_retry',
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
# Step G: Markdown config summary
# ============================================================
echo ""; echo "=== Step G: Write benchmark config summary ==="
"${PYBIN_HOST}" - << PYSUMMARY
import json, pathlib, datetime

bench   = pathlib.Path('${BENCHMARK_DIR}')
final   = pathlib.Path('${FINAL_DIR}')
dataset = '${DATASET}'
gpu     = '${GPU_MODEL}'
job_id  = '${SLURM_JOB_ID:-local}'

rt_path = bench / 'runtime_memory.json'
rt = json.loads(rt_path.read_text()) if rt_path.exists() else {}
stages  = rt.get('stages', {})
pre_s   = stages.get('preprocess', {}).get('wall_clock_seconds')
trn_s   = stages.get('train',      {}).get('wall_clock_seconds')
prd_s   = stages.get('predict',    {}).get('wall_clock_seconds')
total_s = rt.get('runtime_seconds')

def _fmt(s):
    if s is None: return 'N/A'
    m, sec = divmod(int(s), 60)
    return f'{m}m {sec}s ({s}s)'

best_epoch = rt.get('training', {}).get('best_epoch', 'N/A')
best_loss  = rt.get('training', {}).get('best_val_loss', 'N/A')
best_auroc = rt.get('training', {}).get('best_val_auroc', 'N/A')
best_f1    = rt.get('training', {}).get('best_val_f1', 'N/A')

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

gpu_mem_mib = rt.get('gpu_peak_memory_mib')
gpu_mem_str = f'{gpu_mem_mib} MiB ({gpu_mem_mib/1024:.2f} GB)' if isinstance(gpu_mem_mib, (int,float)) else 'N/A'
rss_gb  = rt.get('peak_rss_gb_observed')
rss_str = f'{rss_gb:.2f} GB' if isinstance(rss_gb, float) else 'not measured (no /usr/bin/time)'
n_tiles = rt.get('n_tiles', 'N/A')

md = f"""# Segger ROI Benchmark — {dataset}

Generated: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
SLURM job:  {job_id}  (predict-retry)
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
echo "[$(date)] Segger predict-retry COMPLETE: ${DATASET}"
echo "============================================================"
echo "Standardized transcript file:"
echo "  ${FINAL_DIR}/segger_transcripts_standardized.parquet"
echo "Benchmark summary:"
echo "  ${BENCHMARK_DIR}/segger_benchmark_summary.md"

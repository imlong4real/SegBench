#!/bin/bash
# ============================================================================
# run_segger_final_TSU20.sbatch
# Publication-grade Segger training + prediction on TSU-20 (full convergence).
#
# This is the FINAL run, distinct from the 3-epoch smoke test in:
#   results/segger_TSU20/trained_model/       (test run, DO NOT DELETE)
#   results/segger_TSU20/segger_output/...    (test run output)
#
# Outputs of this job:
#   results/segger_TSU20/trained_model_final/   -- convergence checkpoint
#   results/segger_TSU20/final/                 -- standardized final outputs
#   results/segger_TSU20/benchmark/             -- runtime/convergence metadata
#   logs/slurm/segger_final_TSU20_<JOBID>.out
# ============================================================================
#SBATCH --job-name=segger_final_TSU20
#SBATCH --partition=nvl,h100
#SBATCH --account=adeshpa6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/segger_final_TSU20_%j.out
#SBATCH --error=logs/slurm/segger_final_TSU20_%j.err

set -euo pipefail

# ── Environment diagnostics (run first; always visible in log) ────────────────
echo "=== Environment diagnostics ==="
echo "  hostname:        $(hostname)"
echo "  date:            $(date)"
echo "  SLURM_JOB_ID:    ${SLURM_JOB_ID:-unset}"
echo "  SLURM_JOB_NODE:  ${SLURM_JOB_NODELIST:-unset}"
echo "  PATH:            $PATH"
echo "  pwd:             $(pwd)"
echo "  shell:           $BASH_VERSION"
echo "  apptainer:       $(apptainer --version 2>&1 || echo NOT FOUND)"
echo "  nvidia-smi:      $(which nvidia-smi 2>&1 || echo NOT FOUND)"
echo "  /usr/bin/time:   $( [[ -x /usr/bin/time ]] && echo "found" || echo "NOT FOUND (will use wall-clock fallback)")"
echo "  bc:              $(which bc 2>&1 || echo "NOT FOUND (will use awk)")"
echo "  awk:             $(which awk 2>&1 || echo NOT FOUND)"
echo "  python3 (host):  $(which python3 2>&1 || echo NOT FOUND)"
echo "================================================"

# ── Portable timing helper ────────────────────────────────────────────────────
# /usr/bin/time (GNU time) is NOT installed on this cluster.
# run_timed <logfile> <command...>
#   - Uses /usr/bin/time -v -o <logfile> when available (gives peak RSS etc.)
#   - Falls back to plain execution when not; wall-clock still captured via
#     TRAIN_WALL_START / TRAIN_WALL_END (date +%s) surrounding each call.
run_timed() {
    local logfile="$1"; shift
    if [[ -x /usr/bin/time ]]; then
        /usr/bin/time -v -o "${logfile}" "$@"
    else
        echo "(NOTE: /usr/bin/time not found; peak RSS unavailable for this stage)" \
            >> "${logfile}" 2>/dev/null || true
        "$@"
    fi
}

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_DIR="${SEGBENCH_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}"
XENIUM_BASE="${REPO_DIR}/results/segger_smoke/input/xenium_bundle/TSU-20"
EXTRA_PKGS="${REPO_DIR}/results/segger_smoke/.site_packages"

# Container-visible paths (Apptainer auto-binds /home/lyuan13 → $HOME)
REPO_CONT="${SEGBENCH_REPO_CONT:-$REPO_DIR}"
XENIUM_BASE_CONT="${REPO_CONT}/results/segger_smoke/input/xenium_bundle/TSU-20"
EXTRA_PKGS_CONT="${REPO_CONT}/results/segger_smoke/.site_packages"

# This run's directories (separate from test-run directories)
OUT_ROOT="${REPO_DIR}/results/segger_TSU20"
OUT_ROOT_CONT="${REPO_CONT}/results/segger_TSU20"
MODEL_DIR="${OUT_ROOT}/trained_model_final"
MODEL_DIR_CONT="${OUT_ROOT_CONT}/trained_model_final"
PREDICT_DIR="${OUT_ROOT}/segger_output_final"
PREDICT_DIR_CONT="${OUT_ROOT_CONT}/segger_output_final"
FINAL_DIR="${OUT_ROOT}/final"
BENCHMARK_DIR="${OUT_ROOT}/benchmark"
TIMING_DIR="${BENCHMARK_DIR}/timing_raw"

cd "${REPO_DIR}"
mkdir -p "${MODEL_DIR}" "${PREDICT_DIR}" "${FINAL_DIR}" \
         "${BENCHMARK_DIR}" "${TIMING_DIR}" logs/slurm

echo "============================================================"
echo "[$(date)] Segger FINAL TSU-20 run — job ${SLURM_JOB_ID:-local}"
echo "[$(date)] Node: $(hostname)"
echo "[$(date)] Apptainer: $(apptainer --version)"
echo "============================================================"

# ── Guard: container must be built ───────────────────────────────────────────
if [[ ! -s "${REPO_DIR}/containers/python_cuda.sif" ]]; then
    echo "ERROR: containers/python_cuda.sif not found."
    echo "  Run: sbatch scripts/slurm/build_python_cuda_container.sbatch"
    exit 1
fi

# ── Guard: preprocessed_data must exist (reuse from test run) ─────────────────
n_tiles=$(find "${OUT_ROOT}/preprocessed_data" -name "*.pt" 2>/dev/null | wc -l)
if [[ "${n_tiles}" -lt 10 ]]; then
    echo "ERROR: Preprocessed tiles missing (found ${n_tiles})."
    echo "  Run the smoke/test job first to generate tiles:"
    echo "    sbatch scripts/slurm/run_segger_smoke.sbatch"
    exit 1
fi
echo "[$(date)] Found ${n_tiles} preprocessed tiles (reusing from test run)"

# ── Wrapper: direct conda python + LD_LIBRARY_PATH + PYTHONPATH ───────────────
WRAP_DIR="/tmp/segger_final_$$"
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

# ── GPU info ──────────────────────────────────────────────────────────────────
echo ""
echo "=== nvidia-smi ==="
nvidia-smi

GPU_MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
GPU_MEM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")
GPU_MEM_GB=$(awk "BEGIN{printf \"%.1f\", ${GPU_MEM_MIB}/1024}" 2>/dev/null || echo "unknown")
CPU_COUNT=$(nproc 2>/dev/null || echo "unknown")
HOST_RAM_GB=$(awk '/MemTotal/{printf "%.0f", $2/1024/1024}' /proc/meminfo 2>/dev/null || echo "unknown")

echo ""
echo "=== Hardware summary ==="
echo "  GPU:        ${GPU_MODEL}"
echo "  GPU VRAM:   ${GPU_MEM_MIB} MiB (${GPU_MEM_GB} GB)"
echo "  CPU count:  ${CPU_COUNT}"
echo "  Host RAM:   ${HOST_RAM_GB} GB"
echo "  SLURM job:  ${SLURM_JOB_ID:-local}"

# ── CUDA preflight ────────────────────────────────────────────────────────────
echo ""
echo "=== CUDA preflight ==="
segger_exec -c "
import torch, sys
assert torch.cuda.is_available(), 'CUDA not available inside container'
print('CUDA OK — device_count =', torch.cuda.device_count())
print('Device 0:', torch.cuda.get_device_name(0))
print('Segger version:')
try:
    import segger; print(' ', segger.__file__)
    import importlib.metadata; print('  version:', importlib.metadata.version('segger'))
except Exception as e:
    print('  (version lookup failed:', e, ')')
"

# ── Start GPU memory monitor ──────────────────────────────────────────────────
GPU_DMON_LOG="${TIMING_DIR}/gpu_dmon.log"
echo "# nvidia-smi dmon output (started: $(date))" > "${GPU_DMON_LOG}"
nvidia-smi dmon -d 5 -s mu -o T >> "${GPU_DMON_LOG}" &
DMON_PID=$!
echo "[$(date)] GPU monitor PID=${DMON_PID} → ${GPU_DMON_LOG}"

# Ensure monitor is killed on exit even if the script fails
cleanup() {
    kill "${DMON_PID}" 2>/dev/null || true
    echo "[$(date)] GPU monitor stopped"
}
trap 'cleanup; rm -rf ${WRAP_DIR}' EXIT

# ── Step 1: Train (full convergence, max_epochs=200) ─────────────────────────
echo ""
echo "=== Segger FINAL train (train_model.py, max_epochs=200, save_best_model=True) ==="
echo "=== Model dir: ${MODEL_DIR} ==="

TRAIN_ARGS=(
    /opt/segger_dev/src/segger/cli/train_model.py
    --dataset_dir   "${OUT_ROOT_CONT}/preprocessed_data"
    --models_dir    "${MODEL_DIR_CONT}"
    --sample_tag    TSU-20
    --num_tx_tokens 500
    --accelerator   cuda
    --devices       1
    --max_epochs    200
    --batch_size    4
    --num_workers   8
    --save_best_model True
    --precision     16-mixed
)

echo "=== Training command ==="
printf '  segger_exec %s\n' "${TRAIN_ARGS[@]}"
echo ""

TRAIN_TIME_LOG="${TIMING_DIR}/train_time_v.txt"
TRAIN_WALL_START=$(date +%s)
run_timed "${TRAIN_TIME_LOG}" \
    apptainer exec --nv \
        --pwd "${REPO_CONT}" \
        containers/python_cuda.sif "${WRAP_DIR}/run.sh" \
        "${TRAIN_ARGS[@]}" 2>&1
TRAIN_WALL_END=$(date +%s)
TRAIN_WALL_SEC=$(( TRAIN_WALL_END - TRAIN_WALL_START ))
echo "[$(date)] Training complete in ${TRAIN_WALL_SEC}s"
echo "TRAIN_WALL_SECONDS=${TRAIN_WALL_SEC}" >> "${BENCHMARK_DIR}/wall_times.env"

# ── Verify checkpoint ─────────────────────────────────────────────────────────
CKPT=$(find "${MODEL_DIR}" -name "*.ckpt" 2>/dev/null | sort | tail -1)
if [[ -z "${CKPT}" ]]; then
    echo "ERROR: No checkpoint found under ${MODEL_DIR}"
    exit 1
fi
CKPT_CONT="${MODEL_DIR_CONT}/lightning_logs/$(realpath --relative-to="${MODEL_DIR}" "${CKPT}")"
echo "[$(date)] Best checkpoint: ${CKPT}"
echo "BEST_CKPT=${CKPT}" >> "${BENCHMARK_DIR}/wall_times.env"

# ── Step 2: Predict ───────────────────────────────────────────────────────────
echo ""
echo "=== Segger FINAL predict (predict_fast.py) ==="
echo "=== Output dir: ${PREDICT_DIR} ==="

PREDICT_ARGS=(
    /opt/segger_dev/src/segger/cli/predict_fast.py
    --segger_data_dir  "${OUT_ROOT_CONT}/preprocessed_data"
    --models_dir       "${MODEL_DIR_CONT}"
    --benchmarks_dir   "${PREDICT_DIR_CONT}"
    --transcripts_file "${XENIUM_BASE_CONT}/transcripts.parquet"
    --batch_size       4
    --num_workers      0
    --model_version    0
    --knn_method       kd_tree
    --cell_id_col      segger_cell_id
    --use_cc           False
)

echo "=== Predict command ==="
printf '  segger_exec %s\n' "${PREDICT_ARGS[@]}"
echo ""

PREDICT_TIME_LOG="${TIMING_DIR}/predict_time_v.txt"
PREDICT_WALL_START=$(date +%s)
run_timed "${PREDICT_TIME_LOG}" \
    apptainer exec --nv \
        --pwd "${REPO_CONT}" \
        containers/python_cuda.sif "${WRAP_DIR}/run.sh" \
        "${PREDICT_ARGS[@]}" 2>&1
PREDICT_WALL_END=$(date +%s)
PREDICT_WALL_SEC=$(( PREDICT_WALL_END - PREDICT_WALL_START ))
echo "[$(date)] Prediction complete in ${PREDICT_WALL_SEC}s"
echo "PREDICT_WALL_SECONDS=${PREDICT_WALL_SEC}" >> "${BENCHMARK_DIR}/wall_times.env"

# Stop GPU monitor now that GPU work is done
kill "${DMON_PID}" 2>/dev/null || true
trap 'rm -rf ${WRAP_DIR}' EXIT

# ── Step 3: Locate and flatten prediction output ──────────────────────────────
# predict_fast.py writes under: PREDICT_DIR/segger_embedding_*/<outputs>
echo ""
echo "=== Locating prediction outputs ==="
PRED_SUBDIR=$(find "${PREDICT_DIR}" -maxdepth 1 -name "segger_embedding_*" -type d 2>/dev/null | sort | tail -1)
if [[ -z "${PRED_SUBDIR}" ]]; then
    echo "ERROR: Could not find segger_embedding_* subdir under ${PREDICT_DIR}"
    ls -lh "${PREDICT_DIR}/" 2>/dev/null || true
    exit 1
fi
echo "[$(date)] Prediction subdir: ${PRED_SUBDIR}"

# ── Step 4: Assemble final/ directory ────────────────────────────────────────
echo ""
echo "=== Assembling final output directory ==="

# Copy core outputs
for f in segger_transcripts.parquet segger_adata.h5ad segmentation_log.json; do
    if [[ -f "${PRED_SUBDIR}/${f}" ]]; then
        cp "${PRED_SUBDIR}/${f}" "${FINAL_DIR}/${f}"
        echo "  Copied: ${f}"
    else
        echo "  WARNING: ${f} not found in ${PRED_SUBDIR}"
    fi
done

# transcripts_df.parquet — may be a directory of parts; concat with pyarrow
if [[ -d "${PRED_SUBDIR}/transcripts_df.parquet" ]]; then
    echo "  Concatenating transcripts_df.parquet parts..."
    PYBIN_HOST="/apps/software/spack/gcc/8.5.0/anaconda3/2024.02-1-jbzsrx3q6jyqoh2f3wy6n7oxhdav64ml/bin/python3"
    "${PYBIN_HOST}" -c "
import pyarrow as pa, pyarrow.parquet as pq, pathlib, sys
src = pathlib.Path('${PRED_SUBDIR}/transcripts_df.parquet')
parts = sorted(src.glob('*.parquet'))
print(f'  Concatenating {len(parts)} parts')
tables = [pq.read_table(p) for p in parts]
pq.write_table(pa.concat_tables(tables), '${FINAL_DIR}/transcripts_df.parquet')
print('  Done: transcripts_df.parquet')
"
elif [[ -f "${PRED_SUBDIR}/transcripts_df.parquet" ]]; then
    cp "${PRED_SUBDIR}/transcripts_df.parquet" "${FINAL_DIR}/transcripts_df.parquet"
    echo "  Copied: transcripts_df.parquet"
fi

echo ""
echo "=== Final directory contents ==="
ls -lh "${FINAL_DIR}/"

# ── Step 5: Create segger_validation.json ─────────────────────────────────────
echo ""
echo "=== Writing segger_validation.json ==="
PYBIN_HOST="/apps/software/spack/gcc/8.5.0/anaconda3/2024.02-1-jbzsrx3q6jyqoh2f3wy6n7oxhdav64ml/bin/python3"
"${PYBIN_HOST}" -c "
import json, pathlib, datetime

final_dir = pathlib.Path('${FINAL_DIR}')
benchmark_dir = pathlib.Path('${BENCHMARK_DIR}')
ckpt = '${CKPT}'
model_dir = '${MODEL_DIR}'

# Collect validation info
val = {
    'timestamp': datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
    'repo_dir': '${REPO_DIR}',
    'run_label': 'final_convergence',
    'slurm_job_id': '${SLURM_JOB_ID:-local}',
    'gpu_model': '${GPU_MODEL}',
    'gpu_vram_mib': ${GPU_MEM_MIB},
    'checks': {},
    'paths': {
        'trained_model_dir': model_dir,
        'best_checkpoint': ckpt,
        'predict_subdir': '${PRED_SUBDIR}',
        'final_dir': str(final_dir),
        'benchmark_dir': str(benchmark_dir),
    },
}

required_files = [
    'segger_transcripts.parquet',
    'segger_adata.h5ad',
    'segmentation_log.json',
]
for fname in required_files:
    p = final_dir / fname
    val['checks'][fname] = {'path': str(p), 'ok': p.exists()}

val['status'] = 'PASS' if all(v['ok'] for v in val['checks'].values()) else 'FAIL'
val['blockers'] = [k for k, v in val['checks'].items() if not v['ok']]

with open(final_dir / 'segger_validation.json', 'w') as f:
    json.dump(val, f, indent=2)
print('Wrote:', final_dir / 'segger_validation.json')
print('Status:', val['status'])
"

# ── Step 6: Standardize for benchmark (benchmark schema) ─────────────────────
echo ""
echo "=== Standardizing transcript output for get_metric.py ==="
PYBIN_HOST="/apps/software/spack/gcc/8.5.0/anaconda3/2024.02-1-jbzsrx3q6jyqoh2f3wy6n7oxhdav64ml/bin/python3"
"${PYBIN_HOST}" \
    workflow/scripts/_segmentation/standardize_segger_output.py \
    --input  "${FINAL_DIR}/segger_transcripts.parquet" \
    --output "${FINAL_DIR}/segger_transcripts_standardized.parquet" \
    --report "${BENCHMARK_DIR}/schema_validation_report.json"

echo "[$(date)] Standardization complete"

# ── Step 7: Collect runtime / convergence metadata ───────────────────────────
echo ""
echo "=== Collecting runtime and convergence metadata ==="
"${PYBIN_HOST}" \
    workflow/scripts/_segmentation/collect_segger_runtime_metadata.py \
    --models-dir          "${MODEL_DIR}" \
    --outdir              "${BENCHMARK_DIR}" \
    --model-version       0 \
    --time-v-log          "${TRAIN_TIME_LOG}" \
    --predict-time-v-log  "${PREDICT_TIME_LOG}" \
    --gpu-dmon-log        "${GPU_DMON_LOG}" \
    --gpu-model           "${GPU_MODEL}" \
    --slurm-job-id        "${SLURM_JOB_ID:-local}" \
    --container-sif       "${REPO_DIR}/containers/python_cuda.sif" \
    --dataset             TSU-20 \
    --run-label           final_convergence

echo ""
echo "============================================================"
echo "[$(date)] Segger FINAL TSU-20 run COMPLETE"
echo "============================================================"
echo ""
echo "=== Key output paths ==="
echo ""
echo "Standardized transcripts (for benchmark):"
echo "  ${FINAL_DIR}/segger_transcripts_standardized.parquet"
echo ""
echo "Segger native outputs:"
echo "  ${FINAL_DIR}/segger_transcripts.parquet"
echo "  ${FINAL_DIR}/segger_adata.h5ad"
echo "  ${FINAL_DIR}/segmentation_log.json"
echo "  ${FINAL_DIR}/segger_validation.json"
echo ""
echo "Benchmark metadata:"
echo "  ${BENCHMARK_DIR}/runtime_memory.json"
echo "  ${BENCHMARK_DIR}/runtime_by_stage.tsv"
echo "  ${BENCHMARK_DIR}/training_convergence.tsv"
echo "  ${BENCHMARK_DIR}/training_summary.md"
echo ""
echo "Best checkpoint:"
echo "  ${CKPT}"
echo ""
echo "Test run preserved at:"
echo "  ${OUT_ROOT}/trained_model/          (3-epoch smoke test)"
echo "  ${OUT_ROOT}/segger_output/          (smoke test prediction output)"
echo ""
echo "=== get_metric.py command (run locally after rsync) ==="
echo ""
echo "python workflow/scripts/get_metric.py \\"
echo "  --method Segger \\"
echo "  --transcripts results/segger_TSU20/final/segger_transcripts_standardized.parquet \\"
echo "  --original datasets/lung_cancer_xenium_10x/lung_cancer_df.parquet \\"
echo "  --reference-h5ad datasets/lung_cancer_scrna_36973297/lung_cancer_50k.h5ad \\"
echo "  --npmi results/benchmark/lung_xenium_ref36973297/reference_npmi/lung_cancer_36973297_npmi.csv.gz \\"
echo "  --runtime-json results/segger_TSU20/benchmark/runtime_memory.json \\"
echo "  --outdir results/benchmark/lung_xenium_ref36973297/metrics/Segger \\"
echo "  --min-transcripts-per-cell 10 \\"
echo "  --max-transcripts-per-cell 900"
echo ""
echo "=== Files to rsync to local machine ==="
echo ""
echo "rsync -avz --progress HPC:${FINAL_DIR}/segger_transcripts_standardized.parquet ."
echo "rsync -avz --progress HPC:${FINAL_DIR}/segmentation_log.json ."
echo "rsync -avz --progress HPC:${FINAL_DIR}/segger_validation.json ."
echo "rsync -avz --progress HPC:${BENCHMARK_DIR}/runtime_memory.json ."
echo "rsync -avz --progress HPC:${BENCHMARK_DIR}/runtime_by_stage.tsv ."
echo "rsync -avz --progress HPC:${BENCHMARK_DIR}/training_convergence.tsv ."
echo "rsync -avz --progress HPC:${BENCHMARK_DIR}/training_summary.md ."
echo "# Optional large files:"
echo "rsync -avz --progress HPC:${FINAL_DIR}/segger_adata.h5ad ."
echo "rsync -avz --progress HPC:${MODEL_DIR}/ trained_model_final/"

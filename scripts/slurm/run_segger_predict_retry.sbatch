#!/bin/bash
#SBATCH --job-name=segger_predict_retry
#SBATCH --partition=l40s,nvl,h100
#SBATCH --account=adeshpa6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/segger_predict_retry_%j.out
#SBATCH --error=logs/slurm/segger_predict_retry_%j.err

set -euo pipefail

REPO_DIR="/weka/home/lyuan13/segmentation_benchmark_pipeline"
REPO_CONT="/home/lyuan13/segmentation_benchmark_pipeline"
XENIUM_BASE_CONT="${REPO_CONT}/results/segger_smoke/input/xenium_bundle/TSU-20"
EXTRA_PKGS="${REPO_DIR}/results/segger_smoke/.site_packages"
EXTRA_PKGS_CONT="${REPO_CONT}/results/segger_smoke/.site_packages"
OUT_ROOT="${REPO_DIR}/results/segger_TSU20"
OUT_ROOT_CONT="${REPO_CONT}/results/segger_TSU20"
DEBUG_DIR="${OUT_ROOT}/debug"

cd "${REPO_DIR}"
mkdir -p "${EXTRA_PKGS}" "${DEBUG_DIR}" logs/slurm

echo "[$(date)] segger_predict_retry on node $(hostname)"
echo "[$(date)] Apptainer: $(apptainer --version)"

if [[ ! -s containers/python_cuda.sif ]]; then
    echo "ERROR: containers/python_cuda.sif not found."
    exit 1
fi

# Guard: preprocessed tiles must already exist from the previous run.
n_tiles=$(find "${OUT_ROOT}/preprocessed_data" -name "*.pt" 2>/dev/null | wc -l)
if [[ "${n_tiles}" -lt 10 ]]; then
    echo "ERROR: Too few preprocessed tiles (${n_tiles}). Re-run full smoke job first."
    exit 1
fi
echo "[$(date)] Found ${n_tiles} preprocessed tiles — skipping create_dataset_fast.py"

# Guard: checkpoint must exist.
CKPT=$(find "${OUT_ROOT}/trained_model" -name "*.ckpt" 2>/dev/null | sort | tail -1)
if [[ -z "${CKPT}" ]]; then
    echo "ERROR: No checkpoint found under ${OUT_ROOT}/trained_model"
    exit 1
fi
echo "[$(date)] Checkpoint: ${CKPT}"

# ── Wrapper: direct conda python + LD_LIBRARY_PATH + PYTHONPATH ──────────────
WRAP_DIR="/tmp/segger_predict_$$"
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

# ── GPU + CUDA preflight ──────────────────────────────────────────────────────
echo ""
echo "=== nvidia-smi ==="
nvidia-smi

echo ""
echo "=== CUDA preflight ==="
segger_exec -c "
import torch, sys
assert torch.cuda.is_available(), 'CUDA not available inside container'
assert torch.cuda.device_count() >= 1, 'No CUDA device inside container'
print('CUDA preflight PASSED: device_count =', torch.cuda.device_count())
print('Device 0:', torch.cuda.get_device_name(0))
"

# ── Step A: Inspect PyG tile schemas ─────────────────────────────────────────
echo ""
echo "=== Inspect PyG dataset schemas ==="
segger_exec \
    workflow/scripts/_segmentation/inspect_segger_pyg_schema.py \
    --dataset-dir "${OUT_ROOT_CONT}/preprocessed_data" \
    --outdir      "${OUT_ROOT_CONT}/debug"

echo "[$(date)] Schema inspection complete. Summary:"
cat "${OUT_ROOT}/debug/pyg_schema_summary.json" 2>/dev/null \
    | python3 -c "
import json, sys
d = json.load(sys.stdin)
for split, s in d['summary'].items():
    print(f\"  {split}: total={s['total']} has_edge_label={s['has_edge_label']} missing={s['missing_edge_label']}\")
" || true

# ── Step B: Patch test_tiles missing edge_label ───────────────────────────────
echo ""
echo "=== Patch test_tiles (add missing edge_label) ==="
segger_exec \
    workflow/scripts/_segmentation/patch_segger_edge_label.py \
    --dataset-dir  "${OUT_ROOT_CONT}/preprocessed_data" \
    --splits       test_tiles \
    --report-csv   "${OUT_ROOT_CONT}/debug/edge_label_patch_report.csv"

# ── Step C: Predict ───────────────────────────────────────────────────────────
# Clear any partial output from the previous crashed run.
rm -rf "${OUT_ROOT}/segger_output"
mkdir -p "${OUT_ROOT}/segger_output"

echo ""
echo "=== Segger predict (predict_fast.py, num_workers=0) ==="
PREDICT_ARGS=(
    /opt/segger_dev/src/segger/cli/predict_fast.py
    --segger_data_dir  "${OUT_ROOT_CONT}/preprocessed_data"
    --models_dir       "${OUT_ROOT_CONT}/trained_model"
    --benchmarks_dir   "${OUT_ROOT_CONT}/segger_output"
    --transcripts_file "${XENIUM_BASE_CONT}/transcripts.parquet"
    --batch_size       4
    --num_workers      0
    --model_version    0
    --knn_method       kd_tree
    --cell_id_col      segger_cell_id
    --use_cc           False
)
echo "=== predict command ==="
printf '  %s\n' "${PREDICT_ARGS[@]}"
echo ""
segger_exec "${PREDICT_ARGS[@]}"

echo ""
echo "[$(date)] Prediction COMPLETE"
echo "Output: ${OUT_ROOT}/segger_output/"
find "${OUT_ROOT}/segger_output" -type f | head -30

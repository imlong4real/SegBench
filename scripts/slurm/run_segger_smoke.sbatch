#!/bin/bash
#SBATCH --job-name=segger_TSU20
#SBATCH --partition=l40s,nvl,h100
#SBATCH --account=adeshpa6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/segger_TSU20_%j.out
#SBATCH --error=logs/slurm/segger_TSU20_%j.err

set -euo pipefail

REPO_DIR="/weka/home/lyuan13/segmentation_benchmark_pipeline"
# Full Xenium bundle — real TSU-20 data, ~28 MB transcripts, ~2.9 MB nucleus_boundaries.
XENIUM_BASE="${REPO_DIR}/results/segger_smoke/input/xenium_bundle/TSU-20"
# extra site-packages already populated here; reuse to avoid re-copying.
EXTRA_PKGS="${REPO_DIR}/results/segger_smoke/.site_packages"
# Container-visible paths: Apptainer auto-binds $HOME=/home/lyuan13, NOT /weka/home/.
REPO_CONT="/home/lyuan13/segmentation_benchmark_pipeline"
XENIUM_BASE_CONT="${REPO_CONT}/results/segger_smoke/input/xenium_bundle/TSU-20"
EXTRA_PKGS_CONT="${REPO_CONT}/results/segger_smoke/.site_packages"
# Output root for this full run.
OUT_ROOT="${REPO_DIR}/results/segger_TSU20"
OUT_ROOT_CONT="${REPO_CONT}/results/segger_TSU20"

cd "${REPO_DIR}"
mkdir -p "${EXTRA_PKGS}" "${OUT_ROOT}" logs/slurm

echo "[$(date)] Segger TSU-20 run on node $(hostname)"
echo "[$(date)] Apptainer: $(apptainer --version)"

# ── Guard: container must be built ───────────────────────────────────────────
if [[ ! -s containers/python_cuda.sif ]]; then
    echo "ERROR: containers/python_cuda.sif not found."
    echo "  Run: sbatch scripts/slurm/build_python_cuda_container.sbatch"
    exit 1
fi

# ── Ensure experiment.xenium exists in TSU-20 bundle ─────────────────────────
# Segger reads this for metadata (run_name, region_name, pixel_size).
if [[ ! -f "${XENIUM_BASE}/experiment.xenium" ]]; then
    echo "[$(date)] Creating experiment.xenium stub for TSU-20"
    cat > "${XENIUM_BASE}/experiment.xenium" << 'XMETA'
{
  "run_name": "2_018",
  "region_name": "TSU-20",
  "num_cells": 58648,
  "pixel_size": 0.2125
}
XMETA
fi

# ── Pre-flight: ensure pkg_resources is in EXTRA_PKGS ────────────────────────
# lightning_fabric uses pkg_resources.declare_namespace; not in segger_cuda env.
HOST_PKG_RES="/apps/software/spack/gcc/8.5.0/anaconda3/2024.02-1-jbzsrx3q6jyqoh2f3wy6n7oxhdav64ml/lib/python3.11/site-packages/pkg_resources"
if [[ ! -d "${EXTRA_PKGS}/pkg_resources" ]]; then
    echo "[$(date)] Copying pkg_resources to ${EXTRA_PKGS}"
    cp -r "${HOST_PKG_RES}" "${EXTRA_PKGS}/"
fi

# ── Wrapper: direct conda python + LD_LIBRARY_PATH + PYTHONPATH ───────────────
# mamba run does not set LD_LIBRARY_PATH for llvmlite (needs CXXABI_1.3.15
# from conda env's libstdc++, not Ubuntu 22.04 GCC 11).
WRAP_DIR="/tmp/segger_run_$$"
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

# ── Step 1: GPU sanity ────────────────────────────────────────────────────────
echo ""
echo "=== nvidia-smi ==="
nvidia-smi

echo ""
echo "=== PyTorch CUDA check ==="
segger_exec -c "
import sys, torch
print('python', sys.version.split()[0])
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('cuda_device', torch.cuda.get_device_name(0))
else:
    print('ERROR: CUDA not available')
    sys.exit(1)
"

# ── Step 2: Segger import check ───────────────────────────────────────────────
echo ""
echo "=== Segger import check ==="
segger_exec -c "
import segger
print('segger OK:', segger.__file__)
"

# ── Step 3: Preprocess ────────────────────────────────────────────────────────
# Skip if tiles already present from a previous run (saves ~1 min on resubmits).
if find "${OUT_ROOT}/preprocessed_data" -name "*.pt" -quit 2>/dev/null | grep -q .; then
    echo "[$(date)] Preprocessed tiles found — skipping create_dataset_fast.py"
else
    rm -rf "${OUT_ROOT}/preprocessed_data"
    mkdir -p "${OUT_ROOT}/preprocessed_data" \
             "${OUT_ROOT}/trained_model" \
             "${OUT_ROOT}/segger_output"

    echo ""
    echo "=== Segger preprocess (TSU-20, create_dataset_fast.py) ==="
    segger_exec \
        /opt/segger_dev/src/segger/cli/create_dataset_fast.py \
        --base_dir    "${XENIUM_BASE_CONT}" \
        --data_dir    "${OUT_ROOT_CONT}/preprocessed_data" \
        --sample_type xenium \
        --tile_width  200 \
        --tile_height 200 \
        --n_workers   8
fi
# NOTE: create_dataset_fast.py --sample_type xenium already writes train/val/test
# splits directly. Do NOT copy test→train/val; that would pollute splits.
mkdir -p "${OUT_ROOT}/trained_model" "${OUT_ROOT}/segger_output"

echo "[$(date)] Tile counts by split:"
find "${OUT_ROOT}/preprocessed_data" -name "*.pt" | \
    awk -F/ '{print $(NF-2)}' | sort | uniq -c

# ── Step 4: Train ─────────────────────────────────────────────────────────────
# Dump CLI help once so we have a reference for valid flags.
echo ""
echo "=== train_model.py --help ==="
segger_exec /opt/segger_dev/src/segger/cli/train_model.py --help \
    2>&1 | tee "${REPO_DIR}/docs/segger_train_model_help.txt" || true

# CUDA preflight — fail fast if GPU not visible inside container.
echo ""
echo "=== CUDA preflight ==="
segger_exec -c "
import torch, sys
assert torch.cuda.is_available(), 'CUDA not available inside container'
assert torch.cuda.device_count() >= 1, 'No CUDA device visible inside container'
print('CUDA preflight PASSED: device_count =', torch.cuda.device_count())
print('Device 0:', torch.cuda.get_device_name(0))
"

echo ""
echo "=== Segger train (train_model.py, 3 smoke epochs) ==="
# FIX: devices=1 means 'use 1 GPU'. devices=0 is invalid under cuda accelerator
# (Lightning interprets 0 as 'zero devices', not 'GPU index 0').
TRAIN_ARGS=(
    /opt/segger_dev/src/segger/cli/train_model.py
    --dataset_dir   "${OUT_ROOT_CONT}/preprocessed_data"
    --models_dir    "${OUT_ROOT_CONT}/trained_model"
    --sample_tag    TSU-20
    --num_tx_tokens 500
    --accelerator   cuda
    --devices       1
    --max_epochs    3
    --batch_size    4
    --num_workers   8
)
echo "=== Segger train command ==="
printf '  %s\n' "${TRAIN_ARGS[@]}"
echo ""
segger_exec "${TRAIN_ARGS[@]}"

# ── Step 5: Predict ───────────────────────────────────────────────────────────
echo ""
echo "=== Segger predict (predict_fast.py) ==="
segger_exec \
    /opt/segger_dev/src/segger/cli/predict_fast.py \
    --segger_data_dir  "${OUT_ROOT_CONT}/preprocessed_data" \
    --models_dir       "${OUT_ROOT_CONT}/trained_model" \
    --benchmarks_dir   "${OUT_ROOT_CONT}/segger_output" \
    --transcripts_file "${XENIUM_BASE_CONT}/transcripts.parquet" \
    --batch_size       4 \
    --num_workers      8 \
    --model_version    0 \
    --knn_method       kd_tree \
    --cell_id_col      segger_cell_id \
    --use_cc           False

echo ""
echo "[$(date)] Segger TSU-20 full run COMPLETE"
echo "Output: ${OUT_ROOT}/segger_output/"
ls -lh "${OUT_ROOT}/segger_output/" 2>/dev/null || echo "(output dir empty)"

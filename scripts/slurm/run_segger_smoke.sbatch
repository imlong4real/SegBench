#!/bin/bash
#SBATCH --job-name=segger_smoke
#SBATCH --partition=nvl
#SBATCH --account=adeshpa6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/segger_smoke_%j.out
#SBATCH --error=logs/slurm/segger_smoke_%j.err

set -euo pipefail

REPO_DIR="/weka/home/lyuan13/segmentation_benchmark_pipeline"
INPUT_PARQUET="/weka/home/lyuan13/TRACER/tutorials/lung_cancer/data/lung_cancer_df.parquet"

cd "${REPO_DIR}"

echo "[$(date)] Segger smoke test on node $(hostname)"
echo "[$(date)] Apptainer: $(apptainer --version)"

# ── Guard: container must already be built ────────────────────────────────────
if [[ ! -s containers/python_cuda.sif ]]; then
    echo "ERROR: containers/python_cuda.sif not found. Run build first:"
    echo "  sbatch scripts/slurm/build_python_cuda_container.sbatch"
    exit 1
fi

# ── Step 1: GPU sanity ────────────────────────────────────────────────────────
echo ""
echo "=== nvidia-smi ==="
nvidia-smi

echo ""
echo "=== PyTorch CUDA check ==="
apptainer exec --nv containers/python_cuda.sif \
    mamba run -n segger_cuda python3 - <<'PY'
import sys, torch
print("python", sys.version)
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda_device", torch.cuda.get_device_name(0))
else:
    print("ERROR: CUDA not available — cannot run Segger")
    sys.exit(1)
PY

# ── Step 2: Segger import check ───────────────────────────────────────────────
echo ""
echo "=== Segger import check ==="
apptainer exec --nv containers/python_cuda.sif \
    mamba run -n segger_cuda python3 - <<'PY'
import segger
print("segger OK:", getattr(segger, '__version__', 'no __version__'))
print("segger file:", segger.__file__)
PY

# ── Step 3: Create smoke input subset ────────────────────────────────────────
echo ""
echo "=== Creating 100k smoke subset ==="
module load anaconda3/2024.02-1 2>/dev/null || true
conda run -n base python workflow/scripts/_segmentation/create_segger_smoke_subset.py \
    --input  "${INPUT_PARQUET}" \
    --output results/segger_smoke/input/transcripts_100k.parquet \
    --summary results/segger_smoke/input/input_summary.json \
    --xenium_bundle results/segger_smoke/input/xenium_bundle \
    --n 100000

echo ""
echo "=== Input summary ==="
cat results/segger_smoke/input/input_summary.json

# ── Step 4: Segger preprocess ────────────────────────────────────────────────
echo ""
echo "=== Segger preprocess (create_dataset_fast.py) ==="
mkdir -p results/segger_smoke/preprocessed_data
apptainer exec --nv containers/python_cuda.sif \
    mamba run -n segger_cuda python3 \
        /opt/segger_dev/src/segger/cli/create_dataset_fast.py \
        --base_dir    results/segger_smoke/input/xenium_bundle \
        --data_dir    results/segger_smoke/preprocessed_data \
        --sample_type xenium \
        --tile_width  200 \
        --tile_height 200 \
        --n_workers   4

# ── Step 5: Segger train (minimal) ───────────────────────────────────────────
echo ""
echo "=== Segger train (train_model.py, 3 epochs) ==="
mkdir -p results/segger_smoke/trained_model
apptainer exec --nv containers/python_cuda.sif \
    mamba run -n segger_cuda python3 \
        /opt/segger_dev/src/segger/cli/train_model.py \
        --dataset_dir   results/segger_smoke/preprocessed_data \
        --models_dir    results/segger_smoke/trained_model \
        --sample_tag    smoke \
        --num_tx_tokens 500 \
        --accelerator   cuda \
        --devices       0 \
        --max_epochs    3 \
        --batch_size    2 \
        --num_workers   4

# ── Step 6: Segger predict ────────────────────────────────────────────────────
echo ""
echo "=== Segger predict (predict_fast.py) ==="
mkdir -p results/segger_smoke/segger_output
apptainer exec --nv containers/python_cuda.sif \
    mamba run -n segger_cuda python3 \
        /opt/segger_dev/src/segger/cli/predict_fast.py \
        --segger_data_dir results/segger_smoke/preprocessed_data \
        --models_dir      results/segger_smoke/trained_model \
        --benchmarks_dir  results/segger_smoke/segger_output \
        --transcripts_file results/segger_smoke/input/transcripts_100k.parquet \
        --batch_size       1 \
        --num_workers      4 \
        --model_version    0 \
        --knn_method       kd_tree \
        --cell_id_col      segger_cell_id \
        --use_cc           False

echo ""
echo "[$(date)] Segger smoke test COMPLETE"
echo "Output dir: results/segger_smoke/segger_output/"
ls -la results/segger_smoke/segger_output/ 2>/dev/null || echo "(output dir empty or missing)"

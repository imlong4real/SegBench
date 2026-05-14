#!/bin/bash
#SBATCH --job-name=segger_cuda_check
#SBATCH --partition=nvl
#SBATCH --account=adeshpa6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/segger_cuda_check_%j.out
#SBATCH --error=logs/slurm/segger_cuda_check_%j.err

set -euo pipefail

REPO_DIR="/weka/home/lyuan13/segmentation_benchmark_pipeline"
cd "${REPO_DIR}"

echo "[$(date)] CUDA check on node $(hostname)"
echo "[$(date)] Apptainer: $(apptainer --version)"

# Step 1: GPU visibility outside container
echo ""
echo "=== nvidia-smi ==="
nvidia-smi

if [[ ! -s containers/python_cuda.sif ]]; then
    echo "ERROR: containers/python_cuda.sif not found. Build it first with:"
    echo "  sbatch scripts/slurm/build_python_cuda_container.sbatch"
    exit 1
fi

# Step 2: PyTorch CUDA check inside container
echo ""
echo "=== PyTorch CUDA check inside container ==="
apptainer exec --nv containers/python_cuda.sif \
    mamba run -n segger_cuda python3 - <<'PY'
import sys
print("python", sys.version)
try:
    import torch
    print("torch", torch.__version__)
    print("cuda_available", torch.cuda.is_available())
    print("cuda_device_count", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("cuda_device", torch.cuda.get_device_name(0))
    else:
        print("WARNING: CUDA not available inside container — check --nv flag")
        sys.exit(1)
except Exception as e:
    print("TORCH_CHECK_FAILED", repr(e))
    sys.exit(1)
PY

# Step 3: Segger importability check
echo ""
echo "=== Segger import check ==="
apptainer exec --nv containers/python_cuda.sif \
    mamba run -n segger_cuda python3 - <<'PY'
import pkgutil
mods = [m.name for m in pkgutil.iter_modules() if "segger" in m.name.lower()]
print("segger-like modules:", mods)
try:
    import segger
    print("segger import OK:", segger)
    print("segger location:", getattr(segger, '__file__', 'unknown'))
except Exception as e:
    print("segger import failed:", repr(e))
    import sys; sys.exit(1)
PY

echo ""
echo "[$(date)] All checks passed."

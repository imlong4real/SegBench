#!/bin/bash
#SBATCH --job-name=build_python_cuda
#SBATCH --partition=nvl
#SBATCH --account=adeshpa6
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/build_python_cuda_%j.out
#SBATCH --error=logs/slurm/build_python_cuda_%j.err

set -euo pipefail

REPO_DIR="/weka/home/lyuan13/segmentation_benchmark_pipeline"

echo "[$(date)] Starting container build on node $(hostname)"
echo "[$(date)] Apptainer version: $(apptainer --version)"

mkdir -p "${REPO_DIR}/containers"

# The python_cuda.def %files section copies segger_cuda.yml and general_cuda.yml
# from the current directory at build time, so we must cd into the def's directory.
cd "${REPO_DIR}/reproducibility/python_cuda"

SIF="${REPO_DIR}/containers/python_cuda.sif"

# Try fakeroot first; fall back without it if not permitted.
if apptainer build --fakeroot --force "${SIF}" python_cuda.def 2>&1; then
    echo "[$(date)] Build with --fakeroot succeeded"
else
    echo "[$(date)] --fakeroot failed, retrying without fakeroot..."
    apptainer build --force "${SIF}" python_cuda.def
fi

if [[ -s containers/python_cuda.sif ]]; then
    echo "[$(date)] SUCCESS: containers/python_cuda.sif exists ($(du -sh containers/python_cuda.sif | cut -f1))"
else
    echo "[$(date)] ERROR: containers/python_cuda.sif is missing or empty"
    exit 1
fi

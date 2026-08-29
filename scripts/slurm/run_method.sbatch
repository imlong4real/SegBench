#!/usr/bin/env bash
# Run one SegBench method on SLURM.
#
#   sbatch --export=ALL,METHOD=proseg,DATASET=nsclc_xenium,OUTDIR=/path \
#          scripts/slurm/run_method.sbatch
#
# Resources are per-method: override with the usual sbatch flags at submit
# time (e.g. --gres=gpu:1 --partition=a100 for segger).
#SBATCH --job-name=segbench
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -uo pipefail

: "${METHOD:?set METHOD}"
: "${DATASET:?set DATASET}"
# SLURM copies this script to a spool dir, so BASH_SOURCE is useless here;
# prefer an explicit SEGBENCH_REPO, then the submit directory.
REPO="${SEGBENCH_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [ ! -x "$REPO/bin/segbench" ]; then
  echo "ERROR: no bin/segbench under REPO=$REPO. Set SEGBENCH_REPO." >&2
  exit 2
fi
cd "$REPO"

# Per-machine environment resolution.
if [ -f configs/environments.local.sh ]; then
  source configs/environments.local.sh
else
  source configs/environments.local.example.sh
fi

OUTDIR="${OUTDIR:-/scratch4/adeshpa6/segbench_runs/${DATASET}/methods/${METHOD}}"
EXTRA="${EXTRA:-}"

echo "=== segbench $METHOD on $DATASET ==="
echo "  repo   : $REPO"
echo "  outdir : $OUTDIR"
echo "  threads: ${SLURM_CPUS_PER_TASK:-8}"
echo "  node   : $(hostname)"
date

./bin/segbench run "$METHOD" \
  --dataset "$DATASET" \
  --outdir  "$OUTDIR" \
  --threads "${SLURM_CPUS_PER_TASK:-8}" \
  --overwrite $EXTRA

rc=$?
date
echo "=== exit $rc ==="
exit $rc

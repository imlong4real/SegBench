#!/usr/bin/env bash
# Build a panel-matched cPMI table on SLURM.
#   sbatch --export=ALL,PMI_SCRIPT=/path/to/run_pmi_nsclc.sh scripts/slurm/build_pmi.sbatch
#SBATCH --job-name=segbench_pmi
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
set -uo pipefail
: "${PMI_SCRIPT:?set PMI_SCRIPT}"
echo "=== cPMI build: $PMI_SCRIPT on $(hostname) ==="; date
bash "$PMI_SCRIPT"; rc=$?
date; echo "=== exit $rc ==="; exit $rc

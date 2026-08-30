# Source this (or copy to environments.local.sh) to point SegBench at the
# environments built by scripts/setup_environments.sh.
export SEGBENCH_ENV_ROOT=/scratch4/adeshpa6/segbench_envs
export TRACER_VENV=/home/lyuan13/scr4_adeshpa6/TRACER/.venv_rcc
export SEGBENCH_DATA=/home/lyuan13/scr4_adeshpa6/TRACER/datasets
# Where benchmark runs and reports are written. Set this explicitly:
# scratch is named for the allocation group here, not the login name.
export SEGBENCH_RUNS=/scratch4/adeshpa6/segbench_runs
# Interpreter that runs the segbench driver itself.
export SEGBENCH_PYTHON="$TRACER_VENV/bin/python"

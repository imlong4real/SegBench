#!/usr/bin/env bash
# Build every method environment SegBench needs, from scratch.
#
#   scripts/setup_environments.sh [ENV_ROOT]
#
# ENV_ROOT defaults to $SEGBENCH_ENV_ROOT, else /scratch4/$USER/segbench_envs.
# Idempotent: each step is skipped when its target already exists, so this can
# be re-run to fill in whatever failed.
#
# Afterwards, point SegBench at the result:
#   export SEGBENCH_ENV_ROOT=<ENV_ROOT>
#   export TRACER_VENV=/path/to/TRACER/.venv
#   ./bin/segbench doctor
#
# NOTE ON MEMORY: conda solves are memory-hungry and the login node here
# enforces a per-user cgroup limit that OOM-killed mamba. Run this under
# `sbatch --mem=48G` on a compute node rather than on a login node.
set -uo pipefail

ROOT="${1:-${SEGBENCH_ENV_ROOT:-/scratch4/$USER/segbench_envs}}"
LOG="$ROOT/logs"; mkdir -p "$LOG" "$ROOT/bin"
MAMBA="$ROOT/miniforge3/bin/mamba"
export MAMBA_ROOT_PREFIX="$ROOT/miniforge3"
step () { echo; echo "=== [$(date +%H:%M:%S)] $* ==="; }

# ---------------------------------------------------------------- miniforge
step "miniforge / mamba"
if [ ! -x "$MAMBA" ]; then
  curl -sSL -o "$ROOT/miniforge.sh" \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
  bash "$ROOT/miniforge.sh" -b -p "$ROOT/miniforge3" >"$LOG/miniforge.log" 2>&1
fi
"$MAMBA" --version | head -1

# ------------------------------------------------------------------- baysor
step "Baysor (prebuilt binary)"
BAYSOR="$ROOT/baysor/bin/baysor/bin/baysor"
if [ ! -x "$BAYSOR" ]; then
  curl -sSL -o "$ROOT/baysor.zip" \
    "https://github.com/kharchenkolab/Baysor/releases/download/v0.7.1/baysor-x86_x64-linux-v0.7.1_build.zip"
  rm -rf "$ROOT/baysor"; mkdir -p "$ROOT/baysor"
  (cd "$ROOT/baysor" && unzip -q ../baysor.zip)
  chmod -R u+x "$ROOT/baysor/bin"
fi
"$BAYSOR" --version 2>/dev/null && echo "  baysor OK" || echo "  baysor FAILED"

# ------------------------------------------------------------------- proseg
step "proseg (cargo; no prebuilt Linux asset is published)"
export RUSTUP_HOME="$ROOT/rustup" CARGO_HOME="$ROOT/cargo"
if [ ! -x "$ROOT/cargo/bin/proseg" ]; then
  [ -x "$ROOT/cargo/bin/cargo" ] || curl -sSf https://sh.rustup.rs | \
    sh -s -- -y --no-modify-path --profile minimal >"$LOG/rust.log" 2>&1
  "$ROOT/cargo/bin/cargo" install proseg --locked >>"$LOG/rust.log" 2>&1
fi
"$ROOT/cargo/bin/proseg" --version 2>/dev/null && echo "  proseg OK" || echo "  proseg FAILED"

# ----------------------------------------------------------------- bin2cell
step "bin2cell (venv; pip is far lighter than a conda solve here)"
if [ ! -x "$ROOT/venvs/bin2cell/bin/python" ]; then
  python3 -m venv "$ROOT/venvs/bin2cell" >"$LOG/bin2cell.log" 2>&1
fi
"$ROOT/venvs/bin2cell/bin/pip" install -q -U pip                        >>"$LOG/bin2cell.log" 2>&1
"$ROOT/venvs/bin2cell/bin/pip" install -q numpy pandas scipy anndata scanpy h5py \
    pyarrow psutil pyyaml scikit-image matplotlib tifffile opencv-python-headless \
    bin2cell "tensorflow-cpu<2.17" stardist csbdeep                     >>"$LOG/bin2cell.log" 2>&1
"$ROOT/venvs/bin2cell/bin/python" -c "import bin2cell,stardist; print('  bin2cell', bin2cell.__version__, '+ stardist OK')" \
  2>/dev/null || echo "  bin2cell FAILED"

# ------------------------------------------------------------------- segger
step "segger (venv; see reproducibility/segger_env_notes.md — upstream is broken)"
if [ ! -x "$ROOT/venvs/segger/bin/python" ]; then
  python3 -m venv "$ROOT/venvs/segger" >"$LOG/segger.log" 2>&1
fi
"$ROOT/venvs/segger/bin/pip" install -q -U pip                          >>"$LOG/segger.log" 2>&1
"$ROOT/venvs/segger/bin/pip" install -q numpy pandas pyarrow scipy anndata scanpy \
    psutil pyyaml h5py scikit-learn                                     >>"$LOG/segger.log" 2>&1
"$ROOT/venvs/segger/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cu121 \
                                                                        >>"$LOG/segger.log" 2>&1
"$ROOT/venvs/segger/bin/pip" install -q torch_geometric lightning       >>"$LOG/segger.log" 2>&1
# Dependencies segger imports but never declares:
"$ROOT/venvs/segger/bin/pip" install -q "dask[distributed]" dask-geopandas pqdm \
    torchmetrics cupy-cuda12x                                           >>"$LOG/segger.log" 2>&1
"$ROOT/venvs/segger/bin/pip" install -q "segger @ git+https://github.com/EliHei2/segger_dev.git" \
                                                                        >>"$LOG/segger.log" 2>&1
# Upstream ships a stray shell command inside train_model.py.
SEG_CLI="$ROOT/venvs/segger/lib/python3*/site-packages/segger/cli/train_model.py"
for f in $SEG_CLI; do
  [ -f "$f" ] && sed -i '/^uv a$/d' "$f" && echo "  patched stray line in $(basename "$f")"
done

# ------------------------------------------------------------------------ R
step "R 4.3 + Seurat (staged; one big solve gets OOM-killed)"
if [ ! -x "$ROOT/envs/r/bin/Rscript" ]; then
  "$MAMBA" create -y -p "$ROOT/envs/r" -c conda-forge r-base=4.3 r-matrix r-rcpp \
    >"$LOG/r1.log" 2>&1
fi
"$MAMBA" install -y -p "$ROOT/envs/r" -c conda-forge \
    r-seurat r-arrow r-jsonlite r-optparse r-remotes r-devtools r-hdf5r r-anndata \
    >"$LOG/r2.log" 2>&1
"$MAMBA" install -y -p "$ROOT/envs/r" -c conda-forge -c bioconda \
    r-rcppeigen r-doparallel r-foreach r-data.table bioconductor-biocparallel \
    bioconductor-sparsematrixstats hdf5 r-reticulate \
    >"$LOG/r3.log" 2>&1
# Dependencies of spacexr/SPLIT/cellAdmix that will not build from source here,
# plus the compilers those GitHub packages need.
"$MAMBA" install -y -p "$ROOT/envs/r" -c conda-forge \
    r-beeswarm r-ggbeeswarm r-proc r-pbmcapply r-cairo r-nnls r-nmf r-ggrastr \
    r-irlba r-rcpparmadillo r-rcppprogress r-matrixstats r-plyr r-reshape2 \
    r-cowplot r-ggrepel r-pheatmap r-rtsne r-uwot r-mgcv r-quadprog \
    gxx_linux-64 gcc_linux-64 gfortran_linux-64 make cmake pkg-config \
    >"$LOG/r4.log" 2>&1

step "R GitHub packages (serialize these — parallel installs fight over the library lock)"
export PATH="$ROOT/envs/r/bin:$PATH"
export CC="$ROOT/envs/r/bin/x86_64-conda-linux-gnu-cc"
export CXX="$ROOT/envs/r/bin/x86_64-conda-linux-gnu-c++"
cat > "$ROOT/install_r_github.R" <<'RS'
options(repos = c(CRAN = "https://cloud.r-project.org"), Ncpus = 8)
ok <- function(p) requireNamespace(p, quietly = TRUE)
tryit <- function(e, nm) {
  cat("\n===", nm, "===\n"); r <- try(eval(e), silent = FALSE)
  cat(if (inherits(r, "try-error")) paste("FAILED:", nm) else paste("OK:", nm), "\n")
}
# CRF is archived on CRAN, so install.packages() cannot resolve it by name.
if (!ok("CRF")) tryit(quote(install.packages(
  "https://cran.r-project.org/src/contrib/Archive/CRF/CRF_0.4-3.tar.gz",
  repos = NULL, type = "source")), "CRF")
if (!ok("sccore"))    tryit(quote(remotes::install_github("kharchenkolab/sccore", upgrade="never")), "sccore")
if (!ok("spacexr"))   tryit(quote(remotes::install_github("dmcable/spacexr", upgrade="never", build_vignettes=FALSE)), "spacexr")
if (!ok("SPLIT"))     tryit(quote(remotes::install_github("bdsc-tds/SPLIT", upgrade="never", build_vignettes=FALSE)), "SPLIT")
if (!ok("cellAdmix")) tryit(quote(remotes::install_github("kharchenkolab/cellAdmix", upgrade="never", build_vignettes=FALSE)), "cellAdmix")
cat("\n--- final ---\n")
for (p in c("anndata","spacexr","SPLIT","cellAdmix","sccore","CRF","Seurat","NMF","nnls"))
  cat(sprintf("%-12s %s\n", p, if (ok(p)) "PRESENT" else "MISSING"))
RS
"$ROOT/envs/r/bin/Rscript" "$ROOT/install_r_github.R" >"$LOG/r_github.log" 2>&1
tail -14 "$LOG/r_github.log"

step "DONE — now run: export SEGBENCH_ENV_ROOT=$ROOT && ./bin/segbench doctor"

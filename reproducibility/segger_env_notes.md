# Segger environment notes

Segger 0.1.0 (`git+https://github.com/EliHei2/segger_dev.git`) does not install
runnable out of the box. Three fixes were needed, all discovered by running it:

## 1. Stray text committed into the source

`segger/cli/train_model.py` contains a literal `uv a` line in the middle of the
module, which makes it unimportable (`SyntaxError`). Removed; the diff is at
`reproducibility/patches/segger_train_model_stray_line.patch`.

## 2. Undeclared dependencies

`train_model` imports `predict_parquet`, which pulls in a chain the package
metadata never declares:

```bash
pip install "dask[distributed]" dask-geopandas pqdm torchmetrics
```

## 3. cupy is imported unconditionally

`predict_parquet.py` does `import cupy as cp` at module scope, so even a
CPU-only run needs cupy importable:

```bash
pip install cupy-cuda12x     # the import succeeds without a GPU present
```

## CPU vs GPU

Segger is designed for CUDA. On this cluster the only account carrying
`qos_gpu` has a lifetime allocation of 60 CPU-minutes / 5 GPU-minutes, so
Segger was run with `--accelerator cpu`.

**Its runtime and peak RSS are therefore not comparable** to the other methods
in the table, and are marked as such. Cell counts, transcript assignment and
all reference-based metrics are unaffected by the accelerator and stay
comparable.

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

---

## Status on this cluster: BLOCKED upstream

Segger did not produce a result. The blocker is not the environment or the
allocation — it is that segger_dev v0.1.0 is internally inconsistent across
its CLI, training module and model. Five distinct failures, each fixed only to
expose the next:

| # | Failure | Fix |
|---|---|---|
| 1 | `SyntaxError` — a literal `uv a` line committed into `cli/train_model.py` | removed (patch recorded) |
| 2 | `ModuleNotFoundError: cupy` — `predict_parquet` imports it unconditionally, even for a CPU run | `pip install cupy-cuda12x` |
| 3 | `ModuleNotFoundError: dask.distributed` — plus `dask_geopandas`, `pqdm`, `torchmetrics`, none declared in package metadata | installed |
| 4 | `TypeError: LitSegger.__init__() got an unexpected keyword argument 'is_token_based'` — the CLI passes 9 model-construction kwargs; the installed `LitSegger` takes `(model, learning_rate)` | attempted a bridge that builds `Segger(...)` and wraps it |
| 5 | `TypeError: nan_to_num(): argument 'input' must be Tensor, not dict` inside `Segger.forward` | **not attempted** |

Failures 4 and 5 show the CLI, `training/train.py` and `models/segger_model.py`
are different generations of the same code. Fixing 5 would mean rewriting
Segger's model/training interface, at which point the numbers would describe
code we had written rather than Segger, so we stopped. The call-site bridge for
(4) was reverted; only the stray-line fix and the missing dependencies remain.

**What did work.** Preprocessing succeeded: `create_dataset_fast.py` tiled the
full TSU-20 sample into **298 PyG tiles** (train/val/test) and those are on
disk. Only training onward is blocked.

**To unblock**, one of: a segger_dev revision whose CLI/training/model agree,
a GPU allocation large enough to test the CUDA path (the maintainers likely
only exercise that), or upstream issues for (4) and (5).

`segbench doctor` reports segger READY because its package and interpreter
resolve — readiness is an import check, not a guarantee that upstream code
runs. The comparison table carries Segger with `status="blocked_upstream"`
rather than a fabricated row.

# Segger: migration to the supported upstream, and why it is still unrun

## Summary

The wrapper targeted `EliHei2/segger_dev` v0.1.0, which is deprecated and
internally inconsistent. It has been migrated to the supported
`dpeerlab/segger` **v0.2.0**. The environment builds cleanly and the v0.1.0
blockers are gone. Segger still does not appear in the benchmark, for a reason
that is neither the wrapper's nor the environment's: **v0.2.0 has no CPU code
path, and no GPU is obtainable on this cluster.**

## Why the old target was abandoned

`segger_dev` carries a migration notice as of March 2026 pointing at
`dpeerlab/segger`. Its v0.1.0 could not be made to run: the CLI, the training
module and the model were different generations of the same code, and five
failures cascaded -- a literal `uv a` line committed into `cli/train_model.py`,
an unconditional `import cupy`, three undeclared dependency chains, a
`LitSegger.__init__()` signature the CLI did not match, and finally
`nan_to_num(): argument 'input' must be Tensor, not dict` inside
`Segger.forward`. Fixing the last would have meant rewriting the model and
training interface, at which point the numbers would no longer describe Segger.

## The pinned target

| | |
|---|---|
| repository | `https://github.com/dpeerlab/segger` |
| tag | `v0.2.0` |
| commit | `ca7bf1caaf9a177dd1f3f9051f020d2e7d3937ac` |
| Python | 3.11 |
| environment | `${SEGBENCH_ENV_ROOT}/venvs/segger_v2` |

Installed and verified: `torch 2.14.0+cpu`, `torch_geometric 2.8.0.post1`,
`lightning 2.6.5`, `segger 0.2.0` (wheel built from the pinned checkout).

The v0.1.0 defects are structurally absent in v0.2.0: there is no `LitSegger`
and no `is_token_based`, the package metadata declares its dependencies, and
the stray-source patch is unnecessary. `reproducibility/patches/segger_train_model_stray_line.patch`
applies only to the abandoned v0.1.0 and is retained for provenance.

The architecture is entirely different -- `LitISTEncoder`, `ISTDataModule`,
`ISTSegmentationWriter`, a `cyclopts` CLI with `segment` / `debug` / `export`
subcommands -- so the wrapper's invocation must be rewritten against the new
CLI, not adapted from the old one.

## The blocker: v0.2.0 is GPU-only

`segger/__init__.py` imports `cupy`, `rmm` and the RMM CuPy/Torch allocators at
module scope, so the package cannot be imported without a CUDA stack present.
That alone would be workable -- `cupy` imports without a GPU attached -- but
CUDA is not confined to initialisation:

    src/segger/geometry/quadtree.py
    src/segger/geometry/query.py
    src/segger/geometry/conversion.py
    src/segger/data/tiling.py
    src/segger/data/utils/neighbors.py

all import `cupy` / `cudf` / `cuspatial`. These are the spatial index, the
tiling and the neighbour search -- the core of the method, not optional
reporting. The `segment` CLI exposes **no** `--device` or `--accelerator`
option; v0.1.0's `--accelerator cpu` has no counterpart in the supported
release.

Running v0.2.0 on CPU would therefore mean replacing its spatial-indexing
implementation. That is rewriting the algorithm to force it to run, and the
resulting numbers would not describe Segger.

## The blocker: no GPU is obtainable

| account | QOS | GPU GrpTRESMins | usable? |
|---|---|---|---|
| `adeshpa6` | `normal` | 500,000 | **no** -- a100/ica100/l40s admit only `qos_gpu`, `qos_gpu_condo`, `urgent`, `qos_gpu_cryo`, `qos_gpu_access` |
| `aszalay1_gpu` | `normal,qos_gpu` | **5** | **no** -- three submissions returned `COMPLETED`, exit `0:0`, elapsed `00:00:00`, and never produced an output file |

The account holding the GPU budget cannot reach the GPU partitions; the account
that can reach them has a five-GPU-minute lifetime cap and its jobs terminate
before executing.

## What is needed

An allocation on an account carrying both `qos_gpu` (or `qos_gpu_condo` /
`qos_gpu_access`) and a non-trivial GPU-minute budget. The environment is built
and pinned; what remains is rewriting the wrapper against the v0.2.0 `segment`
CLI and running it on the shared input
(`filtered_df_standardized.parquet`, SHA-256 `c95693ae…62e7e`).

Until then Segger is **not benchmarked**, and no Segger row should appear in any
comparison table. Its runtime and peak memory would in any case not be
comparable to the other five imaging methods, all of which ran on CPU.

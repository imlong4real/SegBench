#!/usr/bin/env python3
"""Targeted fix: recompute Segger NPMI relative purity/conflict (the only cells
that failed in the first stage-1 pass because Segger's X was dense). Updates the
matching rows in block_ab_long.tsv in place. Run in the `spatial` env.
"""
from __future__ import annotations
import sys, logging
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "workflow" / "scripts"))
sys.path.insert(0, str(HERE))
import registry as R          # noqa
import get_metric as gm       # noqa
import build_matrices_and_metrics as B  # noqa

log = logging.getLogger("patch"); logging.basicConfig(level=logging.WARNING)
long_path = R.OUT / "block_ab_long.tsv"
df = pd.read_csv(long_path, sep="\t")

for ds in R.DATASET_ORDER:
    cfg = R.DATASETS[ds]
    a = B.build_entity_adata(ds, "segger")
    if a is None:
        continue
    nd = pd.read_csv(cfg["npmi"]); rev = nd.copy()
    rev["gene_i"], rev["gene_j"] = nd["gene_j"].values, nd["gene_i"].values
    panel = pd.concat([nd, rev], ignore_index=True); panel = panel[panel.gene_i != panel.gene_j]
    outdir = R.METRICS / ds / "segger"; outdir.mkdir(parents=True, exist_ok=True)
    nm = gm.metric_npmi_coherence(a, panel, outdir=outdir, log=log)
    for metric, key in (("relative_purity", "median_relative_purity"),
                        ("relative_conflict", "median_relative_conflict")):
        m = (df.dataset == ds) & (df.entity == "segger") & (df.metric == metric)
        df.loc[m, "value"] = nm[key]
        print(f"{ds}/segger {metric} -> {nm[key]:.4f}")

df.to_csv(long_path, sep="\t", index=False)
print("patched", long_path)

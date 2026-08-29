#!/usr/bin/env python3
"""Update Block A (size & compute) of summary_heatmaps_resegment with TRACER
metrics from the LATEST tracer_resegment run (run in the `spatial` env).

Replaces the TRACER-derived Block A values (total cells/profiles, transcripts
per cell/profile for refined & reconstructed, runtime, peak memory) in
summary_heatmaps_resegment/block_ab_long.tsv, then regenerates
source_data_blockA.tsv and benchmark_heatmap_blockA_compute.{png,svg} there.

TRACER outputs read from dataset/<ds>/tracer_resegment/:
  outputs/cell_by_gene_tracer.h5ad      -> total cells/profiles (combined)
  outputs/transcripts_tracer_refined.parquet (_etype cell/partial, label stitched)
                                        -> median transcripts/cell refined / recon
  runtime_memory.json                   -> runtime + peak memory (combined TRACER)
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
import registry as R  # noqa

# Repoint all outputs at the resegment directory so the plotter writes there.
R.OUT = R.FIG3 / "summary_heatmaps_resegment"
import consolidate_and_plot as C  # imported AFTER repointing R.OUT  # noqa


def resegment_blockA(ds):
    base = REPO / f"dataset/{ds}/tracer_resegment"
    comb = ad.read_h5ad(base / "outputs/cell_by_gene_tracer.h5ad").n_obs
    df = pd.read_parquet(base / "outputs/transcripts_tracer_refined.parquet")
    et = df["_etype"].astype(str)

    def med_tx(sub):
        g = sub.groupby("stitched").size()
        return float(g.median()) if len(g) else np.nan

    ref_tx = med_tx(df[et == "cell"])
    rec_tx = med_tx(df[et == "partial"])
    rt = json.loads((base / "runtime_memory.json").read_text())
    runtime = float(rt.get("total_seconds"))
    peak = float(rt.get("peak_rss_gb_observed", rt.get("peak_rss_gb")))
    return dict(total_cells=float(comb), refined_tx=ref_tx, recon_tx=rec_tx,
                runtime=runtime, peak=peak)


def main():
    long_path = R.OUT / "block_ab_long.tsv"
    L = pd.read_csv(long_path, sep="\t")

    def setv(ds, ent, metric, val):
        m = (L.dataset == ds) & (L.entity == ent) & (L.metric == metric)
        if m.any():
            L.loc[m, "value"] = val
            L.loc[m, "note"] = "tracer_resegment"
        else:  # ensure a row exists
            L.loc[len(L)] = dict(dataset=ds, entity=ent, metric=metric, value=val,
                                 note="tracer_resegment")

    print(f"{'dataset':20s} {'cells':>6s} {'ref_tx':>7s} {'rec_tx':>7s} {'runtime':>8s} {'peak':>6s}")
    for ds in R.DATASET_ORDER:
        v = resegment_blockA(ds)
        setv(ds, "TRACER", "total_cells", v["total_cells"])
        setv(ds, "TRACER_refined", "transcripts_per_cell", v["refined_tx"])
        setv(ds, "TRACER_reconstructed", "transcripts_per_cell", v["recon_tx"])
        setv(ds, "TRACER", "runtime_seconds", v["runtime"])
        setv(ds, "TRACER", "peak_memory_gb", v["peak"])
        print(f"{ds:20s} {v['total_cells']:6.0f} {v['refined_tx']:7.1f} {v['recon_tx']:7.1f} "
              f"{v['runtime']:8.1f} {v['peak']:6.3f}")

    L.to_csv(long_path, sep="\t", index=False)
    print("patched", long_path)

    # Regenerate Block A only (source table + figure) into the resegment dir.
    long_all = L  # block A needs only the long table
    metrics = ["total_cells", "transcripts_per_cell", "runtime_seconds", "peak_memory_gb"]
    table = C.build_block_table(long_all, metrics)
    src = R.OUT / "source_data_blockA.tsv"
    table.to_csv(src, sep="\t", index=False)
    print("wrote", src)
    C.plot_block("A", metrics, table, str(R.OUT / "benchmark_heatmap_blockA_compute"),
                 "Block A — Cell/profile size & compute (TRACER = resegment)")
    print("wrote Block A figure (png/svg)")


if __name__ == "__main__":
    main()

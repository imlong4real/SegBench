#!/usr/bin/env python3
"""Update Block B (biological coherence) of summary_heatmaps_resegment with the
latest tracer_resegment outputs (run in the `spatial` env).

Pipeline:
  1. Rebuild the resegment TRACER _work matrices (combined / refined / reconstructed)
     from dataset/<ds>/tracer_resegment/ (cell_by_gene_tracer.h5ad + the _etype
     split of transcripts_tracer_refined.parquet). Non-TRACER methods are
     independent of TRACER mode and reuse the existing resegment _work matrices.
  2. Recompute Block B (marker log2FC, NPMI relative purity/conflict, Kendall) via
     the established full-panel methodology (revise_biological): every method is
     completed to the full platform-panel gene universe (cellAdmix inherits its
     non-HVG genes from the matched original cell; all others zero-filled), KNN
     label transfer on the completed panel, 3 canonical scRNA markers/type, the
     cervical 'Unannotated' reference cluster dropped, refined/reconstructed scored
     separately, NPMI purity/conflict on QC-filtered (10-900) completed profiles.
  3. Regenerate source_data_blockB.tsv + benchmark_heatmap_blockB_biological.{png,svg}
     and all_metrics_long.tsv in the resegment dir.
  4. Emit the per-method gene-universe audit (_gene_universe_audit.tsv).

Block A and Block C figures/data are left untouched.
"""
from __future__ import annotations
import sys, logging
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "workflow" / "scripts"))
sys.path.insert(0, str(HERE))
import registry as R  # noqa

# --- Repoint every output/work path at the resegment namespace BEFORE the other
#     stage modules read them. ---
R.OUT = R.FIG3 / "summary_heatmaps_resegment"
R.WORK = R.OUT / "_work"
R.METRICS = R.OUT / "metrics"

import get_metric as gm                 # noqa: E402
import build_matrices_and_metrics as B  # noqa: E402
import revise_biological as RB          # noqa: E402
import consolidate_and_plot as C        # noqa: E402

log = logging.getLogger("resegB"); logging.basicConfig(level=logging.ERROR)
TRACER_RESEG = "tracer_resegment"


def rebuild_tracer_work(ds):
    base = REPO / f"dataset/{ds}/{TRACER_RESEG}"
    # combined
    comb = ad.read_h5ad(base / "outputs/cell_by_gene_tracer.h5ad")
    if "counts" in comb.layers:
        comb.X = comb.layers["counts"].copy()
    comb.X = sp.csr_matrix(comb.X) if not sp.issparse(comb.X) else comb.X.tocsr()
    B.save_work_h5ad(ds, "TRACER", comb)
    # refined / reconstructed from transcripts via _etype
    df = pd.read_parquet(base / "outputs/transcripts_tracer_refined.parquet")
    df["feature_name"] = df["feature_name"].astype(str)
    et = df["_etype"].astype(str)
    for ent, kind in (("TRACER_refined", "cell"), ("TRACER_reconstructed", "partial")):
        a = gm.build_cellxgene(df.loc[et == kind], "stitched", keep_ids=None, log=log)
        B.save_work_h5ad(ds, ent, a)
    print(f"  [{ds}] rebuilt TRACER work: combined={comb.n_obs}, "
          f"refined={ad.read_h5ad(R.work_h5ad(ds,'TRACER_refined')).n_obs}, "
          f"recon={ad.read_h5ad(R.work_h5ad(ds,'TRACER_reconstructed')).n_obs}")


def gene_universe_audit():
    rows = []
    HVG_METHODS = {"celladmix"}  # methods known to emit an HVG subset
    for ds in R.DATASET_ORDER:
        panels = {e: set(map(str, ad.read_h5ad(R.work_h5ad(ds, e)).var_names))
                  for e in R.ENTITY_ORDER if R.work_h5ad(ds, e).exists()}
        universe = set.union(*panels.values())
        for ent, genes in panels.items():
            frac = len(genes) / len(universe)
            is_hvg = ent in HVG_METHODS and frac < 0.5
            rows.append(dict(
                dataset=ds, platform=R.DATASETS[ds]["platform"], method=ent,
                n_genes_measured=len(genes), universe_size=len(universe),
                frac_of_universe=round(frac, 3),
                panel_type=("HVG-subset" if is_hvg else "full-measured-panel"),
                completion=("inherit non-HVG genes from matched original cell"
                            if is_hvg else
                            ("none (already full panel)" if frac > 0.95
                             else "zero-fill unreported genes (de-novo, genuine 0)")),
            ))
    aud = pd.DataFrame(rows)
    aud.to_csv(R.OUT / "_gene_universe_audit.tsv", sep="\t", index=False)
    return aud


def regenerate_blockB():
    long_ab = pd.read_csv(R.OUT / "block_ab_long.tsv", sep="\t")
    block_c = C.load_block_c()
    long_all = pd.concat([long_ab, block_c], ignore_index=True)
    long_all.to_csv(R.OUT / "all_metrics_long.tsv", sep="\t", index=False)
    metrics = ["marker_log2fc", "relative_purity", "relative_conflict", "kendall_tau"]
    table = C.build_block_table(long_all, metrics)
    table.to_csv(R.OUT / "source_data_blockB.tsv", sep="\t", index=False)
    C.plot_block("B", metrics, table, str(R.OUT / "benchmark_heatmap_blockB_biological"),
                 "Block B — Biological coherence (TRACER = resegment)")


def main():
    print("=== rebuild TRACER work matrices from tracer_resegment ===")
    for ds in R.DATASET_ORDER:
        rebuild_tracer_work(ds)
    print("\n=== gene-universe audit ===")
    aud = gene_universe_audit()
    print(aud.to_string(index=False))
    print("\n=== recompute Block B (full-panel methodology) ===")
    # snapshot previous Block B for the change report
    prev = pd.read_csv(R.OUT / "block_ab_long.tsv", sep="\t")
    RB.main()
    print("\n=== regenerate Block B figure + source ===")
    regenerate_blockB()
    # change report (TRACER entities, Block B metrics)
    new = pd.read_csv(R.OUT / "block_ab_long.tsv", sep="\t")

    def val(df, ds, ent, m):
        s = df[(df.dataset == ds) & (df.entity == ent) & (df.metric == m)]
        return float(s["value"].iloc[0]) if len(s) and pd.notna(s["value"].iloc[0]) else np.nan
    rows = []
    for ds in R.DATASET_ORDER:
        for ent in R.ENTITY_ORDER:
            for m in ["marker_log2fc", "relative_purity", "relative_conflict", "kendall_tau"]:
                o, n = val(prev, ds, ent, m), val(new, ds, ent, m)
                if pd.isna(o) and pd.isna(n):
                    continue
                rows.append(dict(dataset=ds, entity=ent, metric=m, previous=o, updated=n,
                                 delta=(n - o) if pd.notna(o) and pd.notna(n) else np.nan))
    chg = pd.DataFrame(rows)
    chg.to_csv(R.OUT / "_blockB_value_changes.tsv", sep="\t", index=False)
    print("\n=== Block B value changes (TRACER entities) ===")
    print(chg[chg.entity.str.startswith("TRACER")].to_string(index=False))
    print("\nWROTE source_data_blockB.tsv, benchmark_heatmap_blockB_biological.{png,svg}, "
          "_gene_universe_audit.tsv, _blockB_value_changes.tsv")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit fixes for the TRACER-derived metrics (run in the `spatial` env).

FIX A (marker specificity fairness): panels differ hugely across methods
(cellAdmix/proseg use reduced panels), so each method was previously scored on a
different marker subset. Recompute marker log2FC for ALL entities on a COMMON
marker set = top-N reference markers restricted to genes present in EVERY method's
panel for that dataset. Saves the common marker list per dataset.

FIX B (biological coherence panel): the purity/conflict metric is recomputed
against the REFERENCE-derived NPMI panel each TRACER run was refined against
(scRNA co-expression; nuclear self-reference for MERFISH) instead of the
transcript-derived spatial panel, which is partly self-referential and confounded
by transcripts-per-cell. NPMI (bounded [-1,1]) is used rather than PMI (unbounded,
many indeterminate pairs in the reference panels).

Updates block_ab_long.tsv in place; writes provenance comparison + marker lists.
Reuses the cell-type annotations saved by build_matrices_and_metrics.py.
"""
from __future__ import annotations
import sys, logging
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "workflow" / "scripts"))
sys.path.insert(0, str(HERE))
import get_metric as gm   # noqa
import registry as R      # noqa
import build_matrices_and_metrics as B  # noqa

log = logging.getLogger("audit"); logging.basicConfig(level=logging.ERROR)
N_TOP = 30


def common_panel(ds):
    import anndata as ad
    panels = []
    for ent in R.ENTITY_ORDER:
        p = R.work_h5ad(ds, ent)
        if p.exists():
            panels.append(set(map(str, ad.read_h5ad(p).var_names)))
    return set.intersection(*panels)


def restricted_reference(ref, keep_genes):
    keep = [i for i, g in enumerate(ref.var_names) if g in keep_genes]
    return gm.ReferenceData(
        counts_csr=ref.counts_csr[:, keep].tocsr(),
        var_names=np.asarray(ref.var_names)[keep],
        obs=ref.obs, celltype_col=ref.celltype_col)


def load_ref_npmi(path):
    d = pd.read_csv(path)
    d = d[np.isfinite(pd.to_numeric(d["NPMI"], errors="coerce"))].copy()
    d["NPMI"] = d["NPMI"].astype(float)
    rev = d.copy(); rev["gene_i"], rev["gene_j"] = d["gene_j"].values, d["gene_i"].values
    o = pd.concat([d[["gene_i", "gene_j", "NPMI"]], rev[["gene_i", "gene_j", "NPMI"]]],
                  ignore_index=True)
    return o[o.gene_i != o.gene_j]


def main():
    import anndata as ad
    long_path = R.OUT / "block_ab_long.tsv"
    L = pd.read_csv(long_path, sep="\t")
    comp_rows = []

    for ds in R.DATASET_ORDER:
        cfg = R.DATASETS[ds]
        cp = common_panel(ds)
        ref = gm.load_reference(Path(cfg["reference_h5ad"]), cfg["reference_celltype_col"], log)
        cp_ref = cp & set(ref.var_names)
        ref_r = restricted_reference(ref, cp_ref)
        markers = gm.compute_reference_markers(ref_r, n_top=N_TOP, log=log)
        markers.to_csv(R.METRICS / ds / "_common_marker_genes.tsv", sep="\t", index=False)
        npmi_ref = load_ref_npmi(cfg["npmi_reference"])
        print(f"\n=== {ds}: common_panel={len(cp)} genes, common∩ref={len(cp_ref)}, "
              f"markers={len(markers)} ({markers['cell_type'].nunique()} types), "
              f"ref_NPMI={Path(cfg['npmi_reference']).name} ===")

        for ent in R.ENTITY_ORDER:
            p = R.work_h5ad(ds, ent)
            ann_f = R.METRICS / ds / ent / "post_celltype_annotations.tsv"
            if not p.exists():
                continue
            a = ad.read_h5ad(p)
            ann = pd.read_csv(ann_f, sep="\t") if ann_f.exists() else pd.DataFrame()
            outdir = R.METRICS / ds / ent
            # --- Fix A: marker on common set
            new_marker = np.nan
            if not ann.empty and not markers.empty:
                md = gm.metric_marker_specificity(a, ann, markers, method=ent, outdir=outdir, log=log)
                md.to_csv(outdir / "marker_specificity_log2fc_commonset.tsv", sep="\t", index=False)
                if not md.empty:
                    new_marker = float(md["spatial_log2fc"].median())
            # --- Fix B: purity/conflict on reference NPMI
            nm = gm.metric_npmi_coherence(a, npmi_ref, outdir=outdir, log=log)
            new_pur, new_conf = nm["median_relative_purity"], nm["median_relative_conflict"]

            def old(metric):
                s = L[(L.dataset == ds) & (L.entity == ent) & (L.metric == metric)]
                return float(s["value"].iloc[0]) if len(s) and pd.notna(s["value"].iloc[0]) else np.nan

            comp_rows.append(dict(dataset=ds, entity=ent,
                                  marker_old=old("marker_log2fc"), marker_new=new_marker,
                                  purity_old=old("relative_purity"), purity_new=new_pur,
                                  conflict_old=old("relative_conflict"), conflict_new=new_conf))
            # update long table
            for metric, val in (("marker_log2fc", new_marker),
                                ("relative_purity", new_pur),
                                ("relative_conflict", new_conf)):
                m = (L.dataset == ds) & (L.entity == ent) & (L.metric == metric)
                if m.any():
                    L.loc[m, "value"] = val
            print(f"  {ent:22s} marker {old('marker_log2fc'):.3f}->{new_marker:.3f}  "
                  f"purity {old('relative_purity'):.3f}->{new_pur:.3f}  "
                  f"conflict {old('relative_conflict'):.3f}->{new_conf:.3f}")

    L.to_csv(long_path, sep="\t", index=False)
    pd.DataFrame(comp_rows).to_csv(R.OUT / "_audit_recompute_comparison.tsv", sep="\t", index=False)
    print("\nUPDATED", long_path)


if __name__ == "__main__":
    main()

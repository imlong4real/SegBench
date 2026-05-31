#!/usr/bin/env python3
"""v2 source-data builder: Section 7 per-cell-type reference-consistency heatmap.

Collects pearson_r per (method, cell_type) from each method's
reference_consistency_by_celltype table (SPLIT uses the repaired v2 table), and
computes the missing TRACER-reconstructed (partial-cell) row de novo: build the
partial-cell x gene matrix from the refined transcripts, KNN-annotate against
lung_cancer_50k.h5ad / Cell_Cluster_level1 (same recipe as every other method),
then pseudobulk Pearson over panel-and-reference shared genes.

Outputs:
  source_data/reference_consistency_heatmap_v2.tsv
  source_data/reference_consistency_gene_cell_counts_v2.tsv
"""
from __future__ import annotations
import sys, logging
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _v2_common as C
import get_metric

log = logging.getLogger("v2refcon")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s",
                    datefmt="%H:%M:%S")


def _tracer_partial_refconsist(ref):
    """Pseudobulk reference consistency for TRACER reconstructed (partial) cells."""
    import anndata as ad
    import scipy.sparse as sp
    from scipy.stats import pearsonr
    # partial-cell x gene from refined transcripts
    df = pd.read_parquet(C.TRANSCRIPTS["TRACER"], columns=["cell_id", "feature_name", "_etype"])
    df["cell_id"] = df["cell_id"].astype(str)
    df = df[(~df["cell_id"].isin(C.UNASSIGNED_TOKENS)) & (df["_etype"].astype(str) == "partial")]
    ct = pd.crosstab(df["cell_id"], df["feature_name"])
    a = ad.AnnData(X=sp.csr_matrix(ct.values.astype(np.float64)),
                   obs=pd.DataFrame(index=ct.index.astype(str)),
                   var=pd.DataFrame(index=ct.columns.astype(str)))
    ann = get_metric.transfer_labels(a, ref, seed=0, k=15, per_type=300, log=log)
    labels = ann.set_index("cell_id").reindex(a.obs_names.astype(str))["predicted_celltype"].astype(str).to_numpy()

    ref_gi = {g: i for i, g in enumerate(map(str, ref.var_names))}
    present = [g for g in a.var_names if g in ref_gi]
    Xq = a[:, present].X
    Xq = np.asarray(Xq.todense() if hasattr(Xq, "todense") else Xq, dtype=np.float64)
    Xr = ref.counts_csr.tocsr() if sp.issparse(ref.counts_csr) else sp.csr_matrix(np.asarray(ref.counts_csr))
    Xr = Xr[:, [ref_gi[g] for g in present]].astype(np.float64)
    sr = np.asarray(Xr.sum(axis=1)).ravel(); sr[sr == 0] = 1.0
    Xr = Xr.multiply(1e4 / sr[:, None]).tocsr(); Xr.data = np.log1p(Xr.data)
    rl = ref.obs[C.REF_CELLTYPE_COL].astype(str).to_numpy()
    rows = []
    for ctype in C.CELLTYPES9:
        qm = labels == ctype; nq = int(qm.sum()); nr = int((rl == ctype).sum())
        r = np.nan; reason = ""
        if nq < 10 or nr < 10:
            reason = "fewer than 10 cells"
        else:
            qb = C.lognorm_pseudobulk(Xq[qm])
            rb = np.asarray(Xr[rl == ctype].mean(axis=0)).ravel()
            r = float(pearsonr(qb, rb)[0]) if qb.std() > 0 and rb.std() > 0 else np.nan
        rows.append(dict(method="TRACER-reconstructed", cell_type=ctype,
                         n_spatial_cells=nq, n_reference_cells=nr,
                         n_genes_used=len(present), pearson_r=r, reason=reason))
    return pd.DataFrame(rows)


def main():
    C.ensure_dirs()
    frames = []
    for raw, disp in [("original", "original"), ("Baysor", "Baysor"),
                      ("proseg", "proseg"), ("segger", "Segger"),
                      ("cellAdmix", "cellAdmix"), ("SPLIT", "SPLIT"),
                      ("TRACER-refined", "TRACER-refined")]:
        p = C.REF_CONSIST[raw]
        if not Path(p).exists():
            log.warning("missing %s", p); continue
        d = pd.read_csv(p, sep="\t")
        d = d[d["cell_type"].isin(C.CELLTYPES9)].copy()
        d["method"] = disp
        keep = ["method", "cell_type", "n_spatial_cells", "n_reference_cells",
                "n_genes_used", "pearson_r"]
        for k in keep:
            if k not in d.columns:
                d[k] = np.nan
        frames.append(d[keep])

    ref = get_metric.load_reference(C.REFERENCE_H5AD, C.REF_CELLTYPE_COL, log)
    tp = _tracer_partial_refconsist(ref)
    frames.append(tp[["method", "cell_type", "n_spatial_cells", "n_reference_cells",
                      "n_genes_used", "pearson_r"]])

    alld = pd.concat(frames, ignore_index=True)
    alld["method"] = pd.Categorical(alld["method"], C.METHOD_ORDER, ordered=True)
    alld["cell_type"] = pd.Categorical(alld["cell_type"], C.CELLTYPES9, ordered=True)
    alld = alld.sort_values(["method", "cell_type"])

    # Heatmap matrix (method x cell_type pearson_r)
    mat = alld.pivot_table(index="method", columns="cell_type",
                           values="pearson_r", observed=False)
    mat = mat.reindex(index=C.METHOD_ORDER, columns=C.CELLTYPES9)
    mat_out = mat.reset_index().rename(columns={"index": "method"})
    C.save_source(mat_out, "reference_consistency_heatmap_v2.tsv")

    counts = alld[["method", "cell_type", "n_spatial_cells", "n_reference_cells",
                   "n_genes_used", "pearson_r"]]
    C.save_source(counts, "reference_consistency_gene_cell_counts_v2.tsv")
    log.info("heatmap methods=%d celltypes=%d; n nonnull=%d/%d",
             mat.shape[0], mat.shape[1], int(mat.notna().sum().sum()), mat.size)
    # compact receipt
    med = alld.groupby("method", observed=True)["pearson_r"].median()
    (C.FIGDIR / "_v2_refcon_receipt.txt").write_text(
        "\n".join(f"{m} med_r={med.get(m, float('nan')):.3f}" for m in C.METHOD_ORDER))
    print("DONE refconsist")


if __name__ == "__main__":
    main()

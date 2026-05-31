#!/usr/bin/env python3
"""v2 source-data builder: Section 6 marker panels.

Derives a FIXED marker set from the scRNA reference (lung_cancer_50k.h5ad,
Cell_Cluster_level1), restricted to the Xenium panel:
  * 45 lineage markers = top-5 per the 9 lineages (by log2FC vs rest), and
  * 5 T-cell markers (the top-5 for the 'T' lineage).
Then scores every method on those genes: spatial log2FC of each marker in its
annotated lineage vs all other annotated cells (CP10k -> log1p over the full
panel; mu_in - mu_out), plus the dedicated T-cell-marker log2FC in cells labelled
'T'. Fixes the v1 bug where the T-cell metric looked for the label 'T-cell'
(annotations use the bare 'T'), so it was empty for every method.

Outputs:
  source_data/lineage_marker_genes_45_v2.tsv
  source_data/tcell_marker_genes_v2.tsv
  source_data/marker_specificity_v2.tsv
  source_data/marker_specificity_stats_v2.tsv
  source_data/tcell_marker_log2fc_v2.tsv
"""
from __future__ import annotations
import sys, logging
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _v2_common as C
import get_metric

log = logging.getLogger("v2markers")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s",
                    datefmt="%H:%M:%S")

N_MARKERS = 5
# Curated lineage-defining / rare-cell markers to flag if present among selected.
LINEAGE_HIGHLIGHT = {
    "CD3D", "CD3E", "CD2", "CD8A", "TRAC", "IL7R",            # T
    "MS4A1", "CD79A", "CD79B", "CD19", "BANK1",               # B
    "MZB1", "IGHG1", "DERL3", "TNFRSF17", "JCHAIN",           # Plasma
    "CPA3", "TPSAB1", "TPSB2", "MS4A2", "KIT",                # Mast
    "PECAM1", "VWF", "CLDN5", "CLEC14A", "CDH5",              # Endothelial
    "DCN", "LUM", "COL1A1", "COL1A2", "PDGFRB", "ACTA2",      # Fibroblasts
    "LYZ", "CD68", "CD14", "C1QA", "C1QB", "ITGAM", "MARCO",  # Myeloid
    "EPCAM", "KRT19", "KRT8", "KRT18", "KRT17",               # Cancer/epithelial
    "FOXJ1", "PIFO", "TPPP3", "CAPS", "SNTN",                 # Ciliated
}


def panel_genes_from_original():
    df = pd.read_parquet(C.TRANSCRIPTS["original"], columns=["feature_name"])
    g = sorted(set(df["feature_name"].astype(str)))
    # drop common control/blank probes
    g = [x for x in g if not x.lower().startswith(("blank", "negcontrol",
                                                   "antisense", "deprecated",
                                                   "unassigned", "control"))]
    return g


def cp10k_log1p_full(adata):
    import scanpy as sc
    a = adata.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    X = a.X
    return np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=np.float64)


def load_method_matrix(method):
    """Return (adata raw counts cells x panel-genes, labels Series cell_id->type)
    for a display-method, with the 10<n<900 filter applied."""
    import anndata as ad, scipy.sparse as sp
    if method in ("original", "Baysor", "proseg", "Segger", "cellAdmix"):
        raw = {"Segger": "segger"}.get(method, method)
        ct = C.build_cell_by_gene(C.TRANSCRIPTS[raw])         # cells x genes (counts)
        a = ad.AnnData(X=sp.csr_matrix(ct.values.astype(np.float64)),
                       obs=pd.DataFrame(index=ct.index.astype(str)),
                       var=pd.DataFrame(index=ct.columns.astype(str)))
        labels = annotation_series(C.POST_ANN[raw])
    elif method in ("TRACER-refined", "TRACER-reconstructed"):
        want_partial = method == "TRACER-reconstructed"
        df = pd.read_parquet(C.TRANSCRIPTS["TRACER"],
                             columns=["cell_id", "feature_name", "_etype"])
        df["cell_id"] = df["cell_id"].astype(str)
        df = df[(~df["cell_id"].isin(C.UNASSIGNED_TOKENS)) &
                (df["_etype"].astype(str).eq("partial") == want_partial) &
                (df["_etype"].astype(str).isin(["cell", "partial"]))]
        ct = pd.crosstab(df["cell_id"], df["feature_name"])
        a = ad.AnnData(X=sp.csr_matrix(ct.values.astype(np.float64)),
                       obs=pd.DataFrame(index=ct.index.astype(str)),
                       var=pd.DataFrame(index=ct.columns.astype(str)))
        labels = annotation_series(C.POST_ANN["TRACER"])
    elif method == "SPLIT":
        pur = ad.read_h5ad(C.SPLIT_PURIFIED_H5AD)
        a = ad.AnnData(X=sp.csr_matrix(np.rint(np.asarray(
                           pur.X.todense() if hasattr(pur.X, "todense") else pur.X)).astype(np.float64)),
                       obs=pd.DataFrame(index=pur.obs_names.astype(str)),
                       var=pd.DataFrame(index=pur.var_names.astype(str)))
        ref = _REF[0]
        ann = get_metric.transfer_labels(a, ref, seed=0, k=15, per_type=300, log=log)
        labels = ann.set_index("cell_id")["predicted_celltype"].astype(str)
    else:
        raise ValueError(method)
    # 10 < n < 900 transcript filter
    tot = np.asarray(a.X.sum(axis=1)).ravel()
    keep = (tot > C.TX_MIN) & (tot < C.TX_MAX)
    a = a[keep].copy()
    lab = labels.reindex(a.obs_names.astype(str))
    return a, lab


_REF = [None]


def annotation_series(path):
    ann = pd.read_csv(path, sep="\t")
    label_col = "cell_type" if "cell_type" in ann.columns else "predicted_celltype"
    return ann.set_index(ann["cell_id"].astype(str))[label_col].astype(str)


def main():
    C.ensure_dirs()
    import scipy.sparse as sp
    ref0 = get_metric.load_reference(C.REFERENCE_H5AD, C.REF_CELLTYPE_COL, log)
    panel = panel_genes_from_original()
    keep_genes = [g for g in panel if g in set(ref0.var_names)]
    ref_pos = {g: i for i, g in enumerate(ref0.var_names)}
    ref = get_metric.ReferenceData(
        counts_csr=ref0.counts_csr[:, [ref_pos[g] for g in keep_genes]].tocsr(),
        var_names=np.asarray(keep_genes, dtype=str),
        obs=ref0.obs.copy(),
        celltype_col=ref0.celltype_col,
    )
    _REF[0] = ref

    # ---- derive markers from reference (reuse the validated routine) ----
    markers = get_metric.compute_reference_markers(ref, n_top=N_MARKERS, log=log)

    # reference expression frequency per gene per type (for the criteria table)
    lab_ref = ref.obs[C.REF_CELLTYPE_COL].astype(str).to_numpy()
    Xr = ref.counts_csr.tocsr() if sp.issparse(ref.counts_csr) else sp.csr_matrix(np.asarray(ref.counts_csr))
    genes_ref = list(ref.var_names)
    gpos = {g: i for i, g in enumerate(genes_ref)}

    def frac_expr(gene, ctype):
        if gene not in gpos:
            return np.nan
        col = Xr[:, gpos[gene]]
        col = np.asarray(col.todense()).ravel() if hasattr(col, "todense") else np.asarray(col).ravel()
        m = lab_ref == ctype
        return float((col[m] > 0).mean()) if m.sum() else np.nan

    rows45 = []
    for ct in C.CELLTYPES9:
        subm = markers[markers["cell_type"] == ct].sort_values("rank").head(N_MARKERS)
        for _, mr in subm.iterrows():
            rank = int(mr["rank"])
            g = str(mr["gene"])
            fc = float(mr["scrna_log2fc"])
            rows45.append(dict(cell_type=ct, gene=g, rank=rank, scrna_log2fc=fc,
                               frac_expr_in_type=frac_expr(g, ct),
                               frac_expr_other=frac_expr(g, "__other__")
                               if False else np.nan,
                               is_highlight_marker=g in LINEAGE_HIGHLIGHT))
    lin45 = pd.DataFrame(rows45)
    # other-type expression frequency
    lin45["frac_expr_other"] = [
        float(((np.asarray(Xr[:, gpos[g]].todense()).ravel() if g in gpos else np.array([np.nan]))[lab_ref != ct] > 0).mean())
        if g in gpos else np.nan
        for g, ct in zip(lin45["gene"], lin45["cell_type"])]
    lin45["selection_criteria"] = ("top-%d by scRNA log2FC (Cell_Cluster_level1 vs rest), "
                                   "restricted to Xenium panel" % N_MARKERS)
    C.save_source(lin45, "lineage_marker_genes_45_v2.tsv")

    tcell = lin45[lin45.cell_type == "T"].copy()
    tcell["selection_criteria"] = ("top-5 T-cell markers by scRNA log2FC (T vs rest), "
                                   "restricted to Xenium panel; expression frequency reported")
    C.save_source(tcell[["cell_type", "gene", "rank", "scrna_log2fc",
                         "frac_expr_in_type", "frac_expr_other", "is_highlight_marker",
                         "selection_criteria"]], "tcell_marker_genes_v2.tsv")
    tcell_genes = list(tcell.sort_values("rank")["gene"])
    log.info("T-cell markers: %s", tcell_genes)

    # ---- per-method spatial log2FC for the fixed marker set ----
    spec_rows = []
    tcell_rows = []
    marker_lookup = {(str(r.cell_type), str(r.gene)): float(r.scrna_log2fc)
                     for r in lin45.itertuples(index=False)}
    for method in C.METHOD_ORDER:
        a, lab = load_method_matrix(method)
        Xn = cp10k_log1p_full(a)
        gidx = {g: i for i, g in enumerate(a.var_names)}
        labv = lab.astype(str).to_numpy()
        for (ct, g), fc in marker_lookup.items():
            if g not in gidx:
                continue
            vals = Xn[:, gidx[g]]
            in_m = labv == ct
            out_m = (labv != ct) & (labv != "nan")
            if in_m.sum() < 3 or out_m.sum() < 3:
                spec_rows.append(dict(method=method, cell_type=ct, gene=g,
                                      scrna_log2fc=fc, spatial_log2fc=np.nan,
                                      n_cells_in=int(in_m.sum()), n_cells_out=int(out_m.sum()),
                                      note="too few cells in lineage"))
                continue
            mu_in = float(vals[in_m].mean()); mu_out = float(vals[out_m].mean())
            spec_rows.append(dict(method=method, cell_type=ct, gene=g, scrna_log2fc=fc,
                                  spatial_log2fc=mu_in - mu_out,
                                  spatial_mean_in=mu_in, spatial_mean_out=mu_out,
                                  n_cells_in=int(in_m.sum()), n_cells_out=int(out_m.sum()),
                                  note=""))
        # dedicated T-cell markers in cells labelled 'T'
        in_t = labv == "T"; out_t = (labv != "T") & (labv != "nan")
        for g in tcell_genes:
            if g not in gidx:
                tcell_rows.append(dict(method=method, gene=g, spatial_log2fc=np.nan,
                                       n_tcells=int(in_t.sum()), n_other=int(out_t.sum()),
                                       note="gene not in panel")); continue
            if in_t.sum() < 3 or out_t.sum() < 3:
                tcell_rows.append(dict(method=method, gene=g, spatial_log2fc=np.nan,
                                       n_tcells=int(in_t.sum()), n_other=int(out_t.sum()),
                                       note="too few T cells")); continue
            vals = Xn[:, gidx[g]]
            tcell_rows.append(dict(method=method, gene=g,
                                   spatial_log2fc=float(vals[in_t].mean() - vals[out_t].mean()),
                                   spatial_mean_in=float(vals[in_t].mean()),
                                   spatial_mean_out=float(vals[out_t].mean()),
                                   n_tcells=int(in_t.sum()), n_other=int(out_t.sum()), note=""))
        log.info("%s scored (%d cells; T cells=%d)", method, a.n_obs, int(in_t.sum()))

    spec = pd.DataFrame(spec_rows)
    C.save_source(spec, "marker_specificity_v2.tsv")
    tdf = pd.DataFrame(tcell_rows)
    C.save_source(tdf, "tcell_marker_log2fc_v2.tsv")

    # ---- stats: each method vs TRACER-refined (paired one-sided Wilcoxon) ----
    piv = spec.pivot_table(index=["cell_type", "gene"], columns="method",
                           values="spatial_log2fc", observed=False)
    stat_rows = []
    if "TRACER-refined" in piv.columns:
        rref = piv["TRACER-refined"]
        for m in C.METHOD_ORDER:
            if m == "TRACER-refined" or m not in piv.columns:
                continue
            # paired by (cell_type, gene); TRACER-refined expected >= others (higher specificity)
            stat, p, n = C.wilcoxon_one_sided(rref.to_numpy(), piv[m].to_numpy(),
                                              alternative="greater")
            stat_rows.append(dict(metric="marker_specificity_log2fc",
                                  comparison=f"TRACER-refined vs {m}",
                                  test="paired one-sided Wilcoxon signed-rank (by gene)",
                                  alternative="greater(TRACER-refined>other)",
                                  n_pairs=n, statistic=stat, p_value=p,
                                  p_label=C.p_label(p), stars=C.p_to_stars(p)))
    C.save_source(pd.DataFrame(stat_rows), "marker_specificity_stats_v2.tsv")

    # compact receipt: per-method median spatial log2fc + n T cells
    med = spec.groupby("method", observed=True)["spatial_log2fc"].median()
    tmed = tdf.groupby("method")["spatial_log2fc"].median()
    (C.FIGDIR / "_v2_markers_receipt.txt").write_text(
        "\n".join(f"{m} spec_med={med.get(m, float('nan')):.3f} "
                  f"tcell_med={tmed.get(m, float('nan')):.3f}"
                  for m in C.METHOD_ORDER))
    print("DONE markers")


if __name__ == "__main__":
    main()

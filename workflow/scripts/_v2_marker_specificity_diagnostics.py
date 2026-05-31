#!/usr/bin/env python3
"""Diagnostics for low TRACER-reconstructed marker specificity in v2.

The original v2 marker panel scored TRACER-reconstructed partial transcripts by
``cell_id``. In the TRACER refined parquet, ``cell_id`` is the source/host
original cell for partial transcripts; the reconstructed entity id is
``tracer_id``. This diagnostic rebuilds true partial-cell matrices by
``tracer_id``, labels them by the same KNN transfer used elsewhere, and tests
coverage/sparsity explanations for the 45-gene marker-specificity panel.
"""
from __future__ import annotations

import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _v2_common as C
import get_metric

log = logging.getLogger("v2_marker_diag")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s",
                    datefmt="%H:%M:%S")

OUT_PREFIX = "marker_specificity_tracer_reconstructed"
MIN_STABLE_CELLS = 10


def cp10k_log1p_dense(counts: np.ndarray) -> np.ndarray:
    M = np.asarray(counts, dtype=np.float64)
    s = M.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return np.log1p(M * 1e4 / s)


def build_true_partial_matrix():
    import anndata as ad
    import scipy.sparse as sp

    df = pd.read_parquet(C.TRANSCRIPTS["TRACER"],
                         columns=["tracer_id", "_etype", "feature_name"])
    df["tracer_id"] = df["tracer_id"].astype(str)
    df["_etype"] = df["_etype"].astype(str)
    df["feature_name"] = df["feature_name"].astype(str)
    df = df[(df["_etype"].eq("partial")) &
            (~df["tracer_id"].isin(C.UNASSIGNED_TOKENS))]
    ct = pd.crosstab(df["tracer_id"], df["feature_name"])
    a = ad.AnnData(X=sp.csr_matrix(ct.values.astype(np.float64)),
                   obs=pd.DataFrame(index=ct.index.astype(str)),
                   var=pd.DataFrame(index=ct.columns.astype(str)))
    return a


def specificity_for_markers(adata, labels: pd.Series, markers: pd.DataFrame,
                            method: str, min_cells: int = 3) -> pd.DataFrame:
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    Xn = cp10k_log1p_dense(X)
    gidx = {g: i for i, g in enumerate(map(str, adata.var_names))}
    lab = labels.reindex(adata.obs_names.astype(str)).astype(str).to_numpy()
    rows = []
    for row in markers.itertuples(index=False):
        ct = str(row.cell_type)
        gene = str(row.gene)
        in_m = lab == ct
        out_m = (lab != ct) & (lab != "nan")
        base = {
            "method": method,
            "cell_type": ct,
            "gene": gene,
            "rank": int(row.rank),
            "scrna_log2fc": float(row.scrna_log2fc),
            "n_cells_in": int(in_m.sum()),
            "n_cells_out": int(out_m.sum()),
        }
        if gene not in gidx:
            rows.append({**base, "spatial_log2fc": np.nan,
                         "spatial_mean_in": np.nan, "spatial_mean_out": np.nan,
                         "note": "gene missing from partial-cell matrix"})
            continue
        if in_m.sum() < min_cells or out_m.sum() < min_cells:
            rows.append({**base, "spatial_log2fc": np.nan,
                         "spatial_mean_in": np.nan, "spatial_mean_out": np.nan,
                         "note": "too few cells in lineage"})
            continue
        vals = Xn[:, gidx[gene]]
        mu_in = float(vals[in_m].mean())
        mu_out = float(vals[out_m].mean())
        rows.append({**base, "spatial_log2fc": mu_in - mu_out,
                     "spatial_mean_in": mu_in, "spatial_mean_out": mu_out,
                     "note": ""})
    return pd.DataFrame(rows)


def detection_tables(adata, labels: pd.Series, markers: pd.DataFrame):
    import scipy.sparse as sp

    X = adata.X.tocsr() if sp.issparse(adata.X) else sp.csr_matrix(np.asarray(adata.X))
    gidx = {g: i for i, g in enumerate(map(str, adata.var_names))}
    marker_genes = list(markers["gene"].astype(str).unique())
    present_genes = [g for g in marker_genes if g in gidx]
    marker_idx = [gidx[g] for g in present_genes]
    Xm = X[:, marker_idx]
    detected_per_cell = np.asarray((Xm > 0).sum(axis=1)).ravel()
    total_tx = np.asarray(X.sum(axis=1)).ravel()
    lab = labels.reindex(adata.obs_names.astype(str)).astype(str).to_numpy()

    per_cell = pd.DataFrame({
        "partial_id": adata.obs_names.astype(str),
        "cell_type": lab,
        "n_transcripts": total_tx.astype(int),
        "n_45_marker_genes_detected": detected_per_cell.astype(int),
        "fraction_45_marker_genes_detected": detected_per_cell / max(len(marker_genes), 1),
        "n_marker_genes_present_in_matrix": len(present_genes),
        "n_marker_genes_total": len(marker_genes),
    })

    rows = []
    for row in markers.itertuples(index=False):
        gene = str(row.gene)
        ct = str(row.cell_type)
        if gene in gidx:
            col = X[:, gidx[gene]]
            det = np.asarray((col > 0).todense()).ravel() if hasattr(col, "todense") else np.asarray(col > 0).ravel()
            n_cells = int(det.sum())
            n_tx = int(np.asarray(col.sum(axis=0)).ravel()[0])
        else:
            n_cells = 0
            n_tx = 0
        rows.append({
            "cell_type": ct,
            "gene": gene,
            "rank": int(row.rank),
            "detected_partial_cells": n_cells,
            "fraction_partial_cells_detected": n_cells / max(adata.n_obs, 1),
            "total_partial_transcripts_for_gene": n_tx,
            "present_in_partial_matrix": gene in gidx,
        })
    per_gene = pd.DataFrame(rows)

    summary = (
        per_cell.groupby("cell_type", dropna=False)
        .agg(n_partial_cells=("partial_id", "size"),
             median_transcripts_per_partial=("n_transcripts", "median"),
             mean_transcripts_per_partial=("n_transcripts", "mean"),
             median_45_marker_genes_detected=("n_45_marker_genes_detected", "median"),
             mean_45_marker_genes_detected=("n_45_marker_genes_detected", "mean"),
             fraction_cells_with_any_45_marker=("n_45_marker_genes_detected",
                                                lambda x: float((x > 0).mean())))
        .reset_index()
    )
    marker_by_lineage = (
        per_gene.groupby("cell_type", dropna=False)
        .agg(n_marker_genes=("gene", "size"),
             n_marker_genes_detected_anywhere=("detected_partial_cells",
                                               lambda x: int((x > 0).sum())),
             fraction_marker_genes_detected_anywhere=("detected_partial_cells",
                                                      lambda x: float((x > 0).mean())),
             total_marker_transcripts=("total_partial_transcripts_for_gene", "sum"))
        .reset_index()
    )
    return per_cell, per_gene, summary, marker_by_lineage


def make_plots(celltype_summary, marker_by_lineage, spec_true, represented_summary,
               small_panel_summary):
    C.apply_style()
    cts = C.CELLTYPES9
    cov = celltype_summary.set_index("cell_type").reindex(cts).reset_index()
    mdet = marker_by_lineage.set_index("cell_type").reindex(cts).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.8), gridspec_kw={"wspace": 0.36})
    x = np.arange(len(cts))
    axes[0].bar(x, cov["n_partial_cells"].fillna(0),
                color=[C.CELLTYPE_COLORS.get(ct, "#999") for ct in cts])
    axes[0].axhline(MIN_STABLE_CELLS, ls="--", lw=0.7, color="#777")
    axes[0].set_title("Partial-Cell Coverage", loc="left", fontweight="bold")
    axes[0].set_ylabel("n partial cells")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(cts, rotation=40, ha="right")
    axes[1].bar(x, cov["median_transcripts_per_partial"].fillna(0),
                color=[C.CELLTYPE_COLORS.get(ct, "#999") for ct in cts])
    axes[1].set_title("Transcript Sparsity", loc="left", fontweight="bold")
    axes[1].set_ylabel("Median tx / partial")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(cts, rotation=40, ha="right")
    axes[2].bar(x, mdet["fraction_marker_genes_detected_anywhere"].fillna(0),
                color=[C.CELLTYPE_COLORS.get(ct, "#999") for ct in cts])
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title("45-Marker Coverage", loc="left", fontweight="bold")
    axes[2].set_ylabel("Fraction marker genes detected")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(cts, rotation=40, ha="right")
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#e6e1dc", lw=0.5)
        ax.set_axisbelow(True)
    C.save_fig(fig, "figure_marker_specificity_tracer_reconstructed_diagnostics_v2")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    plot_rows = []
    for label, df in [
        ("true partial\n45 markers", spec_true),
        ("represented\nlineages", represented_summary),
        ("top1", small_panel_summary[small_panel_summary["panel"] == "top1"]),
        ("top2", small_panel_summary[small_panel_summary["panel"] == "top2"]),
        ("top3", small_panel_summary[small_panel_summary["panel"] == "top3"]),
    ]:
        vals = pd.to_numeric(df["spatial_log2fc"], errors="coerce").dropna()
        for v in vals:
            plot_rows.append((label, float(v)))
    p = pd.DataFrame(plot_rows, columns=["panel", "spatial_log2fc"])
    order = ["true partial\n45 markers", "represented\nlineages", "top1", "top2", "top3"]
    rng = np.random.default_rng(3)
    for i, panel in enumerate(order):
        vals = p.loc[p["panel"].eq(panel), "spatial_log2fc"].to_numpy()
        if vals.size == 0:
            continue
        ax.boxplot(vals, positions=[i], widths=0.45, showfliers=False,
                   patch_artist=True,
                   boxprops=dict(facecolor="#f8f7f5", edgecolor="#c0306a", lw=0.9),
                   medianprops=dict(color="#333", lw=1.0),
                   whiskerprops=dict(color="#777", lw=0.7),
                   capprops=dict(color="#777", lw=0.7))
        ax.scatter(i + rng.normal(0, 0.045, vals.size), vals, s=16,
                   color="#c0306a", alpha=0.65, edgecolor="white", lw=0.25)
    ax.axhline(0, lw=0.7, color="#777")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_ylabel("Marker specificity log2FC")
    ax.set_title("TRACER-Reconstructed Sensitivity Checks", loc="left",
                 fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e6e1dc", lw=0.5)
    C.save_fig(fig, "figure_marker_specificity_tracer_reconstructed_sensitivity_v2")
    plt.close(fig)


def main():
    C.ensure_dirs()
    markers = pd.read_csv(C.SRCDIR / "lineage_marker_genes_45_v2.tsv", sep="\t")
    spec_v2 = pd.read_csv(C.SRCDIR / "marker_specificity_v2.tsv", sep="\t")
    partial = build_true_partial_matrix()
    total_tx = np.asarray(partial.X.sum(axis=1)).ravel()
    partial_filt = partial[(total_tx > C.TX_MIN) & (total_tx < C.TX_MAX)].copy()

    ref = get_metric.load_reference(C.REFERENCE_H5AD, C.REF_CELLTYPE_COL, log)
    ann = get_metric.transfer_labels(partial_filt, ref, seed=0, k=15,
                                     per_type=300, log=log)
    labels = ann.set_index("cell_id")["predicted_celltype"].astype(str)

    per_cell, per_gene, celltype_summary, marker_by_lineage = detection_tables(
        partial_filt, labels, markers)
    spec_true = specificity_for_markers(partial_filt, labels, markers,
                                        "TRACER-reconstructed.true_tracer_id")

    represented = (celltype_summary[celltype_summary["n_partial_cells"] >= MIN_STABLE_CELLS]
                   ["cell_type"].astype(str).tolist())
    represented_markers = markers[markers["cell_type"].isin(represented)]
    spec_represented_true = spec_true[spec_true["cell_type"].isin(represented)].copy()

    # Recalculate all-method summaries from the existing source table after
    # restricting to lineages represented by true reconstructed partial cells.
    represented_all_methods = spec_v2[spec_v2["cell_type"].isin(represented)].copy()
    represented_all_methods = pd.concat([
        represented_all_methods[~represented_all_methods["method"].eq("TRACER-reconstructed")],
        spec_represented_true.assign(method="TRACER-reconstructed.true_tracer_id"),
    ], ignore_index=True)
    represented_summary = (
        represented_all_methods.groupby("method", dropna=False)
        .agg(n_rows=("gene", "size"),
             n_scored=("spatial_log2fc", lambda x: int(pd.Series(x).notna().sum())),
             median_spatial_log2fc=("spatial_log2fc", "median"),
             mean_spatial_log2fc=("spatial_log2fc", "mean"),
             represented_cell_types=("cell_type",
                                     lambda x: ",".join(sorted(set(map(str, x))))))
        .reset_index()
    )

    small_rows = []
    small_specs = []
    for n in (1, 2, 3):
        mk = represented_markers[represented_markers["rank"] <= n].copy()
        sp = spec_true[(spec_true["cell_type"].isin(represented)) &
                       (spec_true["rank"] <= n)].copy()
        sp["panel"] = f"top{n}"
        small_specs.append(sp)
        small_rows.append({
            "panel": f"top{n}",
            "n_lineages": len(represented),
            "n_marker_rows": len(mk),
            "n_scored_rows": int(sp["spatial_log2fc"].notna().sum()),
            "median_spatial_log2fc": float(sp["spatial_log2fc"].median()),
            "mean_spatial_log2fc": float(sp["spatial_log2fc"].mean()),
            "cell_types": ",".join(represented),
            "genes": ",".join(mk["gene"].astype(str)),
        })
    small_specs = pd.concat(small_specs, ignore_index=True)
    small_summary = pd.DataFrame(small_rows)

    v2_old_partial = spec_v2[spec_v2["method"].eq("TRACER-reconstructed")].copy()
    old_vs_true = pd.DataFrame([
        {
            "source": "v2_plotted_host_cell_id_grouping",
            "n_rows": len(v2_old_partial),
            "n_scored": int(v2_old_partial["spatial_log2fc"].notna().sum()),
            "median_spatial_log2fc": float(v2_old_partial["spatial_log2fc"].median()),
            "represented_cell_types_scored": ",".join(sorted(
                set(v2_old_partial.loc[v2_old_partial["spatial_log2fc"].notna(),
                                       "cell_type"].astype(str)))),
            "entity_id_basis": "cell_id (source/host original cell for partial transcripts)",
        },
        {
            "source": "diagnostic_true_tracer_id_grouping",
            "n_rows": len(spec_true),
            "n_scored": int(spec_true["spatial_log2fc"].notna().sum()),
            "median_spatial_log2fc": float(spec_true["spatial_log2fc"].median()),
            "represented_cell_types_scored": ",".join(sorted(
                set(spec_true.loc[spec_true["spatial_log2fc"].notna(),
                                  "cell_type"].astype(str)))),
            "entity_id_basis": "tracer_id (_etype=partial reconstructed entity)",
        },
    ])

    C.save_source(per_cell, f"{OUT_PREFIX}_per_partial_cell_detection_v2.tsv")
    C.save_source(per_gene, f"{OUT_PREFIX}_marker_gene_detection_v2.tsv")
    C.save_source(celltype_summary,
                  f"{OUT_PREFIX}_celltype_coverage_v2.tsv")
    C.save_source(marker_by_lineage,
                  f"{OUT_PREFIX}_marker_detection_by_lineage_v2.tsv")
    C.save_source(spec_true,
                  f"{OUT_PREFIX}_true_entity_marker_specificity_45_v2.tsv")
    C.save_source(old_vs_true,
                  f"{OUT_PREFIX}_old_vs_true_entity_audit_v2.tsv")
    C.save_source(represented_all_methods,
                  f"{OUT_PREFIX}_represented_lineages_specificity_rows_v2.tsv")
    C.save_source(represented_summary,
                  f"{OUT_PREFIX}_represented_lineages_specificity_summary_v2.tsv")
    C.save_source(small_specs,
                  f"{OUT_PREFIX}_small_panel_specificity_rows_v2.tsv")
    C.save_source(small_summary,
                  f"{OUT_PREFIX}_small_panel_specificity_summary_v2.tsv")

    make_plots(celltype_summary, marker_by_lineage, spec_true,
               spec_represented_true, small_specs)

    low_tx = float(per_cell["n_transcripts"].median())
    med_markers = float(per_cell["n_45_marker_genes_detected"].median())
    represented_text = ", ".join(represented)
    note = f"""# Marker Specificity v2 Diagnostic Notes

TRACER-reconstructed partial cells have low 45-gene marker specificity for two reasons.

First, the plotted v2 source data grouped partial transcripts by `cell_id`, but in the TRACER parquet `cell_id` is the source/host original cell for `_etype=partial` records. The reconstructed entity ID is `tracer_id`. Therefore the plotted TRACER-reconstructed marker-specificity row should be treated as a host-cell aggregated diagnostic, not a true reconstructed-partial-cell specificity metric.

Second, true reconstructed partial cells are sparse and unevenly distributed across lineages after the 10-900 transcript filter. The median true partial entity has {low_tx:.0f} transcripts and detects a median of {med_markers:.0f} of the 45 marker genes. Stable partial-cell coverage (n >= {MIN_STABLE_CELLS}) is limited to: {represented_text}.

Restricting the calculation to represented lineages and reducing the panel to top 1-3 markers per represented lineage improves interpretability but does not make the 45-gene marker-specificity benchmark directly comparable to whole-cell methods, because reconstructed partial cells are expected to be transcript-sparse subcellular/anuclear entities.

Recommendation: exclude TRACER-reconstructed from the main 45-gene marker-specificity comparison. Show it separately, if needed, as a reconstructed partial-cell diagnostic with explicit coverage/sparsity annotations. For main evidence, report reconstructed partial cells primarily with purity/conflict, RCTD entropy/max-weight, recovery/count, and transcript-fate metrics rather than whole-cell marker-specificity log2FC.
"""
    (C.FIGDIR / "figure_marker_specificity_v2_notes.md").write_text(note)

    print("DONE marker specificity diagnostics")
    print(old_vs_true.to_string(index=False))
    print(small_summary.to_string(index=False))


if __name__ == "__main__":
    main()

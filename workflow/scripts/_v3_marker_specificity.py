#!/usr/bin/env python3
"""Corrected v3 marker-specificity analysis.

Fixes TRACER partial-cell entity IDs by using ``tracer_id`` for
TRACER-reconstructed partial cells, audits marker coverage in TRACER-refined and
TRACER-reconstructed entities, filters poorly covered marker genes, and renders
updated marker-specificity and T-cell marker figures.
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

log = logging.getLogger("v3markers")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s",
                    datefmt="%H:%M:%S")
logging.getLogger("fontTools").setLevel(logging.WARNING)

METHODS_MAIN = ["original", "Baysor", "proseg", "Segger", "cellAdmix", "SPLIT",
                "TRACER-refined"]
METHODS_DIAG = METHODS_MAIN + ["TRACER-reconstructed"]
MIN_TARGET_CELLS = 10
MIN_RECON_TARGET_DETECTED = 3
MIN_RECON_TARGET_FRACTION = 0.10
MIN_REFINED_TARGET_DETECTED = 5
MIN_TCELL_RECON_CELLS_DIAGNOSTIC = 3
MIN_TCELL_RECON_DETECTED = 2
MIN_TCELL_REFINED_DETECTED = 5


def cp10k_log1p_dense(X):
    M = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    M = np.asarray(M, dtype=np.float64)
    s = M.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return np.log1p(M * 1e4 / s)


def tracer_matrix(component: str):
    """Return AnnData for TRACER component grouped by correct entity id."""
    import anndata as ad
    import scipy.sparse as sp

    etype = "cell" if component == "TRACER-refined" else "partial"
    df = pd.read_parquet(C.TRANSCRIPTS["TRACER"],
                         columns=["tracer_id", "_etype", "feature_name"])
    df["tracer_id"] = df["tracer_id"].astype(str)
    df["_etype"] = df["_etype"].astype(str)
    df["feature_name"] = df["feature_name"].astype(str)
    df = df[(df["_etype"].eq(etype)) &
            (~df["tracer_id"].isin(C.UNASSIGNED_TOKENS))]
    ct = pd.crosstab(df["tracer_id"], df["feature_name"])
    a = ad.AnnData(X=sp.csr_matrix(ct.values.astype(np.float64)),
                   obs=pd.DataFrame(index=ct.index.astype(str)),
                   var=pd.DataFrame(index=ct.columns.astype(str)))
    total = np.asarray(a.X.sum(axis=1)).ravel()
    keep = (total > C.TX_MIN) & (total < C.TX_MAX)
    return a[keep].copy()


def label_by_knn(adata, ref):
    ann = get_metric.transfer_labels(adata, ref, seed=0, k=15, per_type=300, log=log)
    return ann.set_index("cell_id")["predicted_celltype"].astype(str)


def specificity_rows(adata, labels: pd.Series, markers: pd.DataFrame,
                     method: str, min_cells: int = 3) -> pd.DataFrame:
    Xn = cp10k_log1p_dense(adata.X)
    gidx = {g: i for i, g in enumerate(map(str, adata.var_names))}
    lab = labels.reindex(adata.obs_names.astype(str)).astype(str).to_numpy()
    rows = []
    for r in markers.itertuples(index=False):
        ct = str(r.cell_type)
        gene = str(r.gene)
        marker_id = f"{ct}|{gene}"
        in_m = lab == ct
        out_m = (lab != ct) & (lab != "nan")
        base = {
            "method": method,
            "cell_type": ct,
            "gene": gene,
            "marker_id": marker_id,
            "rank": int(r.rank),
            "scrna_log2fc": float(r.scrna_log2fc),
            "n_cells_in": int(in_m.sum()),
            "n_cells_out": int(out_m.sum()),
        }
        if gene not in gidx:
            rows.append({**base, "spatial_log2fc": np.nan,
                         "spatial_mean_in": np.nan, "spatial_mean_out": np.nan,
                         "note": "gene missing from entity matrix"})
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


def tcell_rows(adata, labels: pd.Series, genes: list[str], method: str) -> pd.DataFrame:
    Xn = cp10k_log1p_dense(adata.X)
    gidx = {g: i for i, g in enumerate(map(str, adata.var_names))}
    lab = labels.reindex(adata.obs_names.astype(str)).astype(str).to_numpy()
    in_t = lab == "T"
    out_t = (lab != "T") & (lab != "nan")
    rows = []
    for gene in genes:
        base = {"method": method, "gene": gene,
                "n_tcells": int(in_t.sum()), "n_other": int(out_t.sum())}
        if gene not in gidx:
            rows.append({**base, "spatial_log2fc": np.nan,
                         "spatial_mean_in": np.nan, "spatial_mean_out": np.nan,
                         "note": "gene missing from entity matrix"})
            continue
        if in_t.sum() < 3 or out_t.sum() < 3:
            rows.append({**base, "spatial_log2fc": np.nan,
                         "spatial_mean_in": np.nan, "spatial_mean_out": np.nan,
                         "note": "too few T cells"})
            continue
        vals = Xn[:, gidx[gene]]
        rows.append({**base,
                     "spatial_log2fc": float(vals[in_t].mean() - vals[out_t].mean()),
                     "spatial_mean_in": float(vals[in_t].mean()),
                     "spatial_mean_out": float(vals[out_t].mean()),
                     "note": ""})
    return pd.DataFrame(rows)


def tracer_marker_coverage(adata, labels: pd.Series, markers: pd.DataFrame,
                           component: str) -> pd.DataFrame:
    import scipy.sparse as sp

    X = adata.X.tocsr() if sp.issparse(adata.X) else sp.csr_matrix(np.asarray(adata.X))
    gidx = {g: i for i, g in enumerate(map(str, adata.var_names))}
    lab = labels.reindex(adata.obs_names.astype(str)).astype(str).to_numpy()
    rows = []
    for r in markers.itertuples(index=False):
        ct = str(r.cell_type)
        gene = str(r.gene)
        marker_id = f"{ct}|{gene}"
        target = lab == ct
        if gene in gidx:
            col = X[:, gidx[gene]]
            det = np.asarray((col > 0).todense()).ravel() if hasattr(col, "todense") else np.asarray(col > 0).ravel()
            n_detected = int(det.sum())
            n_detected_target = int((det & target).sum())
        else:
            n_detected = 0
            n_detected_target = 0
        rows.append({
            "component": component,
            "cell_type": ct,
            "gene": gene,
            "marker_id": marker_id,
            "rank": int(r.rank),
            "n_entities": int(adata.n_obs),
            "n_detected_entities": n_detected,
            "fraction_detected_entities": n_detected / max(adata.n_obs, 1),
            "n_target_entities": int(target.sum()),
            "n_detected_target_entities": n_detected_target,
            "fraction_detected_target_entities": n_detected_target / max(int(target.sum()), 1),
        })
    return pd.DataFrame(rows)


def decide_marker_retention(cov: pd.DataFrame, markers: pd.DataFrame):
    refined = cov[cov["component"].eq("TRACER-refined")].set_index("marker_id")
    recon = cov[cov["component"].eq("TRACER-reconstructed")].set_index("marker_id")
    rows = []
    for r in markers.itertuples(index=False):
        marker_id = f"{r.cell_type}|{r.gene}"
        rf = refined.loc[marker_id]
        rc = recon.loc[marker_id]
        reasons = []
        if int(rc["n_target_entities"]) < MIN_TARGET_CELLS:
            reasons.append(f"TRACER-reconstructed target lineage n<{MIN_TARGET_CELLS}")
        if int(rf["n_target_entities"]) < MIN_TARGET_CELLS:
            reasons.append(f"TRACER-refined target lineage n<{MIN_TARGET_CELLS}")
        if int(rc["n_detected_target_entities"]) < MIN_RECON_TARGET_DETECTED:
            reasons.append(f"TRACER-reconstructed target detections <{MIN_RECON_TARGET_DETECTED}")
        if float(rc["fraction_detected_target_entities"]) < MIN_RECON_TARGET_FRACTION:
            reasons.append(f"TRACER-reconstructed target detection fraction <{MIN_RECON_TARGET_FRACTION:.2f}")
        if int(rf["n_detected_target_entities"]) < MIN_REFINED_TARGET_DETECTED:
            reasons.append(f"TRACER-refined target detections <{MIN_REFINED_TARGET_DETECTED}")
        rows.append({
            "cell_type": r.cell_type,
            "gene": r.gene,
            "marker_id": marker_id,
            "rank": int(r.rank),
            "scrna_log2fc": float(r.scrna_log2fc),
            "retained": not reasons,
            "drop_reason": "; ".join(reasons),
            "tracer_refined_n_target": int(rf["n_target_entities"]),
            "tracer_refined_n_detected_target": int(rf["n_detected_target_entities"]),
            "tracer_refined_fraction_detected_target": float(rf["fraction_detected_target_entities"]),
            "tracer_reconstructed_n_target": int(rc["n_target_entities"]),
            "tracer_reconstructed_n_detected_target": int(rc["n_detected_target_entities"]),
            "tracer_reconstructed_fraction_detected_target": float(rc["fraction_detected_target_entities"]),
        })
    return pd.DataFrame(rows)


def decide_tcell_retention(cov: pd.DataFrame, tgenes: pd.DataFrame):
    sub = cov[cov["cell_type"].eq("T")].copy()
    refined = sub[sub["component"].eq("TRACER-refined")].set_index("gene")
    recon = sub[sub["component"].eq("TRACER-reconstructed")].set_index("gene")
    rows = []
    for r in tgenes.itertuples(index=False):
        gene = str(r.gene)
        rf = refined.loc[gene]
        rc = recon.loc[gene]
        reasons = []
        if int(rc["n_target_entities"]) < MIN_TCELL_RECON_CELLS_DIAGNOSTIC:
            reasons.append(f"TRACER-reconstructed T cells n<{MIN_TCELL_RECON_CELLS_DIAGNOSTIC}")
        if int(rc["n_detected_target_entities"]) < MIN_TCELL_RECON_DETECTED:
            reasons.append(f"TRACER-reconstructed T-cell detections <{MIN_TCELL_RECON_DETECTED}")
        if int(rf["n_detected_target_entities"]) < MIN_TCELL_REFINED_DETECTED:
            reasons.append(f"TRACER-refined T-cell detections <{MIN_TCELL_REFINED_DETECTED}")
        rows.append({
            "cell_type": "T",
            "gene": gene,
            "rank": int(r.rank),
            "scrna_log2fc": float(r.scrna_log2fc),
            "retained": not reasons,
            "drop_reason": "; ".join(reasons),
            "tracer_refined_n_tcells": int(rf["n_target_entities"]),
            "tracer_refined_n_detected_tcells": int(rf["n_detected_target_entities"]),
            "tracer_refined_fraction_detected_tcells": float(rf["fraction_detected_target_entities"]),
            "tracer_reconstructed_n_tcells": int(rc["n_target_entities"]),
            "tracer_reconstructed_n_detected_tcells": int(rc["n_detected_target_entities"]),
            "tracer_reconstructed_fraction_detected_tcells": float(rc["fraction_detected_target_entities"]),
        })
    return pd.DataFrame(rows)


def merge_specificity(v2: pd.DataFrame, tracer_spec: pd.DataFrame,
                      retained: pd.DataFrame):
    keep_ids = set(retained.loc[retained["retained"], "marker_id"])
    v2 = v2.copy()
    if "marker_id" not in v2.columns:
        v2["marker_id"] = v2["cell_type"].astype(str) + "|" + v2["gene"].astype(str)
    non_tracer = v2[~v2["method"].isin(["TRACER-refined", "TRACER-reconstructed"])]
    out = pd.concat([non_tracer, tracer_spec], ignore_index=True, sort=False)
    return out[out["marker_id"].isin(keep_ids)].copy()


def plot_marker_specificity_main(spec: pd.DataFrame):
    df = spec[spec["method"].isin(METHODS_MAIN)].dropna(subset=["spatial_log2fc"]).copy()
    fig, ax = plt.subplots(figsize=(7.2, 3.45))
    rng = np.random.default_rng(8)
    for i, method in enumerate(METHODS_MAIN):
        vals = df.loc[df["method"].eq(method), "spatial_log2fc"].to_numpy()
        if vals.size == 0:
            continue
        ax.boxplot(vals, positions=[i], widths=0.48, patch_artist=True, showfliers=False,
                   boxprops=dict(facecolor="#f8f7f5", edgecolor=C.PALETTE.get(method, "#999"), lw=0.9),
                   medianprops=dict(color="#333", lw=1.0),
                   whiskerprops=dict(color="#777", lw=0.7),
                   capprops=dict(color="#777", lw=0.7))
        sub = df[df["method"].eq(method)]
        colors = [C.CELLTYPE_COLORS.get(ct, "#999") for ct in sub["cell_type"]]
        ax.scatter(i + rng.normal(0, 0.07, size=len(sub)), sub["spatial_log2fc"],
                   s=16, c=colors, alpha=0.75, edgecolor="white", lw=0.2)
    ax.axhline(0, color="#777", lw=0.7)
    ax.set_xticks(range(len(METHODS_MAIN)))
    ax.set_xticklabels(METHODS_MAIN, rotation=35, ha="right", fontsize=6.4)
    ax.set_ylabel("Marker log2FC in lineage vs rest")
    ax.set_title("Marker Specificity: TRACER-Covered Gene Set",
                 loc="left", fontweight="bold")
    ax.text(0.01, 0.96, "Main comparison excludes reconstructed partial cells.",
            transform=ax.transAxes, va="top", fontsize=6.4, color="#555")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e6e1dc", lw=0.5)
    ax.set_axisbelow(True)
    C.save_fig(fig, "figure_marker_specificity_v3")
    plt.close(fig)


def plot_marker_specificity_diagnostic(spec: pd.DataFrame):
    df = spec[spec["method"].isin(METHODS_DIAG)].dropna(subset=["spatial_log2fc"]).copy()
    med = (df.groupby("method")["spatial_log2fc"].median()
             .reindex(METHODS_DIAG))
    fig, ax = plt.subplots(figsize=(7.2, 3.25))
    x = np.arange(len(METHODS_DIAG))
    ax.bar(x, med, color=[C.PALETTE.get(m, "#999") for m in METHODS_DIAG],
           edgecolor="white", width=0.62)
    for i, v in enumerate(med):
        if pd.notna(v):
            ax.text(i, v + 0.08, f"{v:.2f}", ha="center", va="bottom", fontsize=6.4)
    ax.axhline(0, color="#777", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(METHODS_DIAG, rotation=35, ha="right", fontsize=6.4)
    ax.set_ylabel("Median marker specificity log2FC")
    ax.set_title("Reconstructed Partial-Cell Diagnostic", loc="left", fontweight="bold")
    ax.text(0.01, 0.96, "Same retained genes; reconstructed partials shown separately.",
            transform=ax.transAxes, va="top", fontsize=6.4, color="#555")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e6e1dc", lw=0.5)
    C.save_fig(fig, "figure_marker_specificity_tracer_reconstructed_diagnostic_v3")
    plt.close(fig)


def plot_tcell_lollipop(tdf: pd.DataFrame, retained_genes: list[str]):
    df = tdf[tdf["gene"].isin(retained_genes)].copy()
    fig_w = max(3.8, 1.15 * max(len(retained_genes), 1) + 2.7)
    fig, ax = plt.subplots(figsize=(fig_w, 3.35))
    x = np.arange(len(retained_genes))
    offsets = np.linspace(-0.23, 0.23, len(METHODS_DIAG))
    for i, gene in enumerate(retained_genes):
        ax.vlines(i, 0, max(0, pd.to_numeric(df.loc[df["gene"].eq(gene), "spatial_log2fc"],
                                             errors="coerce").max(skipna=True)),
                  colors="#b8b0aa", linestyles="--", lw=0.8, zorder=1)
    for j, method in enumerate(METHODS_DIAG):
        sub = df[df["method"].eq(method)].set_index("gene").reindex(retained_genes)
        y = pd.to_numeric(sub["spatial_log2fc"], errors="coerce")
        ax.scatter(x + offsets[j], y, s=30, color=C.PALETTE.get(method, "#999"),
                   edgecolor="white", lw=0.4, label=method, zorder=3)
    ax.axhline(0, color="#777", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(retained_genes, fontstyle="italic")
    ax.set_ylabel("Spatial log2FC: T vs other")
    ax.set_title("T-Cell Marker Log2FC: Covered Markers", loc="left",
                 fontweight="bold")
    ax.legend(frameon=False, ncol=4, fontsize=5.8, loc="upper center",
              bbox_to_anchor=(0.5, -0.22))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e6e1dc", lw=0.5)
    C.save_fig(fig, "figure_tcell_marker_log2fc_v3")
    plt.close(fig)


def write_note(retained: pd.DataFrame, dropped: pd.DataFrame,
               t_retained: pd.DataFrame, t_dropped: pd.DataFrame,
               spec: pd.DataFrame):
    recon = spec[spec["method"].eq("TRACER-reconstructed")]
    recon_med = float(recon["spatial_log2fc"].median()) if recon["spatial_log2fc"].notna().any() else np.nan
    retained_genes = retained.loc[retained["retained"], ["cell_type", "gene"]]
    dropped_genes = dropped[["cell_type", "gene", "drop_reason"]]
    lines = [
        "# Marker Specificity v3 Notes",
        "",
        "TRACER marker specificity was recomputed with corrected TRACER entity IDs: `tracer_id` for both `_etype=cell` refined whole-cell records and `_etype=partial` reconstructed partial-cell records.",
        "",
        f"Retained marker rows for the main specificity comparison: {len(retained_genes)}.",
        "Retained 45-gene panel rows:",
        ", ".join(f"{r.cell_type}:{r.gene}" for r in retained_genes.itertuples(index=False)) or "none",
        "",
        f"Dropped marker rows: {len(dropped_genes)}.",
        "Dropped rows and reasons:",
        "\n".join(f"- {r.cell_type}:{r.gene} — {r.drop_reason}" for r in dropped_genes.itertuples(index=False)) or "- none",
        "",
        f"Retained T-cell markers: {', '.join(t_retained['gene'].astype(str)) or 'none'}.",
        "Dropped T-cell markers:",
        "\n".join(f"- {r.gene} — {r.drop_reason}" for r in t_dropped.itertuples(index=False)) or "- none",
        "",
        "Conclusion: TRACER-reconstructed remains too sparse and lineage-imbalanced for the main whole-cell 45-gene marker-specificity comparison. It is now shown only as a separate diagnostic panel. The main marker-specificity figure uses whole-cell methods plus TRACER-refined; reconstructed partial cells should be emphasized through purity/conflict, RCTD entropy/max-weight, recovery/count, transcript fate, and explicit partial-cell diagnostics.",
        f"With the retained covered genes, TRACER-reconstructed diagnostic median log2FC is {recon_med:.3f}.",
    ]
    (C.FIGDIR / "figure_marker_specificity_v3_notes.md").write_text("\n".join(lines))
    (C.FIGDIR / "figure_marker_specificity_v2_notes.md").write_text("\n".join(lines))


def main():
    C.ensure_dirs()
    C.apply_style()
    markers = pd.read_csv(C.SRCDIR / "lineage_marker_genes_45_v2.tsv", sep="\t")
    markers["marker_id"] = markers["cell_type"].astype(str) + "|" + markers["gene"].astype(str)
    tgenes = pd.read_csv(C.SRCDIR / "tcell_marker_genes_v2.tsv", sep="\t")
    v2_spec = pd.read_csv(C.SRCDIR / "marker_specificity_v2.tsv", sep="\t")
    v2_tcell = pd.read_csv(C.SRCDIR / "tcell_marker_log2fc_v2.tsv", sep="\t")

    ref = get_metric.load_reference(C.REFERENCE_H5AD, C.REF_CELLTYPE_COL, log)
    tracer_specs = []
    tracer_tcells = []
    coverages = []
    tracer_labels = {}
    tracer_mats = {}
    for component in ("TRACER-refined", "TRACER-reconstructed"):
        a = tracer_matrix(component)
        labels = label_by_knn(a, ref)
        tracer_mats[component] = a
        tracer_labels[component] = labels
        tracer_specs.append(specificity_rows(a, labels, markers, component))
        tracer_tcells.append(tcell_rows(a, labels, list(tgenes["gene"].astype(str)), component))
        coverages.append(tracer_marker_coverage(a, labels, markers, component))
        log.info("%s: %d filtered entities; T=%d", component, a.n_obs,
                 int((labels == "T").sum()))

    tracer_spec = pd.concat(tracer_specs, ignore_index=True)
    tracer_tcell = pd.concat(tracer_tcells, ignore_index=True)
    coverage = pd.concat(coverages, ignore_index=True)
    retention = decide_marker_retention(coverage, markers)
    retained = retention[retention["retained"]].copy()
    dropped = retention[~retention["retained"]].copy()
    t_retention = decide_tcell_retention(coverage, tgenes)
    t_retained = t_retention[t_retention["retained"]].copy()
    t_dropped = t_retention[~t_retention["retained"]].copy()

    marker_spec = merge_specificity(v2_spec, tracer_spec, retention)
    # T-cell rows: non-TRACER from v2, corrected TRACER rows from this script.
    tcell = pd.concat([
        v2_tcell[~v2_tcell["method"].isin(["TRACER-refined", "TRACER-reconstructed"])],
        tracer_tcell,
    ], ignore_index=True, sort=False)
    tcell = tcell[tcell["gene"].isin(t_retained["gene"].astype(str))]

    piv = marker_spec.pivot_table(index=["cell_type", "gene"], columns="method",
                                  values="spatial_log2fc", observed=False)
    stat_rows = []
    if "TRACER-refined" in piv.columns:
        ref_vals = piv["TRACER-refined"].to_numpy()
        for method in METHODS_MAIN:
            if method == "TRACER-refined" or method not in piv.columns:
                continue
            stat, p, n = C.wilcoxon_one_sided(ref_vals, piv[method].to_numpy(),
                                              alternative="greater")
            stat_rows.append({
                "metric": "marker_specificity_log2fc_v3",
                "comparison": f"TRACER-refined vs {method}",
                "test": "paired one-sided Wilcoxon signed-rank (by retained gene)",
                "alternative": "greater(TRACER-refined>other)",
                "n_pairs": n,
                "statistic": stat,
                "p_value": p,
                "p_label": C.p_label(p),
                "stars": C.p_to_stars(p),
            })

    C.save_source(coverage, "marker_gene_coverage_tracer_v3.tsv")
    C.save_source(retained, "marker_genes_retained_v3.tsv")
    C.save_source(dropped, "marker_genes_dropped_v3.tsv")
    C.save_source(t_retention, "tcell_marker_gene_coverage_retention_v3.tsv")
    C.save_source(t_retained, "tcell_marker_genes_retained_v3.tsv")
    C.save_source(t_dropped, "tcell_marker_genes_dropped_v3.tsv")
    C.save_source(tracer_spec, "marker_specificity_tracer_corrected_entities_v3.tsv")
    C.save_source(marker_spec, "marker_specificity_v3.tsv")
    C.save_source(pd.DataFrame(stat_rows), "marker_specificity_stats_v3.tsv")
    C.save_source(tcell, "tcell_marker_log2fc_v3.tsv")

    plot_marker_specificity_main(marker_spec)
    plot_marker_specificity_diagnostic(marker_spec)
    plot_tcell_lollipop(tcell, list(t_retained.sort_values("rank")["gene"].astype(str)))
    write_note(retention, dropped, t_retained, t_dropped, marker_spec)
    print("DONE marker specificity v3")
    print(retention["retained"].value_counts().to_string())
    print(t_retention[["gene", "retained", "drop_reason"]].to_string(index=False))


if __name__ == "__main__":
    main()

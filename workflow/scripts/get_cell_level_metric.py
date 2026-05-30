#!/usr/bin/env python3
"""Cell-level benchmark metrics for profile-purification methods (SPLIT/RCTD).

SPLIT/RCTD does NOT produce transcript-level molecule assignments — it purifies a
cell-by-gene matrix of expected (fractional) counts. It must therefore be scored
at the CELL level, NOT with the transcript-level get_metric.py.

This script compares the ORIGINAL Xenium cell-by-gene matrix against the SPLIT
PURIFIED cell-by-gene matrix (over shared cells & genes), using an external
scRNA reference for biological-quality metrics. It reports, before vs after
purification:

  * count-level retained / pruned transcript counts + pseudo-unassigned count
  * reference correlation   (cell profile vs its assigned cell-type reference)
  * marker leakage          (fraction of counts in OTHER cell-types' markers)
  * marker specificity      (own-type marker counts / all-marker counts)
  * RCTD entropy / doublet  (aggregate mixture statistics)
  * runtime and memory      (from the runner's runtime_memory.json)

Wording for the manuscript: "SPLIT/RCTD was evaluated as a cell-level profile
purification method. Removed counts were estimated at the cell-by-gene level and
treated as pseudo-unassigned for count-based comparisons. Exact transcript
coordinates of removed molecules were not available and were not used for
transcript-coordinate metrics."

EXAMPLE
=======
    python workflow/scripts/get_cell_level_metric.py \\
      --method SPLIT \\
      --original-cell-by-gene results/benchmark_runs/tsu20/SPLIT/outputs/split_original_cell_by_gene.h5ad \\
      --purified-cell-by-gene results/benchmark_runs/tsu20/SPLIT/outputs/split_cell_by_gene.h5ad \\
      --removed-counts results/benchmark_runs/tsu20/SPLIT/outputs/split_removed_counts_by_cell_gene.tsv.gz \\
      --pruning-summary results/benchmark_runs/tsu20/SPLIT/outputs/split_pruning_summary.json \\
      --rctd-weights results/benchmark_runs/tsu20/SPLIT/outputs/split_rctd_weights.tsv.gz \\
      --reference-h5ad dataset/lung_cancer_scrna_10x/lung_cancer_50k.h5ad \\
      --reference-celltype-col cell_type \\
      --runtime-json results/benchmark_runs/tsu20/SPLIT/runtime_memory.json \\
      --outdir results/benchmark/lung_xenium_ref36973297/metrics/split_cell_level
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", default="SPLIT")
    p.add_argument("--original-cell-by-gene", required=True, type=Path)
    p.add_argument("--purified-cell-by-gene", required=True, type=Path)
    p.add_argument("--removed-counts", type=Path, default=None,
                   help="split_removed_counts_by_cell_gene.tsv.gz (optional).")
    p.add_argument("--pruning-summary", type=Path, default=None,
                   help="split_pruning_summary.json (count-level pruning totals).")
    p.add_argument("--rctd-weights", type=Path, default=None,
                   help="split_rctd_weights.tsv.gz (per-cell first/second type + entropy).")
    p.add_argument("--reference-h5ad", required=True, type=Path)
    p.add_argument("--reference-celltype-col", default="Cell_Cluster_level1",
                   help="obs column with reference labels. Default Cell_Cluster_level1 "
                        "(9 coarse types). Per-cell RCTD first_type labels are mapped "
                        "to this vocabulary via an ontology crosswalk.")
    p.add_argument("--runtime-json", type=Path, default=None)
    p.add_argument("--npmi", type=Path, default=None,
                   help="NPMI csv(.gz) panel. If given, NPMI-derived relative purity / "
                        "relative conflict are computed before (original) vs after "
                        "(purified) on the shared cell-by-gene matrices. Purified "
                        "(fractional) counts are rounded for the transcript-count step.")
    p.add_argument("--purity-tau", type=float, default=0.05,
                   help="tau for relu purity/conflict (matches get_metric.py default).")
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--n-markers", type=int, default=15,
                   help="Top markers per reference cell type (by fold-change).")
    return p


# Ordered crosswalk: RCTD ontology first_type -> Cell_Cluster_level1 (9 coarse types).
# Rules are checked in order, so more specific keywords (e.g. dendritic -> Myeloid)
# precede broader ones (e.g. plasma -> Plasma) to resolve 'plasmacytoid dendritic'.
_LEVEL1_RULES = (
    ("Mast", ("mast",)),
    ("Myeloid", ("dendritic", "monocyte", "macrophage", "granulocyte", "neutrophil",
                 "myeloid", "kupffer")),
    ("Plasma", ("plasma cell", "plasmablast")),
    ("B", ("b cell", "b-cell")),
    ("T", ("t cell", "t-cell", "t-regulatory", "cd4", "cd8", "cytotoxic",
           "natural killer", "nk cell", "lymphocyte")),
    ("Ciliated", ("ciliated",)),
    ("Endothelial", ("endotheli",)),
    ("Fibroblasts", ("fibroblast", "smooth muscle", "pericyte", "stromal", "myofibro",
                     "mesenchym")),
    ("Cancer", ("epitheli", "malignant", "cancer", "tumor", "tumour", "alveolar",
                "club", "secretory", "basal")),
)


def map_to_reference_label(ontology_label: str, valid_labels: set[str]) -> str:
    """Map an RCTD ontology cell type to a coarse reference label.

    If the label is already one of the reference labels, keep it. Otherwise apply
    the ordered keyword crosswalk; only return a mapped label if it exists in the
    reference vocabulary, else 'NA'.
    """
    s = str(ontology_label)
    if s in valid_labels:
        return s
    low = s.lower()
    for coarse, kws in _LEVEL1_RULES:
        if coarse in valid_labels and any(k in low for k in kws):
            return coarse
    return "NA"


def _read_h5ad(path: Path):
    import anndata as ad
    return ad.read_h5ad(path)


def _load_npmi_panel(path: Path):
    """Load + symmetrically expand an NPMI panel (same as get_metric.py)."""
    np_df = pd.read_csv(path)
    rev = np_df.copy()
    rev["gene_i"], rev["gene_j"] = np_df["gene_j"].values, np_df["gene_i"].values
    panel = pd.concat([np_df, rev], ignore_index=True)
    return panel.loc[panel["gene_i"] != panel["gene_j"]]


def _purity_conflict(block: np.ndarray, cells: list[str], genes: list[str],
                     npmi_panel, tau: float, round_counts: bool, log):
    """Median relative purity / conflict for a (cells x genes) count block.

    Reuses get_metric._compute_purity_conflict. Fractional (purified) counts are
    rounded first because that routine replicates each count as an integer
    transcript; truncation would silently drop sub-1 expected counts."""
    import anndata as ad
    import scipy.sparse as sp
    import get_metric  # sibling script on sys.path; reuse the exact metric
    X = np.rint(block) if round_counts else block
    X = sp.csr_matrix(np.asarray(X, dtype=np.float64))
    a = ad.AnnData(X=X, obs=pd.DataFrame(index=pd.Index(cells, name="cell_id")),
                   var=pd.DataFrame(index=pd.Index(genes, name="feature_name")))
    pur, conf, rel_pur, rel_conf, _ = get_metric._compute_purity_conflict(
        a, npmi_panel, tau=tau, log=log)
    return {"median_relative_purity": float(np.nanmedian(rel_pur)),
            "median_relative_conflict": float(np.nanmedian(rel_conf)),
            "median_purity_score": float(np.nanmedian(pur)),
            "median_conflict_score": float(np.nanmedian(conf)),
            "n_cells_scored": int(np.sum(~np.isnan(rel_pur)))}


def _dense(adata, cells, genes) -> np.ndarray:
    sub = adata[cells, genes]
    X = sub.X
    return np.asarray(X.toarray() if hasattr(X, "toarray") else X, dtype=np.float64)


def _l1_normalize(mat: np.ndarray) -> np.ndarray:
    s = mat.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return mat / s


def _reference_profiles(ref, celltype_col: str, genes: list[str], log):
    """Per-cell-type mean L1-normalized expression profile over `genes`."""
    import scipy.sparse as sp
    if celltype_col not in ref.obs.columns:
        raise SystemExit(f"--reference-celltype-col {celltype_col!r} not in reference obs "
                         f"({list(ref.obs.columns)[:10]}…)")
    ref_genes = [str(g) for g in ref.var_names]
    gidx = {g: i for i, g in enumerate(ref_genes)}
    shared = [g for g in genes if g in gidx]
    cols = [gidx[g] for g in shared]
    X = ref.X
    X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(np.asarray(X))
    X = X[:, cols]
    labels = ref.obs[celltype_col].astype(str).to_numpy()
    profiles = {}
    for ct in pd.unique(labels):
        rows = np.where(labels == ct)[0]
        m = np.asarray(X[rows].mean(axis=0)).ravel()
        profiles[ct] = m
    prof = pd.DataFrame(profiles, index=shared).T  # celltypes x shared_genes
    prof_norm = prof.div(prof.sum(axis=1).replace(0, 1.0), axis=0)
    log.info("Reference: %d cell types over %d shared genes", prof.shape[0], len(shared))
    return prof_norm, shared


def _markers_per_type(prof_norm: pd.DataFrame, n_markers: int) -> dict[str, list[str]]:
    """Top-N marker genes per cell type by fold-change vs mean of other types."""
    overall = prof_norm.mean(axis=0) + 1e-9
    markers = {}
    for ct in prof_norm.index:
        fc = (prof_norm.loc[ct] + 1e-9) / overall
        markers[ct] = fc.sort_values(ascending=False).head(n_markers).index.tolist()
    return markers


def _per_cell_profile(mat: np.ndarray, genes: list[str], cell_ids: np.ndarray,
                      cell_types: np.ndarray, prof_norm: pd.DataFrame,
                      markers: dict[str, list[str]]) -> pd.DataFrame:
    """Per-cell reference correlation, marker leakage, marker specificity.

    `mat` is cells x genes (same order as `genes`). Returns one row per scorable
    cell with columns: cell_id, cell_type, ref_corr, marker_specificity,
    marker_leakage."""
    gpos = {g: i for i, g in enumerate(genes)}
    matn = _l1_normalize(mat)
    own_idx = {ct: np.array([gpos[g] for g in mk if g in gpos], dtype=int)
               for ct, mk in markers.items()}
    all_marker_genes = sorted({g for mk in markers.values() for g in mk if g in gpos})
    all_marker_idx = np.array([gpos[g] for g in all_marker_genes], dtype=int)
    ref_vec = {}
    for ct in prof_norm.index:
        v = np.zeros(len(genes))
        common = [g for g in prof_norm.columns if g in gpos]
        v[[gpos[g] for g in common]] = prof_norm.loc[ct, common].to_numpy()
        ref_vec[ct] = v

    out = []
    for i in range(mat.shape[0]):
        ct = cell_types[i]
        c = matn[i]
        if ct not in ref_vec or c.sum() == 0:
            continue
        rv = ref_vec[ct]
        corr = float(np.corrcoef(c, rv)[0, 1]) if (c.std() > 0 and rv.std() > 0) else np.nan
        all_mass = c[all_marker_idx].sum()
        own_mass = c[own_idx[ct]].sum() if ct in own_idx and len(own_idx[ct]) else 0.0
        spec = float(own_mass / all_mass) if all_mass > 0 else np.nan
        leak = float((all_mass - own_mass) / max(c.sum(), 1e-12)) if all_mass > 0 else np.nan
        out.append((cell_ids[i], ct, corr, spec, leak))
    return pd.DataFrame(out, columns=["cell_id", "cell_type", "ref_corr",
                                      "marker_specificity", "marker_leakage"])


def _pseudobulk_refconsist(block: np.ndarray, genes: list[str], cell_types: np.ndarray,
                           ref, celltype_col: str, log, min_cells: int = 10) -> pd.DataFrame:
    """Per-cell-type PSEUDOBULK reference consistency, matching get_metric.py:
    mean log-normalized profile of cells assigned to a type vs the mean log-
    normalized reference profile of that type (Pearson over shared genes).

    This is the cross-method-COMPARABLE metric (get_metric uses the same pseudobulk
    + log-normalization). The per-cell averaged correlation used elsewhere in this
    script is NOT comparable to it and is reported separately."""
    import scipy.sparse as sp
    from scipy.stats import pearsonr

    def _lognorm_dense(M):
        M = np.asarray(M, dtype=np.float64)
        s = M.sum(axis=1, keepdims=True); s[s == 0] = 1.0
        return np.log1p(M * 1e4 / s)

    rg = list(map(str, ref.var_names)); gi = {g: i for i, g in enumerate(rg)}
    present = [g for g in genes if g in gi]
    qcols = [i for i, g in enumerate(genes) if g in gi]
    q = _lognorm_dense(block[:, qcols])
    Xr = ref.X.tocsr() if sp.issparse(ref.X) else sp.csr_matrix(np.asarray(ref.X))
    Xr = Xr[:, [gi[g] for g in present]].astype(np.float64)
    sr = np.asarray(Xr.sum(axis=1)).ravel(); sr[sr == 0] = 1.0
    Xr = Xr.multiply(1e4 / sr[:, None]).tocsr(); Xr.data = np.log1p(Xr.data)
    rl = ref.obs[celltype_col].astype(str).to_numpy()

    rows = []
    for ct in sorted(set(cell_types) | set(rl)):
        if ct == "NA":
            continue
        qm = cell_types == ct; nq = int(qm.sum()); nr = int((rl == ct).sum())
        if nq < min_cells or nr < min_cells:
            rows.append(dict(method="SPLIT", cell_type=ct, n_spatial_cells=nq,
                             n_reference_cells=nr, n_genes_used=len(present),
                             pearson_r=np.nan,
                             reason=("not in SPLIT/RCTD reference vocabulary (0 cells)"
                                     if nq == 0 else "fewer than %d cells" % min_cells)))
            continue
        qb = q[qm].mean(axis=0)
        rb = np.asarray(Xr[rl == ct].mean(axis=0)).ravel()
        r = float(pearsonr(qb, rb)[0]) if (qb.std() > 0 and rb.std() > 0) else np.nan
        rows.append(dict(method="SPLIT", cell_type=ct, n_spatial_cells=nq,
                         n_reference_cells=nr, n_genes_used=len(present),
                         pearson_r=r, reason=""))
    df = pd.DataFrame(rows)
    log.info("[pseudobulk ref-consistency] %d types scored; median r = %.3f",
             int(df["pearson_r"].notna().sum()),
             float(df["pearson_r"].median()) if df["pearson_r"].notna().any() else float("nan"))
    return df


def _agg(df: pd.DataFrame) -> dict:
    return {
        "reference_correlation_mean": float(df["ref_corr"].mean()),
        "reference_correlation_median": float(df["ref_corr"].median()),
        "marker_specificity_mean": float(df["marker_specificity"].mean()),
        "marker_leakage_mean": float(df["marker_leakage"].mean()),
        "n_cells_scored": int(df["ref_corr"].notna().sum()),
    }


def main() -> int:
    args = build_argparser().parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    import logging
    log = logging.getLogger("get_cell_level_metric")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s",
                        datefmt="%H:%M:%S")
    log.info("=== get_cell_level_metric.py === method=%s", args.method)

    orig = _read_h5ad(args.original_cell_by_gene)
    pur = _read_h5ad(args.purified_cell_by_gene)
    ref = _read_h5ad(args.reference_h5ad)

    # Shared cells & genes (fair before/after on identical cells/genes).
    shared_cells = [c for c in map(str, pur.obs_names) if c in set(map(str, orig.obs_names))]
    shared_genes = [g for g in map(str, pur.var_names) if g in set(map(str, orig.var_names))]
    log.info("Aligned %d cells x %d genes (orig=%dx%d, purified=%dx%d)",
             len(shared_cells), len(shared_genes), orig.n_obs, orig.n_vars,
             pur.n_obs, pur.n_vars)

    # Reference profiles keyed by the reference label vocabulary (e.g. the 9
    # Cell_Cluster_level1 types).
    prof_norm, shared_ref_genes = _reference_profiles(ref, args.reference_celltype_col,
                                                      shared_genes, log)
    valid_labels = set(prof_norm.index.astype(str))

    # Per-cell assigned type from RCTD weights (first_type), mapped to the
    # reference vocabulary via the ontology crosswalk.
    cell_type_map = {}
    weights_df = None
    if args.rctd_weights and Path(args.rctd_weights).exists():
        weights_df = pd.read_csv(args.rctd_weights, sep="\t")
        if {"cell_id", "first_type"}.issubset(weights_df.columns):
            cell_type_map = dict(zip(weights_df["cell_id"].astype(str),
                                     weights_df["first_type"].astype(str)))
    raw_types = np.array([cell_type_map.get(c, "NA") for c in shared_cells])
    cell_types = np.array([map_to_reference_label(t, valid_labels) for t in raw_types])
    n_mapped = int((cell_types != "NA").sum())
    log.info("Mapped RCTD first_type -> reference labels (%s): %d/%d cells mapped to %d labels",
             args.reference_celltype_col, n_mapped, len(cell_types),
             len(set(cell_types) - {"NA"}))
    markers = _markers_per_type(prof_norm, args.n_markers)

    orig_block = _dense(orig, shared_cells, shared_genes)
    pur_block = _dense(pur, shared_cells, shared_genes)

    cells_arr = np.asarray(shared_cells)
    before_pc = _per_cell_profile(orig_block, shared_genes, cells_arr, cell_types, prof_norm, markers)
    after_pc = _per_cell_profile(pur_block, shared_genes, cells_arr, cell_types, prof_norm, markers)
    before = _agg(before_pc)
    after = _agg(after_pc)

    # Cross-method-COMPARABLE pseudobulk reference consistency (matches get_metric).
    pseudobulk_rc = _pseudobulk_refconsist(pur_block, shared_genes, cell_types, ref,
                                           args.reference_celltype_col, log)
    pseudobulk_median_r = float(pseudobulk_rc["pearson_r"].median()) \
        if pseudobulk_rc["pearson_r"].notna().any() else float("nan")

    # --- NPMI relative purity / conflict (before=original, after=purified) ---
    purity_conflict = {}
    if args.npmi and Path(args.npmi).exists():
        try:
            panel = _load_npmi_panel(args.npmi)
            pc_before = _purity_conflict(orig_block, shared_cells, shared_genes,
                                         panel, args.purity_tau, round_counts=False, log=log)
            pc_after = _purity_conflict(pur_block, shared_cells, shared_genes,
                                        panel, args.purity_tau, round_counts=True, log=log)
            purity_conflict = {
                "relative_purity": {"before": pc_before["median_relative_purity"],
                                    "after": pc_after["median_relative_purity"],
                                    "delta": pc_after["median_relative_purity"]
                                             - pc_before["median_relative_purity"]},
                "relative_conflict": {"before": pc_before["median_relative_conflict"],
                                      "after": pc_after["median_relative_conflict"],
                                      "delta": pc_after["median_relative_conflict"]
                                               - pc_before["median_relative_conflict"]},
                "purified_counts_rounded_for_purity": True,
                "n_cells_scored": pc_after["n_cells_scored"],
            }
            log.info("Relative purity:  before=%.4f after=%.4f (Δ=%+.4f)",
                     pc_before["median_relative_purity"], pc_after["median_relative_purity"],
                     purity_conflict["relative_purity"]["delta"])
            log.info("Relative conflict: before=%.4f after=%.4f (Δ=%+.4f)",
                     pc_before["median_relative_conflict"], pc_after["median_relative_conflict"],
                     purity_conflict["relative_conflict"]["delta"])
        except Exception as e:
            log.warning("Skipping NPMI purity/conflict (%s: %s)", type(e).__name__, str(e)[:160])

    # --- count-level pruning totals -----------------------------------------
    pruning = {}
    if args.pruning_summary and Path(args.pruning_summary).exists():
        pruning = json.load(open(args.pruning_summary))

    # --- RCTD entropy / doublet aggregates ----------------------------------
    rctd_agg = {}
    if weights_df is not None:
        ent = pd.to_numeric(weights_df.get("rctd_weights_entropy"), errors="coerce")
        spot = weights_df.get("spot_class")
        rctd_agg = {
            "mean_rctd_entropy": float(ent.mean()) if ent is not None and ent.notna().any() else None,
            "median_rctd_entropy": float(ent.median()) if ent is not None and ent.notna().any() else None,
            "n_doublet_or_mixed": int(spot.astype(str).str.lower().str.contains("doublet").sum())
                                  if spot is not None else None,
            "n_singlet": int(spot.astype(str).str.lower().eq("singlet").sum())
                         if spot is not None else None,
            "n_cells": int(len(weights_df)),
        }

    # --- runtime / memory ---------------------------------------------------
    runtime = {}
    if args.runtime_json and Path(args.runtime_json).exists():
        rj = json.load(open(args.runtime_json))
        runtime = {k: rj.get(k) for k in
                   ("total_seconds", "peak_rss_gb_observed", "method_version",
                    "mode", "runtime_valid_for_benchmark")}
        rm = [s for s in rj.get("stages", []) if s.get("name") == "run_method"]
        if rm:
            runtime["run_method_seconds"] = rm[0].get("seconds")
            runtime["run_method_external_max_rss_gb"] = rm[0].get("external_max_rss_gb")

    metrics = {
        "method": args.method,
        "evaluation_level": "cell_level_profile_purification",
        "exact_transcript_coordinates_available": False,
        "count_level": {
            "original_assigned_count": pruning.get("original_assigned_count"),
            "purified_retained_count": pruning.get("purified_retained_count"),
            "estimated_removed_count": pruning.get("estimated_removed_count"),
            "fraction_removed_of_original_assigned":
                pruning.get("fraction_removed_of_original_assigned"),
            "original_unassigned_count": pruning.get("original_unassigned_count"),
            "pseudo_post_cleaning_unassigned_count":
                pruning.get("pseudo_post_cleaning_unassigned_count"),
            "purified_counts_are_fractional": pruning.get("purified_counts_are_fractional"),
        },
        "reference_correlation": {"before": before["reference_correlation_mean"],
                                  "after": after["reference_correlation_mean"],
                                  "delta": after["reference_correlation_mean"]
                                           - before["reference_correlation_mean"],
                                  "note": "per-cell averaged L1 correlation (purification "
                                          "effect); NOT comparable to pseudobulk."},
        "median_reference_pearson_r_pseudobulk": pseudobulk_median_r,
        "median_reference_pearson_r_pseudobulk_note":
            "cross-method-comparable (pseudobulk + log-norm, matches get_metric.py)",
        "marker_leakage": {"before": before["marker_leakage_mean"],
                           "after": after["marker_leakage_mean"],
                           "delta": after["marker_leakage_mean"] - before["marker_leakage_mean"]},
        "marker_specificity": {"before": before["marker_specificity_mean"],
                               "after": after["marker_specificity_mean"],
                               "delta": after["marker_specificity_mean"]
                                        - before["marker_specificity_mean"]},
        "relative_purity": purity_conflict.get("relative_purity"),
        "relative_conflict": purity_conflict.get("relative_conflict"),
        "purity_conflict_detail": purity_conflict,
        "n_cells_scored": after["n_cells_scored"],
        "n_cells_aligned": len(shared_cells),
        "n_genes_aligned": len(shared_genes),
        "rctd": rctd_agg,
        "runtime_memory": runtime,
        "manuscript_note": (
            "SPLIT/RCTD was evaluated as a cell-level profile purification method. "
            "Removed counts were estimated at the cell-by-gene level and treated as "
            "pseudo-unassigned for count-based comparisons. Exact transcript "
            "coordinates of removed molecules were not available and were not used "
            "for transcript-coordinate metrics."),
    }
    # cell_level_method_summary.json (+ legacy cell_level_metrics.json alias)
    out_json = args.outdir / "cell_level_method_summary.json"
    out_json.write_text(json.dumps(metrics, indent=2, default=str))
    (args.outdir / "cell_level_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    # cell_level_benchmark_summary.tsv (one-row flat summary)
    flat = {
        "method": args.method, "method_class": "cell_level_cleaning",
        "reference_correlation_before": before["reference_correlation_mean"],
        "reference_correlation_after": after["reference_correlation_mean"],
        "median_reference_pearson_r_pseudobulk": pseudobulk_median_r,
        "marker_leakage_before": before["marker_leakage_mean"],
        "marker_leakage_after": after["marker_leakage_mean"],
        "marker_specificity_before": before["marker_specificity_mean"],
        "marker_specificity_after": after["marker_specificity_mean"],
        "relative_purity_before": (purity_conflict.get("relative_purity") or {}).get("before"),
        "relative_purity_after": (purity_conflict.get("relative_purity") or {}).get("after"),
        "relative_conflict_before": (purity_conflict.get("relative_conflict") or {}).get("before"),
        "relative_conflict_after": (purity_conflict.get("relative_conflict") or {}).get("after"),
        "estimated_removed_count": pruning.get("estimated_removed_count"),
        "pseudo_post_cleaning_unassigned_count": pruning.get("pseudo_post_cleaning_unassigned_count"),
        "fraction_removed_of_original_assigned": pruning.get("fraction_removed_of_original_assigned"),
        "mean_rctd_entropy": rctd_agg.get("mean_rctd_entropy"),
        "n_doublet_or_mixed": rctd_agg.get("n_doublet_or_mixed"),
        "n_cells_scored": after["n_cells_scored"],
        "total_seconds": runtime.get("total_seconds"),
        "peak_rss_gb_observed": runtime.get("peak_rss_gb_observed"),
    }
    pd.DataFrame([flat]).to_csv(args.outdir / "cell_level_benchmark_summary.tsv", sep="\t", index=False)
    pd.DataFrame([flat]).to_csv(args.outdir / "cell_level_metrics.tsv", sep="\t", index=False)  # legacy alias

    # reference_consistency_by_celltype.tsv (per coarse cell type, before/after)
    b = before_pc.groupby("cell_type").agg(
        n_cells=("ref_corr", "size"),
        ref_corr_before=("ref_corr", "mean"),
        marker_specificity_before=("marker_specificity", "mean"),
        marker_leakage_before=("marker_leakage", "mean"))
    a = after_pc.groupby("cell_type").agg(
        ref_corr_after=("ref_corr", "mean"),
        marker_specificity_after=("marker_specificity", "mean"),
        marker_leakage_after=("marker_leakage", "mean"))
    bycelltype = b.join(a, how="outer").reset_index()
    bycelltype.insert(0, "method", args.method)
    bycelltype["ref_corr_delta"] = bycelltype["ref_corr_after"] - bycelltype["ref_corr_before"]
    # reference_consistency_by_celltype.tsv: primary column `pearson_r` is the
    # cross-method-COMPARABLE pseudobulk value (matches get_metric); per-cell
    # before/after correlations are carried as extra columns for the purification
    # effect. Merge so downstream plots that read `pearson_r` get a fair number.
    rc_out = pseudobulk_rc.merge(
        bycelltype[["cell_type", "ref_corr_before", "ref_corr_after", "ref_corr_delta",
                    "marker_specificity_after", "marker_leakage_after"]],
        on="cell_type", how="outer")
    rc_out["method"] = args.method
    rc_out.to_csv(args.outdir / "reference_consistency_by_celltype.tsv", sep="\t",
                  index=False, float_format="%.6g")
    # Per-cell version kept separately for transparency.
    bycelltype.to_csv(args.outdir / "reference_consistency_by_celltype_percell.tsv",
                      sep="\t", index=False, float_format="%.6g")

    # marker_specificity_log2fc.tsv (per cell type; log2 after/before specificity)
    spec = bycelltype[["method", "cell_type", "n_cells",
                       "marker_specificity_before", "marker_specificity_after"]].copy()
    spec["specificity_log2fc_after_vs_before"] = np.log2(
        (spec["marker_specificity_after"] + 1e-9) / (spec["marker_specificity_before"] + 1e-9))
    spec.to_csv(args.outdir / "marker_specificity_log2fc.tsv", sep="\t",
                index=False, float_format="%.6g")

    # marker_leakage_metrics.tsv (per cell type before/after + delta)
    leak = bycelltype[["method", "cell_type", "n_cells",
                       "marker_leakage_before", "marker_leakage_after"]].copy()
    leak["leakage_delta_after_minus_before"] = leak["marker_leakage_after"] - leak["marker_leakage_before"]
    leak.to_csv(args.outdir / "marker_leakage_metrics.tsv", sep="\t",
                index=False, float_format="%.6g")

    # pruning_summary_standardized.tsv (tidy one-row-per-field)
    if pruning:
        pr = pd.DataFrame([{"method": args.method, "field": k, "value": v}
                           for k, v in pruning.items()
                           if not isinstance(v, (dict, list))])
        pr.to_csv(args.outdir / "pruning_summary_standardized.tsv", sep="\t", index=False)

    # rctd_entropy_metrics.tsv (per-cell, from RCTD weights)
    if weights_df is not None and "rctd_weights_entropy" in weights_df.columns:
        ent_tab = weights_df.copy()
        ent_tab.insert(0, "method", args.method)
        if "weight_first_type" in ent_tab.columns and "weight_second_type" in ent_tab.columns:
            ent_tab["max_weight"] = ent_tab[["weight_first_type", "weight_second_type"]].max(axis=1)
        ent_tab.to_csv(args.outdir / "rctd_entropy_metrics.tsv", sep="\t", index=False)

    log.info("Reference correlation: before=%.4f after=%.4f (Δ=%+.4f)",
             before["reference_correlation_mean"], after["reference_correlation_mean"],
             metrics["reference_correlation"]["delta"])
    log.info("Marker leakage:       before=%.4f after=%.4f (Δ=%+.4f)",
             before["marker_leakage_mean"], after["marker_leakage_mean"],
             metrics["marker_leakage"]["delta"])
    log.info("Marker specificity:   before=%.4f after=%.4f (Δ=%+.4f)",
             before["marker_specificity_mean"], after["marker_specificity_mean"],
             metrics["marker_specificity"]["delta"])
    log.info("Wrote %s", out_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())

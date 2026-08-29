#!/usr/bin/env python3
"""Rebuild the Fig-3 cross-platform split heatmaps using the BEST TRACER
**resegment** parameter setting (from the tuning sweep), with a cleaner main
layout and the TRACER refined/reconstructed entities handled separately.

Design (per the figure spec):
  * Main "Whole-cell / profile benchmark" heatmaps (Block A + Block B) carry a
    SINGLE TRACER-resegment column (best params) + an optional TRACER-refine
    column, instead of TRACER / refined / reconstructed (fewer NA cells).
  * Method order: Original/Baseline, Baysor, proseg, Segger*, cellAdmix, SPLIT,
    TRACER-resegment, TRACER-refine. MERFISH baseline is labelled "Baysor".
  * Block B marker log2FC & Kendall τ are aggregated as the MEDIAN OVER ABUNDANT
    cell types (n>=50 profiles in >= half the methods); a companion per-cell-type
    heatmap (abundant types × methods, grouped by dataset) is also rendered, and
    low-abundance types go to the supplement.
  * TRACER-reconstructed partial cells get their own "partial-cell recovery"
    panel (count, median tx/partial, cell-type distribution, purity/conflict,
    representative spatial example) — NOT mixed into the whole-cell metrics.
  * A full NA-rich version (all 9 entity columns incl. reconstructed) is kept as
    a supplementary audit figure.

Data sources (all apples-to-apples with compute_block_b methodology):
  * best TRACER resegment + refine-in-place: results/tracer_resegment_tuning/runs/
  * cached methods: summary_heatmaps/all_metrics_long.tsv (cells/tx/purity/
    conflict/marker/kendall) + per-cell-type metrics/<ds>/<ent>/ + benchmark_
    comparison.tsv (runtime/mem) + segger_roi/<ds>/benchmark/runtime_memory.json.

Run with the `spatial` env. Outputs ->
  results/fig3_cross_platform_roi_benchmark/summary_heatmaps_best_resegment/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import registry as R  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "axes.linewidth": 0.6,
})
CMAP_GOOD = "mako_r"
try:
    import seaborn  # noqa
except Exception:
    CMAP_GOOD = "viridis"
CMAP_NEUTRAL = "cividis"
NA_COLOR = "#e8e8e8"
GRID_COLOR = "white"

# Per-dataset display name for the row/platform label. atera_cervical is a Xenium
# ROI but is labelled "Atera" to distinguish it from the Xenium5K dataset.
PLATFORM_DISPLAY = {"atera_cervical": "Atera", "xenium5k_cervical": "Xenium5K",
                    "cosmx_nsclc": "CosMx", "merfish_mouse_ileum": "MERFISH"}


def platform_label(ds):
    return PLATFORM_DISPLAY[ds]


# Cell types excluded from all visualization rows/columns and the abundant-type
# aggregates (junk/unannotated cluster; already in registry reference_exclude).
EXCLUDE_CELLTYPES = {"Unannotated", "unannotated", "UNANNOTATED", "nan", "NA"}

# Shared definition appended to every figure so the two TRACER modes are
# unambiguous to anyone reading the heatmap in isolation.
TRACER_MODE_DEF = (
    "TRACER Resegmentation = full re-segmentation (may merge/reform entities across "
    "original cell boundaries), best per-platform tuned parameters, combined matrix.   "
    "TRACER Refine-in-Place = refinement-only (keeps the original vendor whole-cell "
    "partition; prunes conflicting transcripts + rebuilds partials, no cross-cell merging).")

# ---------------------------------------------------------------------------
REPO = R.REPO
TUNE = REPO / "results" / "tracer_resegment_tuning"
RUNS = TUNE / "runs"
FIG3 = REPO / "results" / "fig3_cross_platform_roi_benchmark"
SHEAT = FIG3 / "summary_heatmaps"
SEGGER = REPO / "results" / "segger_roi"
OUT = FIG3 / "summary_heatmaps_best_resegment"
OUT.mkdir(parents=True, exist_ok=True)

BEST_COMBO = {
    "atera_cervical": "specificity_preset",
    "xenium5k_cervical": "stitch_maha_off",
    "cosmx_nsclc": "stitch_dC0.10",
    "merfish_mouse_ileum": "specificity_preset",
}

# Main whole-cell benchmark column order + labels.
MAIN_METHODS = ["original", "baysor", "proseg", "segger", "celladmix", "split",
                "TRACER_resegment", "TRACER_refine"]
LABELS = {
    "original": "Original /\nbaseline", "baysor": "Baysor", "proseg": "proseg",
    "segger": "Segger*", "celladmix": "cellAdmix", "split": "SPLIT",
    "TRACER_resegment": "TRACER\nResegmentation", "TRACER_refine": "TRACER\nRefine-in-Place",
    "TRACER_reconstructed": "TRACER\nReconstructed\n(partial)",
}
GPU_METHODS = {"segger"}
# cached-method name maps
BC_NAME = {"baysor": "baysor", "proseg": "proseg", "celladmix": "cellAdmix",
           "split": "SPLIT"}
METRICS_DIR_NAME = {"original": "original", "baysor": "baysor", "proseg": "proseg",
                    "segger": "segger", "celladmix": "celladmix", "split": "split",
                    "TRACER_reconstructed": "TRACER_reconstructed"}
ABUND_MIN_CELLS = 50

METRIC_LABEL = {
    "total_cells": "Total cells / profiles",
    "transcripts_per_cell": "Transcripts per cell / profile (median)",
    "runtime_seconds": "Runtime (s)",
    "peak_memory_gb": "Peak memory (GB)",
    "marker_log2fc": "Marker specificity log2FC (median over abundant types)",
    "relative_purity": "Relative purity (NPMI)",
    "relative_conflict": "Relative conflict (NPMI)",
    "kendall_tau": "Kendall τ vs scRNA (median over abundant types)",
}
METRIC_DIR = {  # higher/lower/neutral
    "total_cells": "neutral", "transcripts_per_cell": "neutral",
    "runtime_seconds": "lower", "peak_memory_gb": "lower",
    "marker_log2fc": "higher", "relative_purity": "higher",
    "relative_conflict": "lower", "kendall_tau": "higher",
}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _aml() -> pd.DataFrame:
    return pd.read_csv(SHEAT / "all_metrics_long.tsv", sep="\t")


def _aml_val(aml, ds, entity, metric):
    sub = aml[(aml.dataset == ds) & (aml.entity == entity) & (aml.metric == metric)]
    if len(sub) and pd.notna(sub["value"].iloc[0]):
        return float(sub["value"].iloc[0])
    return np.nan


def _tune_metrics(ds, run):
    p = RUNS / ds / run / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _tune_runtime(ds, run):
    p = RUNS / ds / run / "runtime_memory.json"
    if p.exists():
        j = json.loads(p.read_text())
        return float(j.get("total_seconds", np.nan)), float(j.get("peak_rss_gb_observed", np.nan))
    return np.nan, np.nan


def _bc():
    return pd.read_csv(FIG3 / "benchmark_comparison.tsv", sep="\t")


def _bc_runtime(bc, ds, method):
    row = bc[(bc.dataset == ds) & (bc.method == BC_NAME[method])]
    if len(row):
        return float(row.runtime_seconds.iloc[0]), float(row.peak_rss_gb.iloc[0])
    return np.nan, np.nan


def _segger_runtime(ds):
    p = SEGGER / ds / "benchmark" / "runtime_memory.json"
    if p.exists():
        j = json.loads(p.read_text())
        return float(j["runtime_seconds"]), float(j.get("peak_gpu_memory_gb") or np.nan)
    return np.nan, np.nan


def _score_dir(ds, method):
    """Directory holding per-cell-type artifacts for (ds, method)."""
    if method == "TRACER_resegment":
        return RUNS / ds / BEST_COMBO[ds] / "score" / "combined"
    if method == "TRACER_refine":
        return RUNS / ds / "__refine_in_place" / "score" / "combined"
    if method == "TRACER_reconstructed":
        return RUNS / ds / BEST_COMBO[ds] / "score" / "reconstructed"
    return SHEAT / "metrics" / ds / METRICS_DIR_NAME[method]


def _annot_counts(ds, method) -> pd.Series:
    p = _score_dir(ds, method) / "post_celltype_annotations.tsv"
    if not p.exists():
        return pd.Series(dtype=int)
    a = pd.read_csv(p, sep="\t")
    return a["predicted_celltype"].astype(str).value_counts()


def _per_type_marker(ds, method) -> pd.Series:
    p = _score_dir(ds, method) / "marker_specificity_log2fc.tsv"
    if not p.exists():
        return pd.Series(dtype=float)
    d = pd.read_csv(p, sep="\t")
    return d.groupby("cell_type")["spatial_log2fc"].median()


def _per_type_kendall(ds, method) -> pd.Series:
    p = _score_dir(ds, method) / "reference_consistency_kendall.tsv"
    if not p.exists():
        return pd.Series(dtype=float)
    d = pd.read_csv(p, sep="\t")
    return d.set_index("cell_type")["kendall_tau"]


def method_present(ds, method) -> bool:
    if method == "baysor" and not R.DATASETS[ds]["has_baysor"]:
        return False
    if method in ("original", "TRACER_resegment", "TRACER_refine"):
        return True
    return (_score_dir(ds, method) / "post_celltype_annotations.tsv").exists()


# ---------------------------------------------------------------------------
# Abundant cell types
# ---------------------------------------------------------------------------
def abundant_types(ds, methods):
    """Cell type is abundant if it has >= ABUND_MIN_CELLS profiles in at least
    half of the methods present for the dataset."""
    present = [m for m in methods if method_present(ds, m)]
    counts = {m: _annot_counts(ds, m) for m in present}
    allcts = sorted({ct for s in counts.values() for ct in s.index}
                    - EXCLUDE_CELLTYPES)
    need = int(np.ceil(len(present) / 2))
    rows, ab = [], []
    for ct in allcts:
        n_meets = sum(int(counts[m].get(ct, 0)) >= ABUND_MIN_CELLS for m in present)
        per = {m: int(counts[m].get(ct, 0)) for m in present}
        is_ab = n_meets >= need
        if is_ab:
            ab.append(ct)
        rows.append(dict(dataset=ds, cell_type=ct, n_methods_meeting=n_meets,
                         n_methods_present=len(present), threshold=ABUND_MIN_CELLS,
                         abundant=is_ab, **{f"n_{m}": per.get(m, 0) for m in present}))
    return ab, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Block A / B value assembly
# ---------------------------------------------------------------------------
def blockA_value(aml, bc, ds, method, metric):
    if metric in ("total_cells", "transcripts_per_cell"):
        if method == "TRACER_resegment":
            m = _tune_metrics(ds, BEST_COMBO[ds])
            return m.get("combined_n_cells") if metric == "total_cells" else m.get("combined_median_tx")
        if method == "TRACER_refine":
            m = _tune_metrics(ds, "__refine_in_place")
            return m.get("combined_n_cells") if metric == "total_cells" else m.get("combined_median_tx")
        if not method_present(ds, method):
            return np.nan
        return _aml_val(aml, ds, method, metric)
    # runtime / memory
    if method == "TRACER_resegment":
        rt, mem = _tune_runtime(ds, BEST_COMBO[ds])
    elif method == "TRACER_refine":
        rt, mem = _tune_runtime(ds, "__refine_in_place")
    elif method == "segger":
        rt, mem = _segger_runtime(ds)
    elif method in BC_NAME:
        if not method_present(ds, method):
            return np.nan
        rt, mem = _bc_runtime(bc, ds, method)
    else:  # original = native segmentation, no benchmarked compute
        rt, mem = np.nan, np.nan
    return rt if metric == "runtime_seconds" else mem


def blockB_value(aml, ds, method, metric, abund):
    """marker/kendall = median over abundant cell types; purity/conflict = aggregate."""
    if metric in ("relative_purity", "relative_conflict"):
        if method == "TRACER_resegment":
            return _tune_metrics(ds, BEST_COMBO[ds]).get(f"combined_{metric}")
        if method == "TRACER_refine":
            return _tune_metrics(ds, "__refine_in_place").get(f"combined_{metric}")
        if not method_present(ds, method):
            return np.nan
        return _aml_val(aml, ds, method, metric)
    if not method_present(ds, method):
        return np.nan
    if metric == "marker_log2fc":
        s = _per_type_marker(ds, method)
    else:
        s = _per_type_kendall(ds, method)
    s = s[s.index.isin(abund)].dropna()
    return float(s.median()) if len(s) else np.nan


# ---------------------------------------------------------------------------
# Generic heatmap renderer (rows = datasets, cols = methods) per metric panel
# ---------------------------------------------------------------------------
def fmt(metric, v):
    if pd.isna(v):
        return "NA"
    if metric in ("total_cells",):
        return f"{int(round(v)):,}"
    if metric in ("transcripts_per_cell", "runtime_seconds"):
        return f"{v:,.0f}"
    if metric == "peak_memory_gb":
        return f"{v:.2f}"
    return f"{v:.2f}"


def plot_blocks(table, metrics, methods, out_base, title, subtitle, footnote):
    col_labels = [LABELS[m] for m in methods]
    ncols, n_ds, npan = len(methods), len(R.DATASET_ORDER), len(metrics)
    cell_h = 0.45
    fig_h = npan * (n_ds * cell_h + 0.6) + 3.0
    fig_w = max(10, ncols * 1.15 + 3.0)
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(npan, 1, hspace=0.95, left=0.17, right=0.88,
                          top=1 - 1.35 / fig_h, bottom=2.5 / fig_h)
    for pi, metric in enumerate(metrics):
        ax = fig.add_subplot(gs[pi, 0])
        direction = METRIC_DIR[metric]
        raw = np.full((n_ds, ncols), np.nan)
        for di, ds in enumerate(R.DATASET_ORDER):
            for ci, m in enumerate(methods):
                sub = table[(table.dataset == ds) & (table.metric == metric)]
                raw[di, ci] = sub[m].iloc[0] if len(sub) and m in sub.columns else np.nan
        g = np.full_like(raw, np.nan)
        for di in range(n_ds):
            vals = raw[di]; fin = np.isfinite(vals)
            if fin.sum() >= 1:
                vmin, vmax = np.nanmin(vals[fin]), np.nanmax(vals[fin])
                norm = (vals - vmin) / (vmax - vmin) if vmax > vmin else np.where(fin, 0.5, np.nan)
                g[di] = (1 - norm) if direction == "lower" else norm
        cmap = plt.get_cmap(CMAP_NEUTRAL if direction == "neutral" else CMAP_GOOD).copy()
        cmap.set_bad(NA_COLOR)
        im = ax.imshow(np.ma.masked_invalid(g), aspect="auto", cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(np.arange(-.5, ncols, 1), minor=True)
        ax.set_yticks(np.arange(-.5, n_ds, 1), minor=True)
        ax.grid(which="minor", color=GRID_COLOR, linewidth=1.4)
        ax.tick_params(which="minor", length=0)
        for di in range(n_ds):
            for ci in range(ncols):
                v = raw[di, ci]
                if pd.isna(v):
                    ax.text(ci, di, "NA", ha="center", va="center", fontsize=6.3,
                            color="#9a9a9a", style="italic"); continue
                gg = g[di, ci]; r, gr, b, _ = cmap(gg if np.isfinite(gg) else 0.5)
                lum = 0.2126 * r + 0.7152 * gr + 0.0722 * b
                ax.text(ci, di, fmt(metric, v), ha="center", va="center",
                        fontsize=6.8, color="white" if lum < 0.55 else "#15161a")
        ax.set_yticks(range(n_ds))
        ax.set_yticklabels([platform_label(ds) for ds in R.DATASET_ORDER], fontsize=8)
        ax.set_xticks(range(ncols))
        if pi == npan - 1:
            ax.set_xticklabels(col_labels, fontsize=7.2, rotation=40, ha="right")
        else:
            ax.set_xticklabels([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        dlab = {"higher": "higher = better", "lower": "lower = better",
                "neutral": "descriptive"}[direction]
        ax.set_title(f"{METRIC_LABEL[metric]}   ·   {dlab}", fontsize=9, loc="left",
                     pad=4, fontweight="bold")
        cax = ax.inset_axes([1.012, 0.0, 0.018, 1.0])
        cb = fig.colorbar(im, cax=cax); cb.set_ticks([0, 1])
        cb.set_ticklabels(["worse", "better"] if direction != "neutral" else ["low", "high"],
                          fontsize=6); cb.outline.set_linewidth(0.4)
    fig.suptitle(title, x=0.17, y=1 - 0.35 / fig_h, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.17, 1 - 0.78 / fig_h, subtitle, fontsize=7.2, color="#333333", va="top")
    fig.text(0.17, 1.55 / fig_h, footnote, fontsize=6.3, color="#444444", va="top", wrap=True)
    for ext in ("png", "svg"):
        fig.savefig(f"{out_base}.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_base + ".png/.svg")


# ---------------------------------------------------------------------------
# Per-cell-type companion heatmap (abundant types × methods, grouped by dataset)
# ---------------------------------------------------------------------------
def plot_celltype_heatmap(metric, per_type, methods, abund_map, out_base, title):
    methods = [m for m in methods]
    rows = []  # (dataset, celltype)
    for ds in R.DATASET_ORDER:
        for ct in abund_map[ds]:
            rows.append((ds, ct))
    n = len(rows)
    mat = np.full((n, len(methods)), np.nan)
    for ri, (ds, ct) in enumerate(rows):
        for ci, m in enumerate(methods):
            v = per_type.get((ds, m, metric), {}).get(ct, np.nan)
            mat[ri, ci] = v
    g = np.full_like(mat, np.nan)
    for ri in range(n):
        vals = mat[ri]; fin = np.isfinite(vals)
        if fin.sum() >= 1:
            vmin, vmax = np.nanmin(vals[fin]), np.nanmax(vals[fin])
            g[ri] = (vals - vmin) / (vmax - vmin) if vmax > vmin else np.where(fin, 0.5, np.nan)
    fig_h = max(4, n * 0.30 + 1.8); fig_w = len(methods) * 1.0 + 4.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    cmap = plt.get_cmap(CMAP_GOOD).copy(); cmap.set_bad(NA_COLOR)
    im = ax.imshow(np.ma.masked_invalid(g), aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(np.arange(-.5, len(methods), 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color=GRID_COLOR, linewidth=1.2); ax.tick_params(which="minor", length=0)
    for ri in range(n):
        for ci in range(len(methods)):
            v = mat[ri, ci]
            if pd.isna(v):
                ax.text(ci, ri, "NA", ha="center", va="center", fontsize=5.5,
                        color="#9a9a9a", style="italic"); continue
            gg = g[ri, ci]; r, gr, b, _ = cmap(gg if np.isfinite(gg) else 0.5)
            lum = 0.2126 * r + 0.7152 * gr + 0.0722 * b
            ax.text(ci, ri, f"{v:.2f}", ha="center", va="center", fontsize=5.6,
                    color="white" if lum < 0.55 else "#15161a")
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{platform_label(ds)} · {ct}" for ds, ct in rows], fontsize=6.5)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([LABELS[m] for m in methods], fontsize=7, rotation=40, ha="right")
    # dataset separators
    prev = None
    for ri, (ds, _ct) in enumerate(rows):
        if ds != prev and ri > 0:
            ax.axhline(ri - 0.5, color="#222", lw=1.0)
        prev = ds
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, fontsize=11, loc="left", fontweight="bold", pad=8)
    cax = ax.inset_axes([1.01, 0.0, 0.02, 1.0])
    cb = fig.colorbar(im, cax=cax); cb.set_ticks([0, 1])
    cb.set_ticklabels(["worse", "better"], fontsize=6)
    fig.text(0.01, 0.005, "Colour = within-row (per cell type, per dataset) min–max across methods. "
             "Abundant cell types only (n≥50 profiles in ≥half the methods). Raw values annotated.",
             fontsize=6.2, color="#444")
    for ext in ("png", "svg"):
        fig.savefig(f"{out_base}.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_base + ".png/.svg")


# ---------------------------------------------------------------------------
# Partial-cell recovery panel
# ---------------------------------------------------------------------------
def partial_recovery_table():
    rows = []
    for ds in R.DATASET_ORDER:
        m = _tune_metrics(ds, BEST_COMBO[ds])
        rows.append(dict(
            dataset=ds, platform=platform_label(ds), combo=BEST_COMBO[ds],
            n_partial_cells=m.get("reconstructed_n_cells"),
            median_tx_per_partial=m.get("reconstructed_median_tx"),
            relative_purity=m.get("reconstructed_relative_purity"),
            relative_conflict=m.get("reconstructed_relative_conflict"),
            marker_log2fc=m.get("reconstructed_marker_log2fc"),
            kendall_tau=m.get("reconstructed_kendall_tau"),
            n_whole_cells=m.get("refined_n_cells"),
        ))
    return pd.DataFrame(rows)


def partial_celltype_dist():
    rows = []
    for ds in R.DATASET_ORDER:
        c = _annot_counts(ds, "TRACER_reconstructed")
        c = c[~c.index.isin(EXCLUDE_CELLTYPES)]
        tot = c.sum()
        for ct, n in c.items():
            rows.append(dict(dataset=ds, cell_type=ct, n_partials=int(n),
                             frac=float(n) / tot if tot else np.nan))
    return pd.DataFrame(rows)


def plot_partial_panel(pr, dist):
    fig = plt.figure(figsize=(15, 8.5))
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.42,
                          left=0.06, right=0.97, top=0.9, bottom=0.09)
    # (1) stats table-as-bars: n partials + median tx
    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(len(pr))
    ax.barh(y, pr["n_partial_cells"], color="tab:orange")
    ax.set_yticks(y); ax.set_yticklabels(pr["platform"], fontsize=8); ax.invert_yaxis()
    for i, v in enumerate(pr["n_partial_cells"]):
        ax.text(v, i, f" {int(v)}", va="center", fontsize=7)
    ax.set_title("Reconstructed partial cells (count)", fontsize=10, fontweight="bold")
    ax = fig.add_subplot(gs[0, 1])
    ax.barh(y, pr["median_tx_per_partial"], color="tab:blue")
    ax.set_yticks(y); ax.set_yticklabels(pr["platform"], fontsize=8); ax.invert_yaxis()
    for i, v in enumerate(pr["median_tx_per_partial"]):
        ax.text(v, i, f" {v:.0f}", va="center", fontsize=7)
    ax.set_title("Median transcripts / partial", fontsize=10, fontweight="bold")
    # (2) purity/conflict
    ax = fig.add_subplot(gs[0, 2])
    w = 0.38
    ax.barh(y - w / 2, pr["relative_purity"], height=w, color="tab:green", label="rel. purity")
    ax.barh(y + w / 2, pr["relative_conflict"], height=w, color="tab:red", label="rel. conflict")
    ax.set_yticks(y); ax.set_yticklabels(pr["platform"], fontsize=8); ax.invert_yaxis()
    ax.legend(fontsize=7); ax.set_xlim(0, 1.02)
    ax.set_title("Partial-cell NPMI purity / conflict", fontsize=10, fontweight="bold")
    # (3) cell-type distribution stacked per dataset
    ax = fig.add_subplot(gs[1, :2])
    piv = dist.pivot_table(index="dataset", columns="cell_type", values="n_partials",
                           fill_value=0).reindex(R.DATASET_ORDER)
    bottom = np.zeros(len(piv))
    cmap = plt.get_cmap("tab20")
    for j, ct in enumerate(piv.columns):
        ax.bar(range(len(piv)), piv[ct].values, bottom=bottom, label=ct,
               color=cmap(j % 20))
        bottom += piv[ct].values
    ax.set_xticks(range(len(piv)))
    ax.set_xticklabels([platform_label(d) for d in piv.index], fontsize=8)
    ax.set_ylabel("partial cells"); ax.set_title("Partial-cell cell-type distribution",
                                                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=5.5, ncol=2, bbox_to_anchor=(1.0, 1.0), loc="upper left")
    # (4) representative spatial example (dataset with most partials)
    ax = fig.add_subplot(gs[1, 2])
    ds_rep = pr.sort_values("n_partial_cells", ascending=False)["dataset"].iloc[0]
    _plot_spatial_partials(ax, ds_rep)
    fig.suptitle("TRACER-reconstructed partial-cell recovery panel (best resegment config)",
                 x=0.06, ha="left", fontsize=13, fontweight="bold")
    for ext in ("png", "svg"):
        fig.savefig(f"{OUT / 'partial_cell_recovery_panel'}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote partial_cell_recovery_panel.png/.svg")


def _plot_spatial_partials(ax, ds):
    pq = RUNS / ds / BEST_COMBO[ds] / "outputs" / "transcripts_tracer_refined.parquet"
    df = pd.read_parquet(pq, columns=["x", "y", "_etype"])
    et = df["_etype"].astype(str)
    whole = df[et == "cell"].sample(min(40000, int((et == "cell").sum())), random_state=1)
    part = df[et == "partial"]
    # crop to a window dense in partials for a representative view
    if len(part):
        cx, cy = part["x"].median(), part["y"].median()
        half = 250
        m = lambda d: ((d.x - cx).abs() < half) & ((d.y - cy).abs() < half)
        ax.scatter(whole[m(whole)].x, whole[m(whole)].y, s=1, c="#cccccc", label="whole-cell tx")
        ax.scatter(part[m(part)].x, part[m(part)].y, s=1.5, c="tab:orange", label="partial tx")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Representative partials — {platform_label(ds)}\n(500 µm window)",
                 fontsize=9, fontweight="bold")
    ax.legend(fontsize=6, markerscale=4, loc="upper right")


# ---------------------------------------------------------------------------
def main():
    aml = _aml(); bc = _bc()

    # --- abundant cell types ---
    abund_map, abund_summaries = {}, []
    for ds in R.DATASET_ORDER:
        ab, summ = abundant_types(ds, MAIN_METHODS)
        abund_map[ds] = ab
        abund_summaries.append(summ)
    pd.concat(abund_summaries, ignore_index=True).to_csv(
        OUT / "abundant_cell_type_filtering_summary.tsv", sep="\t", index=False)

    # --- Block A table ---
    a_metrics = ["total_cells", "transcripts_per_cell", "runtime_seconds", "peak_memory_gb"]
    recsA = []
    for metric in a_metrics:
        for ds in R.DATASET_ORDER:
            row = dict(dataset=ds, platform=platform_label(ds), metric=metric,
                       baseline_method=R.DATASETS[ds]["original_label"])
            for m in MAIN_METHODS:
                row[m] = blockA_value(aml, bc, ds, m, metric)
            recsA.append(row)
    tblA = pd.DataFrame(recsA)
    tblA.to_csv(OUT / "source_data_blockA.tsv", sep="\t", index=False)

    # Block A companion: TRACER refined vs reconstructed counts (kept out of main
    # to avoid NA-heavy columns; combined = what the main TRACER-reseg column shows).
    comp_rows = []
    for ds in R.DATASET_ORDER:
        mm = _tune_metrics(ds, BEST_COMBO[ds])
        comp_rows.append(dict(
            dataset=ds, platform=platform_label(ds), combo=BEST_COMBO[ds],
            combined_cells=mm.get("combined_n_cells"), combined_median_tx=mm.get("combined_median_tx"),
            refined_whole_cells=mm.get("refined_n_cells"), refined_median_tx=mm.get("refined_median_tx"),
            reconstructed_partials=mm.get("reconstructed_n_cells"),
            reconstructed_median_tx=mm.get("reconstructed_median_tx")))
    pd.DataFrame(comp_rows).to_csv(OUT / "blockA_tracer_refined_reconstructed_companion.tsv",
                                   sep="\t", index=False)

    # --- Block B table (+ per-type cache for companion) ---
    b_metrics = ["marker_log2fc", "relative_purity", "relative_conflict", "kendall_tau"]
    per_type = {}
    for ds in R.DATASET_ORDER:
        for m in MAIN_METHODS:
            per_type[(ds, m, "marker_log2fc")] = _per_type_marker(ds, m).to_dict()
            per_type[(ds, m, "kendall_tau")] = _per_type_kendall(ds, m).to_dict()
    recsB = []
    for metric in b_metrics:
        for ds in R.DATASET_ORDER:
            row = dict(dataset=ds, platform=platform_label(ds), metric=metric,
                       baseline_method=R.DATASETS[ds]["original_label"])
            for m in MAIN_METHODS:
                row[m] = blockB_value(aml, ds, m, metric, abund_map[ds])
            recsB.append(row)
    tblB = pd.DataFrame(recsB)
    tblB.to_csv(OUT / "source_data_blockB.tsv", sep="\t", index=False)

    # --- per-cell-type long tables (marker + kendall) ---
    ct_rows = []
    for metric in ("marker_log2fc", "kendall_tau"):
        for ds in R.DATASET_ORDER:
            cts = sorted({ct for m in MAIN_METHODS for ct in per_type[(ds, m, metric)]}
                         - EXCLUDE_CELLTYPES)
            for ct in cts:
                row = dict(dataset=ds, platform=platform_label(ds), metric=metric,
                           cell_type=ct, abundant=ct in abund_map[ds])
                for m in MAIN_METHODS:
                    row[m] = per_type[(ds, m, metric)].get(ct, np.nan)
                ct_rows.append(row)
    ct_long = pd.DataFrame(ct_rows)
    ct_long[ct_long.metric == "marker_log2fc"].to_csv(
        OUT / "celltype_marker_log2fc.tsv", sep="\t", index=False)
    ct_long[ct_long.metric == "kendall_tau"].to_csv(
        OUT / "celltype_kendall.tsv", sep="\t", index=False)

    # --- partial-cell recovery ---
    pr = partial_recovery_table(); pr.to_csv(OUT / "partial_cell_recovery_stats.tsv", sep="\t", index=False)
    dist = partial_celltype_dist(); dist.to_csv(OUT / "partial_cell_celltype_distribution.tsv", sep="\t", index=False)

    # =================== FIGURES ===================
    footA = ("Original/baseline: Atera/Xenium5K = 10x Xenium · CosMx = CosMx SMI · MERFISH = Baysor.   "
             "* Segger = GPU (runtime/peak memory are GPU; H100).   "
             "Original runtime/memory = NA (native vendor segmentation).   "
             "Baysor NA on MERFISH (it IS the baseline).   "
             "TRACER refined/reconstructed split kept in companion table.\n" + TRACER_MODE_DEF)
    plot_blocks(tblA, a_metrics, MAIN_METHODS,
                str(OUT / "main_blockA_compute"),
                "Block A — Whole-cell / profile size & compute (best TRACER resegment)",
                "Single TRACER-resegment column (best params) + TRACER-refine. "
                "TRACER refined/reconstructed split kept in companion table.", footA)

    footB = ("Marker log2FC & Kendall τ = median over ABUNDANT cell types "
             f"(n≥{ABUND_MIN_CELLS} in ≥half the methods; 'Unannotated' excluded); "
             "see companion per-cell-type heatmaps. "
             "Relative purity/conflict are NPMI-coherence aggregates.\n" + TRACER_MODE_DEF)
    plot_blocks(tblB, b_metrics, MAIN_METHODS,
                str(OUT / "main_blockB_biological"),
                "Block B — Biological coherence (best TRACER resegment)",
                "Whole-cell / profile benchmark. Marker & Kendall aggregated over abundant cell types.",
                footB)

    plot_celltype_heatmap("marker_log2fc", per_type, MAIN_METHODS, abund_map,
                          str(OUT / "companion_celltype_marker_log2fc"),
                          "Companion — marker specificity log2FC per abundant cell type")
    plot_celltype_heatmap("kendall_tau", per_type, MAIN_METHODS, abund_map,
                          str(OUT / "companion_celltype_kendall"),
                          "Companion — Kendall τ per abundant cell type")

    plot_partial_panel(pr, dist)

    # --- supplementary full NA-rich audit (all entity columns incl reconstructed) ---
    sup_methods = MAIN_METHODS + ["TRACER_reconstructed"]
    recsA2 = []
    for metric in a_metrics:
        for ds in R.DATASET_ORDER:
            row = dict(dataset=ds, metric=metric)
            for m in sup_methods:
                if m == "TRACER_reconstructed":
                    if metric in ("total_cells", "transcripts_per_cell"):
                        mm = _tune_metrics(ds, BEST_COMBO[ds])
                        row[m] = mm.get("reconstructed_n_cells") if metric == "total_cells" else mm.get("reconstructed_median_tx")
                    else:
                        row[m] = np.nan
                else:
                    row[m] = blockA_value(aml, bc, ds, m, metric)
            recsA2.append(row)
    tblA2 = pd.DataFrame(recsA2)
    recsB2 = []
    for metric in b_metrics:
        for ds in R.DATASET_ORDER:
            row = dict(dataset=ds, metric=metric)
            for m in sup_methods:
                if m == "TRACER_reconstructed":
                    mm = _tune_metrics(ds, BEST_COMBO[ds])
                    if metric in ("relative_purity", "relative_conflict"):
                        row[m] = mm.get(f"reconstructed_{metric}")
                    else:
                        row[m] = mm.get(f"reconstructed_{metric}")
                else:
                    row[m] = blockB_value(aml, ds, m, metric, abund_map[ds])
            recsB2.append(row)
    tblB2 = pd.DataFrame(recsB2)
    tblA2.to_csv(OUT / "supp_source_data_blockA_full.tsv", sep="\t", index=False)
    tblB2.to_csv(OUT / "supp_source_data_blockB_full.tsv", sep="\t", index=False)
    plot_blocks(tblA2, a_metrics, sup_methods, str(OUT / "supp_blockA_full_audit"),
                "Supplementary audit — Block A (full, NA-rich: TRACER reseg + refine + reconstructed)",
                "Full entity split retained for audit. NA cells expected (reconstructed has no compute; original native).",
                footA)
    plot_blocks(tblB2, b_metrics, sup_methods, str(OUT / "supp_blockB_full_audit"),
                "Supplementary audit — Block B (full, NA-rich: TRACER reseg + refine + reconstructed)",
                "Full entity split retained for audit. Reconstructed = partial-cell metrics (also in partial panel).",
                footB)

    print("DONE ->", OUT)


if __name__ == "__main__":
    main()

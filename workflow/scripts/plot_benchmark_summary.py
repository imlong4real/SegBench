#!/usr/bin/env python3
"""Publication-quality cross-method benchmark visualizations (TSU20 lung Xenium).

Consumes the standardized summary tables produced by the downstream metric step
and renders Nature-style multi-panel figures plus per-figure source-data TSVs.

USAGE
=====
    python workflow/scripts/plot_benchmark_summary.py \\
      --summary-dir  results/benchmark/lung_xenium_ref36973297/summary \\
      --metrics-root results/benchmark/lung_xenium_ref36973297/metrics \\
      --outdir       results/benchmark/lung_xenium_ref36973297/figures/benchmark_summary \\
      --format both

Caveats preserved in every panel:
  * SPLIT is cell-level profile purification — NOT transcript-level segmentation.
    Its removed transcripts are count-level pseudo-unassigned estimates, not exact
    molecule-coordinate removals. ovrlpy is n/a for SPLIT (never imputed to 0).
  * cellAdmix is retained-transcript cleaning on ORIGINAL Xenium cell IDs, not de
    novo segmentation.
  * ovrlpy and RCTD are orthogonal diagnostics, not optimization targets / ground truth.
  * Reference label column used: Cell_Cluster_level1.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ===========================================================================
# Method registry — display order, labels, colors, classes, marker shapes
# ===========================================================================
# Internal method keys as they appear in the summary tables.
METHOD_ORDER = ["original", "Baysor", "proseg", "Segger", "cellAdmix", "SPLIT", "TRACER"]
LABEL = {"original": "10X", "Baysor": "Baysor", "proseg": "proseg", "Segger": "Segger",
         "cellAdmix": "cellAdmix", "SPLIT": "SPLIT", "TRACER": "TRACER"}
CLASS = {
    "original": "10X_multimodal_segmentation",
    "TRACER": "transcript_refinement",
    "Baysor": "transcript_segmentation",
    "proseg": "transcript_segmentation",
    "Segger": "transcript_segmentation_gpu",
    "cellAdmix": "transcript_cleaning",
    "SPLIT": "cell_level_cleaning",
}
CLASS_LABEL = {
    "10X_multimodal_segmentation": "10X multimodal segmentation",
    "transcript_segmentation": "Transcript segmentation",
    "transcript_segmentation_gpu": "Transcript segmentation (GPU)",
    "transcript_cleaning": "Transcript cleaning",
    "cell_level_cleaning": "Cell-level cleaning",
    "transcript_refinement": "Transcript refinement",
}
# Muted families; TRACER is the magenta accent.
COLOR = {
    "original": "#7f7f7f",   # grey
    "Baysor":   "#2c7fb8",   # blue
    "proseg":   "#41b6c4",   # teal
    "Segger":   "#225ea8",   # dark blue
    "cellAdmix":"#d95f02",   # orange
    "SPLIT":    "#7570b3",   # purple
    "TRACER":   "#e7298a",   # magenta accent
}
# Marker shape per method class (subtle class cue in scatter/legend).
CLASS_MARKER = {
    "10X_multimodal_segmentation": "s",
    "transcript_segmentation": "o",
    "transcript_segmentation_gpu": "D",
    "transcript_cleaning": "^",
    "cell_level_cleaning": "P",
    "transcript_refinement": "*",
}
ACCENT = "TRACER"
# Metrics directory name overrides (case / alternate roots).
DIR_ALIAS = {"Segger": "segger"}


def nature_style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.bbox": "tight",
        "font.family": "sans-serif",
        # DejaVu Sans first: clean sans-serif that includes arrow glyphs (↑/↓);
        # Arial/Helvetica lack them and would render missing-glyph boxes.
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 8, "axes.titlesize": 9, "axes.titleweight": "bold",
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "figure.dpi": 150, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })


# ===========================================================================
# IO helpers
# ===========================================================================
def load_summaries(summary_dir: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for key in ["method_runtime_summary", "transcript_assignment_summary",
                "cell_count_summary", "biological_metric_summary",
                "ovrlpy_summary", "rctd_summary"]:
        p = summary_dir / f"{key}.tsv"
        out[key] = pd.read_csv(p, sep="\t") if p.exists() else pd.DataFrame()
    return out


def ordered(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex a per-method frame to METHOD_ORDER (method column)."""
    d = df.set_index("method")
    d = d.reindex([m for m in METHOD_ORDER if m in d.index])
    return d


def num(s):
    return pd.to_numeric(s, errors="coerce")


def _metric_dirs(method: str, metrics_root: Path) -> list[Path]:
    """All candidate per-method metric dirs (benchmark tree + TRACER_nsclc tree)."""
    tnsclc = metrics_root.parent.parent.parent / "TRACER_nsclc" / "metrics"
    return [metrics_root / method, metrics_root / DIR_ALIAS.get(method, method),
            tnsclc / method, tnsclc / DIR_ALIAS.get(method, method)]


def metric_dir(method: str, metrics_root: Path) -> Path | None:
    for c in _metric_dirs(method, metrics_root):
        if c and c.exists():
            return c
    return None


def find_metric_file(method: str, fname: str, metrics_root: Path) -> Path | None:
    """First candidate dir that actually CONTAINS `fname` (get_metric outputs for
    TRACER/original live under TRACER_nsclc/metrics, while their ovrlpy/rctd dirs
    were created under the benchmark tree — so search for the file, not the dir)."""
    for d in _metric_dirs(method, metrics_root):
        if d and (d / fname).exists():
            return d / fname
    return None


class Saver:
    def __init__(self, outdir: Path, fmt: str):
        self.outdir = outdir
        self.sd = outdir / "source_data"
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.sd.mkdir(parents=True, exist_ok=True)
        # "both" = raster PNG + vector PDF + vector SVG.
        self.formats = ["png", "pdf", "svg"] if fmt == "both" else [fmt]
        self.figures: list[str] = []

    def fig(self, fig, name: str):
        for ext in self.formats:
            fig.savefig(self.outdir / f"{name}.{ext}", dpi=300)
        plt.close(fig)
        self.figures.append(name)

    def source(self, df: pd.DataFrame, name: str):
        df.to_csv(self.sd / name, sep="\t", index=False)


def mcolor(m): return COLOR.get(m, "#999999")
def mmarker(m): return CLASS_MARKER.get(CLASS.get(m, ""), "o")
def present(df, method): return method in set(df.get("method", []))


# ===========================================================================
# Combined per-method metric table (source: plot_method_metrics.tsv)
# ===========================================================================
def build_method_metrics(S: dict) -> pd.DataFrame:
    bio = S["biological_metric_summary"].set_index("method")
    rt = S["method_runtime_summary"].set_index("method")
    ov = S["ovrlpy_summary"].set_index("method")
    rc = S["rctd_summary"].set_index("method")
    cc = S["cell_count_summary"].set_index("method")
    rows = []
    for m in METHOD_ORDER:
        r = {"method": m, "label": LABEL[m], "method_class": CLASS[m],
             "class_label": CLASS_LABEL[CLASS[m]], "color": COLOR[m]}
        for src, cols in [(bio, ["median_reference_pearson_r", "mean_reference_pearson_r",
                                 "median_marker_log2fc", "median_tcell_marker_log2fc",
                                 "median_relative_purity", "median_relative_conflict",
                                 "median_marker_leakage", "n_celltypes_detected"]),
                          (rt, ["runtime_seconds", "peak_memory_gb", "peak_gpu_memory_gb", "gpu_used"]),
                          (ov, ["median_vsi", "mean_vsi", "fraction_low_vsi", "ovrlpy_status"]),
                          (rc, ["median_entropy", "mean_entropy", "median_max_weight",
                                "mixed_or_doublet_fraction", "rctd_status"]),
                          (cc, ["n_cells_after_filter", "n_cells_before_filter",
                                "n_partial_cells", "median_transcripts_per_cell",
                                "median_genes_per_cell"])]:
            for c in cols:
                r[c] = src.loc[m, c] if (m in src.index and c in src.columns) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


# ===========================================================================
# Panel helpers
# ===========================================================================
def _bar(ax, mm, col, title, ylabel, na_methods=()):
    vals = num(mm.set_index("method")[col]).reindex(METHOD_ORDER)
    x = np.arange(len(METHOD_ORDER))
    colors = [mcolor(m) for m in METHOD_ORDER]
    bars = ax.bar(x, vals.fillna(0).values, color=colors,
                  edgecolor=["black" if m == ACCENT else "none" for m in METHOD_ORDER],
                  linewidth=[1.4 if m == ACCENT else 0 for m in METHOD_ORDER])
    for i, m in enumerate(METHOD_ORDER):
        if pd.isna(vals.iloc[i]) or m in na_methods:
            ax.text(i, 0, " n/a", rotation=90, va="bottom", ha="center",
                    fontsize=6, color="#888888")
    ax.set_xticks(x); ax.set_xticklabels([LABEL[m] for m in METHOD_ORDER], rotation=40, ha="right")
    ax.set_title(title); ax.set_ylabel(ylabel)
    return bars


def class_legend(ax):
    handles = [Line2D([0], [0], marker=CLASS_MARKER[c], color="0.3", linestyle="none",
                      markersize=6, label=CLASS_LABEL[c]) for c in
               ["10X_multimodal_segmentation", "transcript_segmentation",
                "transcript_segmentation_gpu", "transcript_cleaning",
                "cell_level_cleaning", "transcript_refinement"]]
    ax.legend(handles=handles, loc="center", frameon=False, ncol=1,
              title="Method class", fontsize=6, title_fontsize=7)
    ax.axis("off")


# ===========================================================================
# FIGURE 1 — benchmark overview (A–F)
# ===========================================================================
def fig_overview(S, mm, saver):
    fig = plt.figure(figsize=(14, 8.4))
    gs = fig.add_gridspec(2, 6, hspace=0.6, wspace=0.9)

    # A. schematic tiles by class
    axA = fig.add_subplot(gs[0, 0:2]); axA.set_title("A  Methods & classes", loc="left")
    axA.set_xlim(0, 1); axA.set_ylim(0, 1); axA.axis("off")
    groups = [("10X multimodal segmentation", ["original"]),
              ("Transcript segmentation", ["Baysor", "proseg", "Segger"]),
              ("Transcript cleaning", ["cellAdmix"]),
              ("Cell-level cleaning", ["SPLIT"]),
              ("Transcript refinement", ["TRACER"])]
    y = 0.93
    for gname, members in groups:
        axA.text(0.0, y, gname, fontsize=7.5, fontweight="bold", va="top")
        y -= 0.085
        x = 0.04
        for m in members:
            w = 0.135 + 0.012 * len(LABEL[m])
            box = mpatches.FancyBboxPatch((x, y - 0.055), w, 0.07,
                    boxstyle="round,pad=0.006,rounding_size=0.02",
                    linewidth=(1.6 if m == ACCENT else 0.8),
                    edgecolor=("black" if m == ACCENT else "0.4"),
                    facecolor=mcolor(m), alpha=0.85)
            axA.add_patch(box)
            axA.text(x + w / 2, y - 0.02, LABEL[m], ha="center", va="center",
                     fontsize=7, color="white", fontweight="bold")
            axA.text(x + w + 0.01, y - 0.02, CLASS_MARKER[CLASS[m]], ha="left",
                     va="center", fontsize=8, color="0.3")
            x += w + 0.06
        y -= 0.105

    # B. runtime vs peak memory (point size ∝ n cells)
    axB = fig.add_subplot(gs[0, 2:4]); axB.set_title("B  Runtime vs peak memory", loc="left")
    sizes = num(mm["n_cells_after_filter"])
    smin, smax = np.nanmin(sizes), np.nanmax(sizes)
    missing = []
    for _, r in mm.iterrows():
        rt = r["runtime_seconds"]; yv = r["peak_memory_gb"]
        if pd.isna(rt) or pd.isna(yv):
            missing.append(LABEL[r["method"]])
            continue
        s = 40 + 360 * (num(pd.Series([r["n_cells_after_filter"]])).iloc[0] - smin) / max(smax - smin, 1)
        axB.scatter(np.log10(float(rt)), float(yv), s=float(np.nan_to_num(s, nan=60)),
                    color=mcolor(r["method"]), marker=mmarker(r["method"]),
                    edgecolor="black" if r["method"] == ACCENT else "white",
                    linewidth=1.3 if r["method"] == ACCENT else 0.6, alpha=0.9, zorder=3)
        axB.annotate(LABEL[r["method"]], (np.log10(float(rt)), float(yv)),
                     textcoords="offset points", xytext=(5, 4), fontsize=7,
                     fontweight="bold" if r["method"] == ACCENT else "normal")
    axB.set_xlabel("log10 runtime (s)"); axB.set_ylabel("peak memory (GB, CPU RSS)")
    if missing:
        axB.text(0.02, 0.02, f"no runtime/mem: {', '.join(missing)}", transform=axB.transAxes,
                 fontsize=6, color="#888888")
    axB.text(0.98, 0.02, "point size ∝ n cells", transform=axB.transAxes,
             ha="right", fontsize=6, color="#888888")

    # C. transcript assignment / cleanup (stacked)
    axC = fig.add_subplot(gs[0, 4:6]); axC.set_title("C  Transcript fate", loc="left")
    fate = transcript_fate_table(S)
    _stacked_fate(axC, fate, footnote=False)

    # D. cell counts after filtering (+ partial stacked) + median tx overlay
    axD = fig.add_subplot(gs[1, 0:2]); axD.set_title("D  Cells after filtering", loc="left")
    after = num(mm.set_index("method")["n_cells_after_filter"]).reindex(METHOD_ORDER)
    partial = num(mm.set_index("method")["n_partial_cells"]).reindex(METHOD_ORDER)
    x = np.arange(len(METHOD_ORDER))
    axD.bar(x, after.fillna(0).values, color=[mcolor(m) for m in METHOD_ORDER],
            edgecolor=["black" if m == ACCENT else "none" for m in METHOD_ORDER],
            linewidth=[1.4 if m == ACCENT else 0 for m in METHOD_ORDER], label="whole cells")
    if partial.notna().any():
        axD.bar(x, partial.fillna(0).values, bottom=after.fillna(0).values,
                color="0.75", edgecolor="0.4", linewidth=0.4, label="partial cells")
    axD.set_xticks(x); axD.set_xticklabels([LABEL[m] for m in METHOD_ORDER], rotation=40, ha="right")
    axD.set_ylabel("n cells")
    axD2 = axD.twinx(); axD2.spines["top"].set_visible(False)
    mt = num(mm.set_index("method")["median_transcripts_per_cell"]).reindex(METHOD_ORDER)
    axD2.plot(x, mt.values, "o", color="black", markersize=4, zorder=5)
    axD2.set_ylabel("median tx / cell", fontsize=7)
    axD.legend(frameon=False, fontsize=6, loc="upper right")

    # E. biological coherence lollipop
    axE = fig.add_subplot(gs[1, 2:4]); axE.set_title("E  Biological coherence", loc="left")
    _bio_lollipop(axE, mm)

    # F. orthogonal diagnostics — two clean side-by-side axes
    axF1 = fig.add_subplot(gs[1, 4]); axF2 = fig.add_subplot(gs[1, 5])
    _ortho(axF1, axF2, mm)

    fig.suptitle("TSU20 lung Xenium — cross-method benchmark overview",
                 fontsize=12, fontweight="bold", y=0.995)
    saver.fig(fig, "figure_benchmark_overview")
    saver.source(mm, "plot_method_metrics.tsv")


def _stacked_fate(ax, fate, footnote=True):
    x = np.arange(len(METHOD_ORDER))
    keep = num(fate.set_index("method")["assigned_or_retained"]).reindex(METHOD_ORDER).fillna(0)
    unas = num(fate.set_index("method")["unassigned"]).reindex(METHOD_ORDER).fillna(0)
    clean = num(fate.set_index("method")["removed_or_cleaned_or_pseudo"]).reindex(METHOD_ORDER).fillna(0)
    sc = 1e6
    ax.bar(x, keep / sc, color="#4daf4a", label="assigned / retained")
    ax.bar(x, unas / sc, bottom=keep / sc, color="#bdbdbd", label="unassigned")
    ax.bar(x, clean / sc, bottom=(keep + unas) / sc, color="#d95f02", label="removed / cleaned")
    # asterisk SPLIT (count-level pseudo)
    for i, m in enumerate(METHOD_ORDER):
        if m == "SPLIT":
            top = (keep + unas + clean).iloc[i] / sc
            ax.text(i, top, " *", fontsize=11, ha="center", va="bottom", color="#7570b3")
    ax.set_xticks(x); ax.set_xticklabels([LABEL[m] for m in METHOD_ORDER], rotation=40, ha="right")
    ax.set_ylabel("transcripts (millions)")
    ax.legend(frameon=False, fontsize=6, loc="upper right")
    if footnote:
        ax.text(0.0, -0.42, "* SPLIT: count-level pseudo-unassigned (not exact molecule removals)",
                transform=ax.transAxes, fontsize=6, color="#7570b3")


def _bio_lollipop(ax, mm):
    metrics = [("median_reference_pearson_r", "ref Pearson r ↑"),
               ("median_marker_log2fc", "marker log2FC ↑"),
               ("median_tcell_marker_log2fc", "T-cell marker log2FC ↑"),
               ("median_relative_purity", "relative purity ↑"),
               ("median_relative_conflict", "relative conflict ↓")]
    # normalize each metric 0..1 across methods for comparable lollipops
    base = mm.set_index("method")
    yt = []
    for j, (col, lab) in enumerate(metrics):
        v = num(base[col]).reindex(METHOD_ORDER)
        vmin, vmax = np.nanmin(v), np.nanmax(v)
        norm = (v - vmin) / (vmax - vmin) if vmax > vmin else v * 0
        yloc = len(metrics) - j
        yt.append((yloc, lab))
        for i, m in enumerate(METHOD_ORDER):
            if pd.isna(norm.iloc[i]):
                continue
            xx = i + norm.iloc[i] * 0.0  # not used; place by method column
        # plot per method as horizontal dots across method columns
        for i, m in enumerate(METHOD_ORDER):
            if pd.isna(v.iloc[i]):
                ax.text(i, yloc, "n/a", fontsize=5, color="#bbbbbb", ha="center", va="center")
                continue
            ax.scatter(i, yloc, s=30 + 90 * float(np.nan_to_num(norm.iloc[i])),
                       color=mcolor(m), marker=mmarker(m),
                       edgecolor="black" if m == ACCENT else "white",
                       linewidth=1.1 if m == ACCENT else 0.4, zorder=3)
    ax.set_yticks([y for y, _ in yt]); ax.set_yticklabels([l for _, l in yt], fontsize=6.5)
    ax.set_xticks(range(len(METHOD_ORDER)))
    ax.set_xticklabels([LABEL[m] for m in METHOD_ORDER], rotation=40, ha="right")
    ax.set_xlim(-0.6, len(METHOD_ORDER) - 0.4)
    ax.text(0.0, -0.32, "dot size ∝ rank within metric; n/a = metric unavailable",
            transform=ax.transAxes, ha="left", fontsize=6, color="#888888")


def _ortho(ax1, ax2, mm):
    base = mm.set_index("method")
    # ovrlpy mean VSI (median VSI is 0 for every method — uninformative)
    v = num(base["mean_vsi"]).reindex(METHOD_ORDER)
    x = np.arange(len(METHOD_ORDER))
    ax1.bar(x, v.fillna(0).values, color=[mcolor(m) for m in METHOD_ORDER])
    for i, m in enumerate(METHOD_ORDER):
        if pd.isna(v.iloc[i]):
            ax1.text(i, 0, " n/a", rotation=90, fontsize=6, color="#888888", ha="center", va="bottom")
    ax1.set_xticks(x); ax1.set_xticklabels([LABEL[m] for m in METHOD_ORDER], rotation=40, ha="right")
    ax1.set_title("F  ovrlpy mean VSI", fontsize=9, fontweight="bold", loc="left"); ax1.set_ylabel("mean VSI")
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
    # RCTD entropy + doublet fraction (twin)
    e = num(base["median_entropy"]).reindex(METHOD_ORDER)
    d = num(base["mixed_or_doublet_fraction"]).reindex(METHOD_ORDER)
    ax2.bar(x, e.fillna(0).values, color=[mcolor(m) for m in METHOD_ORDER])
    ax2.set_xticks(x); ax2.set_xticklabels([LABEL[m] for m in METHOD_ORDER], rotation=40, ha="right")
    ax2.set_title("RCTD entropy (↓) + doublet frac", fontsize=8); ax2.set_ylabel("median entropy")
    ax2.spines["top"].set_visible(False)
    axt = ax2.twinx(); axt.plot(x, d.values, "D", color="black", markersize=3.5)
    axt.set_ylabel("doublet frac", fontsize=7); axt.spines["top"].set_visible(False)


# ===========================================================================
# Transcript fate composition (per-method logic)
# ===========================================================================
def transcript_fate_table(S) -> pd.DataFrame:
    ta = S["transcript_assignment_summary"].set_index("method")
    rows = []
    for m in METHOD_ORDER:
        if m not in ta.index:
            continue
        r = ta.loc[m]
        cls = CLASS[m]
        total = num(pd.Series([r.get("total_transcripts")])).iloc[0]
        unas = num(pd.Series([r.get("unassigned_transcripts")])).iloc[0]
        assigned = num(pd.Series([r.get("assigned_transcripts")])).iloc[0]
        retained = num(pd.Series([r.get("retained_transcripts")])).iloc[0]
        removed = num(pd.Series([r.get("removed_or_cleaned_transcripts")])).iloc[0]
        pseudo = num(pd.Series([r.get("pseudo_unassigned_transcripts")])).iloc[0]
        note = ""
        if cls == "transcript_cleaning":   # cellAdmix
            keep = assigned if not pd.isna(assigned) else (total - removed if not pd.isna(total) else np.nan)
            rows.append(dict(method=m, assigned_or_retained=keep, unassigned=0,
                             removed_or_cleaned_or_pseudo=removed,
                             note="cleaned-to-unassigned on original Xenium cell IDs"))
        elif cls == "cell_level_cleaning":  # SPLIT
            rows.append(dict(method=m, assigned_or_retained=retained, unassigned=0,
                             removed_or_cleaned_or_pseudo=(pseudo if not pd.isna(pseudo) else removed),
                             note="count-level pseudo-unassigned (not exact molecule removals)"))
        else:  # segmentation / refinement / original
            asg = assigned if not pd.isna(assigned) else (total - unas if not pd.isna(total) and not pd.isna(unas) else np.nan)
            rows.append(dict(method=m, assigned_or_retained=asg, unassigned=unas,
                             removed_or_cleaned_or_pseudo=0, note="transcript-level assignment"))
    return pd.DataFrame(rows)


# ===========================================================================
# FIGURE 2 — biological metric heatmap
# ===========================================================================
def fig_heatmap(S, mm, saver):
    # (column, source_value, higher_is_better)
    specs = [
        ("ref Pearson r", "median_reference_pearson_r", True),
        ("marker log2FC", "median_marker_log2fc", True),
        ("T-cell marker log2FC", "median_tcell_marker_log2fc", True),
        ("relative purity", "median_relative_purity", True),
        ("low conflict", "median_relative_conflict", False),
        ("low marker leakage", "median_marker_leakage", False),
        ("ovrlpy mean VSI", "mean_vsi", True),
        ("low RCTD entropy", "median_entropy", False),
        ("low doublet frac", "mixed_or_doublet_fraction", False),
    ]
    base = mm.set_index("method")
    raw = pd.DataFrame({lab: num(base[col]).reindex(METHOD_ORDER) for lab, col, _ in specs})
    # orient (negate lower-is-better), then z-score per column
    Z = pd.DataFrame(index=raw.index, columns=raw.columns, dtype=float)
    for lab, col, hib in specs:
        v = raw[lab].astype(float).copy()
        if not hib:
            v = -v
        valid = v.dropna()
        if len(valid) >= 2 and valid.std(ddof=0) > 0:
            Z[lab] = (v - valid.mean()) / valid.std(ddof=0)
        else:
            Z[lab] = np.where(v.notna(), 0.0, np.nan)

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    arr = np.ma.masked_invalid(Z.values.astype(float))
    cmap = plt.cm.RdBu_r.copy(); cmap.set_bad("#eeeeee")
    vmax = np.nanmax(np.abs(arr)) if np.isfinite(arr).any() else 1
    im = ax.imshow(arr, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(Z.columns))); ax.set_xticklabels(Z.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(METHOD_ORDER))); ax.set_yticklabels([LABEL[m] for m in METHOD_ORDER])
    # annotate values / n/a
    for i, m in enumerate(METHOD_ORDER):
        for j, lab in enumerate(Z.columns):
            if pd.isna(Z.iloc[i, j]):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=6, color="#999999")
            else:
                rawv = raw.iloc[i, j]
                ax.text(j, i, f"{rawv:.2f}" if pd.notna(rawv) else "", ha="center", va="center",
                        fontsize=5.5, color="black")
    # class annotation strip (left)
    for i, m in enumerate(METHOD_ORDER):
        ax.add_patch(mpatches.Rectangle((-1.4, i - 0.5), 0.5, 1.0, color=mcolor(m),
                     clip_on=False, transform=ax.transData))
    ax.text(-1.15, -1.0, "class", fontsize=6, color="0.3")
    ax.set_xlim(-1.6, len(Z.columns) - 0.5)
    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("z-score (oriented so higher = better)", fontsize=7)
    ax.set_title("Biological & diagnostic metrics (z-scored, higher = better; grey = n/a)",
                 loc="left")
    fig.text(0.01, -0.02, "Annotations show raw values. SPLIT ovrlpy = n/a (cell-level; not imputed). "
             "marker leakage available for SPLIT only.", fontsize=6, color="#666666")
    saver.fig(fig, "figure_biological_metric_heatmap")
    src = Z.copy(); src.insert(0, "method", [LABEL[m] for m in METHOD_ORDER])
    saver.source(src, "plot_biological_heatmap.tsv")


# ===========================================================================
# FIGURE 3 — transcript fate
# ===========================================================================
def fig_transcript_fate(S, saver):
    fate = transcript_fate_table(S)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    _stacked_fate(ax, fate)
    ax.set_title("Transcript fate / cleanup by method", loc="left")
    saver.fig(fig, "figure_transcript_fate")
    saver.source(fate.assign(label=lambda d: d["method"].map(LABEL)), "plot_transcript_fate.tsv")


# ===========================================================================
# FIGURE 4 — runtime & memory
# ===========================================================================
def fig_runtime_memory(S, mm, saver):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    base = mm.set_index("method")
    # A runtime
    _bar(axes[0], mm, "runtime_seconds", "A  Runtime", "runtime (s)")
    axes[0].set_yscale("log")
    # B memory (CPU + GPU side by side)
    x = np.arange(len(METHOD_ORDER)); w = 0.38
    cpu = num(base["peak_memory_gb"]).reindex(METHOD_ORDER)
    gpu = num(base["peak_gpu_memory_gb"]).reindex(METHOD_ORDER)
    axes[1].bar(x - w/2, cpu.fillna(0).values, width=w, color=[mcolor(m) for m in METHOD_ORDER], label="CPU peak")
    axes[1].bar(x + w/2, gpu.fillna(0).values, width=w, color="0.6", hatch="//", label="GPU peak")
    for i, m in enumerate(METHOD_ORDER):
        if pd.isna(cpu.iloc[i]):
            axes[1].text(i - w/2, 0, " n/a", rotation=90, fontsize=6, color="#888888", ha="center", va="bottom")
    axes[1].set_xticks(x); axes[1].set_xticklabels([LABEL[m] for m in METHOD_ORDER], rotation=40, ha="right")
    axes[1].set_title("B  Peak memory", loc="left"); axes[1].set_ylabel("GB")
    axes[1].legend(frameon=False, fontsize=6)
    # C runtime vs bio
    axC = axes[2]
    for _, r in mm.iterrows():
        rt = r["runtime_seconds"]; yv = r["median_reference_pearson_r"]
        if pd.isna(rt) or pd.isna(yv):
            continue
        axC.scatter(np.log10(float(rt)), float(yv), s=70, color=mcolor(r["method"]),
                    marker=mmarker(r["method"]), edgecolor="black" if r["method"] == ACCENT else "white",
                    linewidth=1.2 if r["method"] == ACCENT else 0.5, zorder=3)
        axC.annotate(LABEL[r["method"]], (np.log10(float(rt)), float(yv)),
                     textcoords="offset points", xytext=(5, 3), fontsize=7)
    axC.set_xlabel("log10 runtime (s)"); axC.set_ylabel("median reference Pearson r")
    axC.set_title("C  Runtime vs reference r", loc="left")
    fig.text(0.01, -0.03, "10X baseline has no recorded runtime/memory (n/a). Segger ran on GPU "
             "(GPU peak shown; CPU peak not recorded). Missing values are not treated as zero.",
             fontsize=6, color="#666666")
    fig.tight_layout()
    saver.fig(fig, "figure_runtime_memory")
    saver.source(mm[["method", "label", "method_class", "runtime_seconds",
                     "peak_memory_gb", "peak_gpu_memory_gb", "gpu_used",
                     "median_reference_pearson_r", "n_cells_after_filter"]],
                 "plot_runtime_memory.tsv")


# ===========================================================================
# FIGURE 5 — RCTD diagnostics
# ===========================================================================
def fig_rctd(S, mm, saver):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    _bar(axes[0], mm, "median_entropy", "A  RCTD median entropy (↓ cleaner)", "median entropy")
    _bar(axes[1], mm, "mixed_or_doublet_fraction", "B  Mixed/doublet fraction (↓ cleaner)", "fraction")
    _bar(axes[2], mm, "median_max_weight", "C  RCTD median max weight (↑ confident)", "median max weight")
    fig.text(0.01, -0.03, "RCTD is an orthogonal mixture diagnostic, not ground truth. SPLIT values "
             "are from SPLIT's internal RCTD (standardized).", fontsize=6, color="#666666")
    fig.tight_layout()
    saver.fig(fig, "figure_rctd_diagnostics")
    saver.source(S["rctd_summary"], "plot_rctd_summary.tsv")


# ===========================================================================
# FIGURE 6 — ovrlpy VSI
# ===========================================================================
def fig_ovrlpy(S, mm, saver):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    order_no_split = [m for m in METHOD_ORDER if m != "SPLIT"]
    base = mm.set_index("method")
    for ax, col, title, ylab in [(axes[0], "mean_vsi", "A  ovrlpy mean VSI", "mean VSI"),
                                 (axes[1], "fraction_low_vsi", "B  Fraction low-VSI", "fraction low VSI")]:
        v = num(base[col]).reindex(order_no_split)
        x = np.arange(len(order_no_split))
        ax.bar(x, v.fillna(0).values, color=[mcolor(m) for m in order_no_split],
               edgecolor=["black" if m == ACCENT else "none" for m in order_no_split],
               linewidth=[1.4 if m == ACCENT else 0 for m in order_no_split])
        ax.set_xticks(x); ax.set_xticklabels([LABEL[m] for m in order_no_split], rotation=40, ha="right")
        ax.set_title(title, loc="left"); ax.set_ylabel(ylab)
    fig.text(0.01, -0.04, "Median VSI is 0 for every method (>97% low-VSI pixels), so mean VSI is shown. "
             "SPLIT excluded: ovrlpy n/a for cell-level purification (no transcript-level representation). "
             "ovrlpy is an orthogonal diagnostic, not an optimization target.",
             fontsize=6, color="#666666")
    fig.tight_layout()
    saver.fig(fig, "figure_ovrlpy_vsi")
    saver.source(S["ovrlpy_summary"], "plot_ovrlpy_summary.tsv")


# ===========================================================================
# FIGURE 7 — per-cell-type reference consistency (optional)
# ===========================================================================
def load_refconsist(metrics_root: Path) -> pd.DataFrame | None:
    rows = []
    for m in METHOD_ORDER:
        p = find_metric_file(m, "reference_consistency_by_celltype.tsv", metrics_root)
        if not p:
            continue
        t = pd.read_csv(p, sep="\t")
        if "pearson_r" in t.columns:
            r = t[["cell_type", "pearson_r"]].copy()
        elif "ref_corr_after" in t.columns:
            r = t[["cell_type", "ref_corr_after"]].rename(columns={"ref_corr_after": "pearson_r"})
        else:
            continue
        r["method"] = m
        rows.append(r)
    return pd.concat(rows, ignore_index=True) if rows else None


_RC_FOOT = ("Pseudobulk Pearson r (log-normalized, get_metric-comparable). SPLIT shown as "
            "purified cell-level profiles; B/Ciliated absent for SPLIT because its RCTD "
            "reference has no such category (not a coordinate bug). Higher = closer to reference.")


def fig_refconsist(metrics_root, saver):
    """Three styles: A grouped bars, B heatmap, C ranked dot plot (+radar supplement)."""
    df = load_refconsist(metrics_root)
    if df is None or df.empty:
        return None
    cts = sorted(df["cell_type"].unique())
    meth = [m for m in METHOD_ORDER if m in set(df["method"])]
    piv = df.pivot_table(index="cell_type", columns="method", values="pearson_r")

    # ---- A. grouped bar (cleaner) -----------------------------------------
    fig, ax = plt.subplots(figsize=(max(9, 1.1 * len(cts) + 2), 4.8))
    n = len(meth); w = 0.82 / n; x = np.arange(len(cts))
    for k, m in enumerate(meth):
        vals = piv.reindex(cts)[m].values if m in piv.columns else np.full(len(cts), np.nan)
        ax.bar(x + (k - n/2) * w + w/2, np.nan_to_num(vals), width=w, color=mcolor(m),
               edgecolor="black" if m == ACCENT else "none",
               linewidth=1.0 if m == ACCENT else 0, label=LABEL[m])
    # mark SPLIT n/a cell types
    if "SPLIT" in piv.columns:
        for j, ct in enumerate(cts):
            if pd.isna(piv.loc[ct, "SPLIT"]):
                ax.text(j, 0.01, "SPLIT n/a", rotation=90, fontsize=5.5,
                        color=COLOR["SPLIT"], ha="center", va="bottom")
    ax.set_xticks(x); ax.set_xticklabels(cts, rotation=35, ha="right")
    ax.set_ylabel("Pearson r to scRNA reference"); ax.set_ylim(0, 1)
    ax.set_title("Per-cell-type reference consistency (Cell_Cluster_level1)", loc="left")
    ax.legend(frameon=False, ncol=min(n, 7), fontsize=6, loc="lower right")
    fig.text(0.01, -0.04, _RC_FOOT, fontsize=6, color="#666666")
    fig.tight_layout()
    saver.fig(fig, "figure_reference_consistency_by_celltype")

    # ---- B. heatmap (methods rows x cell types cols) ----------------------
    H = piv.reindex(index=cts, columns=meth).T  # rows=methods, cols=cell types
    fig, ax = plt.subplots(figsize=(max(8, 1.0 * len(cts) + 2), 0.6 * len(meth) + 2))
    arr = np.ma.masked_invalid(H.values.astype(float))
    cmap = plt.cm.viridis.copy(); cmap.set_bad("#dddddd")
    im = ax.imshow(arr, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cts))); ax.set_xticklabels(cts, rotation=35, ha="right")
    ax.set_yticks(range(len(meth))); ax.set_yticklabels([LABEL[m] for m in meth])
    for i, m in enumerate(meth):
        for j, ct in enumerate(cts):
            v = H.iloc[i, j]
            ax.text(j, i, "n/a" if pd.isna(v) else f"{v:.2f}", ha="center", va="center",
                    fontsize=6, color=("#999999" if pd.isna(v) else
                                       ("white" if v < 0.55 else "black")))
        if m == ACCENT:  # highlight TRACER row
            ax.add_patch(mpatches.Rectangle((-0.5, i - 0.5), len(cts), 1, fill=False,
                         edgecolor=COLOR[ACCENT], lw=2, clip_on=False))
    fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02).set_label("Pearson r", fontsize=7)
    ax.set_title("Reference consistency heatmap (grey = n/a)", loc="left")
    fig.text(0.01, -0.02, _RC_FOOT, fontsize=6, color="#666666")
    fig.tight_layout()
    saver.fig(fig, "figure_reference_consistency_heatmap")

    # ---- C. ranked dot plot (median r per method, TRACER highlighted) -----
    med = piv.median(axis=0, skipna=True).reindex(meth).sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    yy = np.arange(len(med))
    for yi, m in zip(yy, med.index):
        ax.plot([0, med[m]], [yi, yi], color=mcolor(m), lw=1.2, alpha=0.5, zorder=1)
        ax.scatter(med[m], yi, s=160 if m == ACCENT else 90, color=mcolor(m),
                   marker=mmarker(m), edgecolor="black" if m == ACCENT else "white",
                   linewidth=1.5 if m == ACCENT else 0.6, zorder=3)
        ax.text(med[m] + 0.01, yi, f"{med[m]:.3f}", va="center", fontsize=7,
                fontweight="bold" if m == ACCENT else "normal")
    ax.set_yticks(yy); ax.set_yticklabels([LABEL[m] for m in med.index])
    ax.set_xlabel("median per-cell-type Pearson r"); ax.set_xlim(0, 1)
    ax.set_title("Reference consistency — method ranking", loc="left")
    fig.text(0.01, -0.04, "Median across cell types (B/Ciliated excluded for SPLIT — n/a). "
             "TRACER highlighted.", fontsize=6, color="#666666")
    fig.tight_layout()
    saver.fig(fig, "figure_reference_consistency_dotplot")

    # ---- radar supplement (SVG only) --------------------------------------
    try:
        _radar_refconsist(piv, cts, meth, saver)
    except Exception:
        pass

    out = df.copy(); out["label"] = out["method"].map(LABEL)
    saver.source(out, "plot_reference_consistency_by_celltype.tsv")
    return True


def _radar_refconsist(piv, cts, meth, saver):
    ang = np.linspace(0, 2 * np.pi, len(cts), endpoint=False)
    ang = np.concatenate([ang, ang[:1]])
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, polar=True)
    for m in meth:
        vals = piv.reindex(cts)[m].values.astype(float)
        vals = np.concatenate([vals, vals[:1]])
        ax.plot(ang, np.nan_to_num(vals), color=mcolor(m),
                lw=2.2 if m == ACCENT else 1.2, label=LABEL[m],
                zorder=3 if m == ACCENT else 2)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(cts, fontsize=7)
    ax.set_ylim(0, 1); ax.set_title("Reference consistency (radar, supplement)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), frameon=False, fontsize=6)
    # SVG only for supplement
    fig.savefig(saver.outdir / "figure_reference_consistency_radar_supplement.svg")
    plt.close(fig)
    saver.figures.append("figure_reference_consistency_radar_supplement")


# ===========================================================================
# FIGURE 8 — marker specificity dotplot (optional, per-gene get_metric methods)
# ===========================================================================
def load_markerspec(metrics_root: Path) -> pd.DataFrame | None:
    rows = []
    for m in METHOD_ORDER:
        p = find_metric_file(m, "marker_specificity_log2fc.tsv", metrics_root)
        if not p:
            continue
        t = pd.read_csv(p, sep="\t")
        if "spatial_log2fc" in t.columns:   # per-gene get_metric schema
            r = t[["cell_type", "gene", "spatial_log2fc"]].copy()
            r["method"] = m
            rows.append(r)
    return pd.concat(rows, ignore_index=True) if rows else None


def fig_markerspec(metrics_root, saver):
    df = load_markerspec(metrics_root)
    if df is None or df.empty:
        return None, "no per-gene marker_specificity files found"
    meth = [m for m in METHOD_ORDER if m in set(df["method"])]
    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(meth) + 2), 5))
    rng = np.random.default_rng(0)
    for i, m in enumerate(meth):
        v = pd.to_numeric(df.loc[df["method"] == m, "spatial_log2fc"], errors="coerce").dropna()
        jitter = rng.normal(0, 0.07, len(v))
        ax.scatter(np.full(len(v), i) + jitter, v, s=8, alpha=0.45, color=mcolor(m),
                   edgecolor="none")
        ax.plot([i - 0.25, i + 0.25], [v.median(), v.median()], color="black", lw=1.5, zorder=4)
        ax.text(i, ax.get_ylim()[1], f"n={len(v)}", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(range(len(meth))); ax.set_xticklabels([LABEL[m] for m in meth], rotation=40, ha="right")
    ax.set_ylabel("marker specificity log2FC (in-type vs out-type)")
    ax.set_title("Marker specificity per gene (black bar = median)", loc="left")
    ax.axhline(0, color="0.7", lw=0.6, ls="--")
    fig.text(0.01, -0.03, "TODO: paired one-sided Wilcoxon (TRACER vs each method) not yet implemented; "
             "plot shown without p-values. SPLIT uses per-cell-type specificity (different schema) and is "
             "not shown here.", fontsize=6, color="#666666")
    fig.tight_layout()
    saver.fig(fig, "figure_marker_specificity_dotplot")
    out = df.copy(); out["label"] = out["method"].map(LABEL)
    saver.source(out, "plot_marker_specificity.tsv")
    return True, "ok (no stats — TODO Wilcoxon)"


# ===========================================================================
# Report
# ===========================================================================
def write_report(outdir: Path, made: list[str], extras: dict, mm: pd.DataFrame):
    main_cands = ["figure_benchmark_overview", "figure_biological_metric_heatmap",
                  "figure_reference_consistency_heatmap",
                  "figure_reference_consistency_by_celltype", "figure_transcript_fate"]
    supp_cands = ["figure_runtime_memory", "figure_peak_memory", "figure_rctd_diagnostics",
                  "figure_ovrlpy_vsi", "figure_reference_consistency_dotplot",
                  "figure_reference_consistency_radar_supplement",
                  "figure_marker_specificity_dotplot"]
    lines = [
        "# Benchmark visualization summary — TSU20 lung Xenium (ref 36973297)", "",
        "Generated by `workflow/scripts/plot_benchmark_summary.py` from the standardized "
        "summary tables. Method order: 10X (original), Baysor, proseg, Segger, cellAdmix, "
        "SPLIT, TRACER (accent).", "",
        "## Figures generated", "",
    ]
    for f in made:
        tag = "MAIN" if f in main_cands else ("SUPP" if f in supp_cands else "")
        lines.append(f"- `{f}` {('['+tag+' candidate]') if tag else ''}")
    lines += ["", "## Source-data tables (figures/benchmark_summary/source_data/)", ""]
    for t in sorted((outdir / "source_data").glob("*.tsv")):
        lines.append(f"- `source_data/{t.name}`")
    lines += ["", "## Main-figure candidates", ""]
    lines += [f"{i+1}. {c}" for i, c in enumerate(main_cands) if c in made]
    lines += ["", "## Supplementary candidates", ""]
    lines += [f"{i+1}. {c}" for i, c in enumerate(supp_cands) if c in made]
    lines += ["", "## Missing / n/a metrics (not imputed)", "",
        "- **10X (original)**: no recorded runtime / peak memory (n/a in runtime figure).",
        "- **Segger**: CPU peak memory not recorded (HPC GPU run); GPU peak shown separately.",
        "- **SPLIT**: ovrlpy n/a (cell-level; no transcript-level representation) — shown as n/a, never 0.",
        "- **marker leakage**: available for SPLIT only (others n/a in heatmap).",
        "- **ovrlpy VSI**: median VSI = 0 for every method (>97% low-VSI pixels), so **mean VSI** is "
        "plotted/z-scored instead (the discriminating signal).",
        "- **ovrlpy problem score**: run_ovrlpy emits VSI only; no aggregate problem score available.",
        "- **T-cell marker log2FC**: not computed by get_metric for these runs (all n/a).",
        "- **mean entropy / mean max weight**: only SPLIT; transcript methods report medians.",
        f"- marker specificity stats: {extras.get('markerspec','n/a')}",
    ]
    lines += ["", "## Caveats preserved", "",
        "1. SPLIT is cell-level profile purification/deconvolution — NOT transcript-level "
        "segmentation. Its removed transcripts are count-level pseudo-unassigned estimates "
        "(marked with `*` and footnoted), not exact molecule-coordinate removals.",
        "2. cellAdmix is retained-transcript cleaning on ORIGINAL Xenium cell IDs, not de novo "
        "segmentation; its 'removed' bar is cleaned-to-unassigned.",
        "3. ovrlpy is not applicable to SPLIT and is an orthogonal diagnostic, not a target.",
        "4. RCTD is an orthogonal mixture diagnostic, not ground truth.",
        "5. Reference label column used: Cell_Cluster_level1 (9 coarse types).",
        "6. RCTD was completed/standardized for every method and is included.",
        "8. Per-cell-type reference consistency: SPLIT is computed with the SAME "
        "pseudobulk + log-normalization as get_metric.py (median r≈0.75, comparable to "
        "Baysor 0.71 / TRACER 0.74). An earlier per-cell-averaged version made SPLIT look "
        "artificially low — that was a methodology artifact, not poor purification (see "
        "metrics/SPLIT/split_missing_celltype_diagnosis.md). B and Ciliated are n/a for "
        "SPLIT because its RCTD reference vocabulary has no such category (real limitation, "
        "not a coordinate bug); the figures annotate this and never impute 0.",
        "7. SPLIT relative purity / conflict are NPMI-derived, computed by "
        "get_cell_level_metric.py on the PURIFIED (after) cell-by-gene matrix; "
        "purified fractional counts were rounded for the transcript-count step. "
        "Reported value is the 'after' (purified) median, comparable to other "
        "methods' post values.",
    ]
    lines += ["", "## Interpretation notes (no overclaiming)", "",
        "- Figures compare TRACER against de novo segmentation (Baysor/proseg/Segger), "
        "transcript cleaning (cellAdmix), and cell-level cleaning (SPLIT) — distinct method "
        "classes, labeled as such.",
        "- Heatmap z-scores are oriented so higher = better (conflict/leakage/entropy/doublet "
        "negated); grey = n/a, not zero.",
        "- Diagnostics (ovrlpy VSI, RCTD entropy/doublet) are orthogonal QC signals, not "
        "definitive rankings.",
    ]
    (outdir / "visualization_summary.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary-dir", required=True, type=Path)
    ap.add_argument("--metrics-root", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--format", choices=["png", "pdf", "svg", "both"], default="both")
    args = ap.parse_args()

    nature_style()
    S = load_summaries(args.summary_dir)
    mm = build_method_metrics(S)
    saver = Saver(args.outdir, args.format)

    fig_overview(S, mm, saver)
    fig_heatmap(S, mm, saver)
    fig_transcript_fate(S, saver)
    fig_runtime_memory(S, mm, saver)
    fig_rctd(S, mm, saver)
    fig_ovrlpy(S, mm, saver)
    extras = {}
    fig_refconsist(args.metrics_root, saver)
    _, extras["markerspec"] = fig_markerspec(args.metrics_root, saver) or (None, "skipped")

    write_report(args.outdir, saver.figures, extras, mm)
    print(f"Done. {len(saver.figures)} figures + source data in {args.outdir}")
    for f in saver.figures:
        print("  ", f)
    return 0


if __name__ == "__main__":
    sys.exit(main())

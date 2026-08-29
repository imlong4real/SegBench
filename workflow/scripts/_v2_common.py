#!/usr/bin/env python3
"""Shared configuration & helpers for the v2 (second-pass) benchmark figure revision.

Lung Xenium TSU-20 benchmarking study (ref lung_cancer_36973297).

Centralises: canonical method order & display names, the Nature-style Morandi
palette (with the TRACER magenta accents), filesystem paths, the 10<n<900
transcript filter, and small reusable IO / stats / matrix helpers so every
source-data builder and the plotting script agree byte-for-byte.
"""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(os.environ.get("SEGBENCH_ROOT",
                           Path(__file__).resolve().parents[2]))
BENCH = ROOT / "results/benchmark/lung_xenium_ref36973297"
METRICS = BENCH / "metrics"
SUMMARY = BENCH / "summary"
TRACER_HOME = ROOT / "results/TRACER_nsclc"

FIGDIR = BENCH / "figures/benchmark_summary_v2"
SRCDIR = FIGDIR / "source_data"

REFERENCE_H5AD = ROOT / "dataset/lung_cancer_scrna_10x/lung_cancer_50k.h5ad"
REF_CELLTYPE_COL = "Cell_Cluster_level1"
ORIG_TRANSCRIPTS = ROOT / "dataset/lung_cancer_xenium_10x/TSU-20/filtered_df_standardized.parquet"
NPMI_PANEL = TRACER_HOME / "reference_npmi/lung_cancer_36973297_npmi.csv.gz"

# Per-method transcript tables. Unassigned tx use a cell-id sentinel from
# UNASSIGNED_TOKENS (original uses "-1"; method outputs use "UNASSIGNED").
TRANSCRIPTS = {
    "original":  ORIG_TRANSCRIPTS,
    "Baysor":    ROOT / "results/benchmark_runs/tsu20/Baysor/outputs/baysor_transcripts_standardized.parquet",
    "proseg":    ROOT / "results/benchmark_runs/tsu20/proseg/outputs/proseg_transcripts_standardized.parquet",
    "segger":    ROOT / "results/segger_nsclc/segger_transcripts_standardized.parquet",
    "cellAdmix": ROOT / "results/benchmark_runs/tsu20/cellAdmix/outputs/celladmix_transcripts_standardized.parquet",
    "TRACER":    TRACER_HOME / "TRACER/outputs/transcripts_tracer_refined.parquet",
}
# Sentinel cell-ids that mark unassigned transcripts (mirrors get_metric.py).
UNASSIGNED_TOKENS = {"", "0", "-1", "NA", "nan", "None", "unassigned",
                     "UNASSIGNED", "background", "noise", "-1.0"}

# Per-method cell-level QC tables (per-cell n_transcripts, purity/conflict, ...).
# NOTE: cell_qc.tsv has NO cell_type column; labels live in POST_ANN.
CELL_QC = {
    "original":  TRACER_HOME / "metrics/original/cell_qc.tsv",
    "Baysor":    METRICS / "Baysor" / "cell_qc.tsv",
    "proseg":    METRICS / "proseg" / "cell_qc.tsv",
    "segger":    METRICS / "segger" / "cell_qc.tsv",
    "cellAdmix": METRICS / "cellAdmix" / "cell_qc.tsv",
    "TRACER":    TRACER_HOME / "metrics/TRACER/cell_qc.tsv",   # has is_partial / cell_category
}

# Per-method post-cleaning cell-type annotations (cell_id + cell_type label).
POST_ANN = {
    "original":  TRACER_HOME / "metrics/original/post_celltype_annotations.tsv",
    "Baysor":    METRICS / "Baysor" / "post_celltype_annotations.tsv",
    "proseg":    METRICS / "proseg" / "post_celltype_annotations.tsv",
    "segger":    METRICS / "segger" / "post_celltype_annotations.tsv",
    "cellAdmix": METRICS / "cellAdmix" / "post_celltype_annotations.tsv",
    "TRACER":    TRACER_HOME / "metrics/TRACER/post_celltype_annotations.tsv",
}

# Per-method per-cell RCTD entropy table (rctd_weights_entropy, weight_first_type,
# weight_second_type; SPLIT also has max_weight). TRACER split into whole/partial.
RCTD_ENTROPY = {
    "original":  TRACER_HOME / "rctd/original/rctd_entropy_metrics.tsv",
    "Baysor":    METRICS / "Baysor" / "rctd" / "rctd_entropy_metrics.tsv",
    "proseg":    METRICS / "proseg" / "rctd" / "rctd_entropy_metrics.tsv",
    "segger":    METRICS / "segger" / "rctd" / "rctd_entropy_metrics.tsv",
    "cellAdmix": METRICS / "cellAdmix" / "rctd" / "rctd_entropy_metrics.tsv",
    "SPLIT":     METRICS / "SPLIT" / "rctd_entropy_metrics.tsv",
    "TRACER-refined":       TRACER_HOME / "rctd/tracer_whole_cells/rctd_entropy_metrics.tsv",
    "TRACER-reconstructed": TRACER_HOME / "rctd/tracer_partial_cells/rctd_entropy_metrics.tsv",
}

# Per-method reference-consistency-by-celltype tables (pearson_r per cell type).
REF_CONSIST = {
    "original":  TRACER_HOME / "metrics/original/reference_consistency_by_celltype.tsv",
    "Baysor":    METRICS / "Baysor" / "reference_consistency_by_celltype.tsv",
    "proseg":    METRICS / "proseg" / "reference_consistency_by_celltype.tsv",
    "segger":    METRICS / "segger" / "reference_consistency_by_celltype.tsv",
    "cellAdmix": METRICS / "cellAdmix" / "reference_consistency_by_celltype.tsv",
    "SPLIT":     METRICS / "SPLIT" / "reference_consistency_by_celltype_v2.tsv",  # repaired (Section 1)
    "TRACER-refined": TRACER_HOME / "metrics/TRACER/reference_consistency_by_celltype.tsv",
    # TRACER-reconstructed (partial) is computed on demand (no precomputed table).
}

SPLIT_PURIFIED_H5AD = ROOT / "results/benchmark_runs/tsu20/SPLIT/outputs/split_cell_by_gene.h5ad"
SPLIT_REMOVED_COUNTS = ROOT / "results/benchmark_runs/tsu20/SPLIT/outputs/split_removed_counts_by_cell_gene.tsv.gz"

# --------------------------------------------------------------------------- #
# Method order, classes, display names
# --------------------------------------------------------------------------- #
# Canonical display order for every panel.
METHOD_ORDER = ["original", "Baysor", "proseg", "Segger", "cellAdmix", "SPLIT",
                "TRACER-refined", "TRACER-reconstructed"]

# Short class tags shown under method labels where useful.
METHOD_CLASS = {
    "original": "Xenium baseline",
    "Baysor": "de novo seg.",
    "proseg": "de novo seg.",
    "Segger": "de novo seg. (GPU)",
    "cellAdmix": "transcript cleaning",
    "SPLIT": "cell-level cleaning",
    "TRACER-refined": "whole-cell refinement",
    "TRACER-reconstructed": "partial-cell reconstruction",
}

# The 9 reference lineages (fixed column order for heatmaps / marker panels).
CELLTYPES9 = ["B", "Cancer", "Ciliated", "Endothelial", "Fibroblasts",
              "Mast", "Myeloid", "Plasma", "T"]

# Transcript-count filter (applied to per-cell metrics).
TX_MIN, TX_MAX = 10, 900

# --------------------------------------------------------------------------- #
# Palette  (clean white bg; Morandi-muted; TRACER magenta accents)
# --------------------------------------------------------------------------- #
PALETTE = {
    "original":            "#9aa0a6",  # neutral grey
    "Baysor":              "#6b8fb4",  # muted blue
    "proseg":              "#6fa69a",  # muted teal
    "Segger":              "#8f86b3",  # muted purple
    "cellAdmix":           "#cf9d6b",  # muted orange
    "SPLIT":               "#7f93a3",  # muted blue-grey
    "TRACER-refined":      "#c0306a",  # strong magenta/red accent
    "TRACER-reconstructed": "#e8a4c4", # lighter, related pink
}

# Per-lineage colours (Morandi muted, distinct) for marker-specificity dots.
CELLTYPE_COLORS = {
    "B":           "#b07aa1",
    "Cancer":      "#c1666b",
    "Ciliated":    "#7fa6a0",
    "Endothelial": "#6f8fb0",
    "Fibroblasts": "#c79a6a",
    "Mast":        "#a78b9b",
    "Myeloid":     "#cdae6f",
    "Plasma":      "#8e7cae",
    "T":           "#5f9e8f",
}


def morandi_cmap():
    """Muted, low-chroma sequential colormap for the reference-consistency heatmap."""
    from matplotlib.colors import LinearSegmentedColormap
    # pale warm stone -> muted sage -> muted slate-teal (low chroma throughout)
    return LinearSegmentedColormap.from_list(
        "morandi", ["#efe7dd", "#cdd2c4", "#9fb0a8", "#6f8a86", "#4f6a6b"])


def gpu_methods():
    return {"Segger"}


# --------------------------------------------------------------------------- #
# Matplotlib house style (thin axes, high-res typography, white bg)
# --------------------------------------------------------------------------- #
def apply_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "svg.fonttype": "none",          # keep text as text in SVG
        "pdf.fonttype": 42,              # editable text in PDF
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.0,
    })


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def ensure_dirs():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    SRCDIR.mkdir(parents=True, exist_ok=True)


def save_source(df: pd.DataFrame, name: str):
    """Write a source-data table under source_data/ (tab-separated)."""
    SRCDIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SRCDIR / name, sep="\t", index=False, float_format="%.6g")


def save_fig(fig, stem: str):
    """Save a figure as svg + pdf + png under the v2 figure dir."""
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(FIGDIR / f"{stem}.{ext}", dpi=400, bbox_inches="tight")


def display_name(raw: str) -> str:
    """Map a raw method tag (incl. 'segger', TRACER variants) to display name."""
    m = {"segger": "Segger", "Segger": "Segger"}
    return m.get(raw, raw)


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #
def wilcoxon_one_sided(a: np.ndarray, b: np.ndarray, alternative: str):
    """Paired one-sided Wilcoxon signed-rank on overlapping finite pairs.

    `alternative` is passed through to scipy ('greater'/'less'): tests whether
    a-b is stochastically greater/less than 0. Returns (stat, p, n_pairs).
    Pairing is by position; callers must align a and b on a common key first.
    """
    from scipy.stats import wilcoxon
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    n = int(a.size)
    if n < 8 or np.allclose(a, b):
        return (np.nan, np.nan, n)
    try:
        stat, p = wilcoxon(a, b, alternative=alternative, zero_method="wilcox")
    except ValueError:
        return (np.nan, np.nan, n)
    return (float(stat), float(p), n)


def p_to_stars(p: float) -> str:
    """Compact significance label, capped at p<0.001 to avoid clutter."""
    if p is None or not np.isfinite(p):
        return "ns"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def p_label(p: float) -> str:
    """Numeric p label, capped at '<0.001'."""
    if p is None or not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return "p<0.001"
    return f"p={p:.3f}"


# --------------------------------------------------------------------------- #
# Matrix builders
# --------------------------------------------------------------------------- #
def build_cell_by_gene(transcripts_path, label_col="cell_id", feature_col="feature_name",
                       drop_unassigned=True, cell_subset=None):
    """Build a raw-count cell x gene DataFrame from a long transcript table.

    Returns (counts_df [cells x genes], genes list). `cell_subset` (iterable of
    cell ids) restricts to those cells. Control/blank probes are kept as-is; the
    caller restricts to the panel∩reference gene set downstream.
    """
    df = pd.read_parquet(transcripts_path, columns=[label_col, feature_col])
    df[label_col] = df[label_col].astype(str)
    if drop_unassigned:
        df = df[~df[label_col].isin(UNASSIGNED_TOKENS)]
    if cell_subset is not None:
        df = df[df[label_col].isin(set(map(str, cell_subset)))]
    ct = pd.crosstab(df[label_col], df[feature_col])
    return ct


def lognorm_pseudobulk(counts: np.ndarray) -> np.ndarray:
    """CP10k -> log1p per row, then mean across rows (pseudobulk profile)."""
    M = np.asarray(counts, dtype=np.float64)
    s = M.sum(axis=1, keepdims=True); s[s == 0] = 1.0
    return np.log1p(M * 1e4 / s).mean(axis=0)


if __name__ == "__main__":
    # Tiny self-check (writes a 1-line receipt that survives the output channel).
    ensure_dirs()
    (FIGDIR / "_v2_common_ok.txt").write_text(
        f"methods={len(METHOD_ORDER)} celltypes={len(CELLTYPES9)} "
        f"palette={len(PALETTE)} figdir_ok={FIGDIR.exists()}")
    print("ok")

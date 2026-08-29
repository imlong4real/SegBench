#!/usr/bin/env python3
"""Stage 3: consolidate Block A/B/C into source tables and render the three
publication split-heatmap figures (Nature-Methods style).

Inputs:
  summary_heatmaps/block_ab_long.tsv            (from stage 1)
  summary_heatmaps/metrics/<ds>/<ent>/rctd/rctd_entropy_metrics.tsv  (stage 2)
  <std>/SPLIT/split_rctd_entropy_metrics.tsv    (SPLIT internal RCTD, per cell)

Outputs:
  source_data_blockA.tsv / blockB.tsv / blockC.tsv
  benchmark_heatmap_blockA_compute.{png,svg}
  benchmark_heatmap_blockB_biological.{png,svg}
  benchmark_heatmap_blockC_rctd.{png,svg}
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import registry as R  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib import font_manager      # noqa: E402

# ---------------------------------------------------------------------------
# Nature-Methods-ish style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "axes.linewidth": 0.6,
})

# Colormaps: perceptually-uniform, colourblind-safe.
CMAP_GOOD = "mako_r" if "mako_r" in plt.colormaps() else "viridis"
CMAP_NEUTRAL = "cividis"
try:
    import seaborn as sns  # noqa
    CMAP_GOOD = "mako_r"
except Exception:
    CMAP_GOOD = "viridis"

NA_COLOR = "#e8e8e8"
GRID_COLOR = "white"


# ---------------------------------------------------------------------------
# Block C loader
# ---------------------------------------------------------------------------
def load_block_c():
    """Block C uses QC-filtered (10<=tx<=900) RCTD: `metrics_qc/` for non-SPLIT
    entities; for SPLIT the internal per-cell RCTD table is filtered to the QC
    cell set."""
    import anndata as ad
    rows = []
    metrics_qc = R.OUT / "metrics_qc"
    for ds in R.DATASET_ORDER:
        for ent in R.ENTITY_ORDER:
            if ent == "split":
                f = R.std_dir(ds, "SPLIT") / "split_rctd_entropy_metrics.tsv"
                qc_h5 = R.OUT / "_work_qc" / ds / "split.h5ad"
                if f.exists():
                    t = pd.read_csv(f, sep="\t")
                    if qc_h5.exists():  # restrict to QC cell set
                        keep = set(ad.read_h5ad(qc_h5).obs_names.astype(str))
                        t = t[t["cell_id"].astype(str).isin(keep)]
                    ent_med = float(pd.to_numeric(t["entropy"], errors="coerce").median())
                    mw_med = float(pd.to_numeric(t["max_weight"], errors="coerce").median())
                    rows += [dict(dataset=ds, entity=ent, metric="rctd_entropy", value=ent_med,
                                  note="SPLIT internal RCTD, QC-filtered cells"),
                             dict(dataset=ds, entity=ent, metric="rctd_max_weight", value=mw_med,
                                  note="SPLIT internal RCTD, QC-filtered cells")]
                continue
            f = metrics_qc / ds / ent / "rctd" / "rctd_entropy_metrics.tsv"
            if f.exists():
                t = pd.read_csv(f, sep="\t")
                post = t[t["tag"] == "post"] if "tag" in t.columns else t
                if len(post):
                    rows += [dict(dataset=ds, entity=ent, metric="rctd_entropy",
                                  value=float(post["median_entropy"].iloc[0]), note="QC-filtered"),
                             dict(dataset=ds, entity=ent, metric="rctd_max_weight",
                                  value=float(post["median_max_weight"].iloc[0]), note="QC-filtered")]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Build wide source table for one block
# ---------------------------------------------------------------------------
def build_block_table(long_df, metrics):
    """rows = (dataset, metric); columns = stable entity KEYS (ordered).

    The 'original' column always holds the per-dataset native segmentation;
    its concrete tool name is recorded in 'baseline_method' (10x / CosMx SMI /
    Baysor) so the table is unambiguous and never collides with a standalone
    method column.
    """
    col_keys = R.ENTITY_ORDER
    records = []
    for metric in metrics:
        for ds in R.DATASET_ORDER:
            row = {"dataset": ds, "platform": R.DATASETS[ds]["platform"], "metric": metric,
                   "metric_label": R.METRICS_SPEC[metric]["label"],
                   "direction": R.METRICS_SPEC[metric]["direction"],
                   "baseline_method": R.DATASETS[ds]["original_label"]}
            for ent in col_keys:
                if not R.applicable(ent, metric, ds):
                    row[ent] = np.nan
                    continue
                sub = long_df[(long_df.dataset == ds) & (long_df.entity == ent)
                              & (long_df.metric == metric)]
                row[ent] = float(sub["value"].iloc[0]) if len(sub) and pd.notna(sub["value"].iloc[0]) else np.nan
            records.append(row)
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Formatting of raw annotations
# ---------------------------------------------------------------------------
def fmt(metric, v):
    if pd.isna(v):
        return "NA"
    if metric == "total_cells":
        return f"{int(round(v)):,}"
    if metric == "transcripts_per_cell":
        return f"{v:,.0f}"
    if metric == "runtime_seconds":
        return f"{v:,.0f}"
    if metric == "peak_memory_gb":
        return f"{v:.2f}"
    return f"{v:.2f}"


# ---------------------------------------------------------------------------
# Plot one block (one stacked figure, one sub-heatmap per metric)
# ---------------------------------------------------------------------------
def display_columns(dataset_independent=True):
    # Use a single consistent display-label set. 'original' label varies per
    # dataset, so we keep the entity-key columns and relabel per-row in cells.
    return R.ENTITY_ORDER


def plot_block(block_letter, metrics, table, out_base, title):
    col_keys = R.ENTITY_ORDER
    ncols = len(col_keys)
    n_ds = len(R.DATASET_ORDER)
    nrows_panels = len(metrics)

    # Column header labels (entity display; 'original' shown generically with
    # per-dataset baseline noted in row labels).
    col_labels = []
    for ent in col_keys:
        if ent == "original":
            col_labels.append("Original\n(baseline)")
        else:
            col_labels.append(R.ENTITY_LABELS[ent])

    cell_h = 0.42
    panel_gap = 0.55
    fig_h = nrows_panels * (n_ds * cell_h + panel_gap) + 2.1
    fig_w = max(9.5, ncols * 1.05 + 3.2)
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(nrows_panels, 1, hspace=0.9,
                          left=0.20, right=0.88, top=1 - 0.95 / fig_h, bottom=1.55 / fig_h)

    for pi, metric in enumerate(metrics):
        ax = fig.add_subplot(gs[pi, 0])
        sub = table[table.metric == metric].set_index("dataset").loc[R.DATASET_ORDER]
        direction = R.METRICS_SPEC[metric]["direction"]

        # Build raw value matrix (rows=datasets, cols=entities) from entity-key cols
        raw = np.full((n_ds, ncols), np.nan)
        for di, ds in enumerate(R.DATASET_ORDER):
            r = sub.loc[ds]
            for ci, ent in enumerate(col_keys):
                raw[di, ci] = r[ent] if ent in r.index else np.nan

        # Normalize within each dataset-row across methods -> goodness g in [0,1]
        g = np.full_like(raw, np.nan)
        for di in range(n_ds):
            vals = raw[di]
            finite = np.isfinite(vals)
            if finite.sum() >= 1:
                vmin, vmax = np.nanmin(vals[finite]), np.nanmax(vals[finite])
                if vmax > vmin:
                    norm = (vals - vmin) / (vmax - vmin)
                else:
                    norm = np.where(finite, 0.5, np.nan)
                if direction == "lower":
                    norm = 1.0 - norm
                g[di] = norm

        cmap = plt.get_cmap(CMAP_NEUTRAL if direction == "neutral" else CMAP_GOOD)
        cmap = cmap.copy(); cmap.set_bad(NA_COLOR)

        # Draw
        masked = np.ma.masked_invalid(g)
        im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0, vmax=1)
        # NA cells: paint grey explicitly (imshow masked already uses set_bad)

        # gridlines
        ax.set_xticks(np.arange(-.5, ncols, 1), minor=True)
        ax.set_yticks(np.arange(-.5, n_ds, 1), minor=True)
        ax.grid(which="minor", color=GRID_COLOR, linewidth=1.4)
        ax.tick_params(which="minor", length=0)

        # annotations (text colour chosen from the actual cell luminance so it
        # stays legible on any colormap, light or dark)
        for di in range(n_ds):
            for ci in range(ncols):
                v = raw[di, ci]
                txt = fmt(metric, v)
                if pd.isna(v):
                    ax.text(ci, di, "NA", ha="center", va="center", fontsize=6.5,
                            color="#9a9a9a", style="italic")
                    continue
                gg = g[di, ci]
                r, gr, b, _ = cmap(gg if np.isfinite(gg) else 0.5)
                lum = 0.2126 * r + 0.7152 * gr + 0.0722 * b
                tc = "white" if lum < 0.55 else "#15161a"
                ax.text(ci, di, txt, ha="center", va="center", fontsize=6.9, color=tc)

        # axes labels
        ax.set_yticks(range(n_ds))
        ax.set_yticklabels([R.DATASETS[ds]["platform"] for ds in R.DATASET_ORDER], fontsize=8)
        ax.set_xticks(range(ncols))
        if pi == nrows_panels - 1:
            ax.set_xticklabels(col_labels, fontsize=7.5, rotation=40, ha="right")
        else:
            ax.set_xticklabels([])
        for sp in ax.spines.values():
            sp.set_visible(False)

        dlabel = {"higher": "higher = better", "lower": "lower = better",
                  "neutral": "descriptive"}[direction]
        ax.set_title(f"{R.METRICS_SPEC[metric]['label']}   ·   {dlabel}",
                     fontsize=9, loc="left", pad=4, fontweight="bold")

        # per-panel colorbar (normalized goodness)
        cax = ax.inset_axes([1.012, 0.0, 0.018, 1.0])
        cb = fig.colorbar(im, cax=cax)
        cb.set_ticks([0, 1])
        cb.set_ticklabels(["worse", "better"] if direction != "neutral" else ["low", "high"],
                          fontsize=6)
        cb.outline.set_linewidth(0.4)

    fig.suptitle(title, x=0.20, y=1 - 0.18 / fig_h, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.20, 0.62 / fig_h,
             "Colour = within-metric, within-dataset min–max normalization across methods; "
             "for ‘lower = better’ metrics the scale is reversed so darker is always better. "
             "Descriptive metrics use a neutral (cividis) scale. Cells annotated with raw values.",
             fontsize=6.5, color="#444444")
    fig.text(0.20, 0.24 / fig_h,
             "Original (baseline): Xenium/Xenium5K = 10x · CosMx = CosMx SMI · MERFISH = Baysor.   "
             "* Segger is GPU-based (runtime/peak memory are GPU; H100).   "
             "NA = metric not applicable to that method (see benchmark_heatmap_summary.md).",
             fontsize=6.5, color="#444444")
    for ext in ("png", "svg"):
        fig.savefig(f"{out_base}.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_base + ".png/.svg")


# ---------------------------------------------------------------------------
def main():
    long_df = pd.read_csv(R.OUT / "block_ab_long.tsv", sep="\t")
    block_c = load_block_c()
    long_all = pd.concat([long_df, block_c], ignore_index=True)
    long_all.to_csv(R.OUT / "all_metrics_long.tsv", sep="\t", index=False)

    blocks = {
        "A": (["total_cells", "transcripts_per_cell", "runtime_seconds", "peak_memory_gb"],
              "benchmark_heatmap_blockA_compute",
              "Block A — Cell/profile size & compute"),
        "B": (["marker_log2fc", "relative_purity", "relative_conflict", "kendall_tau"],
              "benchmark_heatmap_blockB_biological",
              "Block B — Biological coherence"),
        "C": (["rctd_entropy", "rctd_max_weight"],
              "benchmark_heatmap_blockC_rctd",
              "Block C — RCTD deconvolution purity"),
    }
    for letter, (metrics, base, title) in blocks.items():
        table = build_block_table(long_all, metrics)
        src = R.OUT / f"source_data_block{letter}.tsv"
        table.to_csv(src, sep="\t", index=False)
        print("wrote", src)
        plot_block(letter, metrics, table, str(R.OUT / base), title)


if __name__ == "__main__":
    main()

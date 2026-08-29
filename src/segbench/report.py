#!/usr/bin/env python3
"""Publication-quality comparison figures from the unified evaluation table.

One figure per metric family, all sharing a method colour map so a reader can
track a method across panels. Everything is driven off the table produced by
:mod:`segbench.evaluate`, so a new method appears in the plots automatically.

Design choices that matter for honesty of the figure:

  * Methods with a NaN for a metric are drawn as an explicit "n/a" tick rather
    than dropped, so a missing bar cannot be misread as a zero.
  * ``entity_kind`` is annotated on the entity-count panel, because a bin count
    and a cell count are not the same quantity and must not read as one.
  * Runtime uses the method-only seconds; the axis label says so.
  * Peak-RSS bars are hatched when the figure came from in-process sampling
    rather than /usr/bin/time, since those are not directly comparable.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

#: Colour-blind-safe qualitative palette (Okabe-Ito), one colour per method.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7",
           "#56B4E9", "#D55E00", "#F0E442", "#999999"]

METRIC_PANELS = [
    ("runtime_method_s",              "Runtime (method only, s)",      True),
    ("peak_rss_gb",                   "Peak RSS (GB)",                 False),
    ("n_entities",                    "Entities produced",             False),
    ("mean_transcripts_per_profile",  "Mean transcripts / profile",    False),
    ("frac_assigned",                 "Fraction of transcripts assigned", False),
    ("rctd_entropy_median",           "RCTD entropy (median)",         False),
    ("rctd_max_weight_median",        "RCTD max weight (median)",      False),
    ("kendall_tau_median",            "Kendall tau vs scRNA (median)",  False),
    ("marker_logfc_median",           "Marker specificity log2FC",     False),
    ("cpmi_conflict",                 "cPMI conflict (median)",        False),
]


def _method_colors(methods: list[str]) -> dict[str, str]:
    return {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(sorted(methods))}


def comparison_figure(df: pd.DataFrame, out_path: Path, *,
                      title: str = "SegBench method comparison") -> Path:
    """Grid of one bar panel per metric, methods on the x axis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [(c, lab, logy) for c, lab, logy in METRIC_PANELS if c in df.columns
              and df[c].notna().any()]
    if not panels:
        raise SystemExit("No plottable metrics in the evaluation table.")

    methods = df["method"].astype(str).tolist()
    colors = _method_colors(methods)
    ncol = 3
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.5 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for ax, (col, label, logy) in zip(axes, panels):
        vals = pd.to_numeric(df[col], errors="coerce")
        x = np.arange(len(df))
        bars = ax.bar(x, vals.fillna(0).to_numpy(),
                      color=[colors[m] for m in methods],
                      edgecolor="black", linewidth=0.6)
        # Hatch peak-RSS bars whose figure is an in-process estimate.
        if col == "peak_rss_gb" and "peak_rss_source" in df.columns:
            for b, src in zip(bars, df["peak_rss_source"].astype(str)):
                if src != "external_time":
                    b.set_hatch("///")
        # A missing value must read as "n/a", never as zero.
        for xi, v in zip(x, vals):
            if pd.isna(v):
                ax.text(xi, 0, "n/a", ha="center", va="bottom",
                        fontsize=8, style="italic", color="#666666")
        if logy and vals.max() and vals.max() / max(vals[vals > 0].min(), 1e-9) > 50:
            ax.set_yscale("log")
        ax.set_title(label, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        if col == "n_entities" and "entity_kind" in df.columns:
            for xi, (v, k) in enumerate(zip(vals, df["entity_kind"].astype(str))):
                if pd.notna(v):
                    ax.text(xi, v, k, ha="center", va="bottom", fontsize=7,
                            color="#444444")

    for ax in axes[len(panels):]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=14, y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_path.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path.with_suffix(".png")


def runtime_memory_scatter(df: pd.DataFrame, out_path: Path) -> Path | None:
    """Runtime vs peak memory — the practical cost of running each method."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    need = {"runtime_method_s", "peak_rss_gb"}
    if not need <= set(df.columns):
        return None
    sub = df.dropna(subset=list(need))
    if sub.empty:
        return None
    colors = _method_colors(df["method"].astype(str).tolist())
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for _, r in sub.iterrows():
        ax.scatter(r["runtime_method_s"] / 60.0, r["peak_rss_gb"], s=130,
                   color=colors[str(r["method"])], edgecolor="black",
                   linewidth=0.7, zorder=3)
        ax.annotate(str(r["method"]),
                    (r["runtime_method_s"] / 60.0, r["peak_rss_gb"]),
                    textcoords="offset points", xytext=(7, 5), fontsize=9)
    ax.set_xlabel("Method runtime (min)")
    ax.set_ylabel("Peak RSS (GB)")
    ax.set_title("Computational cost by method")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.5)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_path.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path.with_suffix(".png")


def _fmt(v) -> str:
    """Render one cell: NaN as an explicit n/a, floats at 4 significant digits."""
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "n/a"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        return f"{v:.4g}"
    return str(v)


def _markdown_table(df: pd.DataFrame) -> str:
    """Small GitHub-flavoured table writer.

    Written out rather than using ``DataFrame.to_markdown`` so the report does
    not need the optional `tabulate` dependency in every method environment.
    """
    cols = list(df.columns)
    rows = [[_fmt(v) for v in rec] for rec in df.itertuples(index=False, name=None)]
    widths = [max(len(str(c)), *(len(r[i]) for r in rows)) if rows else len(str(c))
              for i, c in enumerate(cols)]
    head = "| " + " | ".join(str(c).ljust(w) for c, w in zip(cols, widths)) + " |"
    rule = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    body = ["| " + " | ".join(cell.ljust(w) for cell, w in zip(r, widths)) + " |"
            for r in rows]
    return "\n".join([head, rule, *body])


def write_markdown_summary(df: pd.DataFrame, out_path: Path, *,
                           dataset: str, excluded_celltypes: list[str] | None = None,
                           min_reference_cells: int | None = None) -> Path:
    """Human-readable summary, including what is NOT comparable and why."""
    lines = [f"# SegBench comparison — {dataset}", ""]
    show = [c for c in ("method", "entity_kind", "runtime_method_s", "peak_rss_gb",
                        "n_entities", "mean_transcripts_per_profile", "frac_assigned",
                        "rctd_entropy_median", "rctd_max_weight_median",
                        "kendall_tau_median", "marker_logfc_median",
                        "cpmi_purity", "cpmi_conflict") if c in df.columns]
    lines += [_markdown_table(df[show]), ""]

    if min_reference_cells is not None:
        lines += [f"Reference cell types with < {min_reference_cells} cells were "
                  f"excluded from RCTD and the marker/Kendall metrics so that "
                  f"rare populations do not dominate the medians.", ""]
    if excluded_celltypes:
        lines += [f"Excluded cell types: {', '.join(excluded_celltypes)}", ""]

    notes = [c for c in df.columns if c.endswith("_note")]
    if notes:
        lines += ["## Non-comparable quantities", "",
                  "Values below are reported as `n/a` rather than coerced into a "
                  "common scale.", ""]
        for col in notes:
            base = col[:-5]
            for _, r in df.iterrows():
                if isinstance(r.get(col), str) and r[col]:
                    lines.append(f"- **{r['method']} / {base}** — {r[col]}")
        lines.append("")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    return out_path

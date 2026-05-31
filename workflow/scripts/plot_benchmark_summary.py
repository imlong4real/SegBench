#!/usr/bin/env python3
"""Second-pass manuscript panels for the TSU20 lung Xenium benchmark.

This v2 renderer consumes source-data tables prepared by the companion
``_v2_*`` scripts and writes individual publication-style SVG/PDF/PNG panels.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize

import _v2_common as C

COMBINED_METHOD_ORDER = ["original", "Baysor", "proseg", "Segger", "cellAdmix",
                         "SPLIT", "TRACER"]
TRACER_WHOLE = "TRACER-refined"
TRACER_PARTIAL = "TRACER-reconstructed"


def read_source(name: str) -> pd.DataFrame:
    return pd.read_csv(C.SRCDIR / name, sep="\t")


def method_colors(methods):
    return [C.PALETTE.get(str(m), "#9aa0a6") for m in methods]


def ordered(df: pd.DataFrame, col: str = "method") -> pd.DataFrame:
    d = df.copy()
    d[col] = pd.Categorical(d[col], C.METHOD_ORDER, ordered=True)
    return d.sort_values(col)


def nice_count(x) -> str:
    if pd.isna(x):
        return "n/a"
    x = float(x)
    if abs(x) >= 1e6:
        return f"{x/1e6:.1f}M"
    if abs(x) >= 1e3:
        return f"{x/1e3:.0f}k"
    return f"{x:.0f}"


def nice_time(minutes) -> str:
    if pd.isna(minutes):
        return "n/a"
    minutes = float(minutes)
    if minutes >= 60:
        return f"{minutes/60:.1f} h"
    return f"{minutes:.1f} min"


def annotate_bars(ax, bars, labels, pad=0.02, fontsize=6, rotation=90):
    ymax = ax.get_ylim()[1]
    for bar, label in zip(bars, labels):
        if label == "n/a":
            continue
        h = bar.get_height()
        if not np.isfinite(h):
            continue
        ax.text(bar.get_x() + bar.get_width() / 2, h + ymax * pad, label,
                ha="center", va="bottom", fontsize=fontsize, rotation=rotation)


def finish_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e6e1dc", lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def figure_runtime_memory():
    df = ordered(read_source("runtime_memory_v2.tsv"))
    methods = df["method"].astype(str).tolist()
    x = np.arange(len(methods))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 2.9), gridspec_kw={"wspace": 0.28})

    runtime = pd.to_numeric(df["runtime_minutes"], errors="coerce")
    bars = ax1.bar(x, runtime.fillna(0), width=0.62, color=method_colors(methods), edgecolor="white")
    ax1.set_ylabel("Runtime (min)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=35, ha="right")
    annotate_bars(ax1, bars, [nice_time(v) for v in runtime], rotation=90)
    for i, v in enumerate(runtime):
        if pd.isna(v):
            ax1.text(i, 0.03, "n/a", ha="center", va="bottom", rotation=90, fontsize=6, color="#777")
    ax1.text(0.98, 0.06, "TRACER rows share one run", transform=ax1.transAxes,
             ha="right", va="bottom", fontsize=6.3, color="#555")
    ax1.set_title("Runtime", loc="left", fontweight="bold")
    finish_axis(ax1)

    cpu = pd.to_numeric(df["peak_cpu_memory_gb"], errors="coerce")
    gpu = pd.to_numeric(df["peak_gpu_memory_gb"], errors="coerce")
    w = 0.33
    b1 = ax2.bar(x - w / 2, cpu.fillna(0), width=w, color="#aab0a8", label="CPU RSS")
    b2 = ax2.bar(x + w / 2, gpu.fillna(0), width=w, color="#8f86b3", label="GPU memory")
    for i, (c, g) in enumerate(zip(cpu, gpu)):
        if pd.notna(c):
            ax2.text(i - w/2, c + ax2.get_ylim()[1] * 0.02, f"{c:.1f}", ha="center", va="bottom", fontsize=6, rotation=90)
        if pd.notna(g):
            ax2.text(i + w/2, g + ax2.get_ylim()[1] * 0.02, f"{g:.1f}", ha="center", va="bottom", fontsize=6, rotation=90)
        if bool(df.iloc[i].get("gpu_based")):
            ax2.text(i, ax2.get_ylim()[1] * 0.92, "GPU", ha="center", va="top", fontsize=6.5, color="#5a4d84")
    ax2.set_ylabel("Peak memory (GB)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=35, ha="right")
    ax2.set_title("Peak Memory", loc="left", fontweight="bold")
    ax2.legend(frameon=False, fontsize=6, loc="upper right")
    finish_axis(ax2)
    C.save_fig(fig, "figure_runtime_memory_v2")
    plt.close(fig)


def figure_runtime_memory_combined_v3():
    df = read_source("runtime_memory_v2.tsv")
    df = df[~df["method"].astype(str).eq("original")].copy()
    df["method"] = pd.Categorical(df["method"], C.METHOD_ORDER, ordered=True)
    df = df.sort_values("method")
    C.save_source(df, "runtime_memory_combined_v3.tsv")

    methods = df["method"].astype(str).tolist()
    x = np.arange(len(methods))
    runtime = pd.to_numeric(df["runtime_minutes"], errors="coerce")
    cpu = pd.to_numeric(df["peak_cpu_memory_gb"], errors="coerce")
    gpu = pd.to_numeric(df["peak_gpu_memory_gb"], errors="coerce")

    fig, ax_rt = plt.subplots(figsize=(7.6, 3.45))
    bars = ax_rt.bar(x, runtime.fillna(0), width=0.58,
                     color=method_colors(methods), edgecolor="white", zorder=2)
    ax_rt.set_ylabel("Runtime (min)")
    ax_rt.set_xticks(x)
    ax_rt.set_xticklabels(methods, rotation=35, ha="right", fontsize=6.5)
    ymax_rt = max(float(runtime.max(skipna=True)) * 1.2, 5)
    ax_rt.set_ylim(0, ymax_rt)
    for bar, val in zip(bars, runtime):
        if pd.notna(val):
            ax_rt.text(bar.get_x() + bar.get_width() / 2,
                       bar.get_height() + ymax_rt * 0.025,
                       nice_time(val), ha="center", va="bottom",
                       fontsize=6.2, rotation=0)

    ax_mem = ax_rt.twinx()
    ax_mem.set_ylabel("Peak memory (GB)")
    mem_max = np.nanmax([cpu.max(skipna=True), gpu.max(skipna=True)])
    ax_mem.set_ylim(0, max(float(mem_max) * 1.35, 1))

    cpu_mask = cpu.notna().to_numpy()
    gpu_mask = gpu.notna().to_numpy()
    ax_mem.plot(x[cpu_mask], cpu[cpu_mask], color="#7f8780", lw=1.0,
                marker="^", markersize=6.0, label="CPU RSS",
                markeredgecolor="white", markeredgewidth=0.45, zorder=5)
    ax_mem.plot(x[gpu_mask], gpu[gpu_mask], color="#8f86b3", lw=1.0,
                marker="s", markersize=6.0, label="GPU memory",
                markeredgecolor="white", markeredgewidth=0.45, zorder=6)

    for i, (c, g, method) in enumerate(zip(cpu, gpu, methods)):
        if pd.notna(c):
            ax_mem.text(i + 0.04, c + ax_mem.get_ylim()[1] * 0.03,
                        f"{c:.1f} GB", ha="left", va="bottom",
                        fontsize=6.0, color="#5e665f")
        if pd.notna(g):
            ax_mem.text(i + 0.04, g + ax_mem.get_ylim()[1] * 0.03,
                        f"{g:.1f} GB GPU", ha="left", va="bottom",
                        fontsize=6.0, color="#5a4d84", fontweight="bold")
            ax_mem.text(i, g - ax_mem.get_ylim()[1] * 0.08, "GPU",
                        ha="center", va="top", fontsize=6.2,
                        color="#5a4d84", fontweight="bold")

    ax_rt.set_title("Runtime and Peak Memory", loc="left", fontweight="bold")
    ax_rt.text(0.01, 0.96,
               "Runtime bars; memory markers/lines. TRACER rows share one run.",
               transform=ax_rt.transAxes, va="top", fontsize=6.4, color="#555")
    ax_rt.spines["top"].set_visible(False)
    ax_mem.spines["top"].set_visible(False)
    ax_rt.grid(axis="y", color="#e6e1dc", lw=0.5, zorder=0)
    ax_rt.set_axisbelow(True)
    handles, labels = ax_mem.get_legend_handles_labels()
    ax_mem.legend(handles, labels, frameon=False, fontsize=6.4,
                  loc="upper right")
    C.save_fig(fig, "figure_runtime_memory_combined_v3")
    plt.close(fig)


def figure_transcript_fate():
    fate = ordered(read_source("transcript_fate_v2.tsv"))
    genes = read_source("unassigned_removed_gene_enrichment_v2.tsv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.7), gridspec_kw={"width_ratios": [0.85, 1.55], "wspace": 0.42})

    labels = fate["method"].astype(str)
    x = np.arange(len(fate))
    bars = ax1.bar(x, fate["count"], color=method_colors(fate["method"]), width=0.66, edgecolor="white")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=35, ha="right", fontsize=6.2)
    ax1.set_ylabel("Affected transcripts")
    annotate_bars(ax1, bars, [nice_count(v) for v in fate["count"]], rotation=90)
    for i, lab in enumerate(fate["fate_label"].astype(str)):
        short = (lab.replace("pseudo-unassigned (count-level estimate)", "pseudo")
                   .replace("cleaned-to-unassigned/removed", "cleaned")
                   .replace("removed (count-level)", "removed")
                   .replace("unassigned", "unassigned"))
        ax1.text(i, -ax1.get_ylim()[1] * 0.055, short, ha="right", va="top",
                 rotation=35, fontsize=5.6, color="#555", clip_on=False)
    ax1.set_title("Unassigned / Removed Only", loc="left", fontweight="bold")
    finish_axis(ax1)

    dot = genes[genes["table"].eq("lineage_marker_overlay")].copy()
    dot = dot[dot["n_affected"].fillna(0) > 0]
    if dot.empty:
        dot = genes[genes["table"].eq("top50_per_method")].copy()
    top_genes = (dot.groupby("feature_name")["n_affected"].sum()
                   .sort_values(ascending=False).head(24).index.tolist())
    dot = dot[dot["feature_name"].isin(top_genes)]
    dot["method"] = pd.Categorical(dot["method"], C.METHOD_ORDER, ordered=True)
    top_genes = list(reversed(top_genes))
    y_map = {g: i for i, g in enumerate(top_genes)}
    x_map = {m: i for i, m in enumerate(C.METHOD_ORDER)}
    size_base = np.sqrt(dot["n_affected"].clip(lower=1))
    sizes = 14 + 260 * size_base / max(float(size_base.max()), 1.0)
    sc = ax2.scatter(dot["method"].map(x_map), dot["feature_name"].map(y_map),
                     c=dot["fraction_affected"], s=sizes,
                     cmap=C.morandi_cmap(), norm=Normalize(0, max(0.01, dot["fraction_affected"].max())),
                     edgecolor="#4a4a4a", linewidth=0.25)
    ax2.set_xticks(range(len(C.METHOD_ORDER)))
    ax2.set_xticklabels(C.METHOD_ORDER, rotation=35, ha="right", fontsize=6.2)
    ax2.set_yticks(range(len(top_genes)))
    ax2.set_yticklabels(top_genes, fontsize=6.2, fontstyle="italic")
    ax2.set_title("Affected Marker Genes", loc="left", fontweight="bold")
    ax2.set_xlabel("Method")
    ax2.set_ylabel("Gene")
    cb = fig.colorbar(sc, ax=ax2, fraction=0.046, pad=0.02)
    cb.set_label("Fraction affected", fontsize=6.5)
    finish_axis(ax2)
    C.save_fig(fig, "figure_transcript_fate_v2")
    plt.close(fig)


def half_violin(ax, values, center, side="left", color="#999999", width=0.75):
    vals = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy()
    if vals.size < 5:
        return
    vals = vals[np.isfinite(vals)]
    if vals.size > 3500:
        rng = np.random.default_rng(0)
        vals = rng.choice(vals, size=3500, replace=False)
    parts = ax.violinplot(vals, positions=[center], widths=width, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.35)
        verts = body.get_paths()[0].vertices
        if side == "left":
            verts[:, 0] = np.minimum(verts[:, 0], center)
        else:
            verts[:, 0] = np.maximum(verts[:, 0], center)
    q = np.nanmedian(vals)
    ax.plot([center - width * 0.25, center + width * 0.25], [q, q], color=color, lw=1.1)


def _valid_ids(s: pd.Series) -> pd.Series:
    return ~s.astype(str).isin(C.UNASSIGNED_TOKENS)


def _tracer_entity_counts_from_parquet() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Audit TRACER entity counts and build unfiltered transcript-count rows.

    ``cell_id`` is the source/original cell for partial transcripts, while
    ``tracer_id`` is the reconstructed entity identifier. For whole-cell
    transcripts, ``tracer_id`` is the original whole-cell id.
    """
    tracer = pd.read_parquet(C.TRANSCRIPTS["TRACER"],
                             columns=["cell_id", "tracer_id", "stitched",
                                      "_etype", "feature_name"])
    for col in ("cell_id", "tracer_id", "stitched", "_etype"):
        tracer[col] = tracer[col].astype(str)

    orig = pd.read_parquet(C.TRANSCRIPTS["original"], columns=["cell_id"])
    orig["cell_id"] = orig["cell_id"].astype(str)
    original_ids = pd.Index(sorted(orig.loc[_valid_ids(orig["cell_id"]), "cell_id"].unique()))
    original_count = int(len(original_ids))

    whole_tx = tracer[(tracer["_etype"].eq("cell")) & _valid_ids(tracer["tracer_id"])]
    partial_tx = tracer[(tracer["_etype"].eq("partial")) & _valid_ids(tracer["tracer_id"])]
    assigned_tx = tracer[tracer["_etype"].isin(["cell", "partial", "component"])]

    whole_by_tracer = whole_tx.groupby("tracer_id").size()
    partial_by_tracer = partial_tx.groupby("tracer_id").size()
    whole_dist = (
        whole_by_tracer.reindex(original_ids, fill_value=0)
        .rename_axis("cell_id").rename("n_transcripts").reset_index()
    )
    whole_dist["method"] = "TRACER"
    whole_dist["component"] = "TRACER-refined whole cells"
    whole_dist["component_short"] = TRACER_WHOLE
    whole_dist["distribution_filtered"] = False
    whole_dist["count_basis"] = "original segmentation cell IDs; zero-filled if no _etype=cell transcript remains"

    partial_dist = (
        partial_by_tracer.rename_axis("cell_id").rename("n_transcripts").reset_index()
    )
    partial_dist["method"] = "TRACER"
    partial_dist["component"] = "TRACER-reconstructed partial cells"
    partial_dist["component_short"] = TRACER_PARTIAL
    partial_dist["distribution_filtered"] = False
    partial_dist["count_basis"] = "unique tracer_id among _etype=partial transcripts"

    previous = read_source("cell_count_raincloud_v2.tsv")
    prev_refined = previous.loc[previous["method"].eq(TRACER_WHOLE), "total_cells"]
    prev_partial = previous.loc[previous["method"].eq(TRACER_PARTIAL), "total_cells"]
    prev_refined_after = previous.loc[previous["method"].eq(TRACER_WHOLE), "n_after_filter"]
    prev_partial_after = previous.loc[previous["method"].eq(TRACER_PARTIAL), "n_after_filter"]

    metric_summary = pd.read_csv(C.TRACER_HOME / "metrics/TRACER/cell_count_summary.tsv",
                                 sep="\t")
    metric_map = metric_summary.set_index("metric")["value"].to_dict()

    audit_rows = [
        {
            "metric": "original_cell_count",
            "value": original_count,
            "basis": "unique valid cell_id in original Xenium transcript table",
            "interpretation": "expected TRACER-refined whole-cell layer size",
        },
        {
            "metric": "tracer_whole_cell_count_expected_for_plot",
            "value": original_count,
            "basis": "TRACER is a refinement of original segmentation cells",
            "interpretation": "used as lower TRACER bar segment in v3",
        },
        {
            "metric": "tracer_whole_cells_with_retained_cell_transcripts_by_tracer_id",
            "value": int(whole_by_tracer.size),
            "basis": "unique tracer_id where _etype == cell",
            "interpretation": "transcript-bearing whole cells in TRACER output parquet",
        },
        {
            "metric": "tracer_whole_cells_with_retained_cell_transcripts_by_stitched",
            "value": int(tracer.loc[tracer["_etype"].eq("cell") & _valid_ids(tracer["stitched"]),
                                    "stitched"].nunique()),
            "basis": "unique stitched where _etype == cell",
            "interpretation": "label column used by get_metric can overlap with partial records",
        },
        {
            "metric": "tracer_partial_cell_count_by_tracer_id",
            "value": int(partial_by_tracer.size),
            "basis": "unique tracer_id where _etype == partial",
            "interpretation": "additional reconstructed entities; used as upper TRACER bar segment",
        },
        {
            "metric": "tracer_total_entities_for_plot",
            "value": int(original_count + partial_by_tracer.size),
            "basis": "original whole-cell layer plus unique reconstructed partial tracer_id",
            "interpretation": "stacked TRACER total in v3",
        },
        {
            "metric": "tracer_output_assigned_entity_count_by_stitched",
            "value": int(assigned_tx.loc[_valid_ids(assigned_tx["stitched"]), "stitched"].nunique()),
            "basis": "unique valid stitched among _etype cell/partial/component",
            "interpretation": "entity count reported by current TRACER metric path before 10-900 filter",
        },
        {
            "metric": "tracer_metric_post_filter_entities",
            "value": int(float(metric_map.get("n_total_entities_post_filter", np.nan))),
            "basis": "results/TRACER_nsclc/metrics/TRACER/cell_count_summary.tsv",
            "interpretation": "10-900 transcript QC-filtered metric count, not used for v3 bars",
        },
        {
            "metric": "previous_plot_tracer_refined_count",
            "value": int(prev_refined.iloc[0]) if len(prev_refined) else np.nan,
            "basis": "old v2 source table grouped TRACER by original cell_id and first _etype",
            "interpretation": "not a true output entity count and not the 10-900 QC count",
        },
        {
            "metric": "previous_plot_tracer_reconstructed_count",
            "value": int(prev_partial.iloc[0]) if len(prev_partial) else np.nan,
            "basis": "old v2 source table grouped TRACER by original cell_id and first _etype",
            "interpretation": "undercounts partial entities because cell_id is a source/host cell id",
        },
        {
            "metric": "previous_plot_tracer_refined_after_10_900_filter",
            "value": int(prev_refined_after.iloc[0]) if len(prev_refined_after) else np.nan,
            "basis": "old v2 source table n_after_filter",
            "interpretation": "QC-filtered count was stored but not used as the v2 total bar",
        },
        {
            "metric": "previous_plot_tracer_reconstructed_after_10_900_filter",
            "value": int(prev_partial_after.iloc[0]) if len(prev_partial_after) else np.nan,
            "basis": "old v2 source table n_after_filter",
            "interpretation": "QC-filtered count was stored but not used as the v2 total bar",
        },
    ]
    audit = pd.DataFrame(audit_rows)
    distributions = pd.concat([whole_dist, partial_dist], ignore_index=True)
    values = {
        "original_count": original_count,
        "whole_with_retained_cell_transcripts": int(whole_by_tracer.size),
        "partial_count": int(partial_by_tracer.size),
        "plot_total": int(original_count + partial_by_tracer.size),
        "previous_refined": int(prev_refined.iloc[0]) if len(prev_refined) else None,
        "previous_partial": int(prev_partial.iloc[0]) if len(prev_partial) else None,
        "metric_pre_filter_stitched": int(float(metric_map.get("n_total_entities_pre_filter", np.nan))),
        "metric_post_filter_stitched": int(float(metric_map.get("n_total_entities_post_filter", np.nan))),
    }
    return audit, distributions, values


def build_cell_count_combined_v3_sources():
    C.SRCDIR.mkdir(parents=True, exist_ok=True)
    audit, tracer_dist, vals = _tracer_entity_counts_from_parquet()
    C.save_source(audit, "tracer_cell_count_audit.tsv")

    old_counts = read_source("cell_count_raincloud_v2.tsv")
    base = old_counts[old_counts["method"].isin(COMBINED_METHOD_ORDER[:-1])].copy()
    base = base[["method", "total_cells", "n_after_filter", "tx_min", "tx_max"]]
    rows = []
    for r in base.itertuples(index=False):
        rows.append({
            "method": r.method,
            "component": "single cell/profile layer",
            "component_short": r.method,
            "bar_segment": "total",
            "bar_count": int(r.total_cells),
            "bar_bottom": 0,
            "bar_total": int(r.total_cells),
            "n_after_10_900_filter": int(r.n_after_filter),
            "tx_min": int(r.tx_min),
            "tx_max": int(r.tx_max),
            "distribution_filtered": False,
            "count_basis": "method total cells/profiles before 10-900 filter",
        })
    rows.extend([
        {
            "method": "TRACER",
            "component": "TRACER-refined whole cells",
            "component_short": TRACER_WHOLE,
            "bar_segment": "lower",
            "bar_count": vals["original_count"],
            "bar_bottom": 0,
            "bar_total": vals["plot_total"],
            "n_after_10_900_filter": int(((tracer_dist["component_short"].eq(TRACER_WHOLE)) &
                                           (tracer_dist["n_transcripts"] > C.TX_MIN) &
                                           (tracer_dist["n_transcripts"] < C.TX_MAX)).sum()),
            "tx_min": C.TX_MIN,
            "tx_max": C.TX_MAX,
            "distribution_filtered": False,
            "count_basis": "original segmentation cell count; TRACER-refined preserves the whole-cell layer",
        },
        {
            "method": "TRACER",
            "component": "TRACER-reconstructed partial cells",
            "component_short": TRACER_PARTIAL,
            "bar_segment": "upper",
            "bar_count": vals["partial_count"],
            "bar_bottom": vals["original_count"],
            "bar_total": vals["plot_total"],
            "n_after_10_900_filter": int(((tracer_dist["component_short"].eq(TRACER_PARTIAL)) &
                                           (tracer_dist["n_transcripts"] > C.TX_MIN) &
                                           (tracer_dist["n_transcripts"] < C.TX_MAX)).sum()),
            "tx_min": C.TX_MIN,
            "tx_max": C.TX_MAX,
            "distribution_filtered": False,
            "count_basis": "unique partial/reconstructed tracer_id in TRACER transcript parquet",
        },
    ])
    counts = pd.DataFrame(rows)
    counts["method"] = pd.Categorical(counts["method"], COMBINED_METHOD_ORDER, ordered=True)
    counts = counts.sort_values(["method", "bar_bottom"])
    C.save_source(counts, "cell_count_combined_v3.tsv")

    old_dist = read_source("transcripts_per_cell_distribution_v2.tsv")
    old_dist = old_dist[old_dist["group"].isin(COMBINED_METHOD_ORDER[:-1])].copy()
    old_dist = old_dist.rename(columns={"group": "method"})
    old_dist["component"] = old_dist["method"]
    old_dist["component_short"] = old_dist["method"]
    old_dist["distribution_filtered"] = False
    old_dist["count_basis"] = "unfiltered per-cell transcript counts from method transcript table"
    dist = pd.concat([
        old_dist[["method", "component", "component_short", "cell_id",
                  "n_transcripts", "distribution_filtered", "count_basis"]],
        tracer_dist[["method", "component", "component_short", "cell_id",
                     "n_transcripts", "distribution_filtered", "count_basis"]],
    ], ignore_index=True)
    dist["method"] = pd.Categorical(dist["method"], COMBINED_METHOD_ORDER, ordered=True)
    dist = dist.sort_values(["method", "component_short", "cell_id"])
    C.save_source(dist, "transcripts_per_cell_combined_v3.tsv")

    whole_equals_original = vals["whole_with_retained_cell_transcripts"] == vals["original_count"]
    why = (
        "The transcript-bearing TRACER whole-cell count is lower because the "
        "TRACER parquet only contains whole cells that retain at least one "
        "_etype=cell transcript. Some original cells have all transcripts "
        "reassigned to partial entities or UNASSIGNED/unknown, but the "
        "segmentation-refinement whole-cell layer is still anchored to the "
        "original segmentation count."
    )
    audit_md = f"""# TRACER Cell Count Audit

| Quantity | Count |
|---|---:|
| Original Xenium cells | {vals['original_count']:,} |
| TRACER whole cells with retained `_etype=cell` transcripts (`tracer_id`) | {vals['whole_with_retained_cell_transcripts']:,} |
| TRACER reconstructed partial cells (`tracer_id`) | {vals['partial_count']:,} |
| TRACER whole-cell count equals original? | {'yes' if whole_equals_original else 'no'} |
| Previous v2 TRACER-refined plotted count | {vals['previous_refined']:,} |
| Previous v2 TRACER-reconstructed plotted count | {vals['previous_partial']:,} |
| Current metric pre-filter stitched entities | {vals['metric_pre_filter_stitched']:,} |
| Current metric post-filter stitched entities (10-900 tx) | {vals['metric_post_filter_stitched']:,} |

The previous ~41k TRACER-refined count was not the true refined whole-cell output and was not the 10-900 QC-filtered count. It came from grouping the TRACER parquet by original `cell_id` and taking the first `_etype`; for partial transcripts, `cell_id` is a source/host original cell rather than the reconstructed entity ID.

{why}
"""
    (C.FIGDIR / "tracer_cell_count_audit.md").write_text(audit_md)

    notes = f"""# Cell Count Raincloud Combined v3 Notes

TRACER-refined is plotted as {vals['original_count']:,} whole cells, matching the original Xenium segmentation count, because TRACER is a segmentation-refinement method and preserves the whole-cell layer. The TRACER parquet contains {vals['whole_with_retained_cell_transcripts']:,} whole-cell `tracer_id`s with retained `_etype=cell` transcripts; the difference reflects original cells with no remaining whole-cell transcripts in the transcript-level output, not a 10-900 QC filter.

TRACER reconstructed {vals['partial_count']:,} partial-cell entities by unique partial `tracer_id`; these are stacked on top of the whole-cell segment in the single TRACER bar.

The half-violin distributions use unfiltered per-cell/profile transcript counts. For TRACER-refined, original whole-cell IDs with no retained whole-cell transcripts are included with zero transcripts; TRACER-reconstructed partial-cell entities have a lower transcript-count distribution, consistent with partial/anuclear reconstructed profiles rather than full cells.
"""
    (C.FIGDIR / "figure_cell_count_raincloud_combined_v3_notes.md").write_text(notes)
    return counts, dist


def figure_cell_count_raincloud_combined_v3():
    counts, dist = build_cell_count_combined_v3_sources()
    fig, ax_count = plt.subplots(figsize=(7.9, 3.75))
    ax_tx = ax_count.twinx()
    x = np.arange(len(COMBINED_METHOD_ORDER), dtype=float)
    x_map = {m: i for i, m in enumerate(COMBINED_METHOD_ORDER)}

    bar_w = 0.32
    bar_xoff = -0.16
    for m in COMBINED_METHOD_ORDER[:-1]:
        row = counts[(counts["method"].astype(str).eq(m)) & counts["bar_segment"].eq("total")].iloc[0]
        ax_count.bar(x_map[m] + bar_xoff, row["bar_count"], width=bar_w,
                     color=C.PALETTE.get(m, "#9aa0a6"), edgecolor="white", zorder=3)
        ax_count.text(x_map[m] + bar_xoff, row["bar_count"] + counts["bar_total"].max() * 0.018,
                      nice_count(row["bar_count"]), ha="center", va="bottom",
                      fontsize=6.1, rotation=90)

    tr = counts[counts["method"].astype(str).eq("TRACER")].sort_values("bar_bottom")
    tx = x_map["TRACER"] + bar_xoff
    whole = tr.iloc[0]
    partial = tr.iloc[1]
    ax_count.bar(tx, whole["bar_count"], width=bar_w, color=C.PALETTE[TRACER_WHOLE],
                 edgecolor="white", zorder=3)
    ax_count.bar(tx, partial["bar_count"], width=bar_w, bottom=whole["bar_count"],
                 color=C.PALETTE[TRACER_PARTIAL], edgecolor="white", zorder=3)
    ax_count.text(tx, whole["bar_count"] * 0.5, nice_count(whole["bar_count"]),
                  ha="center", va="center", fontsize=6.1, color="white", rotation=90)
    ax_count.text(tx, whole["bar_count"] + partial["bar_count"] * 0.5,
                  nice_count(partial["bar_count"]), ha="center", va="center",
                  fontsize=6.1, color="#5f3150", rotation=90)
    ax_count.text(tx, partial["bar_total"] + counts["bar_total"].max() * 0.018,
                  nice_count(partial["bar_total"]), ha="center", va="bottom",
                  fontsize=6.1, rotation=90, fontweight="bold")

    # Unfiltered transcript-count distributions on the right axis.
    for m in COMBINED_METHOD_ORDER[:-1]:
        vals = dist.loc[dist["method"].astype(str).eq(m), "n_transcripts"]
        half_violin(ax_tx, vals, x_map[m] + 0.18, side="right",
                    color=C.PALETTE.get(m, "#9aa0a6"), width=0.34)
    refined = dist[dist["component_short"].eq(TRACER_WHOLE)]["n_transcripts"]
    partial_vals = dist[dist["component_short"].eq(TRACER_PARTIAL)]["n_transcripts"]
    half_violin(ax_tx, refined, x_map["TRACER"] + 0.10, side="left",
                color=C.PALETTE[TRACER_WHOLE], width=0.30)
    half_violin(ax_tx, partial_vals, x_map["TRACER"] + 0.33, side="right",
                color=C.PALETTE[TRACER_PARTIAL], width=0.30)

    tx_vals = pd.to_numeric(dist["n_transcripts"], errors="coerce")
    ymax_tx = min(1100, max(80, tx_vals.quantile(0.995) * 1.12))
    ax_tx.set_ylim(0, ymax_tx)
    ax_count.set_ylim(0, counts["bar_total"].max() * 1.16)
    ax_count.set_xlim(-0.65, len(COMBINED_METHOD_ORDER) - 0.35)
    ax_count.set_xticks(x)
    ax_count.set_xticklabels(COMBINED_METHOD_ORDER, rotation=28, ha="right")
    ax_count.set_ylabel("Total cells / profiles")
    ax_tx.set_ylabel("Transcripts per cell/profile (unfiltered)")
    ax_count.set_title("Cell Count and Transcript-Per-Cell Distribution",
                       loc="left", fontweight="bold")
    ax_count.text(0.01, 0.96,
                  "Bars: total profiles. Half violins: unfiltered transcript counts.",
                  transform=ax_count.transAxes, ha="left", va="top",
                  fontsize=6.4, color="#555")
    ax_count.grid(axis="y", color="#e6e1dc", lw=0.5, zorder=0)
    for ax in (ax_count, ax_tx):
        ax.spines["top"].set_visible(False)
    ax_count.spines["right"].set_visible(False)
    ax_tx.spines["left"].set_visible(False)
    ax_count.set_axisbelow(True)

    handles = [
        patches.Patch(facecolor=C.PALETTE[TRACER_WHOLE], label="TRACER-refined whole"),
        patches.Patch(facecolor=C.PALETTE[TRACER_PARTIAL], label="TRACER-reconstructed partial"),
    ]
    ax_count.legend(handles=handles, frameon=False, fontsize=6.2, loc="upper right",
                    bbox_to_anchor=(1.0, 1.18))
    C.save_fig(fig, "figure_cell_count_raincloud_combined_v3")
    plt.close(fig)


def figure_cell_count_raincloud():
    counts = ordered(read_source("cell_count_raincloud_v2.tsv"))
    dist = read_source("transcripts_per_cell_distribution_v2.tsv")
    methods = counts["method"].astype(str).tolist()
    x = np.arange(len(methods))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.35), gridspec_kw={"width_ratios": [0.9, 1.25], "wspace": 0.34})

    bars = ax1.bar(x, counts["total_cells"], color=method_colors(methods), width=0.65, edgecolor="white")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=35, ha="right", fontsize=6.3)
    ax1.set_ylabel("Total cells / profiles")
    annotate_bars(ax1, bars, [nice_count(v) for v in counts["total_cells"]], rotation=90)
    ax1.text(0.03, 0.96, "Bars use total cells; filter shown at right", transform=ax1.transAxes,
             va="top", fontsize=6.5, color="#555")
    ax1.set_title("Cell Counts", loc="left", fontweight="bold")
    finish_axis(ax1)

    for i, m in enumerate(C.METHOD_ORDER):
        sub = dist.loc[dist["group"].astype(str).eq(m), "n_transcripts"]
        side = "left" if m == "TRACER-refined" else "right"
        half_violin(ax2, sub, i, side=side, color=C.PALETTE.get(m, "#999"))
    ax2.axhspan(C.TX_MIN, C.TX_MAX, color="#e8e0d6", alpha=0.26, zorder=0)
    ax2.axhline(C.TX_MIN, color="#8a8178", lw=0.7, ls="--")
    ax2.axhline(C.TX_MAX, color="#8a8178", lw=0.7, ls="--")
    ax2.set_ylim(0, min(1100, pd.to_numeric(dist["n_transcripts"], errors="coerce").quantile(0.995) * 1.15))
    ax2.set_xticks(range(len(C.METHOD_ORDER)))
    ax2.set_xticklabels(C.METHOD_ORDER, rotation=35, ha="right", fontsize=6.3)
    ax2.set_ylabel("Transcripts per cell")
    ax2.set_title("Transcript-Per-Cell Rainclouds", loc="left", fontweight="bold")
    finish_axis(ax2)
    C.save_fig(fig, "figure_cell_count_raincloud_v2")
    plt.close(fig)


def figure_npmi_concept():
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("NPMI Purity / Conflict Concept", loc="left", fontweight="bold")

    ax.add_patch(patches.Ellipse((3.2, 3.4), 4.0, 2.4, angle=8, fc="#cfd8dc", ec="#6b7c85", lw=1.0, alpha=0.65))
    ax.add_patch(patches.Ellipse((5.0, 3.2), 3.5, 2.2, angle=-12, fc="#f0d8b8", ec="#a27847", lw=1.0, alpha=0.65))
    ax.add_patch(patches.Ellipse((6.7, 2.0), 2.1, 1.25, angle=-20, fc="#e8a4c4", ec="#c0306a", lw=1.1, alpha=0.75))
    rng = np.random.default_rng(2)
    for cx, cy, color in [(2.8, 3.5, "#6f8fb0"), (4.9, 3.25, "#cf9d6b"), (6.7, 2.0, "#c0306a")]:
        pts = rng.normal([cx, cy], [0.55, 0.3], size=(24, 2))
        ax.scatter(pts[:, 0], pts[:, 1], s=13, color=color, edgecolor="white", lw=0.25, zorder=3)
    ax.annotate("hidden partial-cell\ncontribution", xy=(6.7, 2.0), xytext=(7.6, 1.15),
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#555"), fontsize=7, ha="left")
    ax.text(1.2, 5.1, "Before separation: mixed lineages inside one segmented cell", fontsize=8)
    ax.text(1.2, 0.75, "After TRACER separates partial-cell contribution:\npositive/coherent pairs are concentrated; negative/conflicting pairs are reduced.",
            fontsize=8)
    formula = (
        "Implemented in workflow/scripts/metrics.py\n"
        "pos_relu = sum(max(NPMI - tau, 0))\n"
        "neg_relu = sum(max(-NPMI - tau, 0))\n"
        "signal_strength = pos_relu + neg_relu\n"
        "relative_purity = pos_relu / signal_strength\n"
        "relative_conflict = neg_relu / signal_strength"
    )
    ax.text(6.05, 5.15, formula, fontsize=7.1, va="top",
            bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.02", fc="white", ec="#d8d3cc", lw=0.8))
    C.save_fig(fig, "figure_npmi_purity_conflict_concept_v2")
    plt.close(fig)


def figure_npmi_bar():
    summ = ordered(read_source("npmi_purity_conflict_summary_v2.tsv"))
    stats = read_source("npmi_purity_conflict_stats_v2.tsv")
    methods = summ["method"].astype(str).tolist()
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    w = 0.36
    b1 = ax.bar(x - w/2, summ["mean_relative_purity"], w, yerr=summ["sem_relative_purity"],
                color=method_colors(methods), edgecolor="white", label="Relative purity")
    b2 = ax.bar(x + w/2, summ["mean_relative_conflict"], w, yerr=summ["sem_relative_conflict"],
                color="#b9aaa2", edgecolor="white", label="Relative conflict")
    for i, m in enumerate(methods):
        stars = stats[(stats["comparison"].eq(f"TRACER-refined vs {m}")) &
                      (stats["metric"].eq("relative_purity"))]["stars"]
        if m != "TRACER-refined" and len(stars):
            ax.text(i, max(summ["mean_relative_purity"].max(), summ["mean_relative_conflict"].max()) + 0.045,
                    stars.iloc[0], ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=6.4)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Mean score over filtered cells")
    ax.set_title("NPMI Purity / Conflict", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="upper right")
    finish_axis(ax)
    C.save_fig(fig, "figure_npmi_purity_conflict_bar_v2")
    plt.close(fig)


def figure_npmi_purity_conflict_stacked_v3():
    summ = ordered(read_source("npmi_purity_conflict_summary_v2.tsv"))
    out = summ[["method", "n_cells", "mean_relative_purity",
                "mean_relative_conflict", "sem_relative_purity",
                "sem_relative_conflict"]].copy()
    out["stack_total"] = out["mean_relative_purity"] + out["mean_relative_conflict"]
    C.save_source(out, "npmi_purity_conflict_stacked_v3.tsv")

    methods = out["method"].astype(str).tolist()
    x = np.arange(len(methods))
    purity = pd.to_numeric(out["mean_relative_purity"], errors="coerce")
    conflict = pd.to_numeric(out["mean_relative_conflict"], errors="coerce")

    fig, ax = plt.subplots(figsize=(7.6, 3.25))
    b_pur = ax.bar(x, purity, width=0.62, color=method_colors(methods),
                   edgecolor="white", label="Relative purity", zorder=2)
    b_con = ax.bar(x, conflict, bottom=purity, width=0.62,
                   color="#b9aaa2", edgecolor="white",
                   label="Relative conflict", zorder=2)

    for i, (p, c) in enumerate(zip(purity, conflict)):
        if pd.notna(p):
            color = "white" if p > 0.45 and methods[i].startswith("TRACER") else "#253238"
            ax.text(i, p / 2, f"{p:.3f}", ha="center", va="center",
                    fontsize=6.2, color=color, rotation=90)
        if pd.notna(c):
            ypos = p + c / 2
            if c < 0.035:
                ax.text(i, p + c + 0.012, f"{c:.3f}", ha="center",
                        va="bottom", fontsize=6.0, color="#6b5f59",
                        rotation=90)
            else:
                ax.text(i, ypos, f"{c:.3f}", ha="center", va="center",
                        fontsize=6.2, color="#253238", rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=6.4)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Relative NPMI contribution")
    ax.set_title("NPMI Purity and Conflict", loc="left", fontweight="bold")
    ax.text(0.01, 0.96,
            "Stack height is purity + conflict; values are means over filtered cells.",
            transform=ax.transAxes, va="top", fontsize=6.4, color="#555")
    ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="lower right",
              bbox_to_anchor=(1.0, 1.08), borderaxespad=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e6e1dc", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    C.save_fig(fig, "figure_npmi_purity_conflict_stacked_v3")
    plt.close(fig)


def figure_tcell_marker_log2fc():
    df = read_source("tcell_marker_log2fc_v2.tsv")
    genes = read_source("tcell_marker_genes_v2.tsv").sort_values("rank")["gene"].tolist()
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    x_map = {g: i for i, g in enumerate(genes)}
    for m in C.METHOD_ORDER:
        sub = df[df["method"].eq(m)].set_index("gene").reindex(genes)
        ax.plot(range(len(genes)), sub["spatial_log2fc"], ls="--", lw=0.7,
                color=C.PALETTE.get(m, "#999"), alpha=0.7)
        ax.scatter(range(len(genes)), sub["spatial_log2fc"], s=26,
                   color=C.PALETTE.get(m, "#999"), edgecolor="white", lw=0.35, label=m)
    ax.axhline(0, color="#777", lw=0.7)
    ax.set_xticks(range(len(genes)))
    ax.set_xticklabels(genes, fontstyle="italic")
    ax.set_ylabel("Spatial log2FC: T vs other")
    ax.set_title("T-Cell Marker Log2FC", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=4, fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    finish_axis(ax)
    C.save_fig(fig, "figure_tcell_marker_log2fc_v2")
    plt.close(fig)


def figure_marker_specificity():
    df = read_source("marker_specificity_v2.tsv").dropna(subset=["spatial_log2fc"])
    stats = read_source("marker_specificity_stats_v2.tsv")
    fig, ax = plt.subplots(figsize=(7.5, 3.55))
    rng = np.random.default_rng(4)
    for i, m in enumerate(C.METHOD_ORDER):
        sub = df[df["method"].eq(m)]
        vals = sub["spatial_log2fc"].to_numpy()
        if len(vals) == 0:
            continue
        ax.boxplot(vals, positions=[i], widths=0.48, patch_artist=True, showfliers=False,
                   boxprops=dict(facecolor="#f8f7f5", edgecolor=C.PALETTE.get(m, "#999"), lw=0.9),
                   medianprops=dict(color="#333", lw=1.0),
                   whiskerprops=dict(color="#777", lw=0.7),
                   capprops=dict(color="#777", lw=0.7))
        xs = i + rng.normal(0, 0.07, size=len(sub))
        colors = [C.CELLTYPE_COLORS.get(ct, "#999") for ct in sub["cell_type"]]
        ax.scatter(xs, sub["spatial_log2fc"], s=15, c=colors, alpha=0.75, edgecolor="white", lw=0.2)
        if m != "TRACER-refined":
            stars = stats[stats["comparison"].eq(f"TRACER-refined vs {m}")]["stars"]
            if len(stars):
                ax.text(i, df["spatial_log2fc"].quantile(0.99) + 0.25, stars.iloc[0], ha="center", fontsize=7)
    ax.axhline(0, color="#777", lw=0.7)
    ax.set_xticks(range(len(C.METHOD_ORDER)))
    ax.set_xticklabels(C.METHOD_ORDER, rotation=35, ha="right", fontsize=6.4)
    ax.set_ylabel("Marker log2FC in lineage vs rest")
    ax.set_title("45-Gene Marker Specificity", loc="left", fontweight="bold")
    finish_axis(ax)
    C.save_fig(fig, "figure_marker_specificity_v2")
    plt.close(fig)


def figure_reference_consistency_heatmap():
    hm = read_source("reference_consistency_heatmap_v2.tsv").set_index("method")
    hm = hm.reindex(C.METHOD_ORDER)[C.CELLTYPES9]
    fig, ax = plt.subplots(figsize=(7.5, 3.35))
    im = ax.imshow(hm.to_numpy(dtype=float), aspect="auto", cmap=C.morandi_cmap(), vmin=0.35, vmax=0.85)
    ax.set_xticks(range(len(C.CELLTYPES9)))
    ax.set_xticklabels(C.CELLTYPES9, rotation=35, ha="right", fontsize=6.5)
    ax.set_yticks(range(len(C.METHOD_ORDER)))
    ax.set_yticklabels(C.METHOD_ORDER, fontsize=6.8)
    for i in range(hm.shape[0]):
        for j in range(hm.shape[1]):
            val = hm.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=5.8, color="#263238")
    ax.set_title("Per-Cell-Type Reference Consistency", loc="left", fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.034, pad=0.02)
    cb.set_label("Pearson r", fontsize=6.5)
    C.save_fig(fig, "figure_reference_consistency_heatmap_v2")
    plt.close(fig)


def figure_rctd_entropy_maxweight():
    df = read_source("rctd_entropy_maxweight_v2.tsv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.25), gridspec_kw={"wspace": 0.3})
    for ax, metric, title, better in [
        (ax1, "rctd_entropy", "RCTD Entropy", "lower is better"),
        (ax2, "rctd_max_weight", "RCTD Max Weight", "higher is better"),
    ]:
        for i, m in enumerate(C.METHOD_ORDER):
            sub = df[df["method"].eq(m)][metric]
            half_violin(ax, sub, i, side="left", color=C.PALETTE.get(m, "#999"), width=0.72)
            val = pd.to_numeric(sub, errors="coerce").median()
            ax.scatter([i + 0.1], [val], s=18, color=C.PALETTE.get(m, "#999"), edgecolor="white", lw=0.4)
        ax.set_xticks(range(len(C.METHOD_ORDER)))
        ax.set_xticklabels(C.METHOD_ORDER, rotation=35, ha="right", fontsize=6.4)
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(f"{title} ({better})", loc="left", fontweight="bold")
        finish_axis(ax)
    C.save_fig(fig, "figure_rctd_entropy_maxweight_v2")
    plt.close(fig)


def write_report():
    out_files = sorted(p.name for p in C.FIGDIR.iterdir()
                       if p.suffix.lower() in {".svg", ".pdf", ".png"})
    split = pd.read_csv(C.METRICS / "SPLIT" / "split_reference_reaudit_v2.tsv", sep="\t")
    tgenes = read_source("tcell_marker_genes_v2.tsv")
    lin = read_source("lineage_marker_genes_45_v2.tsv")
    caveats = [
        "cellAdmix is transcript-cleaning on original Xenium cell IDs, not de novo segmentation.",
        "SPLIT is cell-level profile purification; removed/pseudo-unassigned values are count-level estimates, not exact transcript-coordinate removals.",
        "Segger is shown as GPU-based where runtime/memory metadata report GPU use.",
        "TRACER-refined is the whole-cell TRACER output; TRACER-reconstructed is the partial-cell output.",
        "TRACER-reconstructed has very few T-labelled partial cells in the filtered marker-scoring set, so its T-marker values should be read cautiously.",
    ]
    lines = [
        "# Visualization Revision Summary",
        "",
        "## Recalculated Metrics",
        "- SPLIT reference consistency was repaired by KNN label transfer against `lung_cancer_50k.h5ad` using `Cell_Cluster_level1`, then pseudobulk Pearson was recomputed.",
        "- Filtered-cell NPMI purity/conflict source data were regenerated for `10 < n_transcripts < 900`; SPLIT purity/conflict was recomputed from purified counts.",
        "- Marker genes were re-derived from the scRNA reference restricted to the Xenium panel.",
        "- TRACER-reconstructed reference consistency was computed de novo from partial-cell transcripts.",
        "",
        "## SPLIT Audit",
        f"- SPLIT was not rerun from scratch; the evaluation/annotation was repaired from existing purified count output.",
        f"- Pearson r is present for {split['pearson_r'].notna().sum()}/9 cell types.",
        f"- Missing SPLIT cell types: {', '.join(split.loc[split['pearson_r'].isna(), 'cell_type']) or 'none'}.",
        "",
        "## Metric Correctness",
        "- `get_metric.py` and the v2 SPLIT recomputation call the same `compute_cell_purity_relu` / `compute_cell_conflict_relu` implementation used in `workflow/scripts/metrics.py`.",
        "- `relative_purity = pos_relu / signal_strength`; `relative_conflict = neg_relu / signal_strength`; `signal_strength = pos_relu + neg_relu`.",
        "",
        "## T-Cell Marker Debug",
        "- The previous T-cell marker log2FC was missing because earlier code searched for T-cell label variants such as `T cells`, while these annotations use the bare label `T`.",
        "- Selected T-cell marker genes: " + ", ".join(tgenes.sort_values("rank")["gene"].astype(str)),
        "",
        "## Selected 45 Lineage Markers",
        ", ".join(f"{ct}: " + ", ".join(lin[lin.cell_type.eq(ct)].sort_values("rank")["gene"].astype(str))
                  for ct in C.CELLTYPES9),
        "",
        "## Generated Files",
        "\n".join(f"- `{f}`" for f in out_files),
        "",
        "## Known Caveats",
        "\n".join(f"- {c}" for c in caveats),
        "",
        "## Recommended Main-Figure Panels",
        "- `figure_runtime_memory_combined_v3`, `figure_transcript_fate_v2`, `figure_cell_count_raincloud_combined_v3`, `figure_npmi_purity_conflict_stacked_v3`, `figure_reference_consistency_heatmap_v2`.",
        "",
        "## Recommended Supplementary Panels",
        "- `figure_npmi_purity_conflict_concept_v2`, `figure_tcell_marker_log2fc_v2`, `figure_marker_specificity_v2`, `figure_rctd_entropy_maxweight_v2`.",
        "",
    ]
    (C.FIGDIR / "visualization_revision_summary.md").write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None,
                        help="Optional subset of figure function suffixes to render.")
    args = parser.parse_args()
    C.ensure_dirs()
    C.apply_style()
    tasks = [
        figure_runtime_memory,
        figure_transcript_fate,
        figure_cell_count_raincloud,
        figure_cell_count_raincloud_combined_v3,
        figure_npmi_concept,
        figure_npmi_bar,
        figure_npmi_purity_conflict_stacked_v3,
        figure_tcell_marker_log2fc,
        figure_marker_specificity,
        figure_reference_consistency_heatmap,
        figure_rctd_entropy_maxweight,
        figure_runtime_memory_combined_v3,
    ]
    selected = set(args.only or [])
    for task in tasks:
        if selected and task.__name__.replace("figure_", "") not in selected:
            continue
        task()
    write_report()
    print(f"WROTE {C.FIGDIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

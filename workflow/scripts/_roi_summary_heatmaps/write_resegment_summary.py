#!/usr/bin/env python3
"""Write the TRACER resegment benchmark summary and comparison tables."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import registry as R  # noqa


OLD_OUT = R.FIG3 / "summary_heatmaps"
COMPARE_METRICS = [
    "total_cells",
    "transcripts_per_cell",
    "marker_log2fc",
    "kendall_tau",
    "relative_purity",
    "relative_conflict",
    "rctd_entropy",
    "rctd_max_weight",
]
TRACER_ENTITIES = ["TRACER", "TRACER_refined", "TRACER_reconstructed"]


def fmt(v, signed=False):
    if pd.isna(v):
        return "NA"
    if signed:
        return f"{v:+.3f}"
    return f"{v:.3f}"


def metric_value(df, ds, ent, metric):
    sub = df[(df.dataset == ds) & (df.entity == ent) & (df.metric == metric)]
    if len(sub) and pd.notna(sub["value"].iloc[0]):
        return float(sub["value"].iloc[0])
    return np.nan


def build_comparison(new_all: pd.DataFrame) -> pd.DataFrame:
    old_all_f = OLD_OUT / "all_metrics_long.tsv"
    old_all = pd.read_csv(old_all_f, sep="\t") if old_all_f.exists() else pd.DataFrame()
    rows = []
    for ds in R.DATASET_ORDER:
        for ent in TRACER_ENTITIES:
            for metric in COMPARE_METRICS:
                old_v = metric_value(old_all, ds, ent, metric) if not old_all.empty else np.nan
                new_v = metric_value(new_all, ds, ent, metric)
                rows.append({
                    "dataset": ds,
                    "platform": R.DATASETS[ds]["platform"],
                    "entity": ent,
                    "metric": metric,
                    "tracer_seg_value": old_v,
                    "tracer_resegment_value": new_v,
                    "delta_resegment_minus_seg": new_v - old_v if pd.notna(new_v) and pd.notna(old_v) else np.nan,
                })
    comp = pd.DataFrame(rows)
    comp.to_csv(R.OUT / "tracer_resegment_vs_tracer_seg_metrics.tsv", sep="\t", index=False)
    return comp


def selected_marker_lines(lines):
    lines.append("## Marker Selection Decisions\n")
    lines.append("Marker specificity was recomputed from audited panels rebuilt from the matched scRNA reference. Candidate markers were checked for platform-panel detection and scRNA specificity; broad stress/interferon genes were excluded when lineage-specific alternatives were available. If canonical lineage markers were absent from a platform panel, the panel builder used scRNA-ranked fallback markers and records that source explicitly. Full audit tables are saved as `metrics/<dataset>/marker_audit_table.tsv`; final panels are saved as `metrics/<dataset>/final_marker_list.tsv`.\n")
    for ds in R.DATASET_ORDER:
        final_f = R.METRICS / ds / "final_marker_list.tsv"
        audit_f = R.METRICS / ds / "marker_audit_table.tsv"
        if not final_f.exists():
            continue
        final = pd.read_csv(final_f, sep="\t")
        audit = pd.read_csv(audit_f, sep="\t") if audit_f.exists() else pd.DataFrame()
        lines.append(f"**{R.DATASETS[ds]['platform']} (`{ds}`)**")
        for ct, g in final.groupby("cell_type", sort=False):
            genes = ", ".join(g.sort_values("rank")["gene"].astype(str))
            lines.append(f"- {ct}: {genes}")
        if not audit.empty:
            fallback_n = int(((audit["selected_for_final_panel"] == "yes") & (audit["candidate_source"] == "scrna_fallback")).sum())
            if fallback_n:
                lines.append(f"  Fallback selected markers: {fallback_n} scRNA-ranked marker rows where preferred canonical candidates were unavailable or insufficient in the platform panel.")
            reasons = audit.loc[audit["selected_for_final_panel"] == "no", "reason_for_exclusion_if_not_selected"].value_counts()
            if len(reasons):
                compact = "; ".join(f"{k}: {v}" for k, v in reasons.head(5).items())
                lines.append(f"  Exclusion summary: {compact}.")
        lines.append("")


def comparison_section(lines, comp: pd.DataFrame):
    lines.append("## Resegment vs Previous TRACER Benchmark\n")
    lines.append("Comparison source: previous benchmark `results/fig3_cross_platform_roi_benchmark/summary_heatmaps/all_metrics_long.tsv` versus this resegment run. Marker deltas compare against the previous reported marker metric and therefore include both the new resegment inputs and the audited marker-panel rebuild.\n")
    for metric, label in [
        ("total_cells", "Total cells/profiles"),
        ("transcripts_per_cell", "Median transcripts per cell/profile"),
        ("marker_log2fc", "Marker specificity log2FC"),
        ("kendall_tau", "Kendall tau vs scRNA"),
        ("relative_purity", "NPMI relative purity"),
        ("relative_conflict", "NPMI relative conflict"),
        ("rctd_entropy", "RCTD entropy"),
        ("rctd_max_weight", "RCTD max weight"),
    ]:
        lines.append(f"**{label}**")
        lines.append("| Platform | Entity | tracer_seg | tracer_resegment | Delta |")
        lines.append("|---|---|--:|--:|--:|")
        sub = comp[comp.metric == metric]
        for ds in R.DATASET_ORDER:
            for ent in TRACER_ENTITIES:
                r = sub[(sub.dataset == ds) & (sub.entity == ent)]
                if not len(r):
                    continue
                old_v = r["tracer_seg_value"].iloc[0]
                new_v = r["tracer_resegment_value"].iloc[0]
                delta = r["delta_resegment_minus_seg"].iloc[0]
                if pd.isna(old_v) and pd.isna(new_v):
                    continue
                lines.append(f"| {R.DATASETS[ds]['platform']} | {ent} | {fmt(old_v)} | {fmt(new_v)} | {fmt(delta, signed=True)} |")
        lines.append("")


def current_metric_section(lines, new_all: pd.DataFrame):
    lines.append("## Current Resegment Metric Values\n")
    lines.append("Block B values are from completed full-panel matrices. NPMI purity/conflict use the reference-derived NPMI panel; RCTD values use QC-filtered 10-900 transcript profiles and spacexr doublet mode, except SPLIT which retains its internal RCTD provenance. The resegment TRACER-refined/reconstructed RCTD runs were recomputed single-core (`RCTD_MAX_CORES=1`) to avoid a current spacexr parallel `chooseSigma` failure; non-TRACER QC RCTD outputs are carried forward unchanged where the input matrices are byte-identical to the previous benchmark.\n")
    lines.append("| Platform | Entity | Marker log2FC | Kendall tau | NPMI purity | NPMI conflict | RCTD entropy | RCTD max weight |")
    lines.append("|---|---|--:|--:|--:|--:|--:|--:|")
    for ds in R.DATASET_ORDER:
        for ent in TRACER_ENTITIES:
            vals = {m: metric_value(new_all, ds, ent, m) for m in [
                "marker_log2fc", "kendall_tau", "relative_purity",
                "relative_conflict", "rctd_entropy", "rctd_max_weight",
            ]}
            if all(pd.isna(v) for v in vals.values()):
                continue
            lines.append(
                f"| {R.DATASETS[ds]['platform']} | {ent} | {fmt(vals['marker_log2fc'])} | "
                f"{fmt(vals['kendall_tau'])} | {fmt(vals['relative_purity'])} | "
                f"{fmt(vals['relative_conflict'])} | {fmt(vals['rctd_entropy'])} | "
                f"{fmt(vals['rctd_max_weight'])} |"
            )
    lines.append("")


def main():
    new_all = pd.read_csv(R.OUT / "all_metrics_long.tsv", sep="\t")
    comp = build_comparison(new_all)
    lines: list[str] = []
    lines.append("# TRACER Resegment Benchmark Summary\n")
    lines.append(f"Run namespace: `{R.SUMMARY_NAME}`. TRACER source: `dataset/<dataset>/{R.TRACER_SOURCE}/`; immutable native baseline input transcripts remain from `tracer_seg` where `tracer_resegment` does not provide a copy.\n")
    lines.append("Recomputed metrics: marker specificity, Kendall correlation, NPMI purity/conflict, RCTD entropy, and RCTD max weight. TRACER combined is used only for combined-mode descriptive metrics; TRACER-refined and TRACER-reconstructed are reported separately for biological and RCTD metrics.\n")
    current_metric_section(lines, new_all)
    selected_marker_lines(lines)
    comparison_section(lines, comp)
    lines.append("## Output Files\n")
    lines.append("- `all_metrics_long.tsv`, `source_data_blockB.tsv`, `source_data_blockC.tsv`")
    lines.append("- `tracer_resegment_vs_tracer_seg_metrics.tsv`")
    lines.append("- `metrics/<dataset>/marker_audit_table.tsv` and `metrics/<dataset>/final_marker_list.tsv`")
    lines.append("- Heatmaps: `benchmark_heatmap_blockB_biological.png/.svg` and `benchmark_heatmap_blockC_rctd.png/.svg`\n")
    out = R.OUT / "tracer_resegment_benchmark_summary.md"
    out.write_text("\n".join(lines))
    # Also provide the expected generic summary filename for this namespace.
    (R.OUT / "benchmark_heatmap_summary.md").write_text("\n".join(lines))
    print("wrote", out)


if __name__ == "__main__":
    main()

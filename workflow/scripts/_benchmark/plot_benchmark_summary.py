#!/usr/bin/env python3
"""Render the standard benchmark figures + an HTML report."""

from __future__ import annotations

import argparse
import base64
import io
import sys
from pathlib import Path

import pandas as pd

# Matplotlib is the safest dependency to assume.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


FIGS = [
    "purity_conflict_paired.pdf",
    "marker_specificity_heatmap.pdf",
    "transcript_retention_vs_conflict.pdf",
    "cell_count_transcript_count.pdf",
    "runtime_summary.pdf",
]


def _fig_to_png_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _figure_with_msg(msg: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis("off")
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=10)
    return fig


def plot_purity_conflict(df: pd.DataFrame) -> plt.Figure:
    if "purity_mean" not in df.columns or "conflict_mean" not in df.columns:
        return _figure_with_msg("purity/conflict not available")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["purity_mean"], df["conflict_mean"], s=60)
    for _, row in df.iterrows():
        ax.annotate(
            row["method"],
            (row["purity_mean"], row["conflict_mean"]),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.set_xlabel("NPMI purity (higher = better)")
    ax.set_ylabel("NPMI conflict (lower = better)")
    ax.set_title("Per-method NPMI purity vs. conflict")
    return fig


def plot_marker_heatmap(df: pd.DataFrame) -> plt.Figure:
    if "marker_specificity_mean" not in df.columns:
        return _figure_with_msg("marker specificity not available")
    fig, ax = plt.subplots(figsize=(7, max(2, 0.4 * len(df))))
    methods = df["method"].tolist()
    vals = df["marker_specificity_mean"].fillna(0).to_numpy().reshape(-1, 1)
    im = ax.imshow(vals, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_xticks([0])
    ax.set_xticklabels(["specificity_mean"])
    fig.colorbar(im, ax=ax)
    ax.set_title("Marker specificity (mean per cell)")
    return fig


def plot_retention_vs_conflict(df: pd.DataFrame) -> plt.Figure:
    if "transcript_retention_fraction" not in df.columns or "conflict_mean" not in df.columns:
        return _figure_with_msg("retention/conflict not available")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["transcript_retention_fraction"], df["conflict_mean"], s=60)
    for _, row in df.iterrows():
        ax.annotate(
            row["method"],
            (row["transcript_retention_fraction"], row["conflict_mean"]),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.set_xlabel("Transcript retention fraction")
    ax.set_ylabel("NPMI conflict")
    ax.set_title("Retention vs. conflict")
    return fig


def plot_cell_vs_transcripts(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["cell_count"], df["assigned_transcript_count"], s=60)
    for _, row in df.iterrows():
        ax.annotate(
            row["method"],
            (row["cell_count"], row["assigned_transcript_count"]),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.set_xlabel("Cells")
    ax.set_ylabel("Assigned transcripts")
    ax.set_title("Cells vs. assigned transcripts")
    return fig


def plot_runtime(df: pd.DataFrame) -> plt.Figure:
    if "runtime_seconds" not in df.columns or df["runtime_seconds"].isna().all():
        return _figure_with_msg("runtime data not available")
    fig, ax = plt.subplots(figsize=(7, max(2, 0.4 * len(df))))
    ax.barh(df["method"], df["runtime_seconds"].fillna(0))
    ax.set_xlabel("Wall seconds")
    ax.set_title("Runtime per method")
    return fig


def render_report(df: pd.DataFrame, fig_paths: dict[str, Path], html_path: Path) -> None:
    rows = "".join(
        f"<tr>{''.join(f'<td>{v}</td>' for v in row)}</tr>" for row in df.itertuples(index=False)
    )
    header = "".join(f"<th>{c}</th>" for c in df.columns)

    imgs_html = []
    for name, p in fig_paths.items():
        # Embed PNG previews if available next to each PDF.
        png = p.with_suffix(".png")
        if png.exists():
            b64 = base64.b64encode(png.read_bytes()).decode("ascii")
            imgs_html.append(
                f'<figure><figcaption>{name}</figcaption><img src="data:image/png;base64,{b64}" /></figure>'
            )
        else:
            imgs_html.append(f"<figure><figcaption>{name}</figcaption><p>{p}</p></figure>")

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>TRACER benchmark report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 2em; }}
  table {{ border-collapse: collapse; margin-bottom: 2em; }}
  th, td {{ border: 1px solid #ddd; padding: 4px 8px; font-size: 12px; }}
  th {{ background: #f4f4f4; }}
  figure {{ margin: 0 0 1.5em 0; }}
  figcaption {{ font-weight: 600; }}
  img {{ max-width: 720px; border: 1px solid #eee; }}
</style></head>
<body>
<h1>TRACER benchmark report</h1>
<p>Generated by <code>plot_benchmark_summary.py</code>. PDF figures are saved alongside this report; PNG previews are embedded here for convenience.</p>
<h2>Metrics</h2>
<table><thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table>
<h2>Figures</h2>
{''.join(imgs_html)}
</body></html>
"""
    html_path.write_text(html, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render benchmark figures + HTML report.")
    p.add_argument("--metrics-csv", required=True)
    p.add_argument("--figures-dir", required=True)
    p.add_argument("--report-html", required=True)
    p.add_argument("--log", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.log:
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(args.log, "w", encoding="utf-8")
        sys.stdout = log_fh
        sys.stderr = log_fh

    df = pd.read_csv(args.metrics_csv)
    fig_dir = Path(args.figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig_makers = {
        "purity_conflict_paired.pdf": plot_purity_conflict,
        "marker_specificity_heatmap.pdf": plot_marker_heatmap,
        "transcript_retention_vs_conflict.pdf": plot_retention_vs_conflict,
        "cell_count_transcript_count.pdf": plot_cell_vs_transcripts,
        "runtime_summary.pdf": plot_runtime,
    }
    fig_paths = {}
    for name, fn in fig_makers.items():
        fig = fn(df)
        out_pdf = fig_dir / name
        fig.savefig(out_pdf, bbox_inches="tight")
        # PNG preview for the report.
        fig.savefig(out_pdf.with_suffix(".png"), dpi=120, bbox_inches="tight")
        plt.close(fig)
        fig_paths[name] = out_pdf

    render_report(df, fig_paths, Path(args.report_html))


if __name__ == "__main__":
    main()

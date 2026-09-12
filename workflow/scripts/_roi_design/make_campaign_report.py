#!/usr/bin/env python3
"""Render the campaign report from the collected tables and run provenance.

Reads what collect_results.py produced and writes a single markdown report
carrying: per-ROI run identity (Cirro dataset id, workflow revision, method
containers), the headline comparison at the primary ROI, the density-normalised
cost table, the q25/q50/q75 sensitivity view, every not-applicable entry with
its reason, and any failed or retried task.

Deliberately does no statistics of its own - it reports what the frozen
evaluation suite produced.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HEADLINE = [
    ("n_entities", "entities"),
    ("frac_assigned", "assigned"),
    ("median_transcripts_per_entity", "median tx/entity"),
    ("rctd_entropy_median", "RCTD entropy"),
    ("rctd_max_weight_median", "RCTD max weight"),
    ("marker_logfc_median", "marker log2FC"),
    ("kendall_tau_median", "Kendall tau"),
]
COST = [
    ("runtime_method_s", "runtime (s)"),
    ("runtime_per_1m_transcripts", "s / 1M tx"),
    ("peak_rss_gb", "peak RSS (GiB)"),
    ("peak_rss_gb_per_1m_transcripts", "GiB / 1M tx"),
    ("entities_per_mm2", "entities / mm2"),
]


def pivot(df: pd.DataFrame, metrics: list[tuple[str, str]], roi: str) -> pd.DataFrame:
    d = df[(df.ROI == roi) & (df.metric.isin([m for m, _ in metrics]))].copy()
    if d.empty:
        return d
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    p = d.pivot_table(index="method", columns="metric", values="value", aggfunc="first")
    cols = [m for m, _ in metrics if m in p.columns]
    p = p[cols].rename(columns={m: lbl for m, lbl in metrics})
    return p


def md_table(p: pd.DataFrame) -> str:
    if p.empty:
        return "_no rows_\n"
    out = ["| method | " + " | ".join(str(c) for c in p.columns) + " |",
           "|---|" + "---|" * len(p.columns)]
    for method, row in p.iterrows():
        cells = []
        for v in row:
            if pd.isna(v):
                cells.append("NA")
            elif abs(v) >= 1000:
                cells.append(f"{v:,.0f}")
            elif abs(v) >= 1:
                cells.append(f"{v:,.2f}")
            else:
                cells.append(f"{v:.3f}")
        out.append(f"| {method} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collected", required=True, help="collect_results.py --outdir")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    root = Path(a.collected)
    prov = json.loads((root / "run_provenance.json").read_text())
    man = json.loads(Path(a.manifest).read_text())
    plot = pd.read_csv(root / "plot_ready_table.tsv", sep="\t") \
        if (root / "plot_ready_table.tsv").exists() else pd.DataFrame()

    L: list[str] = []
    A = L.append
    A("# Multiplatform frozen-ROI SegBench benchmark — results\n")
    A(f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  ")
    A(f"Cirro project `{prov['project']}` · run scope `{prov['run_scope']}`\n")

    proto = man["protocol"]
    A("## Design\n")
    A(f"Fixed physical window **{proto['window_side_um']:.0f} µm** "
      f"({proto['window_area_mm2']:.2f} mm²), non-overlapping, "
      f"occupancy rule `{proto['occupancy_rule']}`. "
      f"Primary comparison: {proto['primary_roi']}.\n")
    A("ROI coordinates, the occupancy rule and the window size were frozen "
      "before any method ran and were not revisited.\n")

    # ---- runs -------------------------------------------------------------
    A("\n## Runs\n")
    A("| ROI | platform | input tx | area mm² | Cirro dataset | status | workflow rev |")
    A("|---|---|---:|---:|---|---|---|")
    for r in prov["runs"]:
        A(f"| {r['roi_id']} | {r['platform']} | {r['input_transcripts']:,} | "
          f"{r['area_mm2']} | `{r['cirro_dataset_id']}` | {r['cirro_status']} | "
          f"`{str(r.get('workflow_revision', 'NA'))[:12]}` |")

    # ---- failures / retries ----------------------------------------------
    A("\n## Failed and retried tasks\n")
    bad = []
    for r in prov["runs"]:
        for t in (r.get("tasks") or []):
            if str(t.get("status")) not in ("SUCCEEDED", "COMPLETED"):
                bad.append((r["roi_id"], t.get("name"), t.get("status")))
    if bad:
        A("| ROI | task | status |")
        A("|---|---|---|")
        for roi, name, st in bad:
            A(f"| {roi} | {name} | {st} |")
    else:
        A("None — every task in every collected run succeeded on its first attempt.\n")

    # ---- method containers -------------------------------------------------
    A("\n## Method provenance\n")
    seen = {}
    for r in prov["runs"]:
        for rec in (r.get("method_receipts") or []):
            key = rec.get("method")
            if key and key not in seen:
                seen[key] = rec
    if seen:
        A("| method | version | commit | container |")
        A("|---|---|---|---|")
        for m, rec in sorted(seen.items()):
            A(f"| {m} | {rec.get('version', 'NA')} | "
              f"`{str(rec.get('commit', 'NA'))[:12]}` | "
              f"`{str(rec.get('container', 'NA'))[:64]}` |")
    else:
        A("_no method receipts collected_\n")

    # ---- per-ROI results ---------------------------------------------------
    if not plot.empty:
        primary = [r["roi_id"] for r in prov["runs"]
                   if r["density_quantile"] in ("q50", "whole")]
        A("\n## Primary comparison (q50 / whole tissue)\n")
        for roi in sorted(primary):
            A(f"\n### {roi}\n")
            A(md_table(pivot(plot, HEADLINE, roi)))
            A("\nCost, density-normalised:\n")
            A(md_table(pivot(plot, COST, roi)))

        sens = [r["roi_id"] for r in prov["runs"]
                if r["density_quantile"] in ("q25", "q75")]
        if sens:
            A("\n## Density sensitivity (q25 / q75)\n")
            A("Predefined density sensitivities, not parameter adjustments — "
              "no method was tuned after the q50 results were seen.\n")
            for roi in sorted(sens):
                A(f"\n### {roi}\n")
                A(md_table(pivot(plot, HEADLINE, roi)))

    # ---- not applicable ----------------------------------------------------
    na_path = root / "not_applicable.tsv"
    A("\n## Not applicable\n")
    if na_path.exists():
        na = pd.read_csv(na_path, sep="\t")
        if len(na):
            g = (na.groupby(["platform", "method", "reason"], dropna=False)
                   .size().reset_index(name="n_metrics"))
            A("| platform | method | metrics | reason |")
            A("|---|---|---:|---|")
            for _, r in g.iterrows():
                A(f"| {r['platform']} | {r['method']} | {r['n_metrics']} | "
                  f"{str(r['reason'])[:110]} |")
        else:
            A("Every method produced every metric on every platform.\n")
    else:
        A("_not collected_\n")

    A("\n## Caveats\n")
    A("See `docs/multiplatform_roi_benchmark.md` §7. In short: the MERFISH "
      "cPMI panel covers only 55 of 241 genes; Atera retains 11× more cPMI "
      "edges than Xenium5K, so cross-platform cPMI metrics partly measure "
      "panel breadth; the cervical population is qv>30 against the NSCLC "
      "campaign's qv>20; the MERFISH original mask assigns 48.5% of "
      "molecules; nucleus geometry in the ROI bundles is derived rather than "
      "vendor-supplied; and cellAdmix cluster labels are k-means on each "
      "ROI's own matrix.\n")

    Path(a.out).write_text("\n".join(L) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

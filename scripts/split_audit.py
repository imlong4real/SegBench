#!/usr/bin/env python3
"""Produce the corrected comparison: native vs matched, plus held-out scoring.

    scripts/split_audit.py <runs_root> [--dataset nsclc_xenium]

Writes into ``<runs>/<dataset>/summary/``:

  ``cell_funnel.csv``          where cells are lost, per method
  ``corrected_comparison.csv`` every metric at native and matched scope
  ``split_audit.md``           the readable writeup

The point of the exercise is that the headline table scores each method on
whatever cells that method chose to emit. A method that declines the hard
cells is measured on an easier population, so its per-cell metrics improve
without anything about its output being better. The matched scope removes that
degree of freedom; the held-out columns remove the reference the purification
step optimised against.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from segbench import audit, evaluate as ev  # noqa: E402

CELL_H5AD_GLOBS = ("outputs/cell_by_gene*.h5ad", "outputs/*cell_by_gene*.h5ad")


def find_cell_h5ad(run_dir: Path) -> Path | None:
    """The method's cell-by-gene matrix, preferring the one RCTD was given."""
    prep = run_dir / "rctd" / "rctd_input_info.json"
    if prep.exists():
        try:
            p = Path(json.loads(prep.read_text()).get("rctd_input_h5ad", ""))
            if p.exists():
                return p
        except Exception:
            pass
    for pat in CELL_H5AD_GLOBS:
        hits = sorted(run_dir.glob(pat))
        # split writes both purified and original; the purified one is the output
        hits = [h for h in hits if "original" not in h.name]
        if hits:
            return hits[0]
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs_root", type=Path)
    ap.add_argument("--dataset", default="nsclc_xenium")
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--celltype-col", default="Cell_Cluster_level1")
    ap.add_argument("--heldout-col", default="id")
    ap.add_argument("--heldout-value", default="Validation")
    ap.add_argument("--cohort-col", default="Study")
    ap.add_argument("--baseline", default="baseline_10x")
    args = ap.parse_args()

    root = args.runs_root / args.dataset / "methods"
    summary = args.runs_root / args.dataset / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    methods = sorted(d.name for d in root.iterdir()
                     if d.is_dir() and (d / "benchmark_stats.json").exists())
    print(f"methods found: {methods}")

    # ---- cell-id universe: the vendor segmentation ------------------------
    import anndata as ad
    base_h5ad = find_cell_h5ad(root / args.baseline)
    if base_h5ad is None:
        print(f"!! no baseline at {root/args.baseline}; matched scope needs it")
        return 2
    vendor_ids = list(map(str, ad.read_h5ad(base_h5ad, backed="r").obs_names))
    print(f"vendor cell-id universe: {len(vendor_ids)}")

    comparable, why = audit.comparable_methods(root, methods, vendor_ids)
    print(f"comparable (share the vendor id space): {comparable}")
    for m, r in why.items():
        print(f"  excluded {m}: {r}")

    matched = audit.matched_cell_set(root, comparable)
    print(f"matched cell set (scored for ALL comparable methods): {len(matched)}")

    # ---- funnels ----------------------------------------------------------
    funnels = []
    for m in methods:
        f = audit.Funnel(m)
        f.add("vendor segmentation", len(vendor_ids))
        h = find_cell_h5ad(root / m)
        if h is not None:
            f.add("method output profiles", ad.read_h5ad(h, backed="r").shape[0])
        pc = audit.read_per_cell(root / m)
        if pc is not None:
            f.add("scored by RCTD", len(pc))
        funnels.append(f.to_frame())
    fun = pd.concat(funnels, ignore_index=True)
    fun.to_csv(summary / "cell_funnel.csv", index=False)
    print(f"wrote {summary/'cell_funnel.csv'}")

    # ---- references -------------------------------------------------------
    panel = list(map(str, ad.read_h5ad(base_h5ad, backed="r").var_names))
    full_ref = ev._load_reference(args.reference, panel)
    kept, dropped = ev.common_celltypes(args.reference, args.celltype_col,
                                        ev.MIN_REFERENCE_CELLS)
    print(f"cell types kept: {len(kept)}  dropped: {dropped}")
    disj = audit.describe_disjointness(args.reference, split_col=args.heldout_col,
                                       heldout_value=args.heldout_value,
                                       cohort_col=args.cohort_col)
    print(f"held-out cohort disjointness: {disj}")
    held_ref, held_info = audit.heldout_reference(
        args.reference, split_col=args.heldout_col,
        heldout_value=args.heldout_value, panel=panel)
    print(f"held-out reference: {held_info}")

    # ---- the corrected table ---------------------------------------------
    rows = []
    for m in methods:
        h = find_cell_h5ad(root / m)
        for scope, restrict in (("native", None),
                                ("matched", matched if m in comparable else None)):
            if scope == "matched" and m not in comparable:
                rows.append({"dataset": args.dataset, "method": m, "scope": "matched",
                             "not_comparable_reason": why.get(m, "")})
                continue
            r = {"dataset": args.dataset, "method": m, "scope": scope}
            r.update(audit.per_cell_medians(root, m, restrict=restrict))
            if h is not None:
                for tag, reference in (("", full_ref), ("_heldout", held_ref)):
                    got = audit.pseudobulk_metrics(
                        cell_h5ad=h, run_root=root, method=m, reference=reference,
                        celltype_col=args.celltype_col, kept_types=kept,
                        restrict=restrict)
                    for k, v in got.items():
                        r[f"{k}{tag}"] = v
            rows.append(r)
            print(f"  scored {m} [{scope}]")

    out = pd.DataFrame(rows)
    from segbench.report import method_labels
    out.insert(2, "display_name", method_labels(out["method"].astype(str)))
    lead = ["dataset", "method", "display_name", "scope", "n_cells_scored",
            "rctd_entropy_median", "rctd_max_weight_median",
            "kendall_tau_median", "marker_logfc_median",
            "kendall_tau_median_heldout", "marker_logfc_median_heldout",
            "frac_singlet", "frac_reject"]
    out = out[[c for c in lead if c in out.columns] +
              [c for c in out.columns if c not in lead]]
    out.to_csv(summary / "corrected_comparison.csv", index=False)
    print(f"wrote {summary/'corrected_comparison.csv'}  ({len(out)} rows)")

    (summary / "split_audit_context.json").write_text(json.dumps({
        "vendor_cells": len(vendor_ids), "matched_cells": len(matched),
        "comparable_methods": comparable, "not_comparable": why,
        "celltypes_kept": kept, "celltypes_dropped": dropped,
        "heldout": held_info, "heldout_disjointness": disj,
        "reference": str(args.reference),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

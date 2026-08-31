#!/usr/bin/env python3
"""Produce the corrected comparison: native-output vs matched-cell metrics.

The headline table scores each method on whatever cells that method chose to
emit. That is not a like-for-like comparison: a purification method which
declines to process low-depth cells is measured on an easier population than a
method that keeps them, and every per-cell metric improves as a result. This
script reports both views side by side so the reader can see how much of a
method's lead survives holding the cell population fixed.

It also re-scores the pseudo-bulk metrics against study-disjoint held-out
donors, because the reference used for evaluation is the same one SPLIT
consumes during purification.

    scripts/audit_selection_bias.py [--runs DIR] [--dataset nsclc_xenium]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from segbench import audit, evaluate as ev, registry  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path,
                    default=Path("/scratch4/adeshpa6/segbench_runs"))
    ap.add_argument("--dataset", default="nsclc_xenium")
    ap.add_argument("--reference", type=Path, default=Path(
        "/home/lyuan13/scr4_adeshpa6/TRACER/datasets/dataset/"
        "lung_cancer_scrna_10x/lung_cancer_50k.h5ad"))
    ap.add_argument("--celltype-col", default="Cell_Cluster_level1")
    ap.add_argument("--heldout-col", default="id")
    ap.add_argument("--heldout-value", default="Validation")
    ap.add_argument("--cohort-col", default="Study")
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()

    root = args.runs / args.dataset / "methods"
    outdir = args.outdir or (args.runs / args.dataset / "audit")
    outdir.mkdir(parents=True, exist_ok=True)

    methods = sorted(d.name for d in root.iterdir()
                     if d.is_dir() and (d / audit.RCTD_PER_CELL).exists())
    print(f"methods with a cached RCTD table: {', '.join(methods)}")

    # ---- vendor id space -------------------------------------------------
    import anndata as ad
    base_h5 = root / "baseline_10x" / "outputs" / "cell_by_gene_baseline.h5ad"
    if not base_h5.exists():
        print(f"!! baseline missing at {base_h5}", file=sys.stderr)
        return 2
    vendor_ids = list(map(str, ad.read_h5ad(base_h5, backed="r").obs_names))
    ok, why = audit.comparable_methods(root, methods, vendor_ids)
    print(f"\ncomparable (share the vendor cell-id space): {', '.join(ok)}")
    for m, r in why.items():
        print(f"  excluded {m}: {r}")

    matched = audit.matched_cell_set(root, ok)
    print(f"\nmatched cell set (scored for every comparable method): {len(matched)}")

    # ---- references ------------------------------------------------------
    kept, dropped = ev.common_celltypes(args.reference, args.celltype_col,
                                        ev.MIN_REFERENCE_CELLS)
    print(f"cell types kept: {len(kept)}  dropped(<{ev.MIN_REFERENCE_CELLS} cells): {dropped}")

    panel = list(map(str, ad.read_h5ad(base_h5, backed="r").var_names))
    full_ref = ev._load_reference(args.reference, panel)
    disj = audit.describe_disjointness(args.reference, split_col=args.heldout_col,
                                       heldout_value=args.heldout_value,
                                       cohort_col=args.cohort_col)
    print(f"\nheld-out slice {args.heldout_col}={args.heldout_value}: "
          f"cohorts {disj['heldout_cohorts']} | disjoint from the rest: {disj['is_disjoint']}")
    held_mask = (full_ref.obs[args.heldout_col].astype(str)
                 == str(args.heldout_value)).to_numpy()
    held_ref = full_ref[held_mask].copy()
    print(f"held-out reference cells: {held_ref.shape[0]}")
    held_kept = [t for t in kept
                 if (held_ref.obs[args.celltype_col].astype(str) == t).sum()
                 >= ev.MIN_REFERENCE_CELLS]
    print(f"cell types with >= {ev.MIN_REFERENCE_CELLS} held-out cells: {len(held_kept)}")

    # ---- per method ------------------------------------------------------
    rows, funnels = [], []
    for m in methods:
        spec = registry.METHODS.get(m)
        label = spec.label if spec else m
        cell_h5 = next((p for p in sorted((root / m / "outputs").glob("*.h5ad"))
                        if "rctd_input" not in p.name), None)
        prep = root / m / "rctd" / "rctd_input.h5ad"
        scored_h5 = prep if prep.exists() else cell_h5
        row: dict = {"dataset": args.dataset, "method": m, "method_label": label,
                     "comparable": m in ok}

        row.update({f"native_{k}": v
                    for k, v in audit.per_cell_medians(root, m).items()})
        if m in ok:
            row.update({f"matched_{k}": v for k, v in
                        audit.per_cell_medians(root, m, restrict=matched).items()})

        if scored_h5 is not None and scored_h5.exists():
            nat = audit.pseudobulk_metrics(
                cell_h5ad=scored_h5, run_root=root, method=m, reference=full_ref,
                celltype_col=args.celltype_col, kept_types=kept)
            row.update({f"native_{k}": v for k, v in nat.items()})
            hel = audit.pseudobulk_metrics(
                cell_h5ad=scored_h5, run_root=root, method=m, reference=held_ref,
                celltype_col=args.celltype_col, kept_types=held_kept)
            row.update({f"heldout_{k}": v for k, v in hel.items()})
            if m in ok:
                mat = audit.pseudobulk_metrics(
                    cell_h5ad=scored_h5, run_root=root, method=m, reference=full_ref,
                    celltype_col=args.celltype_col, kept_types=kept, restrict=matched)
                row.update({f"matched_{k}": v for k, v in mat.items()})
                mh = audit.pseudobulk_metrics(
                    cell_h5ad=scored_h5, run_root=root, method=m, reference=held_ref,
                    celltype_col=args.celltype_col, kept_types=held_kept,
                    restrict=matched)
                row.update({f"matched_heldout_{k}": v for k, v in mh.items()})
        rows.append(row)
        print(f"  scored {label}")

        f = audit.Funnel(label)
        f.add("vendor segmentation", len(vendor_ids))
        outs = root / m / "outputs"
        emitted = None
        for p in sorted(outs.glob("*.h5ad")):
            if "rctd_input" in p.name or "original" in p.name:
                continue
            emitted = ad.read_h5ad(p, backed="r").shape[0]
            break
        if emitted is not None:
            f.add("emitted by method", emitted)
        pc = audit.read_per_cell(root / m)
        if pc is not None:
            f.add("scored by benchmark RCTD", len(pc))
        funnels.append(f.to_frame())

    df = pd.DataFrame(rows)
    lead = ["dataset", "method", "method_label", "comparable"]
    df = df[lead + [c for c in df.columns if c not in lead]]
    csv = outdir / "corrected_comparison.csv"
    df.to_csv(csv, index=False)
    fun = pd.concat(funnels, ignore_index=True)
    fun.to_csv(outdir / "cell_funnel.csv", index=False)
    (outdir / "audit_provenance.json").write_text(json.dumps({
        "dataset": args.dataset, "n_matched_cells": len(matched),
        "comparable_methods": ok, "excluded_methods": why,
        "heldout": {**disj, "col": args.heldout_col, "value": args.heldout_value,
                    "n_cells": int(held_ref.shape[0]),
                    "celltypes_scored": held_kept},
        "celltypes_kept": kept, "celltypes_dropped": dropped,
    }, indent=2))
    print(f"\nWrote {csv}")
    print(f"Wrote {outdir/'cell_funnel.csv'}")
    print(f"Wrote {outdir/'audit_provenance.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

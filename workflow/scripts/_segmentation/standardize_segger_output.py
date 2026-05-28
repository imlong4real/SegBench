#!/usr/bin/env python3
"""Convert Segger predict_fast.py output to the benchmark transcript schema.

Input (segger_transcripts.parquet) columns:
  transcript_id, cell_id (10x original), overlaps_nucleus, feature_name,
  x_location, y_location, z_location, qv, fov_name, nucleus_distance,
  score, segger_cell_id, bound

Output (segger_transcripts_standardized.parquet) columns:
  Required : x, y, feature_name, cell_id   (cell_id = segger_cell_id)
  Optional : z, transcript_id, qv, overlaps_nucleus
  Added    : method = "Segger"

The output is compatible with workflow/scripts/get_metric.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REQUIRED_IN = {"feature_name", "x_location", "y_location", "segger_cell_id"}
SCHEMA_REPORT_KEYS = [
    "n_transcripts_total",
    "n_transcripts_assigned",
    "n_transcripts_unassigned",
    "frac_assigned",
    "n_unique_cells",
    "columns_in",
    "columns_out",
    "schema_valid",
]

UNASSIGNED_TOKENS = frozenset({
    "UNASSIGNED", "Unassigned", "unassigned",
    "DROP", "nan", "None", "", "0", "-1", "NA",
})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path,
                   help="segger_transcripts.parquet from predict_fast.py")
    p.add_argument("--output", required=True, type=Path,
                   help="Destination for segger_transcripts_standardized.parquet")
    p.add_argument("--report", type=Path, default=None,
                   help="Optional JSON schema-validation report")
    p.add_argument("--method-label", default="Segger",
                   help="Value for the 'method' constant column (default: Segger)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    print(f"[standardize] Reading: {args.input}")
    df = pd.read_parquet(args.input)
    cols_in = list(df.columns)
    print(f"[standardize] Columns in: {cols_in}")
    print(f"[standardize] Shape: {df.shape}")

    missing = REQUIRED_IN - set(cols_in)
    if missing:
        print(f"ERROR: required input columns missing: {missing}", file=sys.stderr)
        return 1

    # ── Build standardized dataframe ──────────────────────────────────────────
    out = pd.DataFrame()

    # Mandatory columns
    out["x"] = df["x_location"].astype("float32")
    out["y"] = df["y_location"].astype("float32")
    out["feature_name"] = df["feature_name"].astype(str)
    out["cell_id"] = df["segger_cell_id"].astype(str)

    # Optional columns (include when present in source)
    if "z_location" in df.columns:
        out["z"] = df["z_location"].astype("float32")
    if "transcript_id" in df.columns:
        out["transcript_id"] = df["transcript_id"]
    if "qv" in df.columns:
        out["qv"] = df["qv"].astype("float32")
    if "overlaps_nucleus" in df.columns:
        out["overlaps_nucleus"] = df["overlaps_nucleus"]

    # Method label
    out["method"] = args.method_label

    # ── Schema validation ─────────────────────────────────────────────────────
    required_out = {"x", "y", "feature_name", "cell_id"}
    schema_valid = required_out.issubset(set(out.columns))

    assigned_mask = ~out["cell_id"].isin(UNASSIGNED_TOKENS)
    n_total = len(out)
    n_assigned = int(assigned_mask.sum())
    n_unassigned = n_total - n_assigned
    n_unique_cells = int(out.loc[assigned_mask, "cell_id"].nunique())

    report = {
        "n_transcripts_total": n_total,
        "n_transcripts_assigned": n_assigned,
        "n_transcripts_unassigned": n_unassigned,
        "frac_assigned": round(n_assigned / max(1, n_total), 4),
        "n_unique_cells": n_unique_cells,
        "columns_in": cols_in,
        "columns_out": list(out.columns),
        "schema_valid": schema_valid,
        "method": args.method_label,
        "input_path": str(args.input),
        "output_path": str(args.output),
    }

    print(f"[standardize] Assigned: {n_assigned}/{n_total} ({100*n_assigned/max(1,n_total):.1f}%)")
    print(f"[standardize] Unique cells: {n_unique_cells}")
    print(f"[standardize] Schema valid: {schema_valid}")
    print(f"[standardize] Output columns: {list(out.columns)}")

    # ── Write output ──────────────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    print(f"[standardize] Wrote: {args.output}")

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[standardize] Report: {args.report}")

    if not schema_valid:
        print(f"ERROR: schema validation failed — missing: {required_out - set(out.columns)}",
              file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Compute marker-set specificity / leakage for one standardized output."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from _metrics_common import (  # noqa: E402
    load_standardized_dir,
    marker_specificity,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Marker specificity / leakage for one standardized output.")
    p.add_argument("--standardized-dir", required=True)
    p.add_argument("--marker-set", required=True)
    p.add_argument("--out-per-cell", required=True)
    p.add_argument("--out-summary", required=True)
    p.add_argument("--log", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.log:
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(args.log, "w", encoding="utf-8")
        sys.stdout = log_fh
        sys.stderr = log_fh

    data = load_standardized_dir(args.standardized_dir)
    cbg = data["cell_by_gene"]
    feats = data["features"]
    barcodes = data["barcodes"]
    info = data["method_info"]

    with open(args.marker_set, "r", encoding="utf-8") as f:
        marker_groups = yaml.safe_load(f) or {}

    per_cell = marker_specificity(cbg, feats, marker_groups)
    if not per_cell.empty:
        per_cell.insert(0, "cell_id_method", [barcodes[i] for i in per_cell["cell_row"]])
    per_cell.insert(0, "method", info["method_name"])
    Path(args.out_per_cell).parent.mkdir(parents=True, exist_ok=True)
    per_cell.to_parquet(args.out_per_cell, index=False)

    if per_cell.empty:
        summary = {
            "method": info["method_name"],
            "n_cells_with_markers": 0,
            "marker_specificity_mean": float("nan"),
            "marker_specificity_median": float("nan"),
            "marker_leakage_mean": float("nan"),
            "marker_leakage_median": float("nan"),
        }
    else:
        summary = {
            "method": info["method_name"],
            "n_cells_with_markers": int(len(per_cell)),
            "marker_specificity_mean": float(per_cell["specificity"].mean()),
            "marker_specificity_median": float(per_cell["specificity"].median()),
            "marker_leakage_mean": float(per_cell["leakage"].mean()),
            "marker_leakage_median": float(per_cell["leakage"].median()),
        }
    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_parquet(args.out_summary, index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

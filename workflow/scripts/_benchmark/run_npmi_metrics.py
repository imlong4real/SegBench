#!/usr/bin/env python3
"""Compute NPMI-based purity / conflict metrics for one standardized output."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from _metrics_common import (  # noqa: E402
    cell_purity_conflict_from_npmi,
    cooccurrence_npmi,
    load_standardized_dir,
    try_import_tracer_npmi,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute NPMI purity/conflict metrics for one method.")
    p.add_argument("--standardized-dir", required=True)
    p.add_argument("--out-per-cell", required=True)
    p.add_argument("--out-summary", required=True)
    p.add_argument("--tracer-repo-path", default=None)
    p.add_argument("--min-molecules-per-cell", type=int, default=10)
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

    # NPMI table.
    tracer_npmi = try_import_tracer_npmi(args.tracer_repo_path)
    if tracer_npmi is not None:
        try:
            npmi_long = tracer_npmi(cbg, features=feats)
            print(f"[npmi] Using TRACER NPMI: shape={npmi_long.shape}")
        except Exception as e:
            print(f"[npmi] TRACER NPMI failed ({e}); falling back to cooccurrence_npmi.")
            npmi_long, _ = cooccurrence_npmi(cbg)
    else:
        npmi_long, _ = cooccurrence_npmi(cbg)

    per_cell = cell_purity_conflict_from_npmi(
        cbg,
        npmi_long,
        feats,
        min_molecules_per_cell=args.min_molecules_per_cell,
    )
    per_cell.insert(0, "cell_id_method", [barcodes[i] for i in per_cell["cell_row"]])
    per_cell.insert(0, "method", info["method_name"])

    Path(args.out_per_cell).parent.mkdir(parents=True, exist_ok=True)
    per_cell.to_parquet(args.out_per_cell, index=False)

    if per_cell.empty:
        summary = {
            "method": info["method_name"],
            "n_cells_scored": 0,
            "purity_mean": float("nan"),
            "purity_median": float("nan"),
            "conflict_mean": float("nan"),
            "conflict_median": float("nan"),
            "signal_strength": float("nan"),
        }
    else:
        summary = {
            "method": info["method_name"],
            "n_cells_scored": int(len(per_cell)),
            "purity_mean": float(per_cell["purity"].mean()),
            "purity_median": float(per_cell["purity"].median()),
            "conflict_mean": float(per_cell["conflict"].mean()),
            "conflict_median": float(per_cell["conflict"].median()),
            "signal_strength": float(per_cell["n_transcripts"].median()),
        }

    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_parquet(args.out_summary, index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

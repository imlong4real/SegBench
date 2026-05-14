"""
Create a tiny Segger smoke-test input from an existing Xenium parquet transcript file.

Writes:
  - <output>           : filtered transcripts parquet (up to --n rows)
  - <summary>          : JSON summary of the subset
  - <xenium_bundle>/   : minimal Xenium bundle directory containing:
      transcripts.parquet
      nucleus_boundaries.parquet  (synthetic, from transcript clusters)
"""

import argparse
import json
import os
import sys

import pandas as pd
import numpy as np


COORD_CANDIDATES = [
    ("x_location", "y_location"),
    ("global_x", "global_y"),
    ("x", "y"),
]

GENE_CANDIDATES = ["feature_name", "gene", "gene_name"]


def parse_args():
    p = argparse.ArgumentParser(description="Create tiny Segger smoke-test input")
    p.add_argument(
        "--input",
        required=True,
        help="Path to source transcripts parquet (e.g. lung_cancer_df.parquet)",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Path to write subset transcripts parquet",
    )
    p.add_argument(
        "--summary",
        required=True,
        help="Path to write JSON summary",
    )
    p.add_argument(
        "--xenium_bundle",
        default=None,
        help="Directory to write minimal Xenium bundle (default: <output_dir>/xenium_bundle)",
    )
    p.add_argument(
        "--n",
        type=int,
        default=100_000,
        help="Max number of transcripts to include (default: 100000)",
    )
    return p.parse_args()


def detect_columns(df):
    x_col = y_col = gene_col = None
    for xc, yc in COORD_CANDIDATES:
        if xc in df.columns and yc in df.columns:
            x_col, y_col = xc, yc
            break
    for gc in GENE_CANDIDATES:
        if gc in df.columns:
            gene_col = gc
            break
    return x_col, y_col, gene_col


def make_nucleus_boundaries(transcripts, x_col, y_col, gene_col):
    """Create nucleus boundaries from actual cell_id assignments.

    Groups transcripts by cell_id and approximates each cell nucleus as a circle
    of radius = max(1.5 × transcript spread, 8 µm) around the centroid.
    Falls back to a synthetic 7×7 grid with 30 µm radius if cell_id is not
    available or too few cells are present.
    """
    n_pts = 8
    angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)

    if "cell_id" in transcripts.columns:
        cid_col = transcripts["cell_id"].astype(str)
        with_cell = transcripts[~cid_col.isin(["", "nan", "None", "0"])]
        unique_cells = with_cell["cell_id"].astype(str).unique()
    else:
        unique_cells = np.array([])

    if len(unique_cells) >= 5:
        records = []
        for cid in unique_cells[:300]:   # cap at 300 nuclei for smoke test
            mask = with_cell["cell_id"].astype(str) == cid
            cx_pts = with_cell.loc[mask, x_col].values.astype(float)
            cy_pts = with_cell.loc[mask, y_col].values.astype(float)
            cx0, cy0 = cx_pts.mean(), cy_pts.mean()
            spread = float(np.sqrt(((cx_pts - cx0) ** 2 + (cy_pts - cy0) ** 2).mean()))
            radius = max(spread * 1.5, 8.0)
            radius = min(radius, 40.0)
            for angle in angles:
                records.append(
                    {
                        "cell_id": str(cid),
                        "vertex_x": float(cx0 + radius * np.cos(angle)),
                        "vertex_y": float(cy0 + radius * np.sin(angle)),
                    }
                )
        if records:
            return pd.DataFrame(records)

    # Fallback: uniform grid with large-enough radius to capture nearby transcripts
    xs = transcripts[x_col].values.astype(float)
    ys = transcripts[y_col].values.astype(float)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    n_grid = 7  # 7x7 = 49 nuclei
    cx = np.linspace(x_min + 20, x_max - 20, n_grid)
    cy = np.linspace(y_min + 20, y_max - 20, n_grid)

    records = []
    cell_id = 1
    radius = 30.0  # 30-µm radius so grid circles actually contain transcripts

    for x0 in cx:
        for y0 in cy:
            for angle in angles:
                records.append(
                    {
                        "cell_id": str(cell_id),
                        "vertex_x": float(x0 + radius * np.cos(angle)),
                        "vertex_y": float(y0 + radius * np.sin(angle)),
                    }
                )
            cell_id += 1

    return pd.DataFrame(records)


def main():
    args = parse_args()

    print(f"Reading {args.input} ...")
    df = pd.read_parquet(args.input)
    print(f"  columns: {df.columns.tolist()}")
    print(f"  rows   : {len(df)}")

    x_col, y_col, gene_col = detect_columns(df)
    if x_col is None:
        sys.exit(f"ERROR: Cannot find coordinate columns. Have: {df.columns.tolist()}")
    if gene_col is None:
        sys.exit(f"ERROR: Cannot find gene column. Have: {df.columns.tolist()}")

    print(f"  x_col  : {x_col}")
    print(f"  y_col  : {y_col}")
    print(f"  gene   : {gene_col}")

    # Subset: take a spatial ROI from the centre of the dataset to keep
    # transcripts dense enough for Segger's tile-graph construction.
    x_vals = df[x_col]
    y_vals = df[y_col]
    x_mid = (x_vals.min() + x_vals.max()) / 2
    y_mid = (y_vals.min() + y_vals.max()) / 2
    span = 500  # 500-micron box around centre

    roi_mask = (
        (x_vals >= x_mid - span) & (x_vals <= x_mid + span) &
        (y_vals >= y_mid - span) & (y_vals <= y_mid + span)
    )
    subset = df[roi_mask]

    if len(subset) > args.n:
        subset = subset.sample(n=args.n, random_state=42)
    elif len(subset) < 1000:
        # ROI too sparse — fall back to first N rows
        print("  WARNING: ROI too sparse, falling back to first N rows")
        subset = df.head(args.n)

    print(f"  subset : {len(subset)} rows")

    # Write subset parquet
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    subset.to_parquet(args.output, index=False)
    print(f"  wrote  : {args.output}")

    # Write summary JSON
    summary = {
        "source_file": os.path.abspath(args.input),
        "source_rows": int(len(df)),
        "subset_rows": int(len(subset)),
        "columns": df.columns.tolist(),
        "x_col": x_col,
        "y_col": y_col,
        "gene_col": gene_col,
        "x_range": [float(subset[x_col].min()), float(subset[x_col].max())],
        "y_range": [float(subset[y_col].min()), float(subset[y_col].max())],
        "n_unique_genes": int(subset[gene_col].nunique()),
        "output_file": os.path.abspath(args.output),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.summary)), exist_ok=True)
    with open(args.summary, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  wrote  : {args.summary}")

    # Build minimal Xenium bundle for Segger's create_dataset_fast.py
    xenium_bundle = args.xenium_bundle
    if xenium_bundle is None:
        xenium_bundle = os.path.join(os.path.dirname(args.output), "xenium_bundle")
    os.makedirs(xenium_bundle, exist_ok=True)

    # Transcripts: rename to standard Xenium column names if needed
    tx = subset.copy()
    rename_map = {}
    if x_col != "x_location":
        rename_map[x_col] = "x_location"
    if y_col != "y_location":
        rename_map[y_col] = "y_location"
    if "z" in tx.columns and "z_location" not in tx.columns:
        rename_map["z"] = "z_location"
    if gene_col != "feature_name":
        rename_map[gene_col] = "feature_name"
    if rename_map:
        tx = tx.rename(columns=rename_map)

    # Ensure required Xenium transcript columns exist
    if "transcript_id" not in tx.columns:
        tx["transcript_id"] = range(len(tx))
    if "qv" not in tx.columns:
        tx["qv"] = 20.0
    if "overlaps_nucleus" not in tx.columns:
        tx["overlaps_nucleus"] = 0
    if "cell_id" not in tx.columns:
        tx["cell_id"] = ""
    if "z_location" not in tx.columns:
        tx["z_location"] = 0.0

    # Cast categorical columns to plain strings; PyArrow's match_substring_regex
    # does not support dictionary-encoded types in the container's PyArrow version.
    for col in tx.select_dtypes(include="category").columns:
        tx[col] = tx[col].astype(str)

    tx_path = os.path.join(xenium_bundle, "transcripts.parquet")
    tx.to_parquet(tx_path, index=False)
    print(f"  wrote  : {tx_path}")

    # Nucleus boundaries (synthetic)
    nb = make_nucleus_boundaries(subset, x_col, y_col, gene_col)
    nb_path = os.path.join(xenium_bundle, "nucleus_boundaries.parquet")
    nb.to_parquet(nb_path, index=False)
    print(f"  wrote  : {nb_path} ({len(nb)} boundary vertices, {nb['cell_id'].nunique()} nuclei)")

    # Write experiment.xenium metadata stub
    xmeta = {
        "num_cells": int(nb["cell_id"].nunique()),
        "pixel_size": 1.0,
        "region_name": "smoke_test",
        "run_name": "smoke_test",
    }
    with open(os.path.join(xenium_bundle, "experiment.xenium"), "w") as fh:
        json.dump(xmeta, fh, indent=2)

    summary["xenium_bundle"] = os.path.abspath(xenium_bundle)
    with open(args.summary, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nDone. Xenium bundle: {xenium_bundle}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

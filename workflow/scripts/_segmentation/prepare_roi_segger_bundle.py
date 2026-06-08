#!/usr/bin/env python3
"""Build a Xenium-format input bundle for Segger from a benchmark roi_transcripts.parquet.

roi_transcripts.parquet columns (benchmark standard):
    x, y, z, feature_name, cell_id, transcript_id, overlaps_nucleus,
    platform, sample, roi_id

Outputs written to --outdir/<dataset>/:
    transcripts.parquet       (Xenium column names: x_location, y_location, ...)
    nucleus_boundaries.parquet (derived from overlaps_nucleus transcripts via
                                convex hull; fallback circle for sparse cells)
    experiment.xenium         (JSON metadata stub required by Segger)

Usage
-----
    python prepare_roi_segger_bundle.py \\
        --input  /path/to/roi_transcripts.parquet \\
        --outdir results/segger_roi/atera_cervical/xenium_bundle \\
        --dataset atera_cervical \\
        --platform Atera
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path,
                   help="roi_transcripts.parquet from benchmark_data/<dataset>/")
    p.add_argument("--outdir", required=True, type=Path,
                   help="Output directory for the Xenium-like bundle")
    p.add_argument("--dataset", required=True,
                   help="Dataset name tag (e.g. atera_cervical)")
    p.add_argument("--platform", default=None,
                   help="Platform string written to experiment.xenium (optional)")
    p.add_argument("--min-nuc-transcripts", type=int, default=1,
                   help="Min nucleus transcripts per cell to use convex hull (default 1)")
    p.add_argument("--fallback-radius", type=float, default=3.0,
                   help="Circle radius (same units as x/y) for cells with too few "
                        "nucleus transcripts (default 3.0)")
    p.add_argument("--circle-n-pts", type=int, default=12,
                   help="Number of polygon vertices for fallback circles (default 12)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Nucleus boundary helpers
# ---------------------------------------------------------------------------

def _circle_polygon(cx: float, cy: float, r: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return cx + r * np.cos(angles), cy + r * np.sin(angles)


def _convex_hull_vertices(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import ConvexHull
    if len(pts) < 3:
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
        return _circle_polygon(cx, cy, r=2.0, n=8)
    try:
        hull = ConvexHull(pts)
        verts = pts[hull.vertices]
        return verts[:, 0].astype(np.float32), verts[:, 1].astype(np.float32)
    except Exception:
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
        return _circle_polygon(cx, cy, r=2.0, n=8)


def build_nucleus_boundaries(
    df: pd.DataFrame,
    *,
    min_nuc_tx: int,
    fallback_radius: float,
    circle_n_pts: int,
) -> pd.DataFrame:
    """
    Build nucleus_boundaries.parquet (cell_id, vertex_x, vertex_y) from
    transcript positions where overlaps_nucleus == 1.
    Falls back to a small circle for cells with fewer nucleus transcripts
    than min_nuc_tx.
    """
    all_cell_ids = df["cell_id"].astype(str).unique()
    all_cell_ids = [c for c in all_cell_ids
                    if c not in {"UNASSIGNED", "nan", "", "None", "0", "-1"}]

    # Group nucleus transcripts per cell
    nuc_mask = (df["overlaps_nucleus"].fillna(0).astype(int) == 1)
    nuc_df = df.loc[nuc_mask, ["cell_id", "x", "y"]].copy()
    nuc_df["cell_id"] = nuc_df["cell_id"].astype(str)
    nuc_by_cell = nuc_df.groupby("cell_id", observed=True)

    # Fallback: all transcripts for centroid (for cells with no nucleus tx)
    all_df = df[["cell_id", "x", "y"]].copy()
    all_df["cell_id"] = all_df["cell_id"].astype(str)
    centroid_by_cell = all_df.groupby("cell_id", observed=True)[["x", "y"]].mean()

    rows_cell_id: list[str] = []
    rows_vx: list[np.ndarray] = []
    rows_vy: list[np.ndarray] = []

    for cid in all_cell_ids:
        if cid in nuc_by_cell.groups and len(nuc_by_cell.get_group(cid)) >= min_nuc_tx:
            grp = nuc_by_cell.get_group(cid)
            pts = grp[["x", "y"]].to_numpy(dtype=np.float64)
            if len(pts) == 1:
                vx, vy = _circle_polygon(pts[0, 0], pts[0, 1], fallback_radius, circle_n_pts)
            elif len(pts) == 2:
                # Use midpoint + small circle
                cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
                vx, vy = _circle_polygon(cx, cy, fallback_radius, circle_n_pts)
            else:
                vx, vy = _convex_hull_vertices(pts)
        else:
            # Fallback: use centroid of all transcripts for this cell
            if cid in centroid_by_cell.index:
                cx = float(centroid_by_cell.loc[cid, "x"])
                cy = float(centroid_by_cell.loc[cid, "y"])
            else:
                continue
            vx, vy = _circle_polygon(cx, cy, fallback_radius, circle_n_pts)

        n = len(vx)
        rows_cell_id.extend([cid] * n)
        rows_vx.append(vx.astype(np.float32))
        rows_vy.append(vy.astype(np.float32))

    nuc_bounds = pd.DataFrame({
        "cell_id": rows_cell_id,
        "vertex_x": np.concatenate(rows_vx).astype(np.float32),
        "vertex_y": np.concatenate(rows_vy).astype(np.float32),
    })
    return nuc_bounds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    print(f"[bundle] Reading: {args.input}")
    df = pd.read_parquet(args.input)
    print(f"[bundle] Shape: {df.shape}, columns: {list(df.columns)}")

    # Validate required columns
    required = {"x", "y", "feature_name", "cell_id", "overlaps_nucleus"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: Missing required columns: {missing}", file=sys.stderr)
        return 1

    args.outdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Write transcripts.parquet (Xenium column names)
    # ------------------------------------------------------------------
    tx = pd.DataFrame()
    # Required by Segger create_dataset_fast.py
    tx["x_location"] = df["x"].astype(np.float32)
    tx["y_location"] = df["y"].astype(np.float32)
    if "z" in df.columns:
        tx["z_location"] = df["z"].astype(np.float32)
    else:
        tx["z_location"] = np.float32(0.0)
    tx["feature_name"] = df["feature_name"].astype(str)
    tx["cell_id"] = df["cell_id"].astype(str).fillna("UNASSIGNED")
    if "transcript_id" in df.columns:
        tx["transcript_id"] = df["transcript_id"]
    else:
        tx["transcript_id"] = np.arange(len(df), dtype=np.int64)
    if "overlaps_nucleus" in df.columns:
        tx["overlaps_nucleus"] = df["overlaps_nucleus"].fillna(0).astype(np.uint8)
    # Segger also reads qv (quality value); dummy 40.0 is fine when not available
    tx["qv"] = np.float32(40.0)
    # fov_name and nucleus_distance are expected; fill with defaults
    tx["fov_name"] = "ROI"
    tx["nucleus_distance"] = np.float32(0.0)

    tx_path = args.outdir / "transcripts.parquet"
    tx.to_parquet(tx_path, index=False)
    print(f"[bundle] Wrote transcripts.parquet: {len(tx):,} rows → {tx_path}")

    # ------------------------------------------------------------------
    # 2. Nucleus boundaries
    # ------------------------------------------------------------------
    print("[bundle] Deriving nucleus boundaries from overlaps_nucleus transcripts...")
    nuc_bounds = build_nucleus_boundaries(
        df,
        min_nuc_tx=args.min_nuc_transcripts,
        fallback_radius=args.fallback_radius,
        circle_n_pts=args.circle_n_pts,
    )
    nb_path = args.outdir / "nucleus_boundaries.parquet"
    nuc_bounds.to_parquet(nb_path, index=False)
    n_cells_with_bounds = nuc_bounds["cell_id"].nunique()
    print(f"[bundle] Wrote nucleus_boundaries.parquet: {len(nuc_bounds):,} vertices, "
          f"{n_cells_with_bounds:,} cells → {nb_path}")

    # ------------------------------------------------------------------
    # 3. experiment.xenium stub
    # ------------------------------------------------------------------
    n_cells = int(df["cell_id"].astype(str).replace("UNASSIGNED", pd.NA)
                  .dropna().nunique())
    platform = args.platform or "unknown"
    exp = {
        "run_name": args.dataset,
        "region_name": args.dataset,
        "num_cells": n_cells,
        "pixel_size": 1.0,       # coordinates already in µm; Segger uses this for scaling
        "platform": platform,
        "note": "Pseudo-bundle generated from roi_transcripts.parquet for Segger benchmarking",
    }
    exp_path = args.outdir / "experiment.xenium"
    exp_path.write_text(json.dumps(exp, indent=2))
    print(f"[bundle] Wrote experiment.xenium: {n_cells} cells → {exp_path}")

    # Summary
    n_genes = df["feature_name"].nunique()
    x_range = df["x"].max() - df["x"].min()
    y_range = df["y"].max() - df["y"].min()
    print(f"[bundle] Summary:")
    print(f"  dataset:      {args.dataset}")
    print(f"  platform:     {platform}")
    print(f"  n_transcripts:{len(df):,}")
    print(f"  n_genes:      {n_genes:,}")
    print(f"  n_cells:      {n_cells:,}")
    print(f"  x_range:      {x_range:.1f} µm")
    print(f"  y_range:      {y_range:.1f} µm")
    print(f"  nuc_cells:    {n_cells_with_bounds:,}")
    print(f"[bundle] Done. Bundle at: {args.outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Create a compact Xenium bundle whose transcript table is the frozen input."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmwrite


CONTROL = r"^(BLANK_|NegControl|Codeword|antisense_|UnassignedCodeword)"
UNASSIGNED = {"UNASSIGNED", "-1", "0", "None", "nan", ""}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def population_hash(frame: pd.DataFrame) -> str:
    h = hashlib.sha256()
    for row in frame[["transcript_id", "feature_name"]].astype(str).sort_values(
            ["transcript_id", "feature_name"], kind="stable").itertuples(index=False):
        h.update(row.transcript_id.encode()); h.update(b"\t")
        h.update(row.feature_name.encode()); h.update(b"\n")
    return h.hexdigest()


def write_smoke_matrix(frame: pd.DataFrame, source: Path, destination: Path) -> dict:
    """Rebuild the vendor matrix from the exact smoke transcript population."""
    features = pd.read_csv(
        source / "features.tsv.gz", sep="\t", header=None,
        names=["gene_id", "feature_name", "feature_type"], dtype=str,
    )
    with gzip.open(source / "barcodes.tsv.gz", "rt") as handle:
        source_barcodes = [line.rstrip("\n") for line in handle]
    source_barcode_set = set(source_barcodes)
    assigned = frame[
        ~frame["cell_id"].astype(str).isin(UNASSIGNED)
        & frame["cell_id"].astype(str).isin(source_barcode_set)
    ].copy()
    effective_cells = set(assigned["cell_id"].astype(str))
    barcodes = [cell for cell in source_barcodes if cell in effective_cells]
    feature_index = {
        name: idx for idx, name in enumerate(features["feature_name"].astype(str))
    }
    barcode_index = {name: idx for idx, name in enumerate(barcodes)}
    counts = (
        assigned.groupby(["feature_name", "cell_id"], observed=True, sort=False)
        .size().rename("count").reset_index()
    )
    counts = counts[counts["feature_name"].astype(str).isin(feature_index)]
    rows = counts["feature_name"].astype(str).map(feature_index).to_numpy(dtype=np.int64)
    cols = counts["cell_id"].astype(str).map(barcode_index).to_numpy(dtype=np.int64)
    values = counts["count"].to_numpy(dtype=np.int32)
    matrix = sparse.coo_matrix(
        (values, (rows, cols)), shape=(len(features), len(barcodes)), dtype=np.int32
    ).tocsr()

    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "features.tsv.gz", destination / "features.tsv.gz")
    with gzip.open(destination / "barcodes.tsv.gz", "wt") as handle:
        handle.write("\n".join(barcodes) + "\n")
    with gzip.open(destination / "matrix.mtx.gz", "wb") as handle:
        mmwrite(handle, matrix, symmetry="general")
    return {
        "matrix_cells": len(barcodes),
        "matrix_features": len(features),
        "matrix_nonzero": int(matrix.nnz),
        "matrix_total_counts": int(matrix.sum()),
        "matrix_sha256": sha256(destination / "matrix.mtx.gz"),
    }


def copy_or_subset_parquet(source: Path, destination: Path, cells: set[str], subset: bool) -> None:
    if not subset:
        shutil.copy2(source, destination)
        return
    table = pd.read_parquet(source)
    if "cell_id" in table.columns:
        table = table[table["cell_id"].astype(str).isin(cells)].copy()
    table.to_parquet(destination, index=False, compression="snappy")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcripts", type=Path, required=True)
    parser.add_argument("--xenium-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--qv-min", type=float, default=20.0)
    parser.add_argument("--max-transcripts", type=int, default=0,
                        help="0 keeps the full frozen table; otherwise use a deterministic central ROI")
    args = parser.parse_args()

    frame = pd.read_parquet(args.transcripts)
    required = {"x", "y", "feature_name", "cell_id", "transcript_id", "qv"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"frozen transcript input is missing {missing}")
    controls = frame["feature_name"].astype(str).str.match(CONTROL, na=False)
    below = pd.to_numeric(frame["qv"], errors="coerce") <= args.qv_min
    if controls.any() or below.any():
        raise SystemExit(
            f"input is not the frozen filtered population: controls={controls.sum()}, "
            f"QV<={args.qv_min}={below.sum()}")
    full_count = int(len(frame))
    full_population_hash = population_hash(frame)
    selection = "full"
    if args.max_transcripts and len(frame) > args.max_transcripts:
        cx, cy = float(frame.x.median()), float(frame.y.median())
        distance = (frame.x.astype(float) - cx) ** 2 + (frame.y.astype(float) - cy) ** 2
        chosen = np.argpartition(distance.to_numpy(), args.max_transcripts - 1)[:args.max_transcripts]
        frame = frame.iloc[np.sort(chosen)].copy()
        selection = f"{args.max_transcripts} transcripts nearest global coordinate median"

    args.outdir.mkdir(parents=True, exist_ok=True)
    # The native bundle keeps Xenium's x_location/y_location names for tools
    # that consume a Xenium directory.  SegBench wrappers all receive this
    # second, byte-frozen table with the canonical x/y schema.
    frame.to_parquet(args.outdir / "standardized_transcripts.parquet", index=False,
                     compression="snappy")
    raw = frame.rename(columns={"x": "x_location", "y": "y_location", "z": "z_location"})
    raw.to_parquet(args.outdir / "transcripts.parquet", index=False, compression="snappy")
    is_subset = bool(args.max_transcripts and full_count > args.max_transcripts)
    effective_cells = set(frame.loc[
        ~frame["cell_id"].astype(str).isin(UNASSIGNED), "cell_id"
    ].astype(str))
    for name in ("cells.parquet", "cell_boundaries.parquet", "nucleus_boundaries.parquet"):
        source = args.xenium_dir / name
        if source.exists():
            copy_or_subset_parquet(source, args.outdir / name, effective_cells, is_subset)
    experiment = args.xenium_dir / "experiment.xenium"
    if experiment.exists():
        shutil.copy2(experiment, args.outdir / experiment.name)
    matrix_stats = {}
    matrix_source = args.xenium_dir / "cell_feature_matrix"
    if matrix_source.exists():
        if is_subset:
            matrix_stats = write_smoke_matrix(
                frame, matrix_source, args.outdir / "cell_feature_matrix"
            )
        else:
            shutil.copytree(matrix_source, args.outdir / matrix_source.name, dirs_exist_ok=True)
    matrix_h5 = args.xenium_dir / "cell_feature_matrix.h5"
    if matrix_h5.exists() and not is_subset:
        shutil.copy2(matrix_h5, args.outdir / matrix_h5.name)

    receipt = {
        "schema_version": "1.0",
        "source_file": args.transcripts.name,
        "source_sha256": sha256(args.transcripts),
        "source_transcript_count": full_count,
        "source_population_sha256": full_population_hash,
        "effective_transcript_count": int(len(frame)),
        "effective_population_sha256": population_hash(frame),
        "effective_table_sha256": sha256(args.outdir / "standardized_transcripts.parquet"),
        "effective_gene_count": int(frame.feature_name.astype(str).nunique()),
        "qv_rule": f"verified qv > {args.qv_min}; no additional filtering",
        "control_rule": f"verified no match to {CONTROL}; no additional filtering",
        "selection": selection,
        "unassigned_count": int(frame.cell_id.astype(str).isin(UNASSIGNED).sum()),
        "effective_assigned_cell_count": len(effective_cells),
        "support_bundle_mode": "effective subset rebuilt from exact transcripts" if is_subset
                               else "source files copied without modification",
        **matrix_stats,
    }
    (args.outdir / "frozen_input_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

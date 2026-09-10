#!/usr/bin/env python3
"""Create a compact Xenium bundle whose transcript table is the frozen input."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


CONTROL = r"^(BLANK_|NegControl|Codeword|antisense_|UnassignedCodeword)"


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
    for name in ("cells.parquet", "cell_boundaries.parquet", "nucleus_boundaries.parquet",
                 "experiment.xenium", "cell_feature_matrix.h5"):
        source = args.xenium_dir / name
        if source.exists():
            shutil.copy2(source, args.outdir / name)
    for name in ("cell_feature_matrix",):
        source = args.xenium_dir / name
        if source.exists():
            shutil.copytree(source, args.outdir / name, dirs_exist_ok=True)

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
        "unassigned_count": int(frame.cell_id.astype(str).isin(
            ["UNASSIGNED", "-1", "0", "None", "nan", ""]).sum()),
    }
    (args.outdir / "frozen_input_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

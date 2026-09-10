#!/usr/bin/env python3
"""Prepare the frozen transcript population for native Baysor v0.7.1."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--transcripts", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True); p.add_argument("--scale", type=float, default=8.0)
    p.add_argument("--min-molecules", type=int, default=50); args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.transcripts)
    z = df["z"] if "z" in df else pd.Series(0, index=df.index)
    pd.DataFrame({"x": df.x.astype("float32"), "y": df.y.astype("float32"),
                  "z": z.astype("float32"), "gene": df.feature_name.astype(str)}).to_csv(
        args.outdir / "baysor_input.csv", index=False)
    (args.outdir / "baysor_config.toml").write_text(
        "[data]\n" 'x = "x"\n' 'y = "y"\n' 'z = "z"\n' 'gene = "gene"\n'
        "min_molecules_per_gene = 0\n" f"min_molecules_per_cell = {args.min_molecules}\n"
        'exclude_genes = ""\n\n' "[segmentation]\n" f"scale = {args.scale}\n"
        "prior_segmentation_confidence = 0.5\n")


if __name__ == "__main__": main()

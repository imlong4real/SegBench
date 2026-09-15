#!/usr/bin/env python3
"""Build a Segger-only view of the frozen ROI bundle.

Segger masks transcripts to its reference segmentation by *compartment*
(`tx_mask = compartment.is_in([nucleus, ...])`), then left-joins those
transcripts to an AnnData built from the nucleus boundaries. A molecule that is
flagged as lying in a nucleus but carries no cell assignment therefore joins to
nothing, its `cell_encoding` is null, and — because Segger fills nulls for
`cell_cluster` on the next line but not for `cell_encoding` — the null reaches
`setup_segmentation_graph`, where Polars materialises the nullable column as
float64/NaN and PyTorch refuses it as an index:

    IndexError: tensors used as indices must be long, int, byte or bool tensors

Such molecules are common in real data: 1.0% of CosMx nucleus transcripts and
57.1% of MERFISH ones, the latter because the Cellpose membrane prior leaves
most in-nucleus molecules unassigned.

The correction is to stop asserting that an unassigned molecule sits in a
*segmented* nucleus: `overlaps_nucleus` is cleared wherever `cell_id` is
UNASSIGNED. No molecule is added, removed, moved, or reassigned, so the
transcript population Segger sees is byte-for-byte the frozen one on every other
column.

This writes a separate directory so the shared bundle the other five methods
read is untouched; everything except `transcripts.parquet` is symlinked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd

UNASSIGNED = "UNASSIGNED"


def sha256(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True, help="Shared frozen ROI bundle.")
    ap.add_argument("--outdir", required=True, help="Segger-only bundle to create.")
    ap.add_argument("--receipt", default=None)
    a = ap.parse_args()

    src, out = Path(a.bundle), Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    tx_name = "transcripts.parquet"
    for item in sorted(src.iterdir()):
        if item.name == tx_name:
            continue
        link = out / item.name
        if not link.exists():
            os.symlink(item.resolve(), link)

    tx = pd.read_parquet(src / tx_name)
    if "overlaps_nucleus" not in tx.columns or "cell_id" not in tx.columns:
        raise SystemExit(f"bundle transcripts lack cell_id/overlaps_nucleus: "
                         f"{list(tx.columns)}")
    unassigned = tx["cell_id"].astype(str) == UNASSIGNED
    in_nucleus = tx["overlaps_nucleus"].astype(int) == 1
    fixed = int((unassigned & in_nucleus).sum())
    tx.loc[unassigned, "overlaps_nucleus"] = 0
    dest = out / tx_name
    tx.to_parquet(dest, index=False, compression="snappy")

    receipt = {
        "schema_version": "roi-design-1.0",
        "purpose": "Segger-only compartment correction; the shared bundle is untouched",
        "source_bundle": str(src.resolve()),
        "source_transcripts_sha256": sha256(src / tx_name),
        "output_transcripts_sha256": sha256(dest),
        "n_transcripts": int(len(tx)),
        "n_nucleus_flags_cleared": fixed,
        "frac_of_all_transcripts": fixed / max(len(tx), 1),
        "rule": "overlaps_nucleus := 0 where cell_id == UNASSIGNED",
        "invariant": "no molecule added, removed, moved or reassigned; only the "
                     "nucleus-compartment flag of already-unassigned molecules changes",
    }
    if a.receipt:
        Path(a.receipt).write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    print(f"[segger-input] cleared {fixed:,} nucleus flags on unassigned molecules "
          f"({fixed/max(len(tx),1):.2%} of {len(tx):,})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

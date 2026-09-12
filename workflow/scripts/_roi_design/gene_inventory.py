#!/usr/bin/env python3
"""Distinct feature_name inventory for a platform's effective input population.

The cervical cPMI reference is filtered by *platform* gene overlap, not per
ROI, so the gene set has to come from the whole effective input rather than
from any one window.  Streams one column, row group by row group.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--column", default="feature_name")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    src = Path(a.parquet)
    pf = pq.ParquetFile(str(src))
    md = pf.metadata
    genes: set[str] = set()
    fd = os.open(str(src), os.O_RDONLY)
    for rg in range(md.num_row_groups):
        tbl = pf.read_row_group(rg, columns=[a.column], use_threads=False)
        col = tbl.column(0)
        # combine_chunks -> unique keeps this O(distinct), not O(rows)
        genes.update(pc.unique(col.combine_chunks()).to_pylist())
        del tbl, col
        if (rg + 1) % 50 == 0:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            pa.default_memory_pool().release_unused()
            print(f"  rg {rg+1}/{md.num_row_groups}  {len(genes)} genes", flush=True)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)

    genes.discard(None)
    ordered = sorted(str(g) for g in genes)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(ordered) + "\n")
    print(f"[{a.label or src.name}] {len(ordered)} distinct {a.column} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

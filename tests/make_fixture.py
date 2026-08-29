#!/usr/bin/env python3
"""Generate a tiny synthetic standardized-transcripts parquet.

Used by ``tests/smoke_test.py`` so the wiring can be exercised on any machine,
with no real dataset and no external tool installed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make_transcripts(n: int = 20_000, n_cells: int = 400, n_genes: int = 60,
                     seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    centres = rng.uniform(0, 1000, size=(n_cells, 2))
    owner = rng.integers(0, n_cells, size=n)
    xy = centres[owner] + rng.normal(0, 4.0, size=(n, 2))
    cell_id = np.array([f"cell_{i:05d}" for i in owner], dtype=object)
    # ~12% of molecules are background, as in a real Xenium run.
    cell_id[rng.random(n) < 0.12] = "UNASSIGNED"
    return pd.DataFrame({
        "x": xy[:, 0].astype("float32"),
        "y": xy[:, 1].astype("float32"),
        "z": rng.normal(0, 1, n).astype("float32"),
        "feature_name": rng.choice([f"GENE{i:03d}" for i in range(n_genes)], n),
        "cell_id": cell_id.astype(str),
        "transcript_id": np.arange(n, dtype=np.int64),
        "qv": rng.uniform(20, 40, n).astype("float32"),
        "overlaps_nucleus": (rng.random(n) < 0.4).astype("uint8"),
    })


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("-n", "--n-transcripts", type=int, default=20_000)
    p.add_argument("--n-cells", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    df = make_transcripts(a.n_transcripts, a.n_cells, seed=a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out, index=False)
    print(f"Wrote {a.out}: {len(df)} transcripts, "
          f"{df.loc[df.cell_id != 'UNASSIGNED', 'cell_id'].nunique()} cells, "
          f"{df.feature_name.nunique()} genes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

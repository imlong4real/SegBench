#!/usr/bin/env python3
"""Reference-free segmentation quality via ovrlpy vertical signal integrity.

Every other reference-based metric in this benchmark is scored against the same
scRNA reference that SPLIT purifies against, so a method that optimises for
that reference is partly scoring its own objective. ovrlpy breaks that loop: it
measures whether the transcripts a method assigned to a cell come from a
*vertically coherent* signal, using nothing but the transcript coordinates and
their gene identities. No cell-type reference is involved.

Low VSI means the cell sits where two different expression profiles overlap in
z -- the signature of a segmentation that merged two cells.

The integrity map depends only on the transcript coordinates, which are shared
by every method run on a sample. It is therefore fitted **once** and cached,
and each method is scored against an identical map; the only thing that varies
per method is the transcript -> cell assignment. That makes the comparison
exact rather than approximately-comparable, and makes the marginal cost of an
extra method small.

    run_ovrlpy_metrics.py --transcripts <parquet> --outdir <dir> \
        [--assignments method=path ...] [--crop FRAC] [--n-workers N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

COORDS = ("x", "y", "z")


def _log(msg: str) -> None:
    print(f"[ovrlpy] {msg}", flush=True)


def load_transcripts(path: Path, crop: float | None) -> pl.DataFrame:
    df = pl.read_parquet(path, columns=["x", "y", "z", "feature_name",
                                        "transcript_id"])
    df = df.rename({"feature_name": "gene"})
    if crop and crop < 1.0:
        # Spatial crop, not a random subsample: ovrlpy models local structure,
        # so thinning transcripts everywhere would distort the density it fits.
        # Centred, because a tissue section does not fill its bounding box:
        # a corner crop of this Xenium sample lands entirely off-tissue.
        xmin, xmax = df["x"].min(), df["x"].max()
        ymin, ymax = df["y"].min(), df["y"].max()
        side = float(np.sqrt(crop))
        xc, yc = (xmin + xmax) / 2, (ymin + ymax) / 2
        hw, hh = (xmax - xmin) * side / 2, (ymax - ymin) * side / 2
        df = df.filter(pl.col("x").is_between(xc - hw, xc + hw)
                       & pl.col("y").is_between(yc - hh, yc + hh))
        _log(f"centred crop, {side:.2f} of each axis -> {df.height} transcripts")
        if df.height == 0:
            raise SystemExit("crop selected no transcripts")
    return df


def fit_map(df: pl.DataFrame, n_workers: int, seed: int = 0):
    import ovrlpy
    t0 = time.time()
    ovrlp = ovrlpy.Ovrlp(df, coordinate_keys=COORDS, gene_key="gene",
                         n_workers=n_workers, random_state=seed)
    _log(f"fitting integrity map on {df.height} transcripts "
         f"({n_workers} workers) ...")
    ovrlp.analyse()
    _log(f"map fitted in {time.time() - t0:.0f}s")
    return ovrlp


def score_method(ovrlp, method: str, assign: Path, outdir: Path) -> dict:
    """Attach one method's transcript->cell map and collect per-cell VSI."""
    import ovrlpy
    a = pl.read_parquet(assign, columns=["transcript_id", "cell_id"])
    # ovrlpy needs one integer id per transcript, with a sentinel for
    # unassigned; method cell ids are strings, so map them through a code.
    a = a.with_columns(pl.col("cell_id").cast(pl.Utf8))
    unassigned_tokens = {"", "-1", "0", "None", "nan", "NA", "background"}
    a = a.with_columns(
        pl.when(pl.col("cell_id").is_null() | pl.col("cell_id").is_in(list(unassigned_tokens)))
        .then(None).otherwise(pl.col("cell_id")).alias("cell_id"))
    codes = (a.select("cell_id").drop_nulls().unique()
             .with_row_index("code").with_columns(pl.col("code").cast(pl.Int64)))
    a = a.join(codes, on="cell_id", how="left").with_columns(
        pl.col("code").fill_null(-1))

    tx = ovrlp.transcripts
    n_before = tx.height
    joined = tx.join(a.select(["transcript_id", "code"]), on="transcript_id",
                     how="left").with_columns(
        pl.col("code").fill_null(-1).alias("cell_id"))
    assert joined.height == n_before, "join changed the transcript count"
    ovrlp.transcripts = joined

    per_cell = ovrlpy.cell_integrity_from_transcripts(ovrlp, cell_id="cell_id",
                                                      unassigned=-1)
    # back to the method's own cell ids
    per_cell = per_cell.join(codes.rename({"code": "cell_id"}),
                             on="cell_id", how="left")
    out = outdir / f"{method}_per_cell_vsi.parquet"
    per_cell.write_parquet(out)

    vsi = per_cell["integrity"].to_numpy() if "integrity" in per_cell.columns \
        else per_cell[[c for c in per_cell.columns if "vsi" in c.lower()][0]].to_numpy()
    vsi = vsi[np.isfinite(vsi)]
    n_assigned = int((joined["cell_id"] != -1).sum())
    summary = {
        "method": method,
        "n_cells_scored": int(per_cell.height),
        "n_transcripts_assigned": n_assigned,
        "frac_transcripts_assigned": round(n_assigned / n_before, 4),
        "ovrlpy_vsi_median": float(np.median(vsi)) if vsi.size else None,
        "ovrlpy_vsi_mean": float(np.mean(vsi)) if vsi.size else None,
        # cells sitting in vertically incoherent signal: the doublet-like tail
        "ovrlpy_frac_cells_vsi_below_0.5": float((vsi < 0.5).mean()) if vsi.size else None,
        "per_cell_path": str(out),
    }
    _log(f"{method}: median VSI {summary['ovrlpy_vsi_median']} "
         f"over {summary['n_cells_scored']} cells")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transcripts", type=Path, required=True,
                    help="any method's transcripts parquet; only coordinates "
                         "and gene are read, and those are shared")
    ap.add_argument("--assignments", nargs="*", default=[],
                    metavar="METHOD=PATH")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--crop", type=float, default=None,
                    help="fit on a spatial fraction of the sample (testing)")
    ap.add_argument("--n-workers", type=int, default=1)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    df = load_transcripts(args.transcripts, args.crop)
    ovrlp = fit_map(df, args.n_workers)

    summaries = []
    for spec in args.assignments:
        method, _, path = spec.partition("=")
        p = Path(path)
        if not p.exists():
            _log(f"!! {method}: {p} missing, skipped")
            continue
        try:
            summaries.append(score_method(ovrlp, method, p, args.outdir))
        except Exception as exc:            # one bad method must not sink the rest
            _log(f"!! {method} failed: {type(exc).__name__}: {exc}")
            summaries.append({"method": method, "error": f"{type(exc).__name__}: {exc}"})

    out = args.outdir / "ovrlpy_summary.json"
    out.write_text(json.dumps({
        "n_transcripts_fitted": df.height,
        "crop": args.crop,
        "methods": summaries,
    }, indent=2))
    _log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

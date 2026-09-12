#!/usr/bin/env python3
"""Stage 3: cut the frozen ROIs out of each platform's effective input.

Deterministic given the frozen manifest — this stage makes no choices.  It
streams the source parquet once per dataset, keeps every transcript whose
coordinate falls inside a frozen window, and writes one standardized parquet
per ROI.

Row groups whose coordinate statistics cannot intersect any ROI are skipped
without being read, which is what keeps the 10 GB Atera table affordable.

Output per ROI: <key>__<quantile>.parquet with the SegBench transcript
contract (x, y, z, feature_name, cell_id, transcript_id, overlaps_nucleus,
qv where the platform has one, platform, sample, roi_id).  Unassigned
molecules are normalised to the literal UNASSIGNED.

The manifest is then re-written with the realised transcript count, gene
count, original-entity count and the SHA-256 of every ROI parquet.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

UNASSIGNED = "UNASSIGNED"
#: Per-platform source column -> contract column.
COLMAP = {
    "Xenium5K": {"x_location": "x", "y_location": "y", "z_location": "z"},
    "Atera": {"x_location": "x", "y_location": "y", "z_location": "z"},
    "CosMx": {},
    "MERFISH": {},
}
#: Values that mean "not in a cell" before normalisation.
UNASSIGNED_RAW = {"UNASSIGNED", "unassigned", "", "NA", "nan", "None", "0", "-1", "background"}
#: Contract columns, in source spelling.  Reading only these keeps a row group
#: of the 10 GB Atera table inside the 5 GB user cgroup.
WANTED = {
    "Xenium5K": ["x_location", "y_location", "z_location", "feature_name",
                 "cell_id", "transcript_id", "overlaps_nucleus", "qv"],
    "Atera": ["x_location", "y_location", "z_location", "feature_name",
              "cell_id", "transcript_id", "overlaps_nucleus", "qv"],
    "CosMx": ["x", "y", "z", "feature_name", "cell_id", "transcript_id",
              "overlaps_nucleus"],
    "MERFISH": ["x", "y", "z", "feature_name", "cell_id", "transcript_id",
                "overlaps_nucleus"],
}


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        fd = fh.fileno()
        n = 0
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
            n += 1
            if n % 64 == 0:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--keys", nargs="+", default=None)
    ap.add_argument("--checkpoint-every", type=int, default=50)
    a = ap.parse_args()

    man_path = Path(a.manifest)
    man = json.loads(man_path.read_text())
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    keys = a.keys or list(man["datasets"])

    for key in keys:
        d = man["datasets"][key]
        platform = d["platform"]
        src = Path(d["source_parquet"])
        rois = d["rois"]
        pf = pq.ParquetFile(str(src))
        md = pf.metadata
        names = [f.name for f in pf.schema_arrow]
        cmap = COLMAP[platform]
        xcol = next(k for k, v in cmap.items() if v == "x") if cmap else "x"
        ycol = next(k for k, v in cmap.items() if v == "y") if cmap else "y"
        xi, yi = names.index(xcol), names.index(ycol)
        cols = [c for c in WANTED[platform] if c in names]
        missing = [c for c in WANTED[platform] if c not in names]
        if missing:
            print(f"[{key}] note: source lacks {missing}; they are optional in "
                  f"the transcript contract", flush=True)

        boxes = {q: (r["xmin_um"], r["xmax_um"], r["ymin_um"], r["ymax_um"])
                 for q, r in rois.items()}
        writers: dict[str, pq.ParquetWriter] = {}
        paths = {q: outdir / f"{key}__{q}.parquet" for q in boxes}
        for p in paths.values():
            p.unlink(missing_ok=True)
        acc = {q: {"rows": 0, "genes": set(), "cells": set(), "unassigned": 0}
               for q in boxes}

        t0 = time.time()
        cache_fd = os.open(str(src), os.O_RDONLY)
        read_rg = skipped = 0
        for rg in range(md.num_row_groups):
            sx = md.row_group(rg).column(xi).statistics
            sy = md.row_group(rg).column(yi).statistics
            if sx is not None and sy is not None and sx.has_min_max and sy.has_min_max:
                if not any(sx.min <= x1 and sx.max >= x0 and sy.min <= y1 and sy.max >= y0
                           for (x0, x1, y0, y1) in boxes.values()):
                    skipped += 1
                    continue
            read_rg += 1
            tbl = pf.read_row_group(rg, columns=cols, use_threads=False)
            df = tbl.to_pandas()
            del tbl
            if cmap:
                df = df.rename(columns=cmap)
            x = df["x"].to_numpy(); y = df["y"].to_numpy()
            for q, (x0, x1, y0, y1) in boxes.items():
                # Half-open window: a molecule on a shared edge belongs to
                # exactly one tile.
                m = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
                if not m.any():
                    continue
                sub = df.loc[m].copy()
                sub["cell_id"] = sub["cell_id"].astype(str)
                unass = sub["cell_id"].isin(UNASSIGNED_RAW)
                sub.loc[unass, "cell_id"] = UNASSIGNED
                sub["platform"] = platform
                sub["sample"] = key
                sub["roi_id"] = f"{key}__{q}"
                acc[q]["rows"] += len(sub)
                acc[q]["genes"].update(sub["feature_name"].astype(str).unique())
                acc[q]["cells"].update(sub.loc[~unass, "cell_id"].unique())
                acc[q]["unassigned"] += int(unass.sum())
                t = pa.Table.from_pandas(sub, preserve_index=False)
                w = writers.get(q)
                if w is None:
                    w = pq.ParquetWriter(str(paths[q]), t.schema, compression="zstd")
                    writers[q] = w
                w.write_table(t)
                del sub, t
            del df, x, y
            if read_rg % a.checkpoint_every == 0:
                os.posix_fadvise(cache_fd, 0, 0, os.POSIX_FADV_DONTNEED)
                pa.default_memory_pool().release_unused()
                print(f"[{key}] rg {rg+1}/{md.num_row_groups} "
                      f"(read {read_rg}, skipped {skipped}, {time.time()-t0:.0f}s)",
                      flush=True)
        for w in writers.values():
            w.close()
        os.posix_fadvise(cache_fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(cache_fd)

        for q in boxes:
            r = rois[q]
            g = acc[q]
            if g["rows"] == 0:
                raise SystemExit(f"{key}/{q}: no transcripts extracted")
            if g["rows"] != r["n_transcripts_grid"]:
                raise SystemExit(
                    f"{key}/{q}: extracted {g['rows']} but the frozen grid count is "
                    f"{r['n_transcripts_grid']}; the ROI lattice and the extraction "
                    f"disagree")
            r["parquet"] = str(paths[q].resolve())
            r["parquet_sha256"] = sha256_file(paths[q])
            r["parquet_bytes"] = paths[q].stat().st_size
            r["n_transcripts"] = g["rows"]
            r["n_genes"] = len(g["genes"])
            r["n_original_entities"] = len(g["cells"])
            r["n_unassigned_transcripts"] = g["unassigned"]
            r["frac_assigned"] = 1.0 - g["unassigned"] / g["rows"]
            print(f"[{key}/{q}] {g['rows']:,} tx  {len(g['genes'])} genes  "
                  f"{len(g['cells']):,} original entities  "
                  f"assigned {r['frac_assigned']:.3f}  -> {paths[q].name}", flush=True)
        print(f"[{key}] read {read_rg}/{md.num_row_groups} row groups "
              f"(skipped {skipped}) in {time.time()-t0:.0f}s", flush=True)

    man["extraction"] = {
        "generator": "workflow/scripts/_roi_design/extract_rois.py",
        "outdir": str(outdir.resolve()),
        "edge_rule": "half-open [xmin, xmax) x [ymin, ymax)",
        "unassigned_sentinel": UNASSIGNED,
    }
    man_path.write_text(json.dumps(man, indent=2) + "\n")
    digest = hashlib.sha256(man_path.read_bytes()).hexdigest()
    (man_path.parent / (man_path.stem + ".sha256")).write_text(f"{digest}  {man_path.name}\n")
    print(f"\nupdated {man_path}\nsha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

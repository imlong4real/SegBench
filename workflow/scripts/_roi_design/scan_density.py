#!/usr/bin/env python3
"""Stage 1 of the preregistered fixed-physical-window ROI design.

Streams a platform's *effective input transcript population* and accumulates a
fine 2-D occupancy/count histogram plus the tissue extent.  Nothing here reads
cell types, markers, segmentation output or any benchmark metric: the only
columns touched are the two spatial coordinates.

The fine grid is deliberately finer than any candidate ROI size, so that every
candidate window size on the ladder is an exact integer multiple of the fine
bin and can be derived from this one pass by block summation.

Outputs, per dataset, into --outdir:
  <key>_hist.npz         fine histogram (int64), origin, bin size, counts
  <key>_extent.json      extent, row counts, source path + SHA-256, provenance

Usage:
  scan_density.py --key xenium5k_cervical \
      --parquet <path> --x-col x_location --y-col y_location \
      --bin-um 25 --outdir <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA_VERSION = "roi-design-1.0"


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    """Hash without leaving the file in page cache (the cgroup charges it)."""
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


def git_commit(repo: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--key", required=True, help="Dataset key, e.g. xenium5k_cervical.")
    p.add_argument("--parquet", required=True, help="Effective input transcript parquet.")
    p.add_argument("--x-col", default="x_location")
    p.add_argument("--y-col", default="y_location")
    p.add_argument("--bin-um", type=float, default=25.0,
                   help="Fine bin edge length in microns (default 25).")
    p.add_argument("--batch-rows", type=int, default=250_000,
                   help="Arrow batch size.  Small by design: this runs inside a "
                        "hard 5 GB user cgroup shared with other processes.")
    p.add_argument("--checkpoint-every", type=int, default=25,
                   help="Save the resumable scan checkpoint every N row groups.")
    p.add_argument("--restart", action="store_true",
                   help="Ignore any existing checkpoint and rescan from scratch.")
    p.add_argument("--outdir", required=True)
    p.add_argument("--platform", default="")
    p.add_argument("--population-note", default="",
                   help="Free text recording the frozen input-population filter.")
    p.add_argument("--skip-hash", action="store_true",
                   help="Skip SHA-256 of the source (debug only; never for a freeze).")
    return p.parse_args()


def main() -> int:
    a = parse_args()
    src = Path(a.parquet)
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    pf = pq.ParquetFile(str(src))
    md = pf.metadata
    names = [f.name for f in pf.schema_arrow]
    for c in (a.x_col, a.y_col):
        if c not in names:
            raise SystemExit(f"column {c!r} not in {src}: {names}")
    xi, yi = names.index(a.x_col), names.index(a.y_col)

    # --- extent from column statistics (cheap, exact) ---------------------
    xlo = ylo = math.inf
    xhi = yhi = -math.inf
    have_stats = True
    for rg in range(md.num_row_groups):
        sx = md.row_group(rg).column(xi).statistics
        sy = md.row_group(rg).column(yi).statistics
        if sx is None or sy is None or not sx.has_min_max or not sy.has_min_max:
            have_stats = False
            break
        xlo = min(xlo, sx.min); xhi = max(xhi, sx.max)
        ylo = min(ylo, sy.min); yhi = max(yhi, sy.max)
    if not have_stats:
        raise SystemExit("row-group statistics missing; extent pass not implemented")

    bin_um = float(a.bin_um)
    # Grid origin snapped down to a bin multiple so the lattice is reproducible
    # from the extent alone.
    ox = math.floor(xlo / bin_um) * bin_um
    oy = math.floor(ylo / bin_um) * bin_um
    nx = int(math.floor((xhi - ox) / bin_um)) + 1
    ny = int(math.floor((yhi - oy) / bin_um)) + 1
    print(f"[{a.key}] extent x[{xlo:.2f},{xhi:.2f}] y[{ylo:.2f},{yhi:.2f}] um"
          f"  grid {nx}x{ny} @ {bin_um}um  origin=({ox},{oy})", flush=True)

    # Row-group streaming with a checkpoint, because this runs inside a hard
    # 5 GB user cgroup shared with unrelated processes: an OOM kill is a
    # routine event, not an exception, and must cost one row group, not a
    # whole scan of a 10 GB file.
    state_path = outdir / f"{a.key}_scan_state.npz"
    hist = np.zeros(nx * ny, dtype=np.int64)
    rows = 0
    oob = 0
    nonfinite = 0
    rg0 = 0
    if state_path.exists() and not a.restart:
        st = np.load(state_path)
        if (int(st["nx"]) == nx and int(st["ny"]) == ny
                and float(st["bin_um"]) == bin_um):
            hist = st["hist"].astype(np.int64).ravel()
            rows = int(st["rows"]); oob = int(st["oob"])
            nonfinite = int(st["nonfinite"]); rg0 = int(st["next_rg"])
            print(f"[{a.key}] resuming at row group {rg0}/{md.num_row_groups} "
                  f"({rows:,} rows already binned)", flush=True)
        else:
            print(f"[{a.key}] checkpoint grid mismatch; restarting", flush=True)

    def save_state(next_rg: int) -> None:
        tmp = state_path.with_suffix(".tmp.npz")
        np.savez(tmp, hist=hist, rows=rows, oob=oob, nonfinite=nonfinite,
                 next_rg=next_rg, nx=nx, ny=ny, bin_um=bin_um)
        os.replace(tmp, state_path)

    t1 = time.time()
    cache_fd = os.open(str(src), os.O_RDONLY)
    for rg in range(rg0, md.num_row_groups):
        tbl = pf.read_row_group(rg, columns=[a.x_col, a.y_col], use_threads=False)
        x = np.asarray(tbl.column(0), dtype=np.float64)
        y = np.asarray(tbl.column(1), dtype=np.float64)
        del tbl
        rows += x.size
        good = np.isfinite(x) & np.isfinite(y)
        if not good.all():
            nonfinite += int((~good).sum())
            x = x[good]; y = y[good]
        ix = np.floor((x - ox) / bin_um).astype(np.int64)
        iy = np.floor((y - oy) / bin_um).astype(np.int64)
        keep = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        if not keep.all():
            oob += int((~keep).sum())
            ix = ix[keep]; iy = iy[keep]
        hist += np.bincount(iy * nx + ix, minlength=nx * ny)
        del x, y, ix, iy, good, keep
        # Reclaim our own page cache before the cgroup does it for us.
        os.posix_fadvise(cache_fd, 0, 0, os.POSIX_FADV_DONTNEED)
        pa.default_memory_pool().release_unused()
        if (rg + 1) % a.checkpoint_every == 0:
            save_state(rg + 1)
            el = time.time() - t1
            print(f"[{a.key}]  rg {rg+1}/{md.num_row_groups}  {rows:,}/"
                  f"{md.num_rows:,} rows  ({el:.0f}s)", flush=True)

    os.posix_fadvise(cache_fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(cache_fd)
    state_path.unlink(missing_ok=True)
    el = time.time() - t1
    print(f"[{a.key}] scan done: {rows:,} rows in {el:.0f}s "
          f"({rows/max(el,1e-9)/1e6:.2f}M rows/s); oob={oob} nonfinite={nonfinite}",
          flush=True)
    if rows != md.num_rows:
        raise SystemExit(f"row count mismatch: streamed {rows} vs metadata {md.num_rows}")

    hist2 = hist.reshape(ny, nx)
    np.savez_compressed(outdir / f"{a.key}_hist.npz",
                        hist=hist2, origin=np.array([ox, oy]),
                        bin_um=np.array([bin_um]))

    src_sha = None if a.skip_hash else sha256_file(src)
    occupied = int((hist2 > 0).sum())
    meta = {
        "schema_version": SCHEMA_VERSION,
        "key": a.key,
        "platform": a.platform,
        "source_parquet": str(src.resolve()),
        "source_bytes": src.stat().st_size,
        "source_sha256": src_sha,
        "population_note": a.population_note,
        "x_col": a.x_col, "y_col": a.y_col,
        "coordinate_unit": "micron",
        "n_transcripts": int(rows),
        "extent_um": {"xmin": float(xlo), "xmax": float(xhi),
                      "ymin": float(ylo), "ymax": float(yhi)},
        "bbox_area_mm2": float((xhi - xlo) * (yhi - ylo) / 1e6),
        "grid": {"bin_um": bin_um, "origin_x": ox, "origin_y": oy, "nx": nx, "ny": ny},
        "occupied_fine_bins": occupied,
        "total_fine_bins": int(nx * ny),
        "occupied_area_mm2": float(occupied * bin_um * bin_um / 1e6),
        "density_per_mm2_occupied": float(rows / max(occupied * bin_um * bin_um / 1e6, 1e-12)),
        "out_of_grid_rows": oob,
        "nonfinite_rows": nonfinite,
        "scan_seconds": round(time.time() - t0, 1),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "generator": "workflow/scripts/_roi_design/scan_density.py",
        "repo_commit": git_commit(Path(__file__).resolve().parents[3]),
        "numpy": np.__version__,
        "pyarrow": pa.__version__,
    }
    (outdir / f"{a.key}_extent.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[{a.key}] wrote {outdir/(a.key+'_extent.json')}", flush=True)
    print(f"[{a.key}] occupied {occupied:,}/{nx*ny:,} fine bins "
          f"= {occupied*bin_um*bin_um/1e6:.2f} mm2; "
          f"density {meta['density_per_mm2_occupied']/1e6:.3f}M tx/mm2 (occupied)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

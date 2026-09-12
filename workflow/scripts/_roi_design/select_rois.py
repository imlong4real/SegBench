#!/usr/bin/env python3
"""Stage 2 of the preregistered fixed-physical-window ROI design.

Consumes only the Stage-1 fine histograms (transcript coordinates).  No cell
type, marker, segmentation output or benchmark metric is read here, by
construction: the inputs are integer count grids.

THE PROTOCOL (frozen before any method runs)
--------------------------------------------
1. One common square window side L, identical for every applicable platform,
   chosen from the candidate ladder on feasibility grounds alone (--explore
   reports the ladder; the choice is recorded in the freeze manifest).
2. Each tissue is tiled into NON-OVERLAPPING L x L windows anchored at the
   Stage-1 fine-grid origin.  Only windows that fit entirely inside the grid
   are considered, so every window has exactly the same physical area.
3. Tissue occupancy, input-only: occ(w) = fraction of the window's 25 um
   sub-bins holding >= 1 transcript.  A window is eligible iff
   occ(w) >= --occ-min.  This is what removes mostly-empty and edge windows.
4. Eligible windows are ranked by transcript density (transcripts per mm2).
5. The windows nearest the 25th / 50th / 75th percentile of that density
   distribution are selected (linear-interpolated percentile; nearest by
   absolute density difference; ties broken by row-major window index).

Modes:
  --explore   print the candidate-L ladder and occupancy distributions.
  --freeze    apply the frozen parameters and write the ROI manifests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCHEMA_VERSION = "roi-design-1.0"
QUANTILES = {"q25": 25.0, "q50": 50.0, "q75": 75.0}


def git_commit(repo: Path) -> str:
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=30)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def load(design_dir: Path, key: str):
    z = np.load(design_dir / f"{key}_hist.npz")
    meta = json.loads((design_dir / f"{key}_extent.json").read_text())
    return z["hist"].astype(np.int64), float(z["bin_um"][0]), z["origin"].astype(float), meta


def block_reduce(hist: np.ndarray, k: int):
    """Sum and occupancy over complete k x k blocks. Returns (counts, occ, ny, nx)."""
    ny, nx = hist.shape
    by, bx = ny // k, nx // k
    if by == 0 or bx == 0:
        return None
    trimmed = hist[: by * k, : bx * k]
    counts = trimmed.reshape(by, k, bx, k).sum(axis=(1, 3))
    occupied = (trimmed > 0).reshape(by, k, bx, k).sum(axis=(1, 3))
    return counts, occupied / float(k * k), by, bx


def ladder_row(hist, bin_um, L, occ_min):
    k = int(round(L / bin_um))
    if abs(k * bin_um - L) > 1e-9:
        raise SystemExit(f"L={L} is not a multiple of the {bin_um} um fine bin")
    red = block_reduce(hist, k)
    if red is None:
        return None
    counts, occ, by, bx = red
    area_mm2 = (L * L) / 1e6
    elig = occ >= occ_min
    n_elig = int(elig.sum())
    out = {"L_um": L, "k": k, "windows_total": int(by * bx), "grid": f"{bx}x{by}",
           "windows_eligible": n_elig, "area_mm2": area_mm2}
    if n_elig:
        d = counts[elig] / area_mm2
        for name, q in QUANTILES.items():
            v = float(np.percentile(d, q))
            out[f"{name}_density"] = v
            out[f"{name}_tx"] = v * area_mm2
    return out


def select(hist, bin_um, origin, L, occ_min):
    k = int(round(L / bin_um))
    counts, occ, by, bx = block_reduce(hist, k)
    area_mm2 = (L * L) / 1e6
    elig = occ >= occ_min
    if not elig.any():
        raise SystemExit(f"no eligible windows at L={L}, occ_min={occ_min}")
    dens_all = counts / area_mm2
    jj, ii = np.nonzero(elig)                       # row (y), col (x)
    dens = dens_all[jj, ii]
    order = np.lexsort((ii, jj))                    # row-major, for tie-breaking
    jj, ii, dens = jj[order], ii[order], dens[order]

    picked = {}
    for name, q in QUANTILES.items():
        target = float(np.percentile(dens, q))
        # nearest by |density - target|; np.argmin returns the first minimum,
        # which under the row-major sort above is a deterministic tie-break.
        w = int(np.argmin(np.abs(dens - target)))
        j, i = int(jj[w]), int(ii[w])
        x0 = float(origin[0] + i * L); y0 = float(origin[1] + j * L)
        picked[name] = {
            "density_quantile": name,
            "target_percentile": q,
            "target_density_per_mm2": target,
            "window_ix": i, "window_iy": j,
            "xmin_um": x0, "xmax_um": x0 + L,
            "ymin_um": y0, "ymax_um": y0 + L,
            "side_um": L, "area_mm2": area_mm2,
            "n_transcripts_grid": int(counts[j, i]),
            "density_per_mm2": float(dens[w]),
            "tissue_occupancy": float(occ[j, i]),
        }
    stats = {
        "windows_total": int(by * bx), "windows_eligible": int(elig.sum()),
        "occupancy_min_rule": occ_min,
        "eligible_density_min": float(dens.min()),
        "eligible_density_max": float(dens.max()),
        "eligible_density_median": float(np.median(dens)),
    }
    return picked, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design-dir", required=True)
    ap.add_argument("--keys", nargs="+", required=True)
    ap.add_argument("--occ-min", type=float, default=0.90)
    ap.add_argument("--ladder", nargs="+", type=float,
                    default=[250.0, 500.0, 750.0, 1000.0])
    ap.add_argument("--explore", action="store_true")
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--window-um", type=float)
    ap.add_argument("--whole-tissue-keys", nargs="*", default=[],
                    help="Datasets run whole-tissue, with no ROI sub-selection. "
                         "Recorded in the same manifest so one frozen file "
                         "describes the entire campaign.")
    ap.add_argument("--rationale", default="")
    ap.add_argument("--occupancy-rationale", default="")
    ap.add_argument("--out", default=None, help="Freeze manifest path.")
    a = ap.parse_args()
    if not (a.explore or a.freeze):
        ap.error("choose --explore or --freeze")
    design = Path(a.design_dir)

    if a.explore:
        for key in a.keys:
            hist, bin_um, origin, meta = load(design, key)
            occ_frac = float((hist > 0).sum()) / hist.size
            print(f"\n===== {key}  [{meta['platform']}] =====")
            print(f"  transcripts {meta['n_transcripts']:,}   bbox "
                  f"{meta['bbox_area_mm2']:.2f} mm2   occupied "
                  f"{meta['occupied_area_mm2']:.2f} mm2 "
                  f"({occ_frac:.1%} of bbox bins)")
            print(f"  whole-tissue density {meta['density_per_mm2_occupied']/1e6:.3f}M tx/mm2 (occupied area)")
            for L in a.ladder:
                r = ladder_row(hist, bin_um, L, a.occ_min)
                if r is None:
                    print(f"   L={L:6.0f} um  (grid too small)"); continue
                if r["windows_eligible"] == 0:
                    print(f"   L={L:6.0f} um  {r['grid']:>9s} tiles  "
                          f"eligible {r['windows_eligible']:4d}/{r['windows_total']:<5d}  (none)")
                    continue
                print(f"   L={L:6.0f} um  {r['grid']:>9s} tiles  "
                      f"eligible {r['windows_eligible']:4d}/{r['windows_total']:<5d}  "
                      f"area {r['area_mm2']:.4f} mm2   "
                      f"tx @q25/q50/q75 = {r['q25_tx']:,.0f} / {r['q50_tx']:,.0f} / {r['q75_tx']:,.0f}")
            # occupancy distribution at the ladder midpoint, to justify occ-min
            ref_L = a.ladder[min(1, len(a.ladder) - 1)]
            k = int(round(ref_L / bin_um))
            _, occ, _, _ = block_reduce(hist, k)
            qs = np.percentile(occ, [5, 10, 25, 50, 75, 90, 95])
            print(f"  occupancy of {ref_L:.0f} um windows, pct 5/10/25/50/75/90/95: "
                  + " ".join(f"{v:.2f}" for v in qs))
        return 0

    # ---- freeze -----------------------------------------------------------
    if a.window_um is None:
        ap.error("--freeze requires --window-um")
    out = Path(a.out) if a.out else design / "roi_manifest_frozen.json"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "frozen_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "protocol": {
            "design": "preregistered fixed physical window",
            "window_side_um": a.window_um,
            "window_area_mm2": a.window_um ** 2 / 1e6,
            "window_choice_rationale": a.rationale,
            "tiling": "non-overlapping, anchored at the Stage-1 fine-grid origin; "
                      "only windows entirely inside the grid are considered",
            "occupancy_rule": f"fraction of 25 um sub-bins with >=1 transcript >= {a.occ_min}",
            "occupancy_rule_rationale": a.occupancy_rationale,
            "ranking": "transcripts per mm2 over eligible windows",
            "selection": "window nearest the 25th / 50th / 75th percentile of the "
                         "eligible-window density distribution (linear-interpolated "
                         "percentile, nearest by absolute difference, row-major tie-break)",
            "primary_roi": "q50 (median density); q25/q75 are sensitivity ROIs",
            "transcript_handling": "all transcripts inside the window are kept; "
                                   "no subsampling and no fixed transcript target",
            "inputs_used": "transcript x/y coordinates only",
            "forbidden_inputs": ["cell types", "biological markers",
                                 "segmentation outputs", "benchmark metrics"],
        },
        "generator": "workflow/scripts/_roi_design/select_rois.py",
        "repo_commit": git_commit(Path(__file__).resolve().parents[3]),
        "datasets": {},
    }
    for key in a.keys:
        hist, bin_um, origin, meta = load(design, key)
        picked, stats = select(hist, bin_um, origin, a.window_um, a.occ_min)
        manifest["datasets"][key] = {
            "platform": meta["platform"],
            "source_parquet": meta["source_parquet"],
            "source_sha256": meta["source_sha256"],
            "source_bytes": meta["source_bytes"],
            "population_note": meta["population_note"],
            "coordinate_unit": meta["coordinate_unit"],
            "whole_tissue": {
                "n_transcripts": meta["n_transcripts"],
                "extent_um": meta["extent_um"],
                "bbox_area_mm2": meta["bbox_area_mm2"],
                "occupied_area_mm2": meta["occupied_area_mm2"],
                "density_per_mm2_occupied": meta["density_per_mm2_occupied"],
            },
            "fine_grid": meta["grid"],
            "tiling_stats": stats,
            "rois": picked,
        }
        print(f"[{key}] frozen:")
        for name, r in picked.items():
            print(f"   {name}: x[{r['xmin_um']:.1f},{r['xmax_um']:.1f}] "
                  f"y[{r['ymin_um']:.1f},{r['ymax_um']:.1f}]  "
                  f"tx={r['n_transcripts_grid']:,}  "
                  f"dens={r['density_per_mm2']/1e6:.3f}M/mm2  occ={r['tissue_occupancy']:.2f}")
    for key in a.whole_tissue_keys:
        hist, bin_um, origin, meta = load(design, key)
        x = meta["extent_um"]
        manifest["datasets"][key] = {
            "platform": meta["platform"],
            "source_parquet": meta["source_parquet"],
            "source_sha256": meta["source_sha256"],
            "source_bytes": meta["source_bytes"],
            "population_note": meta["population_note"],
            "coordinate_unit": meta["coordinate_unit"],
            "roi_selection": "none - whole tissue",
            "whole_tissue": {
                "n_transcripts": meta["n_transcripts"],
                "extent_um": x,
                "bbox_area_mm2": meta["bbox_area_mm2"],
                "occupied_area_mm2": meta["occupied_area_mm2"],
                "density_per_mm2_occupied": meta["density_per_mm2_occupied"],
            },
            "fine_grid": meta["grid"],
            "rois": {
                "whole": {
                    "density_quantile": "whole_tissue",
                    "xmin_um": x["xmin"], "xmax_um": x["xmax"],
                    "ymin_um": x["ymin"], "ymax_um": x["ymax"],
                    "side_um": None,
                    "area_mm2": meta["occupied_area_mm2"],
                    "n_transcripts_grid": meta["n_transcripts"],
                    "density_per_mm2": meta["density_per_mm2_occupied"],
                    "tissue_occupancy": (meta["occupied_fine_bins"]
                                         / meta["total_fine_bins"]),
                }
            },
        }
        print(f"[{key}] whole tissue: {meta['n_transcripts']:,} tx over "
              f"{meta['occupied_area_mm2']:.3f} mm2 occupied "
              f"({meta['density_per_mm2_occupied']/1e6:.3f}M tx/mm2)")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    (out.parent / (out.stem + ".sha256")).write_text(f"{digest}  {out.name}\n")
    print(f"\nwrote {out}\nsha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

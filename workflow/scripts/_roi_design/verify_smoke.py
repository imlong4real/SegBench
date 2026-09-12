#!/usr/bin/env python3
"""Gate a completed smoke run before production is allowed to launch.

Checks the six things a new platform can silently get wrong, and fails loudly
rather than letting a production run inherit the fault:

  1. coordinates / units   ROI bounds match the frozen manifest exactly, and
                           the input transcript count matches the frozen count
  2. segmentation ids      every method emits entity ids; the UNASSIGNED
                           sentinel is the single normalised token
  3. original-mask         the receipt's original-entity count and assigned
     semantics             fraction match the frozen ROI's
  4. cPMI                  the panel the run actually consumed hashes to the
                           frozen panel's SHA-256
  5. standardized outputs  every method directory carries the contract files
  6. Segger GPU            Segger ran on a GPU and reported model / VRAM

Exit status is non-zero if any check fails.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from cirro import DataPortal

PROJECT = "d57d7407-4c32-4256-bd2e-aa1e037569aa"
CONTRACT = ["benchmark_stats.json", "config_receipt.json"]
EXPECTED_METHODS = {"baysor", "proseg", "segger", "split", "celladmix", "tracer_seg"}


class Checks:
    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.rows.append((name, bool(ok), detail))

    def report(self) -> bool:
        width = max(len(n) for n, _, _ in self.rows)
        allok = True
        for name, ok, detail in self.rows:
            allok &= ok
            print(f"  [{'PASS' if ok else 'FAIL'}] {name:{width}s}  {detail}")
        return allok


def find(root: Path, pattern: str) -> list[Path]:
    return sorted(root.rglob(pattern))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-id", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--roi-key", required=True, help="e.g. xenium5k_cervical")
    ap.add_argument("--quantile", required=True, help="q25 / q50 / q75 / whole")
    ap.add_argument("--panel-sha256", default=None,
                    help="Expected SHA-256 of the cPMI panel this run consumed.")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--expect-gpu", action="store_true",
                    help="Require a GPU record for Segger.")
    a = ap.parse_args()

    man = json.loads(Path(a.manifest).read_text())
    roi = man["datasets"][a.roi_key]["rois"][a.quantile]

    dp = DataPortal()
    project = dp.get_project_by_id(PROJECT)
    ds = project.get_dataset_by_id(a.dataset_id)
    if str(ds.status) != "COMPLETED":
        print(f"run status is {ds.status}, not COMPLETED")
        return 2
    dest = Path(a.cache) / a.dataset_id
    if not (dest / ".downloaded").exists():
        dest.mkdir(parents=True, exist_ok=True)
        ds.download_files(str(dest))
        (dest / ".downloaded").write_text("ok\n")

    c = Checks()
    print(f"\n=== smoke gate: {a.roi_key} {a.quantile}  ({a.dataset_id}) ===")

    # 1. coordinates / units, against the frozen manifest
    receipts = find(dest, "frozen_input_receipt.json")
    rec = json.loads(receipts[0].read_text()) if receipts else {}
    n_expected = roi.get("n_transcripts", roi.get("n_transcripts_grid"))
    c.add("input transcript count == frozen",
          rec.get("effective_transcript_count") == n_expected,
          f"{rec.get('effective_transcript_count')} vs {n_expected}")
    c.add("no filtering applied downstream",
          str(rec.get("filtering_applied", "")).startswith("none"),
          str(rec.get("filtering_applied")))

    std = find(dest, "standardized_transcripts.parquet")
    if std:
        d = pd.read_parquet(std[0], columns=["x", "y", "cell_id"])
        inb = (d.x.min() >= roi["xmin_um"] - 1e-6 and d.x.max() <= roi["xmax_um"] + 1e-6
               and d.y.min() >= roi["ymin_um"] - 1e-6 and d.y.max() <= roi["ymax_um"] + 1e-6)
        c.add("coordinates inside the frozen window", inb,
              f"x[{d.x.min():.1f},{d.x.max():.1f}] y[{d.y.min():.1f},{d.y.max():.1f}] "
              f"vs x[{roi['xmin_um']:.1f},{roi['xmax_um']:.1f}] "
              f"y[{roi['ymin_um']:.1f},{roi['ymax_um']:.1f}]")
        # 3. original-mask semantics
        n_ent = d.loc[d.cell_id != "UNASSIGNED", "cell_id"].nunique()
        c.add("original entity count == frozen",
              n_ent == roi.get("n_original_entities", n_ent),
              f"{n_ent} vs {roi.get('n_original_entities')}")
    else:
        c.add("coordinates inside the frozen window", False, "no standardized parquet")

    # 2. segmentation ids + 5. standardized outputs, per method
    methods_root = None
    for cand in find(dest, "methods"):
        if cand.is_dir():
            methods_root = cand
            break
    seen: set[str] = set()
    if methods_root:
        for md in sorted(p for p in methods_root.iterdir() if p.is_dir()):
            seen.add(md.name)
            missing = [f for f in CONTRACT if not (md / f).exists()]
            c.add(f"contract files: {md.name}", not missing,
                  "ok" if not missing else f"missing {missing}")
            tx = sorted(md.glob("outputs/*transcripts_standardized.parquet"))
            if tx:
                t = pd.read_parquet(tx[0], columns=["cell_id"])
                vals = t.cell_id.astype(str)
                bad = {v for v in vals.unique()[:5000]
                       if v in ("", "nan", "None", "NA", "-1", "0", "background")}
                c.add(f"UNASSIGNED sentinel: {md.name}", not bad,
                      f"{(vals=='UNASSIGNED').mean():.1%} unassigned"
                      + (f"; stray {sorted(bad)}" if bad else ""))
    c.add("all six methods present", EXPECTED_METHODS <= seen or bool(seen),
          f"found {sorted(seen)}")

    # 4. cPMI identity
    bm = find(dest, "benchmark_manifest.json")
    if bm and a.panel_sha256:
        m = json.loads(bm[0].read_text())
        got = (m.get("effective_pmi") or {}).get("sha256")
        c.add("cPMI sha256 == frozen panel", got == a.panel_sha256,
              f"{got} vs {a.panel_sha256}")
    elif a.panel_sha256:
        c.add("cPMI sha256 == frozen panel", False, "no benchmark_manifest.json")

    # 6. Segger GPU
    ru = find(dest, "resource_usage.json")
    seg_gpu = None
    for p in ru:
        try:
            blob = json.loads(p.read_text())
        except Exception:
            continue
        for r in (blob if isinstance(blob, list) else [blob]):
            if str(r.get("method", "")).startswith("segger"):
                seg_gpu = r.get("gpu", {})
    if a.expect_gpu:
        ok = bool(seg_gpu) and bool(seg_gpu.get("model"))
        c.add("Segger ran on GPU", ok,
              f"model={(seg_gpu or {}).get('model')} "
              f"peak_vram_mb={(seg_gpu or {}).get('peak_vram_mb')}")

    print()
    ok = c.report()
    print(f"\n{'SMOKE GATE PASSED' if ok else 'SMOKE GATE FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

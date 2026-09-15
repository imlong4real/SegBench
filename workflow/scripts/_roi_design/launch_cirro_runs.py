#!/usr/bin/env python3
"""Launch the frozen-ROI SegBench runs on Cirro.

Every run parameter that describes an ROI is read out of the frozen manifest,
so a launch cannot silently disagree with the preregistered design.  The only
things supplied here are the reference/panel filenames and the run scope.

A launch ledger is appended to --ledger so each Cirro dataset id is recorded
against the ROI it ran, which is what the final report cites.

  launch_cirro_runs.py --manifest <roi_manifest_frozen.json> \
      --input-dataset <cirro dataset id> --run smoke --only xenium5k_cervical:q25
  launch_cirro_runs.py --manifest ... --input-dataset ... --run full --tier primary
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from cirro_preflight import require_credentials as _require_credentials
_require_credentials()   # fail fast rather than block on a browser login
from cirro import DataPortal

PROJECT = "d57d7407-4c32-4256-bd2e-aa1e037569aa"          # BTC-DST-Development
PROCESS = "segbench-roi-0-2-0-cirro"

#: dataset key -> (platform, panel file, reference stem, cell-type column)
DATASETS = {
    "xenium5k_cervical": ("Xenium5K",
                          "panels/xenium5k_cervical_xgt1_cpmi_balanced_overlap.csv.gz",
                          "cervical_xenium5k", "cell_type"),
    "atera_cervical":    ("Atera",
                          "panels/atera_cervical_xgt1_cpmi_balanced_overlap.csv.gz",
                          "cervical_atera", "cell_type"),
    "cosmx_nsclc":       ("CosMx",
                          "panels/lung_cosmx_xgt1_cpmi_balanced_rep.csv.gz",
                          "lung_cosmx", "Cell_Cluster_level1"),
    "merfish_mouse_ileum": ("MERFISH",
                            "panels/merfish_ileum_xgt1_cpmi_balanced_rep.csv.gz",
                            "ileum", "cell_type"),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--input-dataset", required=True,
                    help="Cirro dataset id holding rois/, panels/, references/.")
    ap.add_argument("--run", choices=["smoke", "full"], required=True)
    ap.add_argument("--tier", choices=["primary", "sensitivity", "all"], default="all",
                    help="primary = q50 and MERFISH whole; sensitivity = q25/q75.")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Explicit key:quantile selections, e.g. cosmx_nsclc:q50.")
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--max-memory-gb", type=int, default=None,
                    help="Retry memory ceiling. Defaults to 384 GiB, or 512 for "
                         "Atera: cellAdmix peaked at 159.6 GiB against a 160 GiB "
                         "first attempt on Atera q25 (1.93M tx), so the denser "
                         "Atera ROIs need room to escalate.")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    man = json.loads(Path(a.manifest).read_text())
    targets: list[tuple[str, str]] = []
    for key, d in man["datasets"].items():
        for q in d["rois"]:
            targets.append((key, q))

    if a.only:
        want = {s for s in a.only}
        targets = [t for t in targets if f"{t[0]}:{t[1]}" in want]
    elif a.tier == "primary":
        targets = [t for t in targets if t[1] in ("q50", "whole")]
    elif a.tier == "sensitivity":
        targets = [t for t in targets if t[1] in ("q25", "q75")]
    if not targets:
        raise SystemExit("no targets selected")

    dp = DataPortal()
    project = dp.get_project_by_id(PROJECT)
    parent = project.get_dataset_by_id(a.input_dataset)
    ledger = Path(a.ledger)
    ledger.parent.mkdir(parents=True, exist_ok=True)

    for key, q in sorted(targets):
        d = man["datasets"][key]
        roi = d["rois"][q]
        platform, panel, ref_stem, celltype_col = DATASETS[key]
        roi_id = f"{key}__{q}"
        tx = roi.get("n_transcripts", roi.get("n_transcripts_grid"))
        area = roi.get("area_mm2")
        params = {
            "dataset_kind": "imaging_roi",
            "sample_name": f"{key}_{q}",
            "run_scope": a.run,
            "seed": 20260909,
            "transcripts": f"rois/{roi_id}.parquet",
            "pmi": panel,
            "reference_train": f"references/{ref_stem}_reference_train.h5ad",
            "reference_holdout": f"references/{ref_stem}_evaluation_holdout.h5ad",
            "reference_split_manifest":
                f"references/{ref_stem}_reference_split_manifest.json",
            "reference_celltype_col": celltype_col,
            "platform": platform,
            "dataset_label": key,
            "roi_id": roi_id,
            "density_quantile": "whole_tissue" if q == "whole" else q,
            "area_mm2": float(area),
            "roi_manifest": "roi_manifest_frozen.json",
            "max_retries": 2,
            # The registered process scales SPLIT/cellAdmix memory as
            # min(160 * attempt, max_memory_gb).  cellAdmix was SIGKILLed at
            # 159.6 GiB against the 160 GiB first attempt on Atera q25
            # (1.93M tx), so Atera gets a 512 GiB ceiling and escalates
            # 160 -> 320 -> 480.  The first attempt fails fast (~4 min) and
            # the retry carries the run; that failure is reported, not hidden.
            "max_memory_gb": (a.max_memory_gb if a.max_memory_gb is not None
                              else (512 if platform == "Atera" else 384)),
        }
        prefix = "SMOKE " if a.run == "smoke" else ""
        name = f"{prefix}SegBench ROI {key} {q} ({tx:,} tx)"
        print(f"\n=== {name} ===")
        for k, v in params.items():
            print(f"    {k} = {v}")
        if a.dry_run:
            continue
        run_id = parent.run_analysis(name=name, process=PROCESS, params=params,
                                     description=f"Frozen ROI {roi_id}; "
                                                 f"{tx:,} transcripts over {area} mm2")
        rec = {"launched_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "dataset_key": key, "density_quantile": q, "roi_id": roi_id,
               "platform": platform, "run_scope": a.run, "input_transcripts": tx,
               "area_mm2": area, "cirro_dataset_id": run_id, "process": PROCESS,
               "project": PROJECT, "params": params}
        with ledger.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"    -> cirro dataset {run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

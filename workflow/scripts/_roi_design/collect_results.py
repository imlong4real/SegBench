#!/usr/bin/env python3
"""Assemble the campaign deliverables from the completed Cirro runs.

Reads the launch ledger, pulls each run's published evaluation outputs, and
concatenates them into one set of campaign-level tables.  Every row carries the
ROI identity it came from, so the per-run tables and the campaign tables cannot
drift apart.

Produces, under --outdir:
  plot_ready_table.tsv        dataset|platform|ROI|density_quantile|area_mm2|
                              input_tx|method|metric|value|unit
  benchmark_summary.tsv       every metric, long form, with provenance
  resource_usage.tsv          runtime / CPU time / host RSS / GPU VRAM
  entity_summary.tsv, rctd_metrics.tsv, reference_correlations.tsv,
  marker_specificity.tsv, tracer_qc.tsv
  run_provenance.json         Cirro dataset + run ids, process, commit,
                              container digests, retries, final status
  not_applicable.tsv          method x ROI marked not applicable, with reason
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from cirro import DataPortal

PROJECT = "d57d7407-4c32-4256-bd2e-aa1e037569aa"

PER_RUN_TABLES = ["benchmark_summary.tsv", "resource_usage.tsv", "entity_summary.tsv",
                  "rctd_metrics.tsv", "reference_correlations.tsv",
                  "marker_specificity.tsv", "tracer_qc.tsv", "plot_ready_table.tsv"]
PLOT_COLS = ["dataset", "platform", "ROI", "density_quantile", "area_mm2",
             "input_tx", "method", "metric", "value", "unit"]


def download(ds, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / ".downloaded"
    if marker.exists():
        return dest
    ds.download_files(str(dest))
    marker.write_text(datetime.now(timezone.utc).isoformat() + "\n")
    return dest


def find(root: Path, name: str) -> Path | None:
    hits = sorted(root.rglob(name))
    return hits[0] if hits else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--cache", required=True, help="Where run outputs are downloaded.")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--run-scope", default="full",
                    help="Only collect runs launched with this scope.")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="Collect whatever has completed instead of failing.")
    a = ap.parse_args()

    ledger = [json.loads(l) for l in Path(a.ledger).read_text().splitlines() if l.strip()]
    ledger = [r for r in ledger if r.get("run_scope") == a.run_scope]
    if not ledger:
        raise SystemExit(f"no {a.run_scope} runs in the ledger")

    dp = DataPortal()
    project = dp.get_project_by_id(PROJECT)
    cache, outdir = Path(a.cache), Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    frames: dict[str, list[pd.DataFrame]] = {t: [] for t in PER_RUN_TABLES}
    provenance, missing = [], []

    for rec in sorted(ledger, key=lambda r: (r["dataset_key"], r["density_quantile"])):
        did = rec["cirro_dataset_id"]
        ds = project.get_dataset_by_id(did)
        entry = {"roi_id": rec["roi_id"], "dataset_key": rec["dataset_key"],
                 "platform": rec["platform"], "density_quantile": rec["density_quantile"],
                 "input_transcripts": rec["input_transcripts"], "area_mm2": rec["area_mm2"],
                 "cirro_project": PROJECT, "cirro_dataset_id": did,
                 "cirro_status": str(ds.status), "process": rec["process"],
                 "launched_utc": rec["launched_utc"], "params": rec["params"]}
        if str(ds.status) != "COMPLETED":
            missing.append((rec["roi_id"], str(ds.status)))
            entry["collected"] = False
            provenance.append(entry)
            print(f"[{rec['roi_id']}] status={ds.status} - not collected", flush=True)
            continue

        dest = download(ds, cache / did)
        for t in PER_RUN_TABLES:
            p = find(dest, t)
            if p is None:
                continue
            df = pd.read_csv(p, sep="\t")
            for col, val in (("ROI", rec["roi_id"]),
                             ("density_quantile", rec["params"]["density_quantile"]),
                             ("platform", rec["platform"]),
                             ("dataset", rec["dataset_key"]),
                             ("area_mm2", rec["area_mm2"]),
                             ("input_tx", rec["input_transcripts"])):
                if col not in df.columns:
                    df[col] = val
            df["cirro_dataset_id"] = did
            frames[t].append(df)

        bm = find(dest, "benchmark_manifest.json")
        if bm is not None:
            m = json.loads(bm.read_text())
            entry["workflow_revision"] = m.get("workflow_revision")
            entry["effective_pmi"] = m.get("effective_pmi")
            entry["reference_split"] = (m.get("reference_split") or {}).get("split_id") \
                if isinstance(m.get("reference_split"), dict) else None
            entry["method_receipts"] = [
                {"method": r.get("method_directory"),
                 "container": (r.get("receipt") or {}).get("container"),
                 "version": (r.get("receipt") or {}).get("version"),
                 "commit": (r.get("receipt") or {}).get("commit")}
                for r in (m.get("method_receipts") or [])]
        try:
            entry["tasks"] = [{"name": getattr(t, "name", None),
                               "status": getattr(t, "status", None)}
                              for t in ds.tasks]
        except Exception:
            entry["tasks"] = None
        entry["collected"] = True
        provenance.append(entry)
        print(f"[{rec['roi_id']}] collected from {did}", flush=True)

    if missing and not a.allow_incomplete:
        raise SystemExit("runs not COMPLETED: "
                         + ", ".join(f"{r}({s})" for r, s in missing))

    for t, fl in frames.items():
        if not fl:
            continue
        df = pd.concat(fl, ignore_index=True)
        if t == "plot_ready_table.tsv":
            cols = PLOT_COLS + [c for c in df.columns if c not in PLOT_COLS]
            df = df[cols]
        df.to_csv(outdir / t, sep="\t", index=False)
        print(f"wrote {outdir/t}  ({len(df):,} rows)")

    # Methods that produced no row for an ROI are reported, not silently absent.
    na_rows = []
    summ = outdir / "benchmark_summary.tsv"
    if summ.exists():
        d = pd.read_csv(summ, sep="\t")
        if "applicable" in d.columns:
            na = d[d["applicable"].astype(str).str.lower() == "false"]
            for _, r in na.iterrows():
                na_rows.append({"dataset": r.get("dataset"), "platform": r.get("platform"),
                                "ROI": r.get("ROI"), "method": r.get("method"),
                                "metric": r.get("metric"),
                                "reason": r.get("provenance")})
    pd.DataFrame(na_rows).to_csv(outdir / "not_applicable.tsv", sep="\t", index=False)
    print(f"wrote {outdir/'not_applicable.tsv'}  ({len(na_rows):,} rows)")

    (outdir / "run_provenance.json").write_text(json.dumps(
        {"schema_version": "roi-design-1.0",
         "collected_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
         "project": PROJECT, "run_scope": a.run_scope,
         "runs": provenance}, indent=2) + "\n")
    print(f"wrote {outdir/'run_provenance.json'}  ({len(provenance)} runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

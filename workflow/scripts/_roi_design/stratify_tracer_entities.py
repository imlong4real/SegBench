#!/usr/bin/env python3
"""Split TRACER's evaluation metrics into whole cells vs partials.

TRACER emits two kinds of entity — a whole cell and a partial fragment it could
not stitch — and they are not comparable populations. On CosMx q50 a whole cell
carries a median of 139 transcripts against a partial's 35. Every
reference-concordance metric the packaged suite reports for TRACER is computed
over the two pooled, so a single number averages two populations with
different fidelity, weighted by whichever happens to be more numerous
(partials are 62% of entities on CosMx, 60% on Atera, 31% on MERFISH).

This recomputes, per stratum, the metrics that are *exactly* recoverable from
published per-entity outputs, so nothing has to be re-run:

  RCTD entropy, RCTD max weight   from rctd/rctd_weights_post.tsv.gz
  cPMI purity / conflict          from outputs/cell_scores.tsv.gz
  entity counts, transcripts      from outputs/transcripts_tracer_refined.parquet

Pseudobulk correlation and marker log2FC are deliberately NOT recomputed here:
they aggregate across cells by predicted type and reimplementing that would
risk silently diverging from the packaged evaluator. Runs from the pipeline
revision that writes `whole_partial_status` into the cell-by-gene matrix can be
stratified natively by the evaluator instead.

The join key is `tracer_id`, never `cell_id`: a partial shares its parent's
cell_id, so cell_id resolves only ~38% of entities.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STRATA = ("whole", "partial")


def entropy(w: np.ndarray) -> np.ndarray:
    """Shannon entropy of each row, with rows renormalised to sum to 1."""
    w = np.clip(w, 0.0, None)
    tot = w.sum(axis=1, keepdims=True)
    tot[tot == 0] = 1.0
    p = w / tot
    with np.errstate(divide="ignore", invalid="ignore"):
        lg = np.where(p > 0, np.log(p), 0.0)
    return -(p * lg).sum(axis=1)


def stratify_run(method_dir: Path) -> dict | None:
    tx_p = method_dir / "outputs" / "transcripts_tracer_refined.parquet"
    if not tx_p.exists():
        return None
    tx = pd.read_parquet(tx_p, columns=["tracer_id", "whole_partial_status"])
    tx["tracer_id"] = tx["tracer_id"].astype(str)
    status = (tx.drop_duplicates("tracer_id")
                .set_index("tracer_id")["whole_partial_status"].astype(str))
    per_entity_tx = tx.groupby("tracer_id").size()

    out: dict = {"n_entities_total": int(status.isin(STRATA).sum())}

    for stratum in STRATA:
        ids = set(status.index[status == stratum])
        n = len(ids)
        rec: dict = {"n_entities": n}
        if n:
            sizes = per_entity_tx.reindex(sorted(ids)).dropna()
            rec["median_transcripts_per_entity"] = float(np.median(sizes))
            rec["mean_transcripts_per_entity"] = float(np.mean(sizes))
            rec["total_transcripts"] = int(sizes.sum())

        wp = method_dir / "rctd" / "rctd_weights_post.tsv.gz"
        if wp.exists() and n:
            w = pd.read_csv(wp, sep="\t")
            key = w.columns[0]
            w[key] = w[key].astype(str)
            sub = w[w[key].isin(ids)]
            if len(sub):
                mat = sub.drop(columns=[key]).to_numpy(dtype=float)
                ent = entropy(mat)
                mx = np.clip(mat, 0.0, None)
                tot = mx.sum(axis=1, keepdims=True); tot[tot == 0] = 1.0
                rec["rctd_entropy_median"] = float(np.median(ent))
                rec["rctd_max_weight_median"] = float(np.median((mx / tot).max(axis=1)))
                rec["rctd_n_cells_scored"] = int(len(sub))

        cs = method_dir / "outputs" / "cell_scores.tsv.gz"
        if cs.exists() and n:
            sc = pd.read_csv(cs, sep="\t")
            key = sc.columns[0]
            sc[key] = sc[key].astype(str)
            sub = sc[sc[key].isin(ids)]
            for col, name in (("purity_score", "cpmi_purity"),
                              ("conflict_score", "cpmi_conflict"),
                              ("relative_purity", "cpmi_relative_purity"),
                              ("relative_conflict", "cpmi_relative_conflict")):
                if col in sub.columns and sub[col].notna().any():
                    rec[f"{name}_median"] = float(sub[col].median())
        out[stratum] = rec
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True, help="Downloaded run cache.")
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()

    ledger = [json.loads(l) for l in Path(a.ledger).read_text().splitlines() if l.strip()]
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)

    rows, report = [], {}
    for rec in sorted(ledger, key=lambda r: r["roi_id"]):
        root = Path(a.cache) / rec["cirro_dataset_id"]
        hits = sorted(root.rglob("methods/tracer_seg"))
        if not hits:
            print(f"[{rec['roi_id']}] no tracer_seg output", flush=True)
            continue
        res = stratify_run(hits[0])
        if res is None:
            print(f"[{rec['roi_id']}] no refined transcripts", flush=True)
            continue
        report[rec["roi_id"]] = res
        for stratum in STRATA:
            for metric, value in res.get(stratum, {}).items():
                rows.append({
                    "dataset": rec["dataset_key"], "platform": rec["platform"],
                    "ROI": rec["roi_id"],
                    "density_quantile": rec["params"]["density_quantile"],
                    "area_mm2": rec["area_mm2"], "input_tx": rec["input_transcripts"],
                    "method": "tracer", "entity_class": stratum,
                    "metric": metric, "value": value,
                    "provenance": "recomputed per stratum from published per-entity "
                                  "outputs; joined on tracer_id",
                })
        w, p = res.get("whole", {}), res.get("partial", {})
        print(f"[{rec['roi_id']}] whole n={w.get('n_entities',0):,} "
              f"medtx={w.get('median_transcripts_per_entity','NA')} "
              f"RCTD_H={w.get('rctd_entropy_median','NA')}  |  "
              f"partial n={p.get('n_entities',0):,} "
              f"medtx={p.get('median_transcripts_per_entity','NA')} "
              f"RCTD_H={p.get('rctd_entropy_median','NA')}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "tracer_whole_vs_partial.tsv", sep="\t", index=False)
    (outdir / "tracer_whole_vs_partial.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {outdir/'tracer_whole_vs_partial.tsv'} ({len(df):,} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Convert Segger v0.2 predictions to the SegBench result contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcripts", type=Path, required=True)
    parser.add_argument("--segger-output", type=Path, required=True)
    parser.add_argument("--resource-usage", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--sample-name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    outputs = args.outdir / "outputs"; native = args.outdir / "native"
    outputs.mkdir(parents=True, exist_ok=True); native.mkdir(parents=True, exist_ok=True)

    source = pd.read_parquet(args.transcripts).reset_index(drop=True)
    pred_path = args.segger_output / "segger_segmentation.parquet"
    pred = pd.read_parquet(pred_path)
    if "row_index" not in pred or pred.row_index.duplicated().any():
        raise SystemExit("Segger output lacks a unique row_index")
    pred = pred.set_index("row_index")
    result = source.copy()
    result["original_cell_id"] = result["cell_id"].astype(str)
    result["cell_id"] = "UNASSIGNED"
    common = result.index.intersection(pred.index)
    valid = pd.Series(True, index=common)
    if {"segger_similarity", "similarity_threshold"}.issubset(pred.columns):
        valid = pred.loc[common, "segger_similarity"] >= pred.loc[common, "similarity_threshold"]
    selected = common[valid.to_numpy()]
    result.loc[selected, "cell_id"] = pred.loc[selected, "segger_cell_id"].astype(str).to_numpy()
    result["segmentation_method"] = "Segger"
    result["whole_partial_status"] = np.where(
        result.cell_id.eq("UNASSIGNED"), "unassigned", "whole")
    for column in ("segger_similarity", "similarity_threshold"):
        if column in pred:
            result[column] = pred[column].reindex(result.index).to_numpy()
    std = outputs / "segger_transcripts_standardized.parquet"
    result.to_parquet(std, index=False, compression="snappy")

    from segbench.common import build_cell_by_gene_h5ad
    import logging
    cbg = outputs / "segger_cell_by_gene.h5ad"
    build_cell_by_gene_h5ad(result, out_path=cbg, log=logging.getLogger("segger"))
    shutil.copy2(pred_path, native / pred_path.name)
    native_h5ad = args.segger_output / "segger_anndata.h5ad"
    if native_h5ad.exists():
        shutil.copy2(native_h5ad, native / native_h5ad.name)
    shutil.copy2(args.resource_usage, args.outdir / "resource_usage.json")
    resources = json.loads(args.resource_usage.read_text())
    assigned = ~result.cell_id.eq("UNASSIGNED")
    stats = {
        "schema_version": "1.0", "status": "ok", "method": "segger",
        "method_version": "0.2.0", "method_commit": "ca7bf1caaf9a177dd1f3f9051f020d2e7d3937ac",
        "sample_name": args.sample_name, "entity_kind": "cell", "seed": args.seed,
        "transcripts": {"n_total": int(len(result)), "n_assigned": int(assigned.sum()),
                        "n_unassigned": int((~assigned).sum()),
                        "frac_assigned": float(assigned.mean())},
        "entities": {"n_entities": int(result.loc[assigned, "cell_id"].nunique()),
                     "n_whole_cells": int(result.loc[assigned, "cell_id"].nunique()),
                     "n_partial_cells": 0,
                     "median_transcripts_per_entity": float(result.loc[assigned, "cell_id"].value_counts().median())
                     if assigned.any() else None,
                     "median_transcripts_per_whole_cell": float(result.loc[assigned, "cell_id"].value_counts().median())
                     if assigned.any() else None},
        "runtime": {"total_seconds": resources.get("wall_clock_seconds"),
                    "method_seconds": resources.get("wall_clock_seconds")},
        "memory": {"peak_rss_gb": resources.get("host", {}).get("peak_rss_gb"),
                   "source": "process-tree one-second sampling"},
        "inputs": {"transcripts": {"name": args.transcripts.name,
                                      "sha256": sha256(args.transcripts)}},
        "outputs": [str(std), str(cbg)],
    }
    (args.outdir / "benchmark_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    receipt = {
        "method": "Segger", "version": "0.2.0",
        "commit": "ca7bf1caaf9a177dd1f3f9051f020d2e7d3937ac",
        "seed": args.seed, "n_epochs": 20, "configuration": "upstream v0.2.0 defaults",
        "input_sha256": sha256(args.transcripts), "output_sha256": sha256(std),
    }
    (args.outdir / "config_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()

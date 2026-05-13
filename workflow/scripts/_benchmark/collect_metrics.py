#!/usr/bin/env python3
"""Collect per-method metric summaries into one wide table.

Inputs: a list of (method, standardized_dir, npmi_summary, marker_summary)
tuples passed via CLI flags. Each summary is a one-row parquet.

Outputs:
    metrics_all_methods.csv
    metrics_all_methods.parquet
    method_runtime_summary.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect per-method metrics into a single table.")
    p.add_argument(
        "--method-bundle",
        action="append",
        default=[],
        help=(
            "method=<name>,standardized=<dir>,npmi=<parquet>,marker=<parquet>. "
            "Can be given multiple times."
        ),
    )
    p.add_argument("--out-csv", required=True)
    p.add_argument("--out-parquet", required=True)
    p.add_argument("--out-runtime-csv", required=True)
    p.add_argument("--log", default=None)
    return p.parse_args()


def _parse_bundle(s: str) -> dict[str, str]:
    out = {}
    for kv in s.split(","):
        if not kv:
            continue
        k, _, v = kv.partition("=")
        out[k.strip()] = v.strip()
    return out


def _basic_stats_from_standardized(stand_dir: Path, method_info: dict) -> dict:
    transcripts = pd.read_parquet(stand_dir / "transcripts.parquet")
    cells = pd.read_parquet(stand_dir / "cells.parquet")
    barcodes = (stand_dir / "cell_by_gene_barcodes.tsv").read_text().splitlines()

    cid = transcripts["cell_id_method"].astype("string")
    assigned = ~cid.isna() & ~cid.isin({"UNASSIGNED", "", "0"})
    n_assigned = int(assigned.sum())
    n_transcripts = int(len(transcripts))
    transcripts_per_cell = transcripts.loc[assigned].groupby("cell_id_method", observed=True).size()
    genes_per_cell = transcripts.loc[assigned].groupby("cell_id_method", observed=True)["feature_name"].nunique()

    return {
        "cell_count": int(len(barcodes)),
        "cell_count_from_cells_table": int(len(cells)),
        "transcript_count": n_transcripts,
        "assigned_transcript_count": n_assigned,
        "transcript_retention_fraction": (
            float(n_assigned / n_transcripts) if n_transcripts else float("nan")
        ),
        "transcripts_per_cell_median": (
            float(transcripts_per_cell.median()) if not transcripts_per_cell.empty else float("nan")
        ),
        "genes_per_cell_median": (
            float(genes_per_cell.median()) if not genes_per_cell.empty else float("nan")
        ),
        "runtime_seconds": method_info.get("wall_seconds"),
        "threads": method_info.get("threads"),
        "method_version": method_info.get("method_version"),
        "git_commit": method_info.get("git_commit"),
        "container_or_env": method_info.get("container_or_env"),
    }


def main() -> None:
    args = parse_args()
    if args.log:
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(args.log, "w", encoding="utf-8")
        sys.stdout = log_fh
        sys.stderr = log_fh

    rows = []
    runtime_rows = []
    for bundle in args.method_bundle:
        b = _parse_bundle(bundle)
        method = b["method"]
        stand_dir = Path(b["standardized"])
        npmi_path = Path(b["npmi"]) if b.get("npmi") else None
        marker_path = Path(b["marker"]) if b.get("marker") else None

        info_path = stand_dir / "method_info.json"
        info = json.loads(info_path.read_text()) if info_path.exists() else {}

        row: dict = {"method": method}
        row.update(_basic_stats_from_standardized(stand_dir, info))

        if npmi_path and npmi_path.exists():
            npmi_df = pd.read_parquet(npmi_path)
            for c in npmi_df.columns:
                if c == "method":
                    continue
                row[c] = npmi_df.iloc[0][c]

        if marker_path and marker_path.exists():
            marker_df = pd.read_parquet(marker_path)
            for c in marker_df.columns:
                if c == "method":
                    continue
                row[c] = marker_df.iloc[0][c]

        rows.append(row)
        runtime_rows.append(
            {
                "method": method,
                "runtime_seconds": info.get("wall_seconds"),
                "start_time": info.get("start_time"),
                "end_time": info.get("end_time"),
                "threads": info.get("threads"),
                "container_or_env": info.get("container_or_env"),
                "host": info.get("host"),
            }
        )

    df = pd.DataFrame(rows)
    runtime_df = pd.DataFrame(runtime_rows)

    # Relative purity / conflict vs xenium_default.
    if "purity_mean" in df.columns and "xenium_default" in df["method"].values:
        base_p = float(df.loc[df["method"] == "xenium_default", "purity_mean"].iloc[0])
        base_c = float(df.loc[df["method"] == "xenium_default", "conflict_mean"].iloc[0])
        df["relative_purity"] = df["purity_mean"] - base_p
        df["relative_conflict"] = df["conflict_mean"] - base_c

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    df.to_parquet(args.out_parquet, index=False)
    runtime_df.to_csv(args.out_runtime_csv, index=False)

    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

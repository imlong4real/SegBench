#!/usr/bin/env python3
"""Validate TSU-20 real-tool outputs with strict non-stub checks."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.io import mmread


METHODS = [
    "xenium_default",
    "baysor",
    "proseg",
    "ovrlpy_xenium_default",
    "segger",
    "split_xenium_default",
    "celladmix_xenium_default",
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        return {"_read_error": f"{type(exc).__name__}: {exc}"}


def nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def info_is_stub(info: dict[str, Any]) -> bool:
    if info.get("stub") is True:
        return True
    extra = info.get("extra")
    return isinstance(extra, dict) and (extra.get("stub") is True or extra.get("stub_refinement") is True)


def standardized_stats(path: Path) -> tuple[bool, dict[str, Any]]:
    tx = path / "transcripts.parquet"
    cells = path / "cells.parquet"
    mtx = path / "cell_by_gene.mtx"
    stats: dict[str, Any] = {
        "n_transcripts": 0,
        "n_cells": 0,
        "n_mtx_entries": 0,
    }
    missing = [str(p) for p in (tx, cells, mtx) if not nonempty(p)]
    if missing:
        stats["error"] = "missing_or_empty: " + ", ".join(missing)
        return False, stats
    try:
        stats["n_transcripts"] = int(len(pd.read_parquet(tx)))
        stats["n_cells"] = int(len(pd.read_parquet(cells)))
        stats["n_mtx_entries"] = int(mmread(str(mtx)).nnz)
    except Exception as exc:
        stats["error"] = f"{type(exc).__name__}: {exc}"
        return False, stats
    ok = stats["n_transcripts"] > 0 and stats["n_cells"] > 0 and stats["n_mtx_entries"] > 0
    if not ok:
        stats["error"] = "zero rows or zero matrix entries"
    return ok, stats


def base_row(method: str) -> dict[str, Any]:
    return {
        "method": method,
        "status": "FAIL",
        "stub": False,
        "method_info_path": "",
        "n_transcripts": 0,
        "n_removed_transcripts": 0,
        "n_retained_transcripts": 0,
        "n_cells": 0,
        "n_mtx_entries": 0,
        "n_shared_genes": 0,
        "external_scrna_reference_used": "",
        "pseudo_reference_debug": "",
        "debug_cluster_labels": "",
        "error": "",
        "notes": "",
    }


def validate_standardized(results: Path, method: str, subdir: str) -> dict[str, Any]:
    row = base_row(method)
    root = results / subdir
    info_path = root / "method_info.json"
    info = read_json(info_path)
    row["method_info_path"] = str(info_path) if info_path.exists() else ""
    row["stub"] = info_is_stub(info)
    ok, stats = standardized_stats(root)
    row.update({k: v for k, v in stats.items() if k in row})
    if ok and info and not row["stub"]:
        row["status"] = "PASS"
    else:
        row["error"] = stats.get("error") or "method_info missing or marked stub"
    return row


def validate_ovrlpy(results: Path) -> dict[str, Any]:
    row = base_row("ovrlpy_xenium_default")
    root = results / "ovrlpy_xenium_default"
    info_path = root / "method_info.json"
    info = read_json(info_path)
    row["method_info_path"] = str(info_path) if info_path.exists() else ""
    row["stub"] = info_is_stub(info)
    required = [root / "signal_integrity.parquet", info_path]
    has_provenance = any(info.get(key) for key in ("method_version", "command", "container_or_env", "git_commit"))
    status_ok = info.get("status") in (None, "PASS")
    if all(nonempty(p) for p in required) and info and not row["stub"] and has_provenance and status_ok:
        row["status"] = "PASS"
    else:
        row["error"] = "missing signal_integrity.parquet, method_info, acceptable status, or non-stub provenance"
    return row


def validate_segger(results: Path) -> dict[str, Any]:
    row = base_row("segger")
    info_path = results / "segger" / "raw" / "method_info.json"
    info = read_json(info_path)
    row["method_info_path"] = str(info_path) if info_path.exists() else ""
    row["stub"] = info_is_stub(info)
    row["status"] = "SKIPPED"
    row["notes"] = "Skipped for this TSU-20 rescue run by request; no TRACER or fake Segger outputs were generated."
    return row


def validate_split(results: Path) -> dict[str, Any]:
    row = base_row("split_xenium_default")
    root = results / "split_xenium_default"
    info_path = root / "method_info.json"
    info = read_json(info_path)
    row["method_info_path"] = str(info_path) if info_path.exists() else ""
    row["stub"] = info_is_stub(info)
    row["external_scrna_reference_used"] = info.get("external_scrna_reference_used", "")
    row["pseudo_reference_debug"] = info.get("pseudo_reference_debug", "")
    row["n_cells"] = int(info.get("n_spatial_cells") or 0)
    row["n_shared_genes"] = int(info.get("n_shared_genes") or 0)
    required = [
        root / "split_result.rds",
        root / "purified_counts.mtx",
        root / "post_processed_RCTD.rds",
        info_path,
    ]
    if nonempty(root / "purified_counts.mtx"):
        try:
            row["n_mtx_entries"] = int(mmread(str(root / "purified_counts.mtx")).nnz)
        except Exception as exc:
            row["error"] = f"could not read purified_counts.mtx: {type(exc).__name__}: {exc}"
    ok = (
        all(nonempty(p) for p in required)
        and info.get("status") == "PASS"
        and info.get("external_scrna_reference_used") is True
        and info.get("pseudo_reference_debug") is False
        and not row["stub"]
    )
    if ok:
        row["status"] = "PASS"
    elif not row["error"]:
        missing = [str(p) for p in required if not nonempty(p)]
        bits = []
        if missing:
            bits.append("missing_or_empty: " + ", ".join(missing))
        if info.get("status") != "PASS":
            bits.append(f"method_info status={info.get('status')!r}")
        if info.get("external_scrna_reference_used") is not True:
            bits.append("external_scrna_reference_used is not true")
        if info.get("pseudo_reference_debug") is not False:
            bits.append("pseudo_reference_debug is not false")
        if row["stub"]:
            bits.append("method_info marks stub")
        row["error"] = "; ".join(bits)
    return row


def validate_celladmix(results: Path) -> dict[str, Any]:
    row = base_row("celladmix_xenium_default")
    root = results / "celladmix_xenium_default"
    info_path = root / "method_info.json"
    info = read_json(info_path)
    row["method_info_path"] = str(info_path) if info_path.exists() else ""
    row["stub"] = info_is_stub(info)
    row["debug_cluster_labels"] = info.get("debug_cluster_labels", "")
    row["n_transcripts"] = int(info.get("n_input_transcripts") or 0)
    row["n_removed_transcripts"] = int(info.get("n_removed_transcripts") or 0)
    row["n_retained_transcripts"] = int(info.get("n_retained_transcripts") or 0)
    row["n_cells"] = int(info.get("n_cells") or 0)
    required = [
        root / "cleaned_transcripts.parquet",
        root / "corrected_counts.mtx",
        info_path,
    ]
    if nonempty(root / "corrected_counts.mtx"):
        try:
            row["n_mtx_entries"] = int(mmread(str(root / "corrected_counts.mtx")).nnz)
        except Exception as exc:
            row["error"] = f"could not read corrected_counts.mtx: {type(exc).__name__}: {exc}"
    ok = (
        all(nonempty(p) for p in required)
        and info.get("status") in {"PASS", "DEBUG_PASS"}
        and not row["stub"]
    )
    if ok:
        row["status"] = "DEBUG_PASS" if info.get("debug_cluster_labels") is True else str(info.get("status"))
    elif info.get("status") == "BLOCKED":
        row["status"] = "BLOCKED"
        row["notes"] = str(info.get("reason") or "")
    elif not row["error"]:
        missing = [str(p) for p in required if not nonempty(p)]
        row["error"] = "missing_or_empty/status/stub check failed"
        if missing:
            row["error"] += ": " + ", ".join(missing)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, help="Top-level TSU-20 result directory.")
    parser.add_argument("--out", required=True, help="Output validation CSV path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = Path(args.results)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        validate_standardized(results, "xenium_default", "standardized/xenium_default"),
        validate_standardized(results, "baysor", "standardized/baysor"),
        validate_standardized(results, "proseg", "standardized/proseg"),
        validate_ovrlpy(results),
        validate_segger(results),
        validate_split(results),
        validate_celladmix(results),
    ]
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote validation CSV: {out}")
    print(pd.DataFrame(rows)[["method", "status", "stub", "error", "notes"]].to_string(index=False))
    failures = [r for r in rows if r["status"] == "FAIL"]
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

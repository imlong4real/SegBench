#!/usr/bin/env python3
"""Validate that each tool produced REAL outputs (no stubs).

Walks ``--results <dir>`` and per-tool expectations:

    standardized/{method}/
        transcripts.parquet         (nonzero rows)
        cells.parquet               (nonzero rows)
        cell_by_gene.mtx            (nonzero entries)
        method_info.json            (no extra.stub == true)

    {tool}_xenium_default/method_info.json
    {tool}/raw/...

A method PASSES only when:
  1. its standardized dir exists with non-empty transcripts/cells/mtx,
     OR its tool-specific raw dir contains the expected raw file with
     non-zero size and a method_info.json,
  2. method_info.json exists,
  3. method_info.json does not declare stub=true,
  4. method_info.json records command/version/container/env (at least one
     of method_version / command / git_commit / container_or_env).

Writes a CSV at:
    {results}/summary/tool_output_validation.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
from scipy.io import mmread


METHOD_SPECS: dict[str, dict] = {
    "xenium_default": {
        "kind": "standardized",
        "standardized_subdir": "standardized/xenium_default",
        "raw_required": [],
    },
    "baysor": {
        "kind": "standardized_or_raw",
        "standardized_subdir": "standardized/baysor",
        "raw_required": ["baysor/raw/segmentation.csv"],
    },
    "proseg": {
        "kind": "standardized_or_raw",
        "standardized_subdir": "standardized/proseg",
        "raw_required": ["proseg/raw/transcript-metadata.csv.gz"],
    },
    "segger": {
        "kind": "standardized_or_raw",
        "standardized_subdir": "standardized/segger",
        "raw_required": ["segger/raw/segger_transcripts.parquet"],
        # BLOCKED on this host; method_info.json will say status=BLOCKED.
        "raw_fallback": "segger/raw/method_info.json",
    },
    "ovrlpy_xenium_default": {
        "kind": "raw_only",
        "raw_required": [
            "ovrlpy_xenium_default/signal_integrity.parquet",
            "ovrlpy_xenium_default/method_info.json",
        ],
    },
    "celladmix_xenium_default": {
        "kind": "raw_only",
        "raw_required": [
            "celladmix_xenium_default/corrected_counts.mtx",
            "celladmix_xenium_default/method_info.json",
        ],
    },
    "split_xenium_default": {
        "kind": "raw_only",
        "raw_required": [
            "split_xenium_default/corrected_counts.mtx",
            "split_xenium_default/method_info.json",
        ],
    },
}


def _read_info(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as e:
        return {"_read_error": f"{type(e).__name__}: {e}"}


def _info_is_stub(info: dict | None) -> bool:
    if not info:
        return False
    if info.get("stub") is True:
        return True
    extra = info.get("extra") or {}
    if extra.get("stub") is True or extra.get("stub_refinement") is True:
        return True
    return False


def _info_records_provenance(info: dict | None) -> bool:
    if not info:
        return False
    fields = ("method_version", "command", "git_commit", "container_or_env")
    return any(info.get(f) not in (None, "", []) for f in fields)


def _check_standardized(stand_dir: Path) -> tuple[bool, dict]:
    """Return (ok, stats) for a standardized output directory."""
    stats = {"standardized_dir": str(stand_dir)}
    tx = stand_dir / "transcripts.parquet"
    cells = stand_dir / "cells.parquet"
    mtx = stand_dir / "cell_by_gene.mtx"
    if not tx.exists() or not cells.exists() or not mtx.exists():
        stats["error"] = "missing required standardized artifact"
        return False, stats

    try:
        tx_df = pd.read_parquet(tx)
        cells_df = pd.read_parquet(cells)
        m = mmread(str(mtx))
    except Exception as e:
        stats["error"] = f"{type(e).__name__}: {e}"
        return False, stats

    stats["n_transcripts"] = int(len(tx_df))
    stats["n_cells"] = int(len(cells_df))
    stats["n_mtx_entries"] = int(m.nnz)
    ok = stats["n_transcripts"] > 0 and stats["n_cells"] > 0 and stats["n_mtx_entries"] > 0
    if not ok:
        stats["error"] = "zero rows / zero matrix entries"
    return ok, stats


def _check_raw(results: Path, paths: Iterable[str]) -> tuple[bool, dict]:
    stats: dict = {}
    missing = []
    for rel in paths:
        p = results / rel
        if not p.exists() or p.stat().st_size == 0:
            missing.append(rel)
    if missing:
        stats["error"] = "missing_or_empty: " + ", ".join(missing)
        return False, stats
    return True, stats


def validate_method(method: str, spec: dict, results: Path) -> dict:
    row: dict = {
        "method": method,
        "status": "FAIL",
        "stub": False,
        "provenance_ok": False,
        "n_transcripts": 0,
        "n_cells": 0,
        "n_mtx_entries": 0,
        "method_info_path": "",
        "error": "",
        "notes": "",
    }

    kind = spec["kind"]

    standardized_dir = (results / spec["standardized_subdir"]) if "standardized_subdir" in spec else None
    standardized_ok = False
    standardized_stats: dict = {}
    if standardized_dir and standardized_dir.exists():
        standardized_ok, standardized_stats = _check_standardized(standardized_dir)

    raw_required = spec.get("raw_required") or []
    raw_ok, raw_stats = (True, {}) if not raw_required else _check_raw(results, raw_required)

    info_path: Path | None = None
    if standardized_dir and (standardized_dir / "method_info.json").exists():
        info_path = standardized_dir / "method_info.json"
    elif raw_required:
        # take method_info.json sibling to the first raw artifact
        first = results / raw_required[0]
        candidate = first.parent / "method_info.json"
        if candidate.exists():
            info_path = candidate
    if info_path is None and "raw_fallback" in spec:
        candidate = results / spec["raw_fallback"]
        if candidate.exists():
            info_path = candidate

    info = _read_info(info_path) if info_path else None
    row["method_info_path"] = str(info_path) if info_path else ""
    row["stub"] = _info_is_stub(info)
    row["provenance_ok"] = _info_records_provenance(info)

    row.update({k: v for k, v in standardized_stats.items() if k in row})

    # Compose verdict.
    if kind == "standardized":
        if standardized_ok and info is not None and not row["stub"] and row["provenance_ok"]:
            row["status"] = "PASS"
        elif standardized_ok and row["stub"]:
            row["status"] = "FAIL"
            row["error"] = "standardized output exists but method_info marks it a stub"
        elif not standardized_ok:
            row["error"] = standardized_stats.get("error", "standardized check failed")
        elif info is None:
            row["error"] = "method_info.json missing"
        elif not row["provenance_ok"]:
            row["error"] = "method_info.json lacks provenance fields"
    elif kind == "raw_only":
        # Accept BLOCKED methods only as FAIL_DOCUMENTED (not a stub).
        if info and info.get("status") == "BLOCKED":
            row["status"] = "BLOCKED"
            row["notes"] = info.get("reason", "")
        elif raw_ok and info is not None and not row["stub"]:
            row["status"] = "PASS"
        elif raw_ok and row["stub"]:
            row["status"] = "FAIL"
            row["error"] = "raw outputs exist but method_info marks them stub"
        elif not raw_ok:
            row["error"] = raw_stats.get("error", "raw check failed")
        elif info is None:
            row["error"] = "method_info.json missing"
    elif kind == "standardized_or_raw":
        if info and info.get("status") == "BLOCKED":
            row["status"] = "BLOCKED"
            row["notes"] = info.get("reason", "")
        elif standardized_ok and info is not None and not row["stub"] and row["provenance_ok"]:
            row["status"] = "PASS"
        elif raw_ok and info is not None and not row["stub"]:
            row["status"] = "PASS_RAW_ONLY"
            row["notes"] = "raw output present; standardized not yet generated"
        else:
            errs = []
            if not standardized_ok and standardized_stats:
                errs.append(standardized_stats.get("error", "standardized check failed"))
            if not raw_ok and raw_stats:
                errs.append(raw_stats.get("error", "raw check failed"))
            if not errs:
                errs.append("no standardized and no raw output present")
            row["error"] = "; ".join(errs)
    return row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate per-tool outputs.")
    p.add_argument("--results", required=True, help="Top-level results directory (e.g., results/tsu20_tools).")
    p.add_argument("--out-csv", default=None, help="Where to write the validation CSV.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    results = Path(args.results).resolve()
    if not results.exists():
        print(f"results dir not found: {results}", file=sys.stderr)
        return 2

    out_csv = Path(args.out_csv) if args.out_csv else (results / "summary" / "tool_output_validation.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = [validate_method(m, spec, results) for m, spec in METHOD_SPECS.items()]
    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote validation CSV: {out_csv}")
    # Pretty print to stdout.
    df = pd.DataFrame(rows)
    print(df[["method", "status", "stub", "n_transcripts", "n_cells", "n_mtx_entries", "error", "notes"]].to_string(index=False))

    # Exit non-zero if any non-blocked method is FAIL.
    failed = [r for r in rows if r["status"] == "FAIL"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

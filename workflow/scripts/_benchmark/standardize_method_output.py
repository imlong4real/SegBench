#!/usr/bin/env python3
"""Convert a raw method output into the benchmark's standardized contract.

Usage:
    python standardize_method_output.py \
        --method xenium_default \
        --xenium-dir dataset/lung_cancer_xenium_10x/TSU-20 \
        --out-dir results/lung_tiny/standardized/xenium_default \
        [--qv-threshold 30] [--max-transcripts 100000] \
        [--roi-xmin .. --roi-xmax .. --roi-ymin .. --roi-ymax ..]

    python standardize_method_output.py \
        --method baysor \
        --baysor-segmentation-csv path/to/segmentation.csv \
        --xenium-dir dataset/lung_cancer_xenium_10x/TSU-20 \
        --out-dir results/lung_small/standardized/baysor

The script is intentionally tolerant of inputs that lack some optional
columns (e.g. z_location, qv). Missing columns are filled with sensible
defaults so the downstream metrics layer can always rely on the contract.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from _io_contract import (  # noqa: E402
    MethodInfo,
    write_standardized_outputs,
)

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _utcnow() -> str:
    return dt.datetime.utcnow().isoformat() + "Z"


def _read_xenium_transcripts(xenium_dir: Path) -> pd.DataFrame:
    p = xenium_dir / "transcripts.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Xenium transcripts.parquet not found at {p}")
    df = pd.read_parquet(p)
    # Newer Xenium bundles use feature_name + cell_id (string-or-bytes).
    if "cell_id" in df.columns and "cell_id_xenium_default" not in df.columns:
        df = df.rename(columns={"cell_id": "cell_id_xenium_default"})
    if isinstance(df["cell_id_xenium_default"].iloc[0], bytes):
        df["cell_id_xenium_default"] = df["cell_id_xenium_default"].str.decode(
            "utf-8", errors="ignore"
        )
    # Make sure feature_name is plain string.
    if "feature_name" in df.columns and isinstance(
        df["feature_name"].iloc[0], bytes
    ):
        df["feature_name"] = df["feature_name"].str.decode("utf-8", errors="ignore")
    if "transcript_id" not in df.columns:
        df["transcript_id"] = np.arange(len(df), dtype=np.int64)
    if "z_location" not in df.columns:
        df["z_location"] = 0.0
    return df


def _read_xenium_cells(xenium_dir: Path) -> pd.DataFrame:
    p = xenium_dir / "cells.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    if "cell_id" in df.columns:
        df = df.rename(columns={"cell_id": "cell_id_method"})
    if (
        df.shape[0] > 0
        and isinstance(df["cell_id_method"].iloc[0], bytes)
    ):
        df["cell_id_method"] = df["cell_id_method"].str.decode("utf-8", errors="ignore")

    # Xenium cells.parquet may use different column names; normalize.
    rename_map = {
        "x_centroid": "x_centroid",
        "y_centroid": "y_centroid",
        "total_counts": "n_transcripts",
        "transcript_counts": "n_transcripts",
        "cell_area": "area",
        "nucleus_area": "nucleus_area",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "n_genes" not in df.columns and "control_codeword_counts" in df.columns:
        df["n_genes"] = np.nan
    return df


def _apply_filters(
    df: pd.DataFrame,
    *,
    qv_threshold: float | None,
    roi: dict | None,
    max_transcripts: int | None,
    seed: int = 0,
) -> pd.DataFrame:
    out = df
    # Drop dummy / negative-control features when the column tells us how.
    if "is_gene" in out.columns:
        out = out[out["is_gene"].astype(bool)]
    else:
        # Best-effort: filter codewords with conventional prefixes.
        if "feature_name" in out.columns:
            pat = r"^(BLANK_|NegControl|Codeword|antisense_|UnassignedCodeword)"
            mask = ~out["feature_name"].astype(str).str.match(pat)
            out = out[mask]

    if qv_threshold is not None and "qv" in out.columns:
        out = out[out["qv"] >= float(qv_threshold)]

    if roi:
        x_min, x_max = roi.get("x_min"), roi.get("x_max")
        y_min, y_max = roi.get("y_min"), roi.get("y_max")
        if x_min is not None:
            out = out[out["x_location"] >= float(x_min)]
        if x_max is not None:
            out = out[out["x_location"] <= float(x_max)]
        if y_min is not None:
            out = out[out["y_location"] >= float(y_min)]
        if y_max is not None:
            out = out[out["y_location"] <= float(y_max)]

    if max_transcripts is not None and len(out) > int(max_transcripts):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(out), size=int(max_transcripts), replace=False)
        out = out.iloc[np.sort(idx)].copy()

    return out.reset_index(drop=True)


# ----------------------------------------------------------------------------
# Adapters
# ----------------------------------------------------------------------------


def standardize_xenium_default(args) -> dict[str, str]:
    xenium_dir = Path(args.xenium_dir)
    start = _utcnow()

    transcripts = _read_xenium_transcripts(xenium_dir)
    cells = _read_xenium_cells(xenium_dir)

    transcripts = _apply_filters(
        transcripts,
        qv_threshold=args.qv_threshold,
        roi=_roi_from_args(args),
        max_transcripts=args.max_transcripts,
    )
    # For Xenium default, cell_id_method == cell_id_xenium_default.
    transcripts["cell_id_method"] = transcripts["cell_id_xenium_default"]
    transcripts["method"] = "xenium_default"
    transcripts["assignment_source"] = "xenium_default"

    if not cells.empty:
        # n_transcripts based on the filtered transcript set, not the raw bundle.
        # Drop any pre-existing n_transcripts / n_genes columns before merging
        # so we don't collide on duplicate names.
        cells = cells.drop(
            columns=[c for c in ("n_transcripts", "n_genes") if c in cells.columns]
        )
        cells["cell_id_method"] = cells["cell_id_method"].astype("string")
        per_cell = (
            transcripts.assign(
                cell_id_method=transcripts["cell_id_method"].astype("string")
            )
            .groupby("cell_id_method", observed=True)
            .agg(
                n_transcripts=("transcript_id", "size"),
                n_genes=("feature_name", "nunique"),
            )
        )
        cells = cells.merge(per_cell, left_on="cell_id_method", right_index=True, how="left")
        cells["n_transcripts"] = cells["n_transcripts"].fillna(0).astype("int64")
        cells["n_genes"] = cells["n_genes"].fillna(0).astype("int64")

    info = MethodInfo(
        method_name="xenium_default",
        command=" ".join(sys.argv),
        input_files=[str(xenium_dir / "transcripts.parquet"), str(xenium_dir / "cells.parquet")],
        start_time=start,
        threads=args.threads,
        container_or_env=os.environ.get("CONDA_DEFAULT_ENV"),
        extra={
            "qv_threshold": args.qv_threshold,
            "max_transcripts": args.max_transcripts,
            "roi": _roi_from_args(args),
        },
    )
    return write_standardized_outputs(
        args.out_dir,
        transcripts=transcripts,
        cells=cells if not cells.empty else None,
        method="xenium_default",
        method_info=info,
    )


def _read_baysor_segmentation(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Baysor segmentation.csv typically has: molecule_id, cell, gene, x, y, z, ...
    # When Baysor is run from Xenium transcripts.parquet, it can also preserve
    # the original Xenium transcript_id. Prefer that real identifier over the
    # sequential Baysor molecule_id so provenance merges stay one-column clean.
    if "transcript_id" not in df.columns and "molecule_id" in df.columns:
        df = df.rename(columns={"molecule_id": "transcript_id"})
    rename = {
        "cell": "cell_id_method",
        "gene": "feature_name",
        "x": "x_location",
        "y": "y_location",
        "z": "z_location",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "z_location" not in df.columns:
        df["z_location"] = 0.0
    return df


def standardize_baysor(args) -> dict[str, str]:
    start = _utcnow()
    df = _read_baysor_segmentation(Path(args.baysor_segmentation_csv))

    # Optional: merge xenium cell_id to preserve provenance.
    if args.xenium_transcripts and Path(args.xenium_transcripts).exists():
        xt = pd.read_parquet(args.xenium_transcripts)
        if "cell_id" in xt.columns:
            xt = xt.rename(columns={"cell_id": "cell_id_xenium_default"})
        if "transcript_id" in xt.columns and "transcript_id" in df.columns:
            df = df.merge(
                xt[["transcript_id", "cell_id_xenium_default"]],
                on="transcript_id",
                how="left",
            )

    df = _apply_filters(
        df,
        qv_threshold=args.qv_threshold,
        roi=_roi_from_args(args),
        max_transcripts=args.max_transcripts,
    )

    df["method"] = "baysor"
    df["assignment_source"] = "baysor"

    info = MethodInfo(
        method_name="baysor",
        command=" ".join(sys.argv),
        input_files=[str(args.baysor_segmentation_csv)],
        start_time=start,
        threads=args.threads,
        container_or_env=os.environ.get("CONDA_DEFAULT_ENV"),
    )
    return write_standardized_outputs(
        args.out_dir,
        transcripts=df,
        cells=None,
        method="baysor",
        method_info=info,
    )


def standardize_proseg(args) -> dict[str, str]:
    start = _utcnow()
    tm_path = Path(args.proseg_transcript_metadata)
    df = pd.read_csv(tm_path, engine="pyarrow")
    rename = {
        "transcript_id": "transcript_id",
        "assignment": "cell_id_method",
        "gene": "feature_name",
        "x": "x_location",
        "y": "y_location",
        "z": "z_location",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    # In proseg, unassigned transcripts get the max integer cell id.
    # We mark those "UNASSIGNED" and force the whole column to string so
    # parquet writers don't fight a mixed int/str column.
    max_int = df["cell_id_method"].max()
    df["cell_id_method"] = df["cell_id_method"].astype("string")
    df.loc[df["cell_id_method"] == str(max_int), "cell_id_method"] = "UNASSIGNED"

    df = _apply_filters(
        df,
        qv_threshold=args.qv_threshold,
        roi=_roi_from_args(args),
        max_transcripts=args.max_transcripts,
    )
    df["method"] = "proseg"
    df["assignment_source"] = "proseg"

    info = MethodInfo(
        method_name="proseg",
        command=" ".join(sys.argv),
        input_files=[str(tm_path)],
        start_time=start,
        threads=args.threads,
        container_or_env=os.environ.get("CONDA_DEFAULT_ENV"),
    )
    return write_standardized_outputs(
        args.out_dir,
        transcripts=df,
        cells=None,
        method="proseg",
        method_info=info,
    )


def standardize_segger(args) -> dict[str, str]:
    start = _utcnow()
    df = pd.read_parquet(args.segger_transcripts)
    rename = {
        "segger_cell_id": "cell_id_method",
        "cell_id": "cell_id_method",
        "gene": "feature_name",
        "feature_name": "feature_name",
        "x_location": "x_location",
        "y_location": "y_location",
        "z_location": "z_location",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df = _apply_filters(
        df,
        qv_threshold=args.qv_threshold,
        roi=_roi_from_args(args),
        max_transcripts=args.max_transcripts,
    )
    df["method"] = "segger"
    df["assignment_source"] = "segger"

    info = MethodInfo(
        method_name="segger",
        command=" ".join(sys.argv),
        input_files=[str(args.segger_transcripts)],
        start_time=start,
        threads=args.threads,
        container_or_env=os.environ.get("CONDA_DEFAULT_ENV"),
    )
    return write_standardized_outputs(
        args.out_dir,
        transcripts=df,
        cells=None,
        method="segger",
        method_info=info,
    )


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def _roi_from_args(args) -> dict | None:
    roi = {
        "x_min": args.roi_xmin,
        "x_max": args.roi_xmax,
        "y_min": args.roi_ymin,
        "y_max": args.roi_ymax,
    }
    if all(v is None for v in roi.values()):
        return None
    return roi


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Standardize a method output for benchmarking.")
    p.add_argument("--method", required=True,
                   choices=["xenium_default", "baysor", "proseg", "segger"])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--xenium-dir", default=None,
                   help="Path to the original Xenium bundle (used by xenium_default and to attach xenium cell ids).")
    p.add_argument("--xenium-transcripts", default=None,
                   help="Optional path to xenium transcripts.parquet for merging xenium ids onto third-party outputs.")
    p.add_argument("--baysor-segmentation-csv", default=None)
    p.add_argument("--proseg-transcript-metadata", default=None)
    p.add_argument("--segger-transcripts", default=None)
    p.add_argument("--qv-threshold", type=float, default=None)
    p.add_argument("--max-transcripts", type=int, default=None)
    p.add_argument("--roi-xmin", type=float, default=None)
    p.add_argument("--roi-xmax", type=float, default=None)
    p.add_argument("--roi-ymin", type=float, default=None)
    p.add_argument("--roi-ymax", type=float, default=None)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--log", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.log:
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(args.log, "w", encoding="utf-8")
        sys.stdout = log_fh
        sys.stderr = log_fh

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    if args.method == "xenium_default":
        if not args.xenium_dir:
            raise SystemExit("xenium_default requires --xenium-dir")
        result = standardize_xenium_default(args)
    elif args.method == "baysor":
        if not args.baysor_segmentation_csv:
            raise SystemExit("baysor requires --baysor-segmentation-csv")
        if not args.xenium_transcripts and args.xenium_dir:
            args.xenium_transcripts = str(Path(args.xenium_dir) / "transcripts.parquet")
        result = standardize_baysor(args)
    elif args.method == "proseg":
        if not args.proseg_transcript_metadata:
            raise SystemExit("proseg requires --proseg-transcript-metadata")
        result = standardize_proseg(args)
    elif args.method == "segger":
        if not args.segger_transcripts:
            raise SystemExit("segger requires --segger-transcripts")
        result = standardize_segger(args)
    else:  # pragma: no cover
        raise SystemExit(f"Unknown method: {args.method}")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Shared IO helpers for the benchmark layer.

Every method-runner writes the same set of files into a "standardized
output" directory. This module centralises that contract so that the
downstream metrics / TRACER / cellAdmix scripts only have to know about
one shape.

Contract (per method):
    {standardized_dir}/
        transcripts.parquet
        cells.parquet
        cell_by_gene.mtx           (MatrixMarket)
        cell_by_gene_barcodes.tsv
        cell_by_gene_features.tsv
        cell_metadata.parquet
        method_info.json
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import socket
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmwrite

# --- Column contract -------------------------------------------------------

TRANSCRIPT_COLUMNS: tuple[str, ...] = (
    "transcript_id",
    "feature_name",
    "x_location",
    "y_location",
    "z_location",
    "qv",
    "cell_id_xenium_default",
    "cell_id_method",
    "method",
    "assignment_source",
)

CELL_COLUMNS: tuple[str, ...] = (
    "cell_id_method",
    "x_centroid",
    "y_centroid",
    "n_transcripts",
    "n_genes",
    "area",
    "method",
)


def _git_commit(repo_path: str | None) -> str | None:
    if not repo_path:
        return None
    try:
        out = subprocess.check_output(
            ["git", "-C", repo_path, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


@dataclass
class MethodInfo:
    method_name: str
    method_version: str | None = None
    git_commit: str | None = None
    command: str | None = None
    input_files: list[str] = field(default_factory=list)
    output_files: list[str] = field(default_factory=list)
    start_time: str | None = None
    end_time: str | None = None
    wall_seconds: float | None = None
    threads: int | None = None
    container_or_env: str | None = None
    host: str = field(default_factory=socket.gethostname)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def ensure_transcript_columns(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """Ensure the standardized transcript-table columns exist.

    Any required column missing is filled with a sensible default. Returns
    a new DataFrame with columns ordered per ``TRANSCRIPT_COLUMNS`` and
    other columns appended afterwards.
    """
    df = df.copy()
    if "transcript_id" not in df.columns:
        df["transcript_id"] = np.arange(len(df), dtype=np.int64)
    if "z_location" not in df.columns:
        df["z_location"] = 0.0
    if "qv" not in df.columns:
        df["qv"] = np.nan
    if "cell_id_xenium_default" not in df.columns:
        df["cell_id_xenium_default"] = pd.NA
    if "cell_id_method" not in df.columns:
        raise ValueError("transcripts table must contain 'cell_id_method'")
    if "feature_name" not in df.columns:
        raise ValueError("transcripts table must contain 'feature_name'")
    if "method" not in df.columns:
        df["method"] = method
    if "assignment_source" not in df.columns:
        df["assignment_source"] = method

    extras = [c for c in df.columns if c not in TRANSCRIPT_COLUMNS]
    return df[list(TRANSCRIPT_COLUMNS) + extras]


def ensure_cell_columns(df: pd.DataFrame, method: str) -> pd.DataFrame:
    df = df.copy()
    if "cell_id_method" not in df.columns:
        raise ValueError("cells table must contain 'cell_id_method'")
    for c in ("x_centroid", "y_centroid", "n_transcripts", "n_genes", "area"):
        if c not in df.columns:
            df[c] = np.nan
    if "method" not in df.columns:
        df["method"] = method

    extras = [c for c in df.columns if c not in CELL_COLUMNS]
    return df[list(CELL_COLUMNS) + extras]


def build_cell_by_gene(
    transcripts: pd.DataFrame,
    *,
    drop_unassigned: bool = True,
    unassigned_values: Iterable[Any] = ("UNASSIGNED", "", "0", 0, -1, None),
) -> tuple[sparse.csr_matrix, list[str], list[str]]:
    """Build a sparse cell x gene count matrix from a standardized transcripts table."""
    if drop_unassigned:
        unassigned = set(map(str, unassigned_values))
        unassigned.discard("nan")
        cid = transcripts["cell_id_method"].astype("string")
        keep = ~cid.isna() & ~cid.isin(unassigned)
        t = transcripts.loc[keep]
    else:
        t = transcripts

    barcodes = pd.Index(sorted(t["cell_id_method"].astype("string").unique()))
    features = pd.Index(sorted(t["feature_name"].astype("string").unique()))

    if len(barcodes) == 0 or len(features) == 0:
        empty = sparse.csr_matrix((len(barcodes), len(features)), dtype=np.int32)
        return empty, list(barcodes), list(features)

    bc_to_idx = {b: i for i, b in enumerate(barcodes)}
    ft_to_idx = {f: i for i, f in enumerate(features)}

    rows = t["cell_id_method"].astype("string").map(bc_to_idx).to_numpy()
    cols = t["feature_name"].astype("string").map(ft_to_idx).to_numpy()
    data = np.ones(len(t), dtype=np.int32)

    mat = sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(len(barcodes), len(features)),
        dtype=np.int32,
    ).tocsr()
    mat.sum_duplicates()
    return mat, list(barcodes), list(features)


def derive_cells_from_transcripts(
    transcripts: pd.DataFrame, method: str
) -> pd.DataFrame:
    """When a method does not provide its own cells table, derive it from transcripts."""
    cid = transcripts["cell_id_method"].astype("string")
    keep = ~cid.isna() & ~cid.isin({"UNASSIGNED", "", "0"})
    t = transcripts.loc[keep]
    if t.empty:
        return pd.DataFrame(columns=list(CELL_COLUMNS))

    grouped = t.groupby("cell_id_method", observed=True)
    cells = pd.DataFrame(
        {
            "x_centroid": grouped["x_location"].mean(),
            "y_centroid": grouped["y_location"].mean(),
            "n_transcripts": grouped.size(),
            "n_genes": grouped["feature_name"].nunique(),
        }
    ).reset_index()
    cells["area"] = np.nan
    cells["method"] = method
    return ensure_cell_columns(cells, method)


def write_standardized_outputs(
    out_dir: str | os.PathLike,
    *,
    transcripts: pd.DataFrame,
    cells: pd.DataFrame | None,
    method: str,
    method_info: MethodInfo,
    cell_by_gene_drop_unassigned: bool = True,
) -> dict[str, str]:
    """Write all files defined by the standardized contract.

    Returns a dict of artifact-path-by-name.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    transcripts = ensure_transcript_columns(transcripts, method)
    if cells is None or cells.empty:
        cells = derive_cells_from_transcripts(transcripts, method)
    else:
        cells = ensure_cell_columns(cells, method)

    transcripts_path = out_dir / "transcripts.parquet"
    cells_path = out_dir / "cells.parquet"
    meta_path = out_dir / "cell_metadata.parquet"
    mtx_path = out_dir / "cell_by_gene.mtx"
    barcodes_path = out_dir / "cell_by_gene_barcodes.tsv"
    features_path = out_dir / "cell_by_gene_features.tsv"
    info_path = out_dir / "method_info.json"

    transcripts.to_parquet(transcripts_path, index=False)
    cells.to_parquet(cells_path, index=False)

    metadata_cols = [
        c
        for c in (
            "cell_id_method",
            "n_transcripts",
            "n_genes",
            "area",
            "x_centroid",
            "y_centroid",
            "cluster",
            "cell_type",
            "method",
        )
        if c in cells.columns
    ]
    cells[metadata_cols].to_parquet(meta_path, index=False)

    mat, barcodes, features = build_cell_by_gene(
        transcripts, drop_unassigned=cell_by_gene_drop_unassigned
    )
    mmwrite(str(mtx_path), mat)
    Path(barcodes_path).write_text("\n".join(map(str, barcodes)) + "\n")
    Path(features_path).write_text("\n".join(map(str, features)) + "\n")

    method_info.output_files = [
        str(transcripts_path),
        str(cells_path),
        str(meta_path),
        str(mtx_path),
        str(barcodes_path),
        str(features_path),
    ]
    if method_info.end_time is None:
        method_info.end_time = _dt.datetime.utcnow().isoformat() + "Z"
    if (
        method_info.start_time is not None
        and method_info.wall_seconds is None
    ):
        try:
            t0 = _dt.datetime.fromisoformat(method_info.start_time.rstrip("Z"))
            t1 = _dt.datetime.fromisoformat(method_info.end_time.rstrip("Z"))
            method_info.wall_seconds = (t1 - t0).total_seconds()
        except Exception:
            pass

    info_path.write_text(json.dumps(method_info.to_dict(), indent=2, default=str))
    return {
        "transcripts": str(transcripts_path),
        "cells": str(cells_path),
        "cell_metadata": str(meta_path),
        "cell_by_gene_mtx": str(mtx_path),
        "cell_by_gene_barcodes": str(barcodes_path),
        "cell_by_gene_features": str(features_path),
        "method_info": str(info_path),
    }


def load_standardized_transcripts(path: str | os.PathLike) -> pd.DataFrame:
    return pd.read_parquet(path)


def load_standardized_cells(path: str | os.PathLike) -> pd.DataFrame:
    return pd.read_parquet(path)

"""Shared helpers for benchmark metrics.

We try to import the *real* NPMI implementation from TRACER when it is
importable; otherwise we fall back to a simple pure-Python implementation
so the benchmark can still produce a runnable metric table.
"""

from __future__ import annotations

import importlib
import math
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse


# ---------------------------------------------------------------------------
# NPMI
# ---------------------------------------------------------------------------


def try_import_tracer_npmi(repo_path: str | None = None):
    """Return a callable for TRACER NPMI computation if available, else None."""
    if repo_path:
        repo_path = str(Path(repo_path).resolve())
        for sub in ("", "src"):
            candidate = str(Path(repo_path) / sub) if sub else repo_path
            if os.path.isdir(candidate) and candidate not in sys.path:
                sys.path.insert(0, candidate)

    for mod_name, attr in (
        ("tracer.metrics", "compute_npmi"),
        ("tracer.npmi", "compute_npmi"),
        ("tracer", "compute_npmi"),
    ):
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        if hasattr(mod, attr):
            return getattr(mod, attr)
    return None


def cooccurrence_npmi(
    cell_by_gene: sparse.csr_matrix,
    *,
    eps: float = 1e-12,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Fallback NPMI: pairwise co-occurrence per cell.

    Treats each (cell, gene) entry as a Bernoulli event (gene present in
    cell). For each pair of genes ``(g_i, g_j)`` returns:

        NPMI(g_i, g_j) = log( p(g_i, g_j) / (p(g_i)*p(g_j)) ) / -log p(g_i, g_j)

    Returns the upper-triangular NPMI matrix as a (gene_i, gene_j, npmi)
    long DataFrame and the per-gene presence rate.
    """
    if cell_by_gene.shape[0] == 0:
        return (
            pd.DataFrame(columns=["gene_i", "gene_j", "npmi"]),
            np.zeros(cell_by_gene.shape[1], dtype=float),
        )

    presence = (cell_by_gene > 0).astype(np.int8)
    n_cells = presence.shape[0]
    gene_presence = np.asarray(presence.sum(axis=0)).flatten()  # (G,)
    p_g = gene_presence / max(n_cells, 1)

    # Pair co-occurrence: P^T P
    cooc = (presence.T @ presence).toarray().astype(float)
    np.fill_diagonal(cooc, 0)  # ignore self pairs
    p_pair = cooc / max(n_cells, 1)

    p_i = p_g[:, None]
    p_j = p_g[None, :]
    pmi_denom = (p_i * p_j) + eps
    npmi = np.zeros_like(p_pair)
    mask = (p_pair > eps) & (pmi_denom > eps)
    with np.errstate(divide="ignore", invalid="ignore"):
        npmi[mask] = np.log(p_pair[mask] / pmi_denom[mask]) / (-np.log(p_pair[mask] + eps))

    # Upper triangle long format
    iu = np.triu_indices_from(npmi, k=1)
    long_df = pd.DataFrame(
        {
            "gene_i": iu[0],
            "gene_j": iu[1],
            "npmi": npmi[iu],
        }
    )
    return long_df, p_g


def cell_purity_conflict_from_npmi(
    cell_by_gene: sparse.csr_matrix,
    npmi_long: pd.DataFrame,
    feature_index: list[str],
    *,
    min_molecules_per_cell: int = 10,
) -> pd.DataFrame:
    """Compute per-cell purity / conflict scores from a precomputed NPMI table.

    For each cell:
      - purity = mean NPMI across pairs of genes the cell actually expresses
      - conflict = mean negative NPMI for pairs of genes the cell expresses
        (i.e. the "anti-correlation" score, restricted to genes both > 0)

    Cells with fewer than ``min_molecules_per_cell`` are dropped.
    """
    G = cell_by_gene.shape[1]
    npmi_dense = np.zeros((G, G), dtype=float)
    if not npmi_long.empty:
        npmi_dense[npmi_long["gene_i"].to_numpy(), npmi_long["gene_j"].to_numpy()] = (
            npmi_long["npmi"].to_numpy()
        )
        npmi_dense = npmi_dense + npmi_dense.T

    cells = []
    presence = (cell_by_gene > 0).astype(np.int8)
    cell_totals = np.asarray(cell_by_gene.sum(axis=1)).flatten()
    for ci in range(cell_by_gene.shape[0]):
        if cell_totals[ci] < min_molecules_per_cell:
            continue
        gi = presence[ci].toarray().flatten().astype(bool)
        if gi.sum() < 2:
            continue
        sub = npmi_dense[np.ix_(gi, gi)]
        iu = np.triu_indices_from(sub, k=1)
        vals = sub[iu]
        if vals.size == 0:
            continue
        purity = float(np.mean(np.clip(vals, 0, None)))
        conflict = float(np.mean(np.clip(-vals, 0, None)))
        cells.append((ci, purity, conflict, int(gi.sum()), int(cell_totals[ci])))

    return pd.DataFrame(
        cells,
        columns=["cell_row", "purity", "conflict", "n_genes", "n_transcripts"],
    )


# ---------------------------------------------------------------------------
# Marker specificity / leakage
# ---------------------------------------------------------------------------


def marker_specificity(
    cell_by_gene: sparse.csr_matrix,
    feature_index: list[str],
    marker_groups: dict[str, list[str]],
) -> pd.DataFrame:
    """For each cell, compute the fraction of marker counts coming from its
    dominant marker group (``specificity``) and from non-dominant groups
    (``leakage``).

    Returns one row per cell with columns:
      cell_row, dominant_group, dominant_count, total_marker_count,
      specificity, leakage
    """
    feat_to_idx = {f: i for i, f in enumerate(feature_index)}

    group_cols: dict[str, list[int]] = {}
    for group, markers in marker_groups.items():
        idx = [feat_to_idx[m] for m in markers if m in feat_to_idx]
        if idx:
            group_cols[group] = idx
    if not group_cols:
        return pd.DataFrame(
            columns=[
                "cell_row",
                "dominant_group",
                "dominant_count",
                "total_marker_count",
                "specificity",
                "leakage",
            ]
        )

    rows = []
    groups = list(group_cols.keys())
    # Pre-extract per-group sub-matrices to avoid repeated slicing.
    per_group_counts = {
        g: np.asarray(cell_by_gene[:, cols].sum(axis=1)).flatten()
        for g, cols in group_cols.items()
    }
    group_matrix = np.vstack([per_group_counts[g] for g in groups]).T  # (n_cells, n_groups)
    total_marker = group_matrix.sum(axis=1)
    dominant_idx = np.argmax(group_matrix, axis=1)
    dominant_count = group_matrix[np.arange(group_matrix.shape[0]), dominant_idx]

    for ci in range(group_matrix.shape[0]):
        if total_marker[ci] == 0:
            continue
        spec = float(dominant_count[ci] / total_marker[ci])
        leak = 1.0 - spec
        rows.append(
            (
                ci,
                groups[dominant_idx[ci]],
                int(dominant_count[ci]),
                int(total_marker[ci]),
                spec,
                leak,
            )
        )
    return pd.DataFrame(
        rows,
        columns=[
            "cell_row",
            "dominant_group",
            "dominant_count",
            "total_marker_count",
            "specificity",
            "leakage",
        ],
    )


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_standardized_dir(stand_dir: str | os.PathLike) -> dict:
    """Load every artifact of the standardized contract from a directory."""
    p = Path(stand_dir)
    from scipy.io import mmread

    transcripts = pd.read_parquet(p / "transcripts.parquet")
    cells = pd.read_parquet(p / "cells.parquet")
    cell_by_gene = mmread(p / "cell_by_gene.mtx").tocsr()
    barcodes = (p / "cell_by_gene_barcodes.tsv").read_text().splitlines()
    features = (p / "cell_by_gene_features.tsv").read_text().splitlines()
    import json as _json

    info = _json.loads((p / "method_info.json").read_text())
    return {
        "transcripts": transcripts,
        "cells": cells,
        "cell_by_gene": cell_by_gene,
        "barcodes": barcodes,
        "features": features,
        "method_info": info,
    }

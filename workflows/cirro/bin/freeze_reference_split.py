#!/usr/bin/env python3
"""Create a deterministic, stratified scRNA train/evaluation split.

The cell lists are the source of truth.  Their hashes and the resulting h5ad
hashes are recorded so a Cirro run cannot silently exchange the SPLIT training
reference for the held-out evaluation reference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp


def sha256(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def write_subset(source_path: Path, obs: pd.DataFrame, var: pd.DataFrame,
                 rows: np.ndarray, columns: np.ndarray, destination: Path,
                 row_chunk: int = 256) -> None:
    """Stream selected CSR rows/columns without loading the 72M-edge matrix.

    anndata's backed mode still materialises sparse layers in this source file.
    Reading the HDF5 CSR arrays directly keeps peak memory bounded while also
    calculating true whole-transcriptome library depth for every selected cell.
    """
    col_map = np.full(len(var), -1, dtype=np.int32)
    col_map[columns] = np.arange(len(columns), dtype=np.int32)
    blocks: list[sp.csr_matrix] = []
    depths: list[np.ndarray] = []
    with h5py.File(source_path, "r") as handle:
        group = handle["layers/counts"] if "counts" in handle["layers"] else handle["X"]
        indptr = group["indptr"]
        data = group["data"]
        indices = group["indices"]
        for start in range(0, len(rows), row_chunk):
            batch = rows[start:start + row_chunk]
            out_data: list[np.ndarray] = []
            out_indices: list[np.ndarray] = []
            out_indptr = [0]
            batch_depth = np.zeros(len(batch), dtype=np.float64)
            for j, source_row in enumerate(batch):
                lo, hi = int(indptr[source_row]), int(indptr[source_row + 1])
                values = np.asarray(data[lo:hi])
                source_cols = np.asarray(indices[lo:hi])
                batch_depth[j] = float(values.sum(dtype=np.float64))
                mapped = col_map[source_cols]
                keep = mapped >= 0
                out_data.append(values[keep])
                out_indices.append(mapped[keep])
                out_indptr.append(out_indptr[-1] + int(keep.sum()))
            block_data = np.concatenate(out_data) if out_data else np.array([], dtype=np.float32)
            block_indices = (np.concatenate(out_indices) if out_indices
                             else np.array([], dtype=np.int32))
            blocks.append(sp.csr_matrix(
                (block_data, block_indices, np.asarray(out_indptr, dtype=np.int32)),
                shape=(len(batch), len(columns))))
            depths.append(batch_depth)
    matrix = sp.vstack(blocks, format="csr")
    subset_obs = obs.iloc[rows].copy()
    subset_obs["_full_library_total_counts"] = np.concatenate(depths)
    subset = ad.AnnData(X=matrix, obs=subset_obs, var=var.iloc[columns].copy())
    subset.layers["counts"] = matrix.copy()
    subset.write_h5ad(destination, compression="gzip")
    del subset, matrix, blocks, depths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--celltype-col", required=True)
    parser.add_argument("--panel-genes", type=Path, required=True,
                        help="Parquet with feature_name or one-gene-per-line text file")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260909)
    args = parser.parse_args()
    if not 0 < args.holdout_fraction < 1:
        raise SystemExit("--holdout-fraction must be between zero and one")

    args.outdir.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.reference, "r") as handle:
        obs = ad.io.read_elem(handle["obs"])
        var = ad.io.read_elem(handle["var"])
    if args.celltype_col not in obs:
        raise SystemExit(f"missing cell-type column: {args.celltype_col}")

    cell_ids = obs.index.astype(str).to_numpy()
    labels = obs[args.celltype_col].astype(str).to_numpy()
    if args.panel_genes.suffix == ".parquet":
        panel_genes = pd.read_parquet(args.panel_genes, columns=["feature_name"])[
            "feature_name"].astype(str).unique().tolist()
    else:
        panel_genes = [line.strip() for line in args.panel_genes.read_text().splitlines()
                       if line.strip()]
    var_names = var.index.astype(str).to_numpy()
    wanted = set(panel_genes)
    columns = np.flatnonzero(np.isin(var_names, list(wanted)))
    if not len(columns):
        raise SystemExit("no panel genes overlap the reference")
    train_idx: list[int] = []
    holdout_idx: list[int] = []
    strata: dict[str, dict[str, int]] = {}
    for label in sorted(set(labels)):
        idx = np.flatnonzero(labels == label)
        # Sort first so the split is independent of source row order.
        idx = idx[np.argsort(cell_ids[idx], kind="stable")]
        label_seed = int.from_bytes(
            hashlib.sha256(f"{args.seed}|{label}".encode()).digest()[:8], "big")
        idx = idx[np.random.default_rng(label_seed).permutation(len(idx))]
        n_holdout = min(len(idx) - 1, max(1, int(round(len(idx) * args.holdout_fraction))))
        holdout_idx.extend(idx[:n_holdout].tolist())
        train_idx.extend(idx[n_holdout:].tolist())
        strata[label] = {
            "total": int(len(idx)),
            "reference_train": int(len(idx) - n_holdout),
            "evaluation_holdout": int(n_holdout),
        }

    train = np.asarray(sorted(train_idx), dtype=np.int64)
    holdout = np.asarray(sorted(holdout_idx), dtype=np.int64)
    if np.intersect1d(train, holdout).size or train.size + holdout.size != len(obs):
        raise RuntimeError("split is not an exact, disjoint partition")

    assignments = pd.DataFrame({
        "cell_id": np.concatenate([cell_ids[train], cell_ids[holdout]]),
        "cell_type": np.concatenate([labels[train], labels[holdout]]),
        "reference_split": (["reference_train"] * len(train)
                            + ["evaluation_holdout"] * len(holdout)),
    }).sort_values(["reference_split", "cell_type", "cell_id"], kind="stable")
    assignment_path = args.outdir / "lung_reference_split.tsv.gz"
    assignments.to_csv(assignment_path, sep="\t", index=False, compression="gzip")

    train_path = args.outdir / "lung_reference_train.h5ad"
    holdout_path = args.outdir / "lung_evaluation_holdout.h5ad"
    write_subset(args.reference, obs, var, train, columns, train_path)
    write_subset(args.reference, obs, var, holdout, columns, holdout_path)

    manifest = {
        "schema_version": "1.0",
        "algorithm": "per-stratum lexicographic-order + PCG64 permutation",
        "seed": args.seed,
        "holdout_fraction_requested": args.holdout_fraction,
        "celltype_column": args.celltype_col,
        "reference_representation": "Xenium-panel-restricted counts with full-library depth in obs",
        "panel": {
            "source_name": args.panel_genes.name,
            "source_sha256": sha256(args.panel_genes),
            "n_input_genes": int(len(set(panel_genes))),
            "n_reference_genes": int(len(columns)),
            "full_library_depth_column": "_full_library_total_counts",
        },
        "source": {"name": args.reference.name, "sha256": sha256(args.reference)},
        "reference_train": {
            "name": train_path.name, "n_cells": int(len(train)),
            "sha256": sha256(train_path),
        },
        "evaluation_holdout": {
            "name": holdout_path.name, "n_cells": int(len(holdout)),
            "sha256": sha256(holdout_path),
            "restriction": "evaluation only; never available to SPLIT or panel building",
        },
        "assignments": {
            "name": assignment_path.name, "n_rows": int(len(assignments)),
            "sha256": sha256(assignment_path),
        },
        "strata": strata,
    }
    (args.outdir / "lung_reference_split_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

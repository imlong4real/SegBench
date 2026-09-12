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
    """Stream selected CSR rows/columns from source h5ad straight to destination.

    anndata's backed mode still materialises sparse layers in this source file,
    and accumulating the subset in memory before writing does not fit the hard
    5 GB user cgroup this runs in: the cervical reference restricted to the
    18k-gene Atera panel is ~1.4 GB as a CSR, and anndata would then want a
    second copy for ``layers["counts"]``.

    So the destination is written in one pass into resizable HDF5 datasets, and
    ``layers/counts`` is an HDF5 hard link to ``X`` rather than a copy.  True
    whole-transcriptome library depth is accumulated on the way through, before
    the gene restriction is applied.
    """
    col_map = np.full(len(var), -1, dtype=np.int32)
    col_map[columns] = np.arange(len(columns), dtype=np.int32)
    n_rows, n_cols = len(rows), len(columns)
    depths = np.zeros(n_rows, dtype=np.float64)
    indptr_acc = np.zeros(n_rows + 1, dtype=np.int64)

    subset_obs = obs.iloc[rows].copy()
    subset_var = var.iloc[columns].copy()
    # Write the scaffold (obs/var/uns) first, then replace X with a streamed one.
    subset_obs["_full_library_total_counts"] = 0.0
    ad.AnnData(obs=subset_obs, var=subset_var,
               X=sp.csr_matrix((n_rows, n_cols), dtype=np.float32)
               ).write_h5ad(destination, compression="gzip")

    with h5py.File(destination, "a") as out:
        del out["X"]
        grp = out.create_group("X")
        grp.attrs["encoding-type"] = "csr_matrix"
        grp.attrs["encoding-version"] = "0.1.0"
        grp.attrs["shape"] = np.array([n_rows, n_cols], dtype="int64")
        d = grp.create_dataset("data", shape=(0,), maxshape=(None,),
                               dtype="float32", chunks=(1 << 18,), compression="gzip")
        i = grp.create_dataset("indices", shape=(0,), maxshape=(None,),
                               dtype="int32", chunks=(1 << 18,), compression="gzip")
        offset = 0
        with h5py.File(source_path, "r") as handle:
            group = (handle["layers/counts"]
                     if "layers" in handle and "counts" in handle["layers"]
                     else handle["X"])
            indptr, data, indices = group["indptr"], group["data"], group["indices"]
            for start in range(0, n_rows, row_chunk):
                batch = rows[start:start + row_chunk]
                out_data: list[np.ndarray] = []
                out_indices: list[np.ndarray] = []
                for j, source_row in enumerate(batch):
                    lo, hi = int(indptr[source_row]), int(indptr[source_row + 1])
                    values = np.asarray(data[lo:hi])
                    source_cols = np.asarray(indices[lo:hi])
                    depths[start + j] = float(values.sum(dtype=np.float64))
                    mapped = col_map[source_cols]
                    keep = mapped >= 0
                    out_data.append(values[keep])
                    out_indices.append(mapped[keep])
                    indptr_acc[start + j + 1] = indptr_acc[start + j] + int(keep.sum())
                bd = (np.concatenate(out_data) if out_data
                      else np.array([], dtype=np.float32))
                bi = (np.concatenate(out_indices) if out_indices
                      else np.array([], dtype=np.int32))
                if bd.size:
                    d.resize((offset + bd.size,))
                    i.resize((offset + bi.size,))
                    d[offset:offset + bd.size] = bd.astype(np.float32)
                    i[offset:offset + bi.size] = bi.astype(np.int32)
                    offset += bd.size
                del out_data, out_indices, bd, bi
        grp.create_dataset("indptr", data=indptr_acc, dtype="int64",
                           compression="gzip")
        # Replace the placeholder depth column now that it is known.
        del out["obs"]["_full_library_total_counts"]
        out["obs"].create_dataset("_full_library_total_counts", data=depths,
                                  dtype="float64", compression="gzip")
        out["obs"]["_full_library_total_counts"].attrs.update(
            {"encoding-type": "array", "encoding-version": "0.2.0"})
        # layers/counts is the same matrix; a hard link avoids a second copy.
        if "layers" not in out:
            g = out.create_group("layers")
            g.attrs.update({"encoding-type": "dict", "encoding-version": "0.1.0"})
        if "counts" in out["layers"]:
            del out["layers"]["counts"]
        out["layers"]["counts"] = out["X"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--celltype-col", required=True)
    parser.add_argument("--panel-genes", type=Path, required=True,
                        help="Parquet with feature_name or one-gene-per-line text file")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--exclude-celltypes", nargs="*", default=[],
                        help="Cell-type labels dropped from BOTH subsets before "
                             "splitting (e.g. Unannotated).")
    parser.add_argument("--prefix", default="lung",
                        help="Output filename prefix.  Defaults to 'lung' so the "
                             "frozen NSCLC artefacts keep their names.")
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
    excluded = set(map(str, args.exclude_celltypes))
    kept_rows = np.flatnonzero(~np.isin(labels, list(excluded))) if excluded \
        else np.arange(len(labels))
    n_excluded = int(len(labels) - len(kept_rows))
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
    for label in sorted(set(labels[kept_rows])):
        idx = kept_rows[labels[kept_rows] == label]
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
    if np.intersect1d(train, holdout).size or train.size + holdout.size != len(kept_rows):
        raise RuntimeError("split is not an exact, disjoint partition")

    assignments = pd.DataFrame({
        "cell_id": np.concatenate([cell_ids[train], cell_ids[holdout]]),
        "cell_type": np.concatenate([labels[train], labels[holdout]]),
        "reference_split": (["reference_train"] * len(train)
                            + ["evaluation_holdout"] * len(holdout)),
    }).sort_values(["reference_split", "cell_type", "cell_id"], kind="stable")
    assignment_path = args.outdir / f"{args.prefix}_reference_split.tsv.gz"
    # mtime=0: gzip otherwise stamps the header with the wall clock, which
    # would change this file's SHA-256 on every rebuild of identical content.
    assignments.to_csv(assignment_path, sep="\t", index=False,
                       compression={"method": "gzip", "mtime": 0})

    train_path = args.outdir / f"{args.prefix}_reference_train.h5ad"
    holdout_path = args.outdir / f"{args.prefix}_evaluation_holdout.h5ad"
    write_subset(args.reference, obs, var, train, columns, train_path)
    write_subset(args.reference, obs, var, holdout, columns, holdout_path)

    manifest = {
        "schema_version": "1.0",
        "algorithm": "per-stratum lexicographic-order + PCG64 permutation",
        "seed": args.seed,
        "holdout_fraction_requested": args.holdout_fraction,
        "celltype_column": args.celltype_col,
        "excluded_celltypes": sorted(excluded),
        "n_cells_excluded": n_excluded,
        "n_cells_considered": int(len(kept_rows)),
        "reference_representation": "panel-restricted counts with full-library depth in obs",
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
    (args.outdir / f"{args.prefix}_reference_split_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

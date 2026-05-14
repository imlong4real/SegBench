#!/usr/bin/env python3
"""Prepare shared TSU-20 inputs for real SPLIT and cellAdmix runs.

Outputs use simple MatrixMarket/TSV/CSV/Parquet files so R scripts can avoid
reticulate/anndata dependency problems.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread, mmwrite


CELLTYPE_PREFERENCES = [
    "cell_type",
    "celltype",
    "cell_type_major",
    "annotation",
    "annotations",
    "predicted_cell_type",
    "lineage",
    "broad_cell_type",
    "subtype",
    "Level3",
    "Level2",
    "Level1",
    "Harmonised_Level4",
]
CONTROL_PREFIX = r"^(BLANK_|NegControl|Codeword|antisense_|UnassignedCodeword)"


def _decode(x: Any) -> Any:
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    if isinstance(x, np.bytes_):
        return x.astype(str)
    return x


def read_h5ad_column(group: h5py.Group, name: str) -> np.ndarray:
    obj = group[name]
    if isinstance(obj, h5py.Dataset):
        return np.array([_decode(x) for x in obj[:]])
    if isinstance(obj, h5py.Group) and "categories" in obj and "codes" in obj:
        categories = np.array([_decode(x) for x in obj["categories"][:]], dtype=object)
        codes = obj["codes"][:]
        out = np.empty(len(codes), dtype=object)
        out[:] = None
        mask = codes >= 0
        out[mask] = categories[codes[mask]]
        return out
    raise TypeError(f"Unsupported h5ad column encoding for {name}")


def choose_celltype_column(obs_cols: list[str], override: str) -> str:
    if override and override != "auto":
        if override not in obs_cols:
            raise SystemExit(f"Requested celltype column not found in h5ad obs: {override}")
        return override
    lower_to_col = {c.lower(): c for c in obs_cols}
    for pref in CELLTYPE_PREFERENCES:
        if pref.lower() in lower_to_col:
            return lower_to_col[pref.lower()]
    for c in obs_cols:
        lc = c.lower()
        if any(x in lc for x in ("cell_type", "celltype", "annotation", "level")):
            return c
    raise SystemExit(
        "No suitable h5ad cell-type column detected. Use --celltype-column."
    )


def h5ad_csr(f: h5py.File, key: str = "X") -> sparse.csr_matrix:
    g = f[key]
    shape = tuple(int(x) for x in g.attrs["shape"])
    return sparse.csr_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=shape)


def read_h5ad_genes(f: h5py.File) -> list[str]:
    if "feature_name" in f["var"]:
        return [str(x) for x in read_h5ad_column(f["var"], "feature_name")]
    return [str(x) for x in read_h5ad_column(f["var"], "_index")]


def read_10x_features(path: Path) -> pd.DataFrame:
    rows = []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            rows.append(
                {
                    "gene_id": parts[0] if len(parts) > 0 else "",
                    "gene_name": parts[1] if len(parts) > 1 else parts[0],
                    "feature_type": parts[2] if len(parts) > 2 else "",
                }
            )
    return pd.DataFrame(rows)


def read_10x_barcodes(path: Path) -> list[str]:
    with gzip.open(path, "rt") as fh:
        return [line.strip() for line in fh]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xenium-dir", required=True)
    ap.add_argument("--scrna-h5ad", required=True)
    ap.add_argument("--clusters", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--celltype-column", default="auto")
    args = ap.parse_args()

    xenium_dir = Path(args.xenium_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    clusters = pd.read_csv(args.clusters)
    if {"Barcode", "Cluster"}.issubset(clusters.columns):
        clusters = clusters.rename(columns={"Barcode": "cell_id", "Cluster": "cluster"})
    elif {"cell_id_method", "cluster"}.issubset(clusters.columns):
        clusters = clusters.rename(columns={"cell_id_method": "cell_id"})
    clusters["cell_id"] = clusters["cell_id"].astype(str)
    clusters["celltype"] = "graphclust_" + clusters["cluster"].astype(str)

    cells = pd.read_parquet(xenium_dir / "cells.parquet")
    cells["cell_id"] = cells["cell_id"].astype(str)
    cell_meta = cells.merge(clusters[["cell_id", "cluster", "celltype"]], on="cell_id", how="left")
    missing_cluster = int(cell_meta["cluster"].isna().sum())
    cell_meta.to_parquet(outdir / "xenium_cell_metadata_with_clusters.parquet", index=False)

    tx = pd.read_parquet(xenium_dir / "transcripts.parquet")
    tx = tx[tx["cell_id"].astype(str).ne("UNASSIGNED")].copy()
    tx = tx[~tx["feature_name"].astype(str).str.match(CONTROL_PREFIX)].copy()
    tx["cell_id"] = tx["cell_id"].astype(str)
    tx = tx.merge(clusters[["cell_id", "cluster", "celltype"]], on="cell_id", how="inner")
    tx_out = pd.DataFrame(
        {
            "x": tx["x_location"].astype("float32"),
            "y": tx["y_location"].astype("float32"),
            "z": tx.get("z_location", pd.Series(0, index=tx.index)).astype("float32"),
            "gene": tx["feature_name"].astype(str),
            "cell": tx["cell_id"].astype(str),
            "celltype": tx["celltype"].astype(str),
            "mol_id": tx["transcript_id"].astype(str),
        }
    )
    tx_out.to_parquet(outdir / "xenium_transcripts_for_celladmix.parquet", index=False)

    matrix_dir = xenium_dir / "cell_feature_matrix"
    features = read_10x_features(matrix_dir / "features.tsv.gz")
    barcodes = read_10x_barcodes(matrix_dir / "barcodes.tsv.gz")
    xenium_gene_to_idx = {
        row.gene_name: i
        for i, row in features.iterrows()
        if row.feature_type == "Gene Expression"
    }

    with h5py.File(args.scrna_h5ad, "r") as f:
        obs_cols = list(f["obs"].keys())
        celltype_col = choose_celltype_column(obs_cols, args.celltype_column)
        ref_cells = [str(x) for x in read_h5ad_column(f["obs"], "_index")]
        ref_genes = read_h5ad_genes(f)
        ref_labels = read_h5ad_column(f["obs"], celltype_col)

        x_integer = np.mean(
            np.abs(f["X/data"][: min(len(f["X/data"]), 200_000)] - np.round(f["X/data"][: min(len(f["X/data"]), 200_000)])) > 1e-6
        ) < 1e-4
        counts_source = "X" if x_integer else "X_normalized_warning"
        ref = h5ad_csr(f, "X")

    shared = sorted(set(xenium_gene_to_idx) & set(ref_genes))
    if len(shared) < 10:
        raise SystemExit(f"Very low Xenium/scRNA gene overlap ({len(shared)}). Refusing to continue.")

    ref_gene_to_idx = {g: i for i, g in enumerate(ref_genes)}
    ref_idx = [ref_gene_to_idx[g] for g in shared]
    xen_idx = [xenium_gene_to_idx[g] for g in shared]

    ref_shared = ref[:, ref_idx].transpose().tocsr()
    if x_integer:
        ref_shared.data = np.rint(ref_shared.data).astype(np.int32)
    mmwrite(outdir / "scrna_reference_counts.mtx", ref_shared)
    (outdir / "scrna_reference_cells.tsv").write_text("\n".join(ref_cells) + "\n")
    (outdir / "scrna_reference_genes.tsv").write_text("\n".join(shared) + "\n")
    pd.DataFrame({"cell_id": ref_cells, "celltype": ref_labels.astype(str)}).to_csv(
        outdir / "scrna_reference_cell_metadata.csv", index=False
    )

    xen = mmread(matrix_dir / "matrix.mtx.gz").tocsr()
    xen_shared = xen[xen_idx, :].tocsr()
    mmwrite(outdir / "xenium_counts.mtx", xen_shared)
    (outdir / "xenium_barcodes.tsv").write_text("\n".join(barcodes) + "\n")
    (outdir / "xenium_features.tsv").write_text("\n".join(shared) + "\n")

    overlap_df = pd.DataFrame(
        {
            "gene": shared,
            "in_xenium": True,
            "in_scrna": True,
            "xenium_feature_index": xen_idx,
            "scrna_feature_index": ref_idx,
        }
    )
    overlap_df.to_csv(outdir / "gene_overlap_report.csv", index=False)

    info = {
        "xenium_dir": str(xenium_dir),
        "scrna_h5ad": str(args.scrna_h5ad),
        "clusters": str(args.clusters),
        "reference_celltype_column": celltype_col,
        "n_reference_cells": len(ref_cells),
        "n_spatial_cells": len(barcodes),
        "n_shared_genes": len(shared),
        "scrna_counts_source": counts_source,
        "missing_spatial_cluster_labels": missing_cluster,
        "outputs": sorted(str(p) for p in outdir.iterdir()),
    }
    (outdir / "common_inputs_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

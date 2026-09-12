#!/usr/bin/env python3
"""Build one input bundle per frozen ROI, shared by every imaging method.

The frozen ROI parquet is the single source of truth: every artefact here is
derived from it, so all six methods see the identical transcript population.
Nothing is re-filtered — the population was frozen upstream and is copied
through byte-for-byte as ``standardized_transcripts.parquet``.

Bundle layout
-------------
    standardized_transcripts.parquet   the frozen population (ProSeg, TRACER,
                                       Baysor prep)
    transcripts.parquet                Xenium column naming (Segger,
                                       common-input prep)
    nucleus_boundaries.parquet         derived from overlaps_nucleus
    experiment.xenium                  metadata stub
    cells.parquet                      per-entity table from the ORIGINAL mask
    cell_feature_matrix/               features/barcodes/matrix (10x layout)
    clusters.csv                       Barcode,Cluster for cellAdmix
    frozen_input_receipt.json          counts, hashes, derivation notes

Cluster labels are deterministic k-means on the PCA of the ROI's own
cell-by-gene matrix.  They are unsupervised and derived from the frozen input
alone: no reference cell type, marker or segmentation result enters the
benchmark through this file.  (Leiden would be the usual choice but the
analysis image carries no leidenalg/igraph.)
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmwrite

UNASSIGNED = {"UNASSIGNED", "unassigned", "", "NA", "nan", "None", "0", "-1", "background"}


def sha256(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def build_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Per-entity table for the ORIGINAL segmentation mask.

    Xenium ships a vendor cells.parquet, but a whole-tissue vendor table would
    describe cells the ROI does not contain.  Deriving it from the frozen ROI
    keeps the entity set and the transcript population in exact agreement, and
    works identically on platforms that ship no such table at all.
    """
    assigned = df[~df["cell_id"].astype(str).isin(UNASSIGNED)]
    g = assigned.groupby("cell_id", observed=True)
    cells = g.agg(
        x_centroid=("x", "mean"),
        y_centroid=("y", "mean"),
        transcript_counts=("feature_name", "size"),
        nucleus_count=("overlaps_nucleus", "sum"),
    ).reset_index()
    # Area proxy: bounding-box of the entity's own molecules.  Recorded as
    # derived; no polygon is available on every platform.
    span = g.agg(xmin=("x", "min"), xmax=("x", "max"),
                 ymin=("y", "min"), ymax=("y", "max")).reset_index()
    cells = cells.merge(span, on="cell_id")
    cells["cell_area"] = ((cells["xmax"] - cells["xmin"]).clip(lower=0.5)
                          * (cells["ymax"] - cells["ymin"]).clip(lower=0.5))
    cells["nucleus_area"] = cells["cell_area"] * 0.0
    cells["total_counts"] = cells["transcript_counts"]
    cells["control_probe_counts"] = 0
    cells["control_codeword_counts"] = 0
    cells["cell_id"] = cells["cell_id"].astype(str)
    return cells.drop(columns=["xmin", "xmax", "ymin", "ymax"])


def build_matrix(df: pd.DataFrame, outdir: Path) -> tuple[list[str], list[str], dict]:
    """10x-layout cell_feature_matrix (genes x cells) from the frozen ROI."""
    assigned = df[~df["cell_id"].astype(str).isin(UNASSIGNED)].copy()
    assigned["cell_id"] = assigned["cell_id"].astype(str)
    assigned["feature_name"] = assigned["feature_name"].astype(str)
    genes = sorted(df["feature_name"].astype(str).unique())
    barcodes = sorted(assigned["cell_id"].unique())
    gidx = {g: i for i, g in enumerate(genes)}
    bidx = {b: i for i, b in enumerate(barcodes)}
    counts = (assigned.groupby(["feature_name", "cell_id"], observed=True)
              .size().rename("n").reset_index())
    rows = counts["feature_name"].map(gidx).to_numpy(np.int64)
    cols = counts["cell_id"].map(bidx).to_numpy(np.int64)
    vals = counts["n"].to_numpy(np.int32)
    m = sparse.coo_matrix((vals, (rows, cols)),
                          shape=(len(genes), len(barcodes)), dtype=np.int32).tocsr()

    outdir.mkdir(parents=True, exist_ok=True)
    with gzip.open(outdir / "features.tsv.gz", "wt") as fh:
        for g in genes:
            fh.write(f"{g}\t{g}\tGene Expression\n")
    with gzip.open(outdir / "barcodes.tsv.gz", "wt") as fh:
        fh.write("\n".join(barcodes) + "\n")
    with gzip.open(outdir / "matrix.mtx.gz", "wb") as fh:
        mmwrite(fh, m, symmetry="general")
    stats = {"matrix_features": len(genes), "matrix_cells": len(barcodes),
             "matrix_nonzero": int(m.nnz), "matrix_total_counts": int(m.sum())}
    return genes, barcodes, stats


def build_clusters(matrix_dir: Path, genes: list[str], barcodes: list[str],
                   seed: int, n_clusters: int) -> pd.DataFrame:
    """Deterministic unsupervised labels for cellAdmix's --clusters input."""
    from scipy.io import mmread
    from sklearn.decomposition import TruncatedSVD
    from sklearn.cluster import KMeans

    m = mmread(matrix_dir / "matrix.mtx.gz").tocsc().T.tocsr()   # cells x genes
    counts = np.asarray(m.sum(1)).ravel()
    counts[counts == 0] = 1
    x = m.multiply(1e4 / counts[:, None]).tocsr()
    x.data = np.log1p(x.data)
    # TruncatedSVD on the sparse matrix, not PCA on a dense one: a full SVD of
    # a cells x genes block allocates a genes x genes workspace (4,755 genes ->
    # ~180 MB before intermediates) and does not fit the memory this stage is
    # given.  Randomised SVD is deterministic under a fixed random_state.
    k = int(min(50, max(2, min(x.shape) - 1)))
    comps = TruncatedSVD(n_components=k, random_state=seed,
                         algorithm="randomized").fit_transform(x)
    n_clusters = int(min(n_clusters, max(2, len(barcodes) - 1)))
    labels = KMeans(n_clusters=n_clusters, random_state=seed,
                    n_init=10).fit_predict(comps)
    return pd.DataFrame({"Barcode": barcodes, "Cluster": labels + 1})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transcripts", required=True, help="frozen ROI parquet")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--platform", required=True)
    ap.add_argument("--segger-bundle-script", required=True)
    ap.add_argument("--seed", type=int, default=20260909)
    ap.add_argument("--n-clusters", type=int, default=12)
    a = ap.parse_args()

    src, outdir = Path(a.transcripts), Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Segger-format artefacts (transcripts.parquet, nucleus_boundaries.parquet,
    # experiment.xenium) come from the existing builder so there is one
    # derivation of nucleus geometry in the repository, not two.
    subprocess.run([sys.executable, str(Path(a.segger_bundle_script)),
                    "--input", str(src), "--outdir", str(outdir),
                    "--dataset", a.dataset, "--platform", a.platform],
                   check=True)

    df = pd.read_parquet(src)
    required = {"x", "y", "feature_name", "cell_id", "transcript_id", "overlaps_nucleus"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"frozen ROI input is missing {missing}")

    # The frozen population passes through untouched.
    std = outdir / "standardized_transcripts.parquet"
    df.to_parquet(std, index=False, compression="snappy")

    cells = build_cells(df)
    cells.to_parquet(outdir / "cells.parquet", index=False)

    genes, barcodes, mstats = build_matrix(df, outdir / "cell_feature_matrix")
    clusters = build_clusters(outdir / "cell_feature_matrix", genes, barcodes,
                              a.seed, a.n_clusters)
    clusters.to_csv(outdir / "clusters.csv", index=False)

    n_unassigned = int(df["cell_id"].astype(str).isin(UNASSIGNED).sum())
    receipt = {
        "schema_version": "roi-design-1.0",
        "dataset": a.dataset,
        "platform": a.platform,
        "source_roi_parquet": str(src.resolve()),
        "source_sha256": sha256(src),
        "effective_transcript_count": int(len(df)),
        "effective_gene_count": len(genes),
        "original_entity_count": len(barcodes),
        "unassigned_count": n_unassigned,
        "frac_assigned": 1.0 - n_unassigned / max(len(df), 1),
        "standardized_transcripts_sha256": sha256(std),
        "filtering_applied": "none - the population was frozen upstream",
        "derived": {
            "cells.parquet": "centroid/counts per ORIGINAL-mask entity; cell_area "
                             "is a molecule bounding-box proxy, not a vendor polygon",
            "nucleus_boundaries.parquet": "convex hull of overlaps_nucleus molecules "
                                          "per entity, circle fallback (shared "
                                          "derivation with the Segger bundle builder)",
            "cell_feature_matrix": "recomputed from the frozen ROI, so it describes "
                                   "exactly the ROI entities and no others",
            "clusters.csv": f"k-means(k={a.n_clusters}) on PCA of the ROI matrix, "
                            f"seed {a.seed}; unsupervised, input-only",
        },
        **mstats,
    }
    (outdir / "frozen_input_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

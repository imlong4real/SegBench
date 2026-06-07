#!/usr/bin/env python3
"""Platform-agnostic `common_inputs/` builder for SPLIT and cellAdmix on the
cross-platform ROI benchmark.

The TSU-20 builder (prepare_tsu20_common_inputs.py) is hard-wired to the 10x
Xenium cellranger directory (cell_feature_matrix/*.mtx.gz, cells.parquet) and a
separate graphclust clusters.csv. The cross-platform ROIs are a single
standardized transcript parquet per dataset (x, y, feature_name, cell_id, ...)
across Xenium / Xenium5K / CosMx / MERFISH, with no cellranger bundle.

This builder reproduces the EXACT `common_inputs/` file contract the two R
wrappers consume, derived solely from the ROI transcript parquet + an scRNA
h5ad:

  * spatial cell-by-gene counts   ← aggregate the ROI transcripts by the input
                                     segmentation's cell_id (excludes UNASSIGNED
                                     and control/blank probes)
  * spatial cluster/celltype label← Leiden clustering of the spatial cells
                                     (replaces 10x graphclust), prefix "leiden_"
  * scRNA reference counts/labels ← raw integer counts from layers['counts']
                                     (falls back to X) restricted to shared genes
  * gene set                       ← intersection of spatial panel ∩ scRNA genes

Output files (identical names/format to the TSU-20 builder so the R scripts and
run_split.py / run_celladmix.py work unchanged):
  scrna_reference_counts.mtx, scrna_reference_genes.tsv, scrna_reference_cells.tsv,
  scrna_reference_cell_metadata.csv,
  xenium_counts.mtx, xenium_features.tsv, xenium_barcodes.tsv,
  xenium_cell_metadata_with_clusters.parquet,
  xenium_transcripts_for_celladmix.parquet,
  gene_overlap_report.csv, common_inputs_info.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmwrite

CONTROL_PREFIX = (
    "blank", "blank_", "negcontrol", "neg_control", "negprb", "negprobe",
    "antisense_", "codeword", "unassignedcodeword", "deprecatedcodeword",
    "control", "falsecode", "systemcontrol",
)


def _decode(x: Any) -> Any:
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode("utf-8", errors="replace")
    return x


def read_h5ad_column(group: h5py.Group, name: str) -> np.ndarray:
    obj = group[name]
    if isinstance(obj, h5py.Dataset):
        return np.array([_decode(x) for x in obj[:]])
    if isinstance(obj, h5py.Group) and "categories" in obj and "codes" in obj:
        cats = np.array([_decode(x) for x in obj["categories"][:]], dtype=object)
        codes = obj["codes"][:]
        out = np.empty(len(codes), dtype=object)
        out[:] = None
        m = codes >= 0
        out[m] = cats[codes[m]]
        return out
    raise TypeError(f"Unsupported h5ad column encoding for {name}")


def var_index_name(f: h5py.File) -> str:
    idx = f["var"].attrs.get("_index", "_index")
    return _decode(idx) if isinstance(idx, (bytes, np.bytes_)) else idx


def read_h5ad_genes(f: h5py.File) -> list[str]:
    return [str(x) for x in read_h5ad_column(f["var"], var_index_name(f))]


def read_obs_index(f: h5py.File) -> list[str]:
    idx = f["obs"].attrs.get("_index", "_index")
    idx = _decode(idx) if isinstance(idx, (bytes, np.bytes_)) else idx
    return [str(x) for x in read_h5ad_column(f["obs"], idx)]


def h5ad_csr(f: h5py.File, key: str) -> sparse.csr_matrix:
    g = f[key]
    shape = tuple(int(x) for x in g.attrs["shape"])
    return sparse.csr_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=shape)


def is_control(gene: str) -> bool:
    g = str(gene).lower()
    return g.startswith(CONTROL_PREFIX)


def leiden_clusters(cbg: sparse.csr_matrix, cells: list[str], *, seed: int,
                    log) -> pd.Series:
    """Leiden clustering of spatial cells (cells x genes counts). Returns a
    Series cell_id -> 'leiden_<k>'. Falls back to a single cluster on failure."""
    try:
        import scanpy as sc
        import anndata as ad
        a = ad.AnnData(X=cbg.copy(),
                       obs=pd.DataFrame(index=pd.Index(cells, name="cell_id")))
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
        n_comps = int(min(50, max(2, min(a.n_obs, a.n_vars) - 1)))
        sc.pp.pca(a, n_comps=n_comps, random_state=seed)
        sc.pp.neighbors(a, n_neighbors=15, random_state=seed)
        sc.tl.leiden(a, resolution=1.0, random_state=seed, flavor="igraph",
                     n_iterations=2, directed=False)
        lab = a.obs["leiden"].astype(str)
        log.info("Leiden: %d clusters over %d cells", lab.nunique(), len(lab))
        return pd.Series("leiden_" + lab.to_numpy(), index=cells, name="celltype")
    except Exception as e:  # pragma: no cover - clustering must not block the run
        log.warning("Leiden clustering failed (%s: %s); using single cluster.",
                    type(e).__name__, str(e)[:160])
        return pd.Series(["leiden_0"] * len(cells), index=cells, name="celltype")


def main() -> int:
    import logging
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roi-transcripts", required=True, type=Path,
                    help="Standardized ROI transcript parquet "
                         "(x, y, feature_name, cell_id [, z]).")
    ap.add_argument("--scrna-h5ad", required=True, type=Path)
    ap.add_argument("--celltype-column", required=True,
                    help="obs column with reference cell-type labels.")
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--min-shared-genes", type=int, default=10)
    ap.add_argument("--gene-subset", type=Path, default=None,
                    help="Optional 1-column gene list (e.g. hvg_gene_list.tsv / "
                         "npmi_gene_list.tsv) to restrict the shared gene set. "
                         "Used to keep cellAdmix tractable on dense panels.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s :: %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("prepare_roi_common_inputs")
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # ----- spatial transcripts → cell-by-gene over the INPUT segmentation -----
    tx = pd.read_parquet(args.roi_transcripts)
    ren = {}
    if "feature_name" not in tx.columns and "gene" in tx.columns:
        ren["gene"] = "feature_name"
    if "x" not in tx.columns and "x_location" in tx.columns:
        ren["x_location"] = "x"
    if "y" not in tx.columns and "y_location" in tx.columns:
        ren["y_location"] = "y"
    if ren:
        tx = tx.rename(columns=ren)
    tx["cell_id"] = tx["cell_id"].astype(str)
    tx["feature_name"] = tx["feature_name"].astype(str)
    if "z" not in tx.columns:
        tx["z"] = 0.0
    if "transcript_id" not in tx.columns:
        tx["transcript_id"] = np.arange(len(tx), dtype=np.int64)

    tx = tx[tx["cell_id"] != "UNASSIGNED"]
    tx = tx[~tx["feature_name"].map(is_control)]
    log.info("Assigned, non-control transcripts: %d over %d cells, %d genes",
             len(tx), tx["cell_id"].nunique(), tx["feature_name"].nunique())

    spatial_genes_all = sorted(tx["feature_name"].unique())

    # ----- scRNA reference (raw integer counts from layers/counts) -----------
    with h5py.File(args.scrna_h5ad, "r") as f:
        obs_cols = list(f["obs"].keys())
        if args.celltype_column not in obs_cols:
            raise SystemExit(f"--celltype-column {args.celltype_column!r} not in "
                             f"h5ad obs ({obs_cols[:12]}…)")
        ref_genes = read_h5ad_genes(f)
        ref_cells = read_obs_index(f)
        ref_labels = read_h5ad_column(f["obs"], args.celltype_column)
        counts_key = "layers/counts" if "counts" in f.get("layers", {}) else "X"
        ref = h5ad_csr(f, counts_key)
        log.info("scRNA reference: %d cells x %d genes from %s",
                 ref.shape[0], ref.shape[1], counts_key)

    shared = sorted(set(spatial_genes_all) & set(ref_genes))
    gene_subset_used = None
    if args.gene_subset is not None:
        sub = pd.read_csv(args.gene_subset, sep="\t")
        # tolerate header 'gene'/'feature_name' or a bare 1-column list
        col = next((c for c in ("gene", "feature_name", "symbol") if c in sub.columns),
                   sub.columns[0])
        subset = set(sub[col].astype(str))
        before = len(shared)
        shared = sorted(set(shared) & subset)
        gene_subset_used = str(args.gene_subset)
        log.info("Gene subset %s: %d shared genes -> %d after restriction",
                 args.gene_subset.name, before, len(shared))
    if len(shared) < args.min_shared_genes:
        raise SystemExit(f"Only {len(shared)} shared genes (< "
                         f"{args.min_shared_genes}); refusing to continue.")
    log.info("Shared genes (spatial ∩ scRNA%s): %d",
             " ∩ subset" if gene_subset_used else "", len(shared))
    gene_pos = {g: i for i, g in enumerate(shared)}

    # ----- spatial cell-by-gene matrix over shared genes ---------------------
    txs = tx[tx["feature_name"].isin(set(shared))].copy()
    cell_cat = pd.Categorical(txs["cell_id"])
    gene_cat = pd.Categorical(txs["feature_name"], categories=shared)
    spatial_cbg = sparse.csr_matrix(
        (np.ones(len(txs), np.float64), (cell_cat.codes, gene_cat.codes)),
        shape=(len(cell_cat.categories), len(shared)))  # cells x genes
    spatial_cells = list(cell_cat.categories.astype(str))

    # centroids from transcript coords (use all assigned tx, incl. control-free)
    cen = tx.groupby("cell_id", observed=True)[["x", "y"]].mean().reindex(spatial_cells)

    # ----- spatial clusters via Leiden (replaces 10x graphclust) -------------
    clusters = leiden_clusters(spatial_cbg, spatial_cells, seed=args.seed, log=log)
    cluster_id = clusters.str.replace("leiden_", "", regex=False)

    cell_meta = pd.DataFrame({
        "cell_id": spatial_cells,
        "x_centroid": cen["x"].to_numpy(np.float32),
        "y_centroid": cen["y"].to_numpy(np.float32),
        "cluster": cluster_id.to_numpy(),
        "celltype": clusters.to_numpy(),
    })
    cell_meta.to_parquet(outdir / "xenium_cell_metadata_with_clusters.parquet",
                         index=False)

    # ----- transcripts-for-cellAdmix table (x,y,z,gene,cell,celltype,mol_id) --
    ct_map = dict(zip(spatial_cells, clusters.to_numpy()))
    cells_kept = set(spatial_cells)
    txc = txs[txs["cell_id"].isin(cells_kept)]
    tx_out = pd.DataFrame({
        "x": txc["x"].astype("float32"),
        "y": txc["y"].astype("float32"),
        "z": txc["z"].astype("float32"),
        "gene": txc["feature_name"].astype(str),
        "cell": txc["cell_id"].astype(str),
        "celltype": txc["cell_id"].map(ct_map).astype(str),
        "mol_id": txc["transcript_id"].astype(str),
    })
    tx_out.to_parquet(outdir / "xenium_transcripts_for_celladmix.parquet",
                      index=False)

    # ----- spatial counts.mtx (genes x cells), barcodes, features ------------
    spatial_gxc = spatial_cbg.transpose().tocsr()  # genes x cells
    if not np.allclose(spatial_gxc.data, np.round(spatial_gxc.data)):
        spatial_gxc.data = np.rint(spatial_gxc.data)
    spatial_gxc.data = spatial_gxc.data.astype(np.int32)
    mmwrite(str(outdir / "xenium_counts.mtx"), spatial_gxc)
    (outdir / "xenium_barcodes.tsv").write_text("\n".join(spatial_cells) + "\n")
    (outdir / "xenium_features.tsv").write_text("\n".join(shared) + "\n")

    # ----- scRNA reference counts.mtx (genes x cells over shared genes) -------
    ref_gene_to_idx = {g: i for i, g in enumerate(ref_genes)}
    ref_idx = [ref_gene_to_idx[g] for g in shared]
    ref_shared = ref[:, ref_idx].transpose().tocsr()  # genes x cells
    if not np.allclose(ref_shared.data[:100000],
                       np.round(ref_shared.data[:100000])):
        ref_shared.data = np.rint(ref_shared.data)
    ref_shared.data = ref_shared.data.astype(np.int32)
    mmwrite(str(outdir / "scrna_reference_counts.mtx"), ref_shared)
    (outdir / "scrna_reference_cells.tsv").write_text("\n".join(map(str, ref_cells)) + "\n")
    (outdir / "scrna_reference_genes.tsv").write_text("\n".join(shared) + "\n")
    pd.DataFrame({"cell_id": ref_cells,
                  "celltype": [str(x) for x in ref_labels]}).to_csv(
        outdir / "scrna_reference_cell_metadata.csv", index=False)

    # ----- gene overlap report ----------------------------------------------
    pd.DataFrame({
        "gene": shared, "in_xenium": True, "in_scrna": True,
        "xenium_feature_index": [gene_pos[g] for g in shared],
        "scrna_feature_index": ref_idx,
    }).to_csv(outdir / "gene_overlap_report.csv", index=False)

    info = {
        "builder": "prepare_roi_common_inputs.py (platform-agnostic)",
        "roi_transcripts": str(args.roi_transcripts),
        "scrna_h5ad": str(args.scrna_h5ad),
        "reference_celltype_column": args.celltype_column,
        "spatial_cluster_source": "leiden(res=1.0) of ROI segmentation cells",
        "scrna_counts_source": counts_key,
        "n_reference_cells": len(ref_cells),
        "n_spatial_cells": len(spatial_cells),
        "n_spatial_genes_panel": len(spatial_genes_all),
        "gene_subset_used": gene_subset_used,
        "n_shared_genes": len(shared),
        "n_spatial_clusters": int(clusters.nunique()),
        "seed": args.seed,
        "outputs": sorted(str(p.name) for p in outdir.iterdir()),
    }
    (outdir / "common_inputs_info.json").write_text(json.dumps(info, indent=2))
    log.info("Wrote common_inputs to %s (%d shared genes, %d spatial cells, "
             "%d clusters)", outdir, len(shared), len(spatial_cells),
             clusters.nunique())
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

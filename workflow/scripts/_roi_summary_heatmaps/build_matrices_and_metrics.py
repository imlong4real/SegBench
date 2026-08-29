#!/usr/bin/env python3
"""Stage 1: build per-entity cell-by-gene matrices and compute Block A + Block B.

For every (dataset, entity):
  * materialize a cells x genes raw-count AnnData -> _work/<ds>/<entity>.h5ad
    (X + layers['counts'] = integer counts; centroids carried when available,
     so run_rctd.R can read it directly for Block C)
  * Block A: n_cells (total profiles), median transcripts/cell, runtime, peak mem
  * Block B: marker specificity log2FC, NPMI relative purity/conflict,
             Kendall tau vs scRNA reference (per-celltype pseudobulk, median)

Biological metrics reuse the canonical get_metric.py implementation (KNN label
transfer -> per-celltype pseudobulk / marker log2FC / tracer NPMI relu purity).
Run with the `spatial` conda env (scanpy + tracer importable).

Outputs:
  _work/<ds>/<entity>.h5ad
  metrics/<ds>/<entity>/...        (per-entity get_metric artifacts)
  block_ab_long.tsv                (dataset, entity, metric, value, note)
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "workflow" / "scripts"))   # get_metric.py
sys.path.insert(0, str(HERE))                            # registry.py

import get_metric as gm  # noqa: E402
import registry as R     # noqa: E402

log = logging.getLogger("roi_block_ab")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s :: %(message)s",
                    datefmt="%H:%M:%S")

KNN_K = 15
ANCHORS_PER_TYPE = 300
N_TOP_MARKERS = 30
MIN_TX_PER_CELL = 10
SEED = 1


# ---------------------------------------------------------------------------
# Matrix construction
# ---------------------------------------------------------------------------
def build_entity_adata(dataset: str, entity: str):
    """Return a cells x genes AnnData with integer-ish counts in X + layers."""
    import anndata as ad
    spec = R.entity_matrix_spec(dataset, entity)
    if spec is None:
        return None
    if spec["kind"] == "h5ad":
        if not spec["path"].exists():
            log.warning("[%s/%s] missing matrix %s", dataset, entity, spec["path"])
            return None
        a = ad.read_h5ad(spec["path"])
        # Prefer raw counts layer
        if "counts" in a.layers:
            a.X = a.layers["counts"].copy()
        # Several sources (e.g. Segger) store a dense ndarray; the tracer NPMI
        # relu and downstream slicing need CSR.
        a.X = sp.csr_matrix(a.X) if not sp.issparse(a.X) else a.X.tocsr()
        a.var_names = a.var_names.astype(str)
        a.obs_names = a.obs_names.astype(str)
        return a
    # transcripts -> cell x gene
    df = pd.read_parquet(spec["path"])
    if "etype" in spec:
        if "_etype" not in df.columns:
            raise SystemExit(f"{spec['path']} has no _etype column")
        df = df.loc[df["_etype"].astype(str) == spec["etype"]].copy()
    df["feature_name"] = df["feature_name"].astype(str)
    a = gm.build_cellxgene(df, spec["label_col"], keep_ids=None, log=log)
    return a


def save_work_h5ad(dataset: str, entity: str, adata) -> Path:
    import anndata as ad
    p = R.work_h5ad(dataset, entity)
    p.parent.mkdir(parents=True, exist_ok=True)
    X = adata.X
    X = sp.csr_matrix(X) if not sp.issparse(X) else X.tocsr()
    out = ad.AnnData(X=X.copy(),
                     obs=pd.DataFrame(index=adata.obs_names.astype(str)),
                     var=pd.DataFrame(index=adata.var_names.astype(str)))
    out.layers["counts"] = X.copy()
    # carry centroids if present (RCTD uses them for coords, else dummy)
    for cx, cy in (("x_centroid", "y_centroid"),
                   ("cell_centroid_x", "cell_centroid_y")):
        if cx in adata.obs and cy in adata.obs:
            out.obs["x_centroid"] = np.asarray(adata.obs[cx], dtype=float)
            out.obs["y_centroid"] = np.asarray(adata.obs[cy], dtype=float)
            break
    out.write_h5ad(p)
    return p


# ---------------------------------------------------------------------------
# Block B: reference consistency with BOTH Pearson and Kendall
# ---------------------------------------------------------------------------
def reference_consistency_kendall(adata_query, ann, ref, *, method, outdir,
                                  min_cells_per_type=5):
    from scipy.stats import pearsonr, kendalltau
    q_genes = np.asarray(adata_query.var_names, dtype=str)
    r_genes = ref.var_names
    shared = np.intersect1d(q_genes, r_genes)
    if len(shared) < 5:
        return pd.DataFrame()
    qmap = {g: i for i, g in enumerate(q_genes)}
    rmap = {g: i for i, g in enumerate(r_genes)}
    q_pos = np.array([qmap[g] for g in shared])
    r_pos = np.array([rmap[g] for g in shared])

    q_X = adata_query.X[:, q_pos]
    q_X = sp.csr_matrix(q_X) if not sp.issparse(q_X) else q_X.tocsr()
    q_dense = gm._log_normalize(q_X)
    q_dense = q_dense.toarray() if sp.issparse(q_dense) else q_dense

    r_X = ref.counts_csr[:, r_pos]
    r_X = sp.csr_matrix(r_X) if not sp.issparse(r_X) else r_X.tocsr()
    r_norm = gm._log_normalize(r_X)
    ref_labels = ref.obs[ref.celltype_col].astype(str).to_numpy()

    name_to_ct = dict(zip(ann["cell_id"].astype(str), ann["predicted_celltype"]))
    q_labels = np.asarray([name_to_ct.get(c, None)
                           for c in adata_query.obs_names.astype(str)], dtype=object)
    rows = []
    for ct in sorted({c for c in q_labels if c is not None and c == c}):
        q_mask = q_labels == ct
        r_mask = ref_labels == ct
        if int(q_mask.sum()) < min_cells_per_type or int(r_mask.sum()) < min_cells_per_type:
            continue
        q_bulk = q_dense[q_mask].mean(axis=0)
        r_bulk = np.asarray(r_norm[r_mask].mean(axis=0)).ravel()
        if np.std(q_bulk) == 0 or np.std(r_bulk) == 0:
            pr, kt = np.nan, np.nan
        else:
            pr = float(pearsonr(q_bulk, r_bulk)[0])
            kt = float(kendalltau(q_bulk, r_bulk)[0])
        rows.append(dict(method=method, cell_type=ct,
                         n_spatial_cells=int(q_mask.sum()),
                         n_reference_cells=int(r_mask.sum()),
                         n_genes_used=int(len(shared)),
                         pearson_r=pr, kendall_tau=kt))
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "reference_consistency_kendall.tsv", sep="\t", index=False)
    return df


def compute_block_b(dataset: str, entity: str, adata, ref, npmi_panel, outdir):
    """Return dict of block-B metric values for one entity."""
    outdir.mkdir(parents=True, exist_ok=True)
    res = {"marker_log2fc": np.nan, "relative_purity": np.nan,
           "relative_conflict": np.nan, "kendall_tau": np.nan,
           "pearson_r": np.nan, "n_celltypes": 0}
    # KNN label transfer
    try:
        ann = gm.transfer_labels(adata, ref, seed=SEED, k=KNN_K,
                                 per_type=ANCHORS_PER_TYPE, log=log)
    except SystemExit as e:
        log.warning("[%s/%s] label transfer failed: %s", dataset, entity, e)
        ann = pd.DataFrame()
    if not ann.empty:
        ann.to_csv(outdir / "post_celltype_annotations.tsv", sep="\t", index=False)
        res["n_celltypes"] = int(ann["predicted_celltype"].nunique())
        # F: reference consistency (pearson + kendall)
        rc = reference_consistency_kendall(adata, ann, ref, method=entity, outdir=outdir)
        if not rc.empty:
            res["kendall_tau"] = float(rc["kendall_tau"].median())
            res["pearson_r"] = float(rc["pearson_r"].median())
        # G: marker specificity log2FC
        markers = gm.compute_reference_markers(ref, n_top=N_TOP_MARKERS, log=log)
        if not markers.empty:
            markers.to_csv(outdir / "reference_markers_used.tsv", sep="\t", index=False)
            md = gm.metric_marker_specificity(adata, ann, markers, method=entity,
                                              outdir=outdir, log=log)
            if not md.empty:
                res["marker_log2fc"] = float(md["spatial_log2fc"].median())
    # H: NPMI relative purity / conflict (tracer relu)
    if npmi_panel is not None:
        try:
            nm = gm.metric_npmi_coherence(adata, npmi_panel, outdir=outdir, log=log)
            res["relative_purity"] = nm["median_relative_purity"]
            res["relative_conflict"] = nm["median_relative_conflict"]
        except Exception as e:
            log.warning("[%s/%s] NPMI failed: %s", dataset, entity, type(e).__name__)
    return res


# ---------------------------------------------------------------------------
# Block A: runtime / memory sourcing
# ---------------------------------------------------------------------------
def load_runtime_table():
    bc = pd.read_csv(R.FIG3 / "benchmark_comparison.tsv", sep="\t")
    return bc


def runtime_mem_for(dataset, entity, bc):
    """Return (runtime_seconds, peak_memory_gb, mem_is_gpu, note)."""
    method_map = {"TRACER": "TRACER", "baysor": "baysor", "proseg": "proseg",
                  "celladmix": "cellAdmix", "split": "SPLIT"}
    if entity in method_map:
        row = bc[(bc.dataset == dataset) & (bc.method == method_map[entity])]
        if len(row):
            return (float(row.runtime_seconds.iloc[0]),
                    float(row.peak_rss_gb.iloc[0]), False, "")
        return (np.nan, np.nan, False, "absent in benchmark_comparison")
    if entity == "segger":
        j = json.loads((R.SEGGER / dataset / "benchmark" / "runtime_memory.json").read_text())
        return (float(j["runtime_seconds"]), float(j["peak_gpu_memory_gb"]), True,
                "GPU (H100); peak is GPU memory")
    # original (native platform) and TRACER refined/reconstructed -> no benchmarked runtime
    return (np.nan, np.nan, False, "")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    R.WORK.mkdir(parents=True, exist_ok=True)
    R.METRICS.mkdir(parents=True, exist_ok=True)
    bc = load_runtime_table()
    long_rows = []

    for dataset in R.DATASET_ORDER:
        cfg = R.DATASETS[dataset]
        log.info("==== dataset %s ====", dataset)
        ref = gm.load_reference(Path(cfg["reference_h5ad"]), cfg["reference_celltype_col"], log)
        # NPMI panel (symmetric expand exactly like get_metric main)
        npmi_panel = None
        if Path(cfg["npmi"]).exists():
            nd = pd.read_csv(cfg["npmi"])
            rev = nd.copy(); rev["gene_i"], rev["gene_j"] = nd["gene_j"].values, nd["gene_i"].values
            npmi_panel = pd.concat([nd, rev], ignore_index=True)
            npmi_panel = npmi_panel.loc[npmi_panel["gene_i"] != npmi_panel["gene_j"]]

        for entity in R.ENTITY_ORDER:
            spec = R.entity_matrix_spec(dataset, entity)
            if spec is None:
                log.info("[%s/%s] absent for dataset -> skip", dataset, entity)
                continue
            adata = build_entity_adata(dataset, entity)
            if adata is None or adata.n_obs == 0:
                log.warning("[%s/%s] no matrix", dataset, entity)
                continue
            outdir = R.METRICS / dataset / entity
            save_work_h5ad(dataset, entity, adata)

            # Block A: cells + transcripts/cell
            counts = np.asarray(adata.X.sum(axis=1)).ravel()
            n_cells = int(adata.n_obs)
            med_tx = float(np.median(counts)) if n_cells else np.nan
            rt, mem, mem_gpu, rt_note = runtime_mem_for(dataset, entity, bc)

            # Block B
            b = compute_block_b(dataset, entity, adata, ref, npmi_panel, outdir)

            vals = {
                "total_cells": n_cells,
                "transcripts_per_cell": med_tx,
                "runtime_seconds": rt,
                "peak_memory_gb": mem,
                "marker_log2fc": b["marker_log2fc"],
                "relative_purity": b["relative_purity"],
                "relative_conflict": b["relative_conflict"],
                "kendall_tau": b["kendall_tau"],
            }
            for metric, value in vals.items():
                note = ""
                if metric in ("runtime_seconds", "peak_memory_gb"):
                    note = rt_note
                if metric == "peak_memory_gb" and mem_gpu:
                    note = (note + "; GPU memory").strip("; ")
                long_rows.append(dict(dataset=dataset, entity=entity, metric=metric,
                                      value=value, note=note))
            # extra provenance
            long_rows.append(dict(dataset=dataset, entity=entity, metric="pearson_r",
                                  value=b["pearson_r"], note="provenance (not plotted)"))
            log.info("[%s/%s] cells=%d tx/cell=%.1f rt=%s mem=%s log2fc=%.3f purity=%.3f "
                     "conflict=%.3f kendall=%.3f",
                     dataset, entity, n_cells, med_tx, rt, mem,
                     b["marker_log2fc"], b["relative_purity"], b["relative_conflict"],
                     b["kendall_tau"])

    df = pd.DataFrame(long_rows)
    out = R.OUT / "block_ab_long.tsv"
    df.to_csv(out, sep="\t", index=False)
    log.info("WROTE %s (%d rows)", out, len(df))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Section 1 (figure revision v2): repair SPLIT per-cell-type reference consistency.

Root cause (confirmed in split_missing_celltype_diagnosis.md): the original
get_cell_level_metric.py labels SPLIT's purified cells ONLY via RCTD `first_type`,
whose 19-type vocabulary has no B-cell or Ciliated category, so SPLIT could never
score those two types. The purified cell-by-gene matrix itself is healthy.

Repair (matches how every OTHER method is labelled in get_metric.py): annotate the
purified SPLIT cells by KNN label transfer against lung_cancer_50k.h5ad using
Cell_Cluster_level1, then compute the cross-method-comparable PSEUDOBULK Pearson
reference consistency (CP10k -> log1p -> per-type mean profile -> Pearson over the
genes shared by the Xenium panel and the scRNA reference).

Outputs:
  metrics/SPLIT/split_reference_reaudit_v2.tsv   (the audit requested in the brief)
  metrics/SPLIT/reference_consistency_by_celltype_v2.tsv  (for the Section-7 heatmap)
  metrics/SPLIT/_v2_split_reaudit_receipt.json   (compact verification receipt)
"""
from __future__ import annotations
import json, logging, sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
from scipy.stats import pearsonr

ROOT = Path(os.environ.get("SEGBENCH_ROOT",
                           Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT / "workflow" / "scripts"))
import get_metric  # reuse the exact KNN label-transfer used for all other methods

PUR = ROOT / "results/benchmark_runs/tsu20/SPLIT/outputs/split_cell_by_gene.h5ad"
REF = ROOT / "dataset/lung_cancer_scrna_10x/lung_cancer_50k.h5ad"
OUTDIR = ROOT / "results/benchmark/lung_xenium_ref36973297/metrics/SPLIT"
CELLTYPE_COL = "Cell_Cluster_level1"
NINE = ["B", "Cancer", "Ciliated", "Endothelial", "Fibroblasts", "Mast",
        "Myeloid", "Plasma", "T"]
MIN_CELLS = 10
log = logging.getLogger("v2_split_reaudit")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s",
                    datefmt="%H:%M:%S")


def lognorm_mean_profile(block: np.ndarray) -> np.ndarray:
    """CP10k -> log1p per cell, then mean across cells (pseudobulk)."""
    M = np.asarray(block, dtype=np.float64)
    s = M.sum(axis=1, keepdims=True); s[s == 0] = 1.0
    return np.log1p(M * 1e4 / s).mean(axis=0)


def main() -> int:
    pur = ad.read_h5ad(PUR)
    pur.var_names = [str(g) for g in pur.var_names]
    ref = get_metric.load_reference(REF, CELLTYPE_COL, log)

    # ---- KNN label transfer (identical recipe to get_metric for other methods) ----
    ann = get_metric.transfer_labels(pur, ref, seed=0, k=15, per_type=300, log=log)
    pred = ann["predicted_celltype"].to_numpy()
    conf = ann["confidence"].to_numpy()
    pur.obs["celltype_knn"] = np.asarray(pred).astype(str)
    pur.obs["celltype_knn_conf"] = np.asarray(conf, dtype=float)

    # ---- genes shared by Xenium panel (SPLIT vars) and scRNA reference ----
    ref_gi = {g: i for i, g in enumerate(ref.var_names)}  # last-wins, matches existing code
    panel_genes = list(pur.var_names)
    present = [g for g in panel_genes if g in ref_gi]
    n_overlap = len(present)

    # purified block over shared genes (dense; 36k x ~292 is small)
    pur_present = pur[:, present]
    Xq = pur_present.X
    Xq = Xq.toarray() if hasattr(Xq, "toarray") else np.asarray(Xq)
    Xq = np.asarray(Xq, dtype=np.float64)

    # reference block over shared genes, lognorm per cell
    import scipy.sparse as sp
    Xr = ref.counts_csr.tocsr() if sp.issparse(ref.counts_csr) else sp.csr_matrix(np.asarray(ref.counts_csr))
    Xr = Xr[:, [ref_gi[g] for g in present]].astype(np.float64)
    sr = np.asarray(Xr.sum(axis=1)).ravel(); sr[sr == 0] = 1.0
    Xr = Xr.multiply(1e4 / sr[:, None]).tocsr(); Xr.data = np.log1p(Xr.data)
    ref_labels = ref.obs[CELLTYPE_COL].astype(str).to_numpy()
    knn_labels = pur.obs["celltype_knn"].to_numpy()

    rows = []
    for ct in NINE:
        qmask = knn_labels == ct
        nq = int(qmask.sum())
        nr = int((ref_labels == ct).sum())
        reason = ""
        r = np.nan
        if nr < MIN_CELLS:
            reason = f"fewer than {MIN_CELLS} reference cells"
        elif nq < MIN_CELLS:
            reason = (f"fewer than {MIN_CELLS} SPLIT cells assigned to {ct} by KNN "
                      f"(n={nq})")
        else:
            qb = lognorm_mean_profile(Xq[qmask])
            rb = np.asarray(Xr[ref_labels == ct].mean(axis=0)).ravel()
            if qb.std() > 0 and rb.std() > 0:
                r = float(pearsonr(qb, rb)[0])
            else:
                reason = "zero variance in pseudobulk profile"
        rows.append(dict(method="SPLIT", cell_type=ct,
                         n_reference_cells=nr,
                         n_split_purified_cells=nq,
                         n_genes_overlap=n_overlap,
                         pearson_r=r, reason=reason))

    audit = pd.DataFrame(rows, columns=["method", "cell_type", "n_reference_cells",
                                        "n_split_purified_cells", "n_genes_overlap",
                                        "pearson_r", "reason"])
    OUTDIR.mkdir(parents=True, exist_ok=True)
    audit.to_csv(OUTDIR / "split_reference_reaudit_v2.tsv", sep="\t", index=False,
                 float_format="%.6f")

    # Heatmap-ready per-celltype table (method, cell_type, pearson_r, n cells/genes)
    hm = audit.rename(columns={"n_split_purified_cells": "n_spatial_cells",
                               "n_reference_cells": "n_reference_cells",
                               "n_genes_overlap": "n_genes_used"})[
        ["method", "cell_type", "n_spatial_cells", "n_reference_cells",
         "n_genes_used", "pearson_r", "reason"]]
    hm.to_csv(OUTDIR / "reference_consistency_by_celltype_v2.tsv", sep="\t",
              index=False, float_format="%.6f")

    # KNN assignment distribution (for the report / transparency)
    knn_counts = pd.Series(knn_labels).value_counts().reindex(NINE, fill_value=0)

    receipt = {
        "n_split_purified_cells_total": int(pur.n_obs),
        "n_panel_genes": len(panel_genes),
        "n_genes_overlap_panel_and_reference": n_overlap,
        "knn_label_transfer": {
            "reference": str(REF.name), "celltype_col": CELLTYPE_COL,
            "recipe": "get_metric.transfer_labels (CP10k+log1p, L2-normalized cosine KNN, "
                      "300 anchors/type, k=15 majority vote, seed 0)",
            "median_confidence": float(np.median(conf)),
        },
        "knn_assignment_counts": {k: int(v) for k, v in knn_counts.items()},
        "per_celltype_pearson": {r["cell_type"]: (None if pd.isna(r["pearson_r"])
                                                  else round(r["pearson_r"], 4))
                                 for r in rows},
        "n_celltypes_with_pearson": int(audit["pearson_r"].notna().sum()),
        "median_pearson_over_scored_types": (float(audit["pearson_r"].median())
                                             if audit["pearson_r"].notna().any() else None),
        "missing_celltypes": [r["cell_type"] for r in rows if pd.isna(r["pearson_r"])],
        "missing_reasons": {r["cell_type"]: r["reason"] for r in rows
                            if pd.isna(r["pearson_r"])},
    }
    (OUTDIR / "_v2_split_reaudit_receipt.json").write_text(json.dumps(receipt, indent=2))
    print("WROTE", OUTDIR / "split_reference_reaudit_v2.tsv")
    return 0


if __name__ == "__main__":
    sys.exit(main())

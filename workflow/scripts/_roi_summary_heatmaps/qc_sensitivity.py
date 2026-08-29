#!/usr/bin/env python3
"""QC sensitivity analysis: uniform 10<=transcripts<=900 cell filter applied to
EVERY method/dataset; recompute marker log2FC, NPMI purity/conflict, and Kendall
pre vs post filter. Also writes filtered matrices to _work_qc/ for the RCTD re-run.

Run in the `spatial` env. Marker log2FC here uses the FULL reference marker set
scored on each method's own panel (per the QC spec: "full available transcriptome
for each method"), so it is comparable to the original (pre-Audit-1) metric and
isolates the filter effect. NPMI purity uses the reference-derived panel (Audit 2).
"""
from __future__ import annotations
import sys, logging
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "workflow" / "scripts"))
sys.path.insert(0, str(HERE))
import get_metric as gm   # noqa
import registry as R      # noqa
import build_matrices_and_metrics as B  # noqa
from audit_recompute import load_ref_npmi  # noqa

log = logging.getLogger("qc"); logging.basicConfig(level=logging.ERROR)
WORK_QC = R.OUT / "_work_qc"
LO, HI = 10, 900


def kendall_median(adata, ann, ref):
    md = B.reference_consistency_kendall(adata, ann, ref, method="x",
                                         outdir=Path("/tmp/qc_rc"))
    return float(md["kendall_tau"].median()) if len(md) else np.nan


def main():
    import anndata as ad
    (Path("/tmp/qc_rc")).mkdir(exist_ok=True)
    for d in ("/tmp/qa", "/tmp/qb"):
        Path(d).mkdir(exist_ok=True)
    rows = []
    for ds in R.DATASET_ORDER:
        cfg = R.DATASETS[ds]
        ref = gm.load_reference(Path(cfg["reference_h5ad"]), cfg["reference_celltype_col"], log)
        full_markers = gm.compute_reference_markers(ref, n_top=30, log=log)
        npmi_ref = load_ref_npmi(cfg["npmi_reference"])
        for ent in R.ENTITY_ORDER:
            p = R.work_h5ad(ds, ent)
            ann_f = R.METRICS / ds / ent / "post_celltype_annotations.tsv"
            if not p.exists():
                continue
            a = ad.read_h5ad(p)
            ann = pd.read_csv(ann_f, sep="\t") if ann_f.exists() else pd.DataFrame()
            tx = np.asarray(a.X.sum(1)).ravel()
            keep = (tx >= LO) & (tx <= HI)
            a_post = a[keep].copy()
            obs_post = set(a_post.obs_names.astype(str))
            ann_post = ann[ann["cell_id"].astype(str).isin(obs_post)] if not ann.empty else ann

            def marker(adata, annv):
                if annv is None or annv.empty or full_markers.empty:
                    return np.nan
                md = gm.metric_marker_specificity(adata, annv, full_markers, method=ent,
                                                  outdir=Path("/tmp/qa"), log=log)
                return float(md["spatial_log2fc"].median()) if len(md) else np.nan

            def purity(adata):
                nm = gm.metric_npmi_coherence(adata, npmi_ref, outdir=Path("/tmp/qb"), log=log)
                return nm["median_relative_purity"], nm["median_relative_conflict"]

            mk_pre = marker(a, ann); mk_post = marker(a_post, ann_post)
            pp_pre = purity(a); pp_post = purity(a_post)
            kd_pre = kendall_median(a, ann, ref) if not ann.empty else np.nan
            kd_post = kendall_median(a_post, ann_post, ref) if not ann_post.empty else np.nan

            # write filtered matrix (float64) for RCTD
            outp = WORK_QC / ds / f"{ent}.h5ad"; outp.parent.mkdir(parents=True, exist_ok=True)
            X = a_post.X.tocsr() if sp.issparse(a_post.X) else sp.csr_matrix(a_post.X)
            X = X.astype(np.float64)
            o = ad.AnnData(X=X.copy(), obs=pd.DataFrame(index=a_post.obs_names.astype(str)),
                           var=pd.DataFrame(index=a_post.var_names.astype(str)))
            o.layers["counts"] = X.copy()
            for cx, cy in (("x_centroid", "y_centroid"), ("cell_centroid_x", "cell_centroid_y")):
                if cx in a_post.obs and cy in a_post.obs:
                    o.obs["x_centroid"] = np.asarray(a_post.obs[cx], float)
                    o.obs["y_centroid"] = np.asarray(a_post.obs[cy], float); break
            o.write_h5ad(outp)

            rows.append(dict(dataset=ds, entity=ent, n_pre=a.n_obs, n_post=int(keep.sum()),
                             pct_removed=100*(a.n_obs-int(keep.sum()))/a.n_obs,
                             marker_pre=mk_pre, marker_post=mk_post,
                             purity_pre=pp_pre[0], purity_post=pp_post[0],
                             conflict_pre=pp_pre[1], conflict_post=pp_post[1],
                             kendall_pre=kd_pre, kendall_post=kd_post))
            print(f"{ds:18s} {ent:22s} rm={rows[-1]['pct_removed']:4.1f}%  "
                  f"mk {mk_pre:+.3f}->{mk_post:+.3f}  pur {pp_pre[0]:.3f}->{pp_post[0]:.3f}  "
                  f"kd {kd_pre:.3f}->{kd_post:.3f}")
    pd.DataFrame(rows).to_csv(R.OUT / "_qc_sensitivity_metrics.tsv", sep="\t", index=False)
    print("\nWROTE _qc_sensitivity_metrics.tsv")


if __name__ == "__main__":
    main()

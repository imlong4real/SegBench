#!/usr/bin/env python3
"""Revised biological-quality metrics (run in `spatial` env).

Separates whole-cell refinement (TRACER-refined) from partial reconstruction
(TRACER-reconstructed) and removes HVG-selection advantages.

Key methodology (documented in benchmark_heatmap_summary.md):
  * Gene universe per dataset = full platform panel (union of all method panels).
  * Full-panel completion: every method is reindexed to the universe. cellAdmix
    (which outputs an HVG subset on the cervical panels) INHERITS its non-HVG
    genes from the matched original cell (same cell_id) — i.e. non-modelled genes
    are assumed unchanged from the input profile. All other methods are de-novo /
    full-panel, so unreported genes are genuine zeros (zero-filled).
  * Markers: top-3 canonical scRNA Wilcoxon markers per cell type, restricted to
    the universe (so all methods are scored on identical, panel-present markers).
  * Cell types: KNN label transfer is re-run on the completed full-panel matrix.
  * Marker log2FC & Kendall: computed on the completed (unfiltered) full panel;
    TRACER-refined and TRACER-reconstructed are scored separately (no combined).
  * Relative purity/conflict: completed matrices, QC-filtered (10<=tx<=900).
  * RCTD entropy/max weight: QC-filtered profiles (metrics_qc/), refined/recon
    separate (RCTD intersects with the reference panel internally, so HVG
    inheritance is not re-applied there).

Writes:
  block_ab_long.tsv          (marker/purity/conflict/kendall updated)
  metrics/<ds>/_marker_genes_3.tsv          (the 3-marker set per dataset)
  _revised_biological.tsv    (per-entity refined/recon/combined provenance)
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
from marker_panels import build_marker_panel  # noqa

log = logging.getLogger("rev"); logging.basicConfig(level=logging.ERROR)
N_MARKERS = 3
LO, HI = 10, 900
INHERIT = {"celladmix"}   # HVG-output methods that inherit non-modelled genes
for d in ("/tmp/rv",):
    Path(d).mkdir(exist_ok=True)
import anndata as ad


def drop_celltypes(ref, exclude):
    """Return a ReferenceData with the excluded cell types removed (e.g. the
    cervical 'Unannotated' junk cluster)."""
    if not exclude:
        return ref
    lab = ref.obs[ref.celltype_col].astype(str).to_numpy()
    keep = ~np.isin(lab, list(exclude))
    n0 = len(lab)
    out = gm.ReferenceData(counts_csr=ref.counts_csr[keep].tocsr(),
                           var_names=ref.var_names,
                           obs=ref.obs.loc[keep].copy(),
                           celltype_col=ref.celltype_col)
    print(f"   [ref] dropped {exclude}: {n0}->{out.n_cells} cells, "
          f"{out.obs[ref.celltype_col].nunique()} cell types")
    return out


def universe_for(ds):
    genes = set()
    for ent in R.ENTITY_ORDER:
        p = R.work_h5ad(ds, ent)
        if p.exists():
            genes |= set(map(str, ad.read_h5ad(p).var_names))
    return sorted(genes)


def completed(a, universe, gpos, orig=None):
    """Reindex AnnData `a` to `universe`; inherit missing genes from `orig`
    (matched by cell_id) when provided, else zero-fill."""
    avar = list(map(str, a.var_names))
    acols = [i for i, g in enumerate(avar) if g in gpos]
    aun = [gpos[avar[i]] for i in acols]
    P = sp.csr_matrix((np.ones(len(acols)), (acols, aun)), shape=(a.n_vars, len(universe)))
    M = a.X.tocsr() @ P
    if orig is not None:
        present = set(avar)
        ovar = list(map(str, orig.var_names)); opos = {g: i for i, g in enumerate(ovar)}
        miss = [(opos[g], gpos[g]) for g in universe if g not in present and g in opos]
        if miss:
            orig_sub = orig[list(a.obs_names.astype(str))]
            orows = [m[0] for m in miss]; ucols = [m[1] for m in miss]
            Q = sp.csr_matrix((np.ones(len(miss)), (orows, ucols)),
                              shape=(orig.n_vars, len(universe)))
            M = M + (orig_sub.X.tocsr() @ Q)
    out = ad.AnnData(X=M.tocsr(),
                     obs=pd.DataFrame(index=a.obs_names.astype(str)),
                     var=pd.DataFrame(index=pd.Index(universe, name="feature_name")))
    return out


def main():
    L = pd.read_csv(R.OUT / "block_ab_long.tsv", sep="\t")
    prov = []
    for ds in R.DATASET_ORDER:
        cfg = R.DATASETS[ds]
        uni = universe_for(ds); gpos = {g: i for i, g in enumerate(uni)}
        ref = gm.load_reference(Path(cfg["reference_h5ad"]), cfg["reference_celltype_col"], log)
        ref = drop_celltypes(ref, cfg.get("reference_exclude_celltypes", []))
        markers, audit = build_marker_panel(
            ds,
            ref,
            set(uni),
            outdir=R.METRICS / ds,
            exclude_celltypes=cfg.get("reference_exclude_celltypes", []),
        )
        npmi_ref = load_ref_npmi(cfg["npmi_reference"])
        orig_a = ad.read_h5ad(R.work_h5ad(ds, "original"))
        print(f"\n=== {ds}: universe={len(uni)} genes, markers={len(markers)} "
              f"({markers['cell_type'].nunique()} types, {markers['gene'].nunique()} genes), "
              f"audit_rows={len(audit)} ===")

        for ent in R.ENTITY_ORDER:
            p = R.work_h5ad(ds, ent)
            if not p.exists():
                continue
            a = ad.read_h5ad(p)
            comp = completed(a, uni, gpos, orig=orig_a if ent in INHERIT else None)
            own_tx = np.asarray(a.X.sum(1)).ravel()
            keep = (own_tx >= LO) & (own_tx <= HI)
            comp_qc = comp[keep].copy()
            # cell types on completed full panel
            ann = gm.transfer_labels(comp, ref, seed=1, k=15, per_type=300, log=log)
            # marker (top-3, completed, unfiltered)
            mk = np.nan
            if not ann.empty and not markers.empty:
                md = gm.metric_marker_specificity(comp, ann, markers, method=ent,
                                                  outdir=Path("/tmp/rv"), log=log)
                mk = float(md["spatial_log2fc"].median()) if len(md) else np.nan
            # kendall (completed, unfiltered, full panel)
            kd = np.nan
            if not ann.empty:
                rc = B.reference_consistency_kendall(comp, ann, ref, method=ent, outdir=Path("/tmp/rv"))
                kd = float(rc["kendall_tau"].median()) if len(rc) else np.nan
            # purity/conflict (completed, QC-filtered)
            nm = gm.metric_npmi_coherence(comp_qc, npmi_ref, outdir=Path("/tmp/rv"), log=log)
            pur, conf = nm["median_relative_purity"], nm["median_relative_conflict"]

            prov.append(dict(dataset=ds, entity=ent, n_cells=a.n_obs, n_qc=int(keep.sum()),
                             marker_log2fc=mk, kendall_tau=kd,
                             relative_purity=pur, relative_conflict=conf))
            for metric, val in (("marker_log2fc", mk), ("kendall_tau", kd),
                                ("relative_purity", pur), ("relative_conflict", conf)):
                m = (L.dataset == ds) & (L.entity == ent) & (L.metric == metric)
                if m.any():
                    L.loc[m, "value"] = val
            print(f"  {ent:22s} marker={mk:+.3f} kendall={kd:.3f} purity={pur:.3f} conflict={conf:.3f}")

    L.to_csv(R.OUT / "block_ab_long.tsv", sep="\t", index=False)
    pd.DataFrame(prov).to_csv(R.OUT / "_revised_biological.tsv", sep="\t", index=False)
    print("\nUPDATED block_ab_long.tsv + _revised_biological.tsv")


if __name__ == "__main__":
    main()

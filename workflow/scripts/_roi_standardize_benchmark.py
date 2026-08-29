#!/usr/bin/env python3
"""Standardize cross-platform ROI segmentation outputs for direct comparison
against TRACER.

For each (dataset, method) it emits a uniform record under
  results/fig3_cross_platform_roi_benchmark/<dataset>/_benchmark_standardized/<method>/
containing:
  * cell_assignments.tsv.gz      transcript_id, cell_id, [original_cell_id], is_assigned
                                 (transcript-level methods only)
  * cell_by_gene.h5ad            cells x genes counts (+ x/y centroids when available)
  * runtime_memory.json          {method, total_seconds, peak_rss_gb, gpu_*}
  * segmentation_metadata.json   level, n_cells, n_genes, assignment stats, params
  * evaluation_summary.tsv       one tidy row of headline numbers

and a cross-method roll-up:
  results/fig3_cross_platform_roi_benchmark/benchmark_comparison.tsv
  results/fig3_cross_platform_roi_benchmark/benchmark_manifest.json

Methods covered per dataset:
  TRACER (reference; from dataset/<ds>/tracer_seg), baysor, proseg (transcript-level),
  SPLIT (cell-level purification), cellAdmix (transcript cleaning).
MERFISH ileum has no standalone baysor; its proseg/cellAdmix/SPLIT are the
Baysor+X cascades (the ROI cell_id is the existing Baysor segmentation).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("SEGBENCH_ROOT",
                           Path(__file__).resolve().parents[2]))
BENCH = ROOT / "results/fig3_cross_platform_roi_benchmark"

DATASETS = {
    "atera_cervical":      {"platform": "Xenium",   "ref_col": "cell_type"},
    "xenium5k_cervical":   {"platform": "Xenium5K",  "ref_col": "cell_type"},
    "cosmx_nsclc":         {"platform": "CosMx",     "ref_col": "Cell_Cluster_level1"},
    "merfish_mouse_ileum": {"platform": "MERFISH",   "ref_col": "cell_type",
                            "baysor_prior": True},
}

# Method class semantics for the comparison table.
METHOD_CLASS = {
    "TRACER":    "transcript_refinement",
    "baysor":    "transcript_segmentation",
    "proseg":    "transcript_segmentation",
    "cellAdmix": "transcript_cleaning",
    "SPLIT":     "cell_level_cleaning",
}

CONTROL_PREFIX = ("blank", "blank_", "negcontrol", "neg_control", "negprb",
                  "negprobe", "antisense_", "codeword", "unassignedcodeword",
                  "deprecatedcodeword", "control", "falsecode", "systemcontrol")


def is_control(g: str) -> bool:
    return str(g).lower().startswith(CONTROL_PREFIX)


def _read_runtime(p: Path) -> dict:
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    return {
        "total_seconds": d.get("total_seconds"),
        "peak_rss_gb": d.get("peak_rss_gb_observed"),
        "method_version": d.get("method_version"),
    }


def _cbg_stats(h5ad: Path) -> dict:
    import anndata as ad
    a = ad.read_h5ad(h5ad)
    X = a.X
    tot = np.asarray(X.sum(axis=1)).ravel()
    ng = np.asarray((X > 0).sum(axis=1)).ravel()
    return {"n_cells": int(a.n_obs), "n_genes": int(a.n_vars),
            "median_counts_per_cell": float(np.median(tot)) if a.n_obs else 0.0,
            "median_genes_per_cell": float(np.median(ng)) if a.n_obs else 0.0,
            "total_counts": float(tot.sum())}


def _transcript_assignment_stats(parquet: Path) -> dict:
    import pyarrow.parquet as pq
    cols = pq.ParquetFile(parquet).schema_arrow.names
    use = ["cell_id"] + [c for c in ("original_cell_id",) if c in cols]
    df = pd.read_parquet(parquet, columns=use)
    cid = df["cell_id"].astype(str)
    n = len(cid); na = int((cid != "UNASSIGNED").sum())
    return {"n_transcripts_total": n, "n_transcripts_assigned": na,
            "n_transcripts_unassigned": n - na,
            "frac_assigned": na / n if n else 0.0,
            "n_unique_cells": int(cid[cid != "UNASSIGNED"].nunique())}


def standardize_transcript_method(ds: str, method: str, src_dir: Path,
                                  std_parquet: Path, out_dir: Path, log) -> dict:
    """Standardize a transcript-level method (TRACER/baysor/proseg/cellAdmix)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    import pyarrow.parquet as pq
    cols = pq.ParquetFile(std_parquet).schema_arrow.names
    take = [c for c in ("transcript_id", "cell_id", "original_cell_id") if c in cols]
    df = pd.read_parquet(std_parquet, columns=take)
    if "transcript_id" not in df.columns:
        df.insert(0, "transcript_id", np.arange(len(df), dtype=np.int64))
    df["cell_id"] = df["cell_id"].astype(str)
    df["is_assigned"] = (df["cell_id"] != "UNASSIGNED").astype(int)
    df.to_csv(out_dir / "cell_assignments.tsv.gz", sep="\t", index=False)

    # cell-by-gene h5ad: reuse the runner's if present, else build.
    cbg_src = src_dir / "outputs" / f"{method.lower()}_cell_by_gene.h5ad"
    if method == "TRACER":
        cbg_src = src_dir / "outputs" / "cell_by_gene_tracer.h5ad"
    cbg_dst = out_dir / "cell_by_gene.h5ad"
    if cbg_src.exists():
        import shutil
        shutil.copy(cbg_src, cbg_dst)
    else:
        _build_cbg(std_parquet, cbg_dst, log)
    cbg = _cbg_stats(cbg_dst)
    asg = _transcript_assignment_stats(std_parquet)
    return {"level": "transcript", **cbg, **asg}


def _build_cbg(std_parquet: Path, out: Path, log):
    import anndata as ad, scipy.sparse as sp
    df = pd.read_parquet(std_parquet, columns=["x", "y", "feature_name", "cell_id"])
    df["cell_id"] = df["cell_id"].astype(str)
    df = df[(df["cell_id"] != "UNASSIGNED") & (~df["feature_name"].astype(str).map(is_control))]
    cc = pd.Categorical(df["cell_id"]); gc = pd.Categorical(df["feature_name"].astype(str))
    X = sp.csr_matrix((np.ones(len(df), np.float32), (cc.codes, gc.codes)),
                      shape=(len(cc.categories), len(gc.categories)))
    obs = pd.DataFrame(index=pd.Index(cc.categories.astype(str), name="cell_id"))
    cen = df.groupby("cell_id", observed=True)[["x", "y"]].mean().reindex(cc.categories.astype(str))
    obs["x_centroid"] = cen["x"].to_numpy(np.float32)
    obs["y_centroid"] = cen["y"].to_numpy(np.float32)
    a = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=pd.Index(gc.categories.astype(str), name="feature_name")))
    a.layers["counts"] = X.copy()
    a.write_h5ad(out)


def standardize_split(ds: str, src_dir: Path, out_dir: Path, log) -> dict:
    """Standardize SPLIT (cell-level purification: no transcript assignments)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    o = src_dir / "outputs"
    for fn in ("split_cell_by_gene.h5ad", "split_original_cell_by_gene.h5ad",
               "split_rctd_weights.tsv.gz", "split_rctd_entropy_metrics.tsv",
               "split_pruning_summary.json", "split_cell_metadata.tsv",
               "not_transcript_level_reason.txt"):
        if (o / fn).exists():
            shutil.copy(o / fn, out_dir / fn)
    cbg = _cbg_stats(out_dir / "split_cell_by_gene.h5ad")
    pr = json.loads((o / "split_pruning_summary.json").read_text()) \
        if (o / "split_pruning_summary.json").exists() else {}
    return {"level": "cell_level_purification",
            "n_cells": cbg["n_cells"], "n_genes": cbg["n_genes"],
            "median_counts_per_cell": cbg["median_counts_per_cell"],
            "median_genes_per_cell": cbg["median_genes_per_cell"],
            "estimated_removed_count": pr.get("estimated_removed_count"),
            "pseudo_unassigned_count": pr.get("pseudo_post_cleaning_unassigned_count"),
            "fraction_removed": pr.get("fraction_removed_of_original_assigned"),
            "purified_counts_are_fractional": pr.get("purified_counts_are_fractional"),
            "exact_transcript_coordinates_available": False}


def gpu_flag(method: str) -> bool:
    return False  # none of baysor/proseg/cellAdmix/SPLIT/TRACER use GPU here


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--only-present", action="store_true",
                    help="Skip method outputs that are not yet produced.")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("roi_standardize")

    rows = []
    manifest = {"datasets": {}, "generated_utc": pd.Timestamp.utcnow().isoformat()}
    for ds in args.datasets:
        meta = DATASETS[ds]
        dsdir = BENCH / ds
        manifest["datasets"][ds] = {"platform": meta["platform"], "methods": {}}
        # ---- TRACER (reference) ----
        tracer_dir = ROOT / "dataset" / ds / "tracer_seg"
        method_specs = [
            ("TRACER", tracer_dir, tracer_dir / "outputs" / "transcripts_tracer_refined.parquet", "transcript"),
            ("baysor", dsdir / "baysor_seg", dsdir / "baysor_seg/outputs/baysor_transcripts_standardized.parquet", "transcript"),
            ("proseg", dsdir / "proseg_seg", dsdir / "proseg_seg/outputs/proseg_transcripts_standardized.parquet", "transcript"),
            ("cellAdmix", dsdir / "cellAdmix_seg", dsdir / "cellAdmix_seg/outputs/celladmix_transcripts_standardized.parquet", "transcript"),
            ("SPLIT", dsdir / "SPLIT_seg", None, "cell"),
        ]
        for method, src_dir, std_parquet, level in method_specs:
            out_dir = dsdir / "_benchmark_standardized" / method
            present = (std_parquet.exists() if std_parquet is not None
                       else (src_dir / "outputs" / "split_cell_by_gene.h5ad").exists())
            # MERFISH ileum is already Baysor-segmented: standalone Baysor is
            # intentionally skipped (its proseg/SPLIT/cellAdmix ARE the Baysor+X
            # cascades). Record that explicitly rather than as "pending".
            if method == "baysor" and meta.get("baysor_prior"):
                log.info("[%s/baysor] intentionally skipped (dataset is already "
                         "Baysor-segmented; cascades carry the Baysor prior)", ds)
                manifest["datasets"][ds]["methods"][method] = {
                    "status": "skipped_baysor_prior",
                    "note": "ROI cell_id is the existing Baysor segmentation; "
                            "proseg/SPLIT/cellAdmix are the Baysor+X cascades."}
                continue
            if not present:
                if args.only_present:
                    log.info("[%s/%s] not present yet — skipping", ds, method)
                    continue
                log.warning("[%s/%s] MISSING output; recording as pending", ds, method)
                manifest["datasets"][ds]["methods"][method] = {"status": "pending"}
                continue
            try:
                if level == "transcript":
                    seg = standardize_transcript_method(ds, method, src_dir, std_parquet, out_dir, log)
                else:
                    seg = standardize_split(ds, src_dir, out_dir, log)
            except Exception as e:
                log.error("[%s/%s] standardize failed: %s: %s", ds, method, type(e).__name__, str(e)[:200])
                manifest["datasets"][ds]["methods"][method] = {"status": "error", "error": str(e)[:300]}
                continue

            rt = _read_runtime(src_dir / "runtime_memory.json")
            seg.update({"method": method, "dataset": ds, "platform": meta["platform"],
                        "method_class": METHOD_CLASS[method], "gpu_used": gpu_flag(method)})
            (out_dir / "runtime_memory.json").write_text(json.dumps(
                {"method": method, "dataset": ds, "gpu_used": gpu_flag(method), **rt}, indent=2))
            (out_dir / "segmentation_metadata.json").write_text(json.dumps(seg, indent=2, default=str))

            row = {"dataset": ds, "platform": meta["platform"], "method": method,
                   "method_class": METHOD_CLASS[method], "level": seg["level"],
                   "n_cells": seg.get("n_cells"), "n_genes": seg.get("n_genes"),
                   "median_counts_per_cell": seg.get("median_counts_per_cell"),
                   "median_genes_per_cell": seg.get("median_genes_per_cell"),
                   "frac_assigned": seg.get("frac_assigned"),
                   "n_transcripts_total": seg.get("n_transcripts_total"),
                   "n_transcripts_assigned": seg.get("n_transcripts_assigned"),
                   "estimated_removed_count": seg.get("estimated_removed_count"),
                   "fraction_removed": seg.get("fraction_removed"),
                   "runtime_seconds": rt.get("total_seconds"),
                   "peak_rss_gb": rt.get("peak_rss_gb"),
                   "gpu_used": gpu_flag(method),
                   "is_baysor_cascade": bool(meta.get("baysor_prior") and method != "TRACER"),
                   "method_version": rt.get("method_version")}
            pd.DataFrame([row]).to_csv(out_dir / "evaluation_summary.tsv", sep="\t", index=False)
            rows.append(row)
            manifest["datasets"][ds]["methods"][method] = {
                "status": "ok", "level": seg["level"],
                "standardized_dir": str(out_dir.relative_to(ROOT))}
            log.info("[%s/%s] OK — %s, n_cells=%s", ds, method, seg["level"], seg.get("n_cells"))

    if rows:
        comp = pd.DataFrame(rows)
        order = ["TRACER", "baysor", "proseg", "cellAdmix", "SPLIT"]
        comp["method"] = pd.Categorical(comp["method"], order, ordered=True)
        comp = comp.sort_values(["dataset", "method"])
        comp.to_csv(BENCH / "benchmark_comparison.tsv", sep="\t", index=False)
        log.info("Wrote benchmark_comparison.tsv (%d rows)", len(comp))
    (BENCH / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    log.info("Wrote benchmark_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

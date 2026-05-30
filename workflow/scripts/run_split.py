#!/usr/bin/env python3
"""Standalone SPLIT/RCTD runner for the TSU20 / NSCLC segmentation benchmark.

SPLIT (built on RCTD/spacexr) is a cell-profile **cleaning / deconvolution**
method. It does NOT emit transcript-level molecule assignments: ``SPLIT::purify``
returns a *purified cell-by-gene matrix of expected (fractional) counts*. Because
those purified counts are fractional expectations, the exact identity of which
individual transcripts were removed from each cell CANNOT be deterministically
reconstructed, and exact x/y coordinates of removed molecules are unavailable.

Per the benchmark spec, SPLIT is therefore evaluated in **cell-level mode**: this
runner does NOT fabricate per-transcript x/y assignments and does NOT write
``outputs/split_transcripts_standardized.parquet``. Instead it estimates how many
transcript *counts* were pruned by comparing the ORIGINAL Xenium cell-by-gene
matrix with the SPLIT purified cell-by-gene matrix (count-level pruning). The
estimated removed counts are treated as a "count-level pseudo-unassigned
transcript count", NOT an exact transcript-level reassignment.

This Python runner wraps the R implementation
``workflow/scripts/_count_correction/run_split_tsu20_real.R`` (run via the
`tracer_benchmark_r` conda env, which has SPLIT + spacexr + Seurat), records
runtime/memory/provenance like run_tracer.py, and standardizes the cell-level
output + count-level pruning estimates.

Modes:
  * RUN mode  — invoke the R SPLIT/RCTD script (default).
  * WRAP mode — only when --split-raw-dir is explicitly provided AND it already
                has purified_counts.mtx; standardize that without re-running R.
                There is NO silent fallback to WRAP.

Cell-level outputs (always):
    outputs/split_cell_by_gene.h5ad            (purified, fractional expected counts)
    outputs/split_original_cell_by_gene.h5ad   (original assigned integer counts)
    outputs/split_cell_metadata.tsv
    outputs/split_rctd_weights.tsv.gz          (when derivable)
    outputs/split_rctd_entropy_metrics.tsv     (when weights derivable)
Count-level pruning estimates (always):
    outputs/split_removed_counts_by_cell_gene.tsv.gz
    outputs/split_removed_counts_by_gene.tsv
    outputs/split_removed_counts_by_cell.tsv
    outputs/split_pruning_summary.json
    outputs/not_transcript_level_reason.txt

A transcript-level standardized parquet is intentionally NOT produced. Use
``workflow/scripts/get_cell_level_metric.py`` for SPLIT; do NOT pass it to the
transcript-level get_metric.py.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# This file lives at <repo>/workflow/scripts/run_split.py, so the repo root is
# parents[2] (parents[1] is the `workflow` dir). R_SCRIPT/provenance paths are
# built relative to the true repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _runner_common as rc  # noqa: E402

METHOD = "split"
R_SCRIPT = _REPO_ROOT / "workflow" / "scripts" / "_count_correction" / "run_split_tsu20_real.R"
_NEG_EPS = 1e-9  # tolerance for treating a difference as a "real" negative

NOT_TX_REASON = (
    "SPLIT is evaluated in CELL-LEVEL mode (not transcript-level).\n\n"
    "SPLIT::purify returns a purified cell-by-gene matrix of *expected (fractional) "
    "counts*, not integer transcript removals. Because the purified counts are "
    "fractional expectations produced by RCTD doublet decomposition + singlet "
    "purification, the identity and exact x/y coordinates of which individual "
    "transcripts were removed from each original cell cannot be deterministically "
    "reconstructed. Per the benchmark spec, no per-transcript cleaned-to-unassigned "
    "table is fabricated and no split_transcripts_standardized.parquet is written.\n\n"
    "COUNT-LEVEL PRUNING ESTIMATE\n"
    "----------------------------\n"
    "We DO estimate how many transcript *counts* were pruned, at the cell-by-gene "
    "level, by comparing the ORIGINAL Xenium cell-by-gene matrix (built from the "
    "original assigned cell_id values) with the SPLIT purified cell-by-gene matrix, "
    "over the cells and genes shared by both:\n"
    "    removed_count[cell, gene] = original_count[cell, gene] - purified_count[cell, gene]\n"
    "Because purified counts are fractional, removed counts may be fractional and may "
    "be slightly negative due to numerical differences (these are recorded, then "
    "clamped to 0 for summary statistics). The total estimated removed count is added "
    "to the original unassigned transcript count to form a 'count-level pseudo-"
    "unassigned transcript count' (pseudo_post_cleaning_unassigned_count). This is a "
    "COUNT-LEVEL estimate, NOT an exact transcript-level reassignment.\n\n"
    "HOW TO EVALUATE\n"
    "---------------\n"
    "Use the cell-level outputs (split_cell_by_gene.h5ad, "
    "split_original_cell_by_gene.h5ad, split_removed_counts_*.tsv[.gz], "
    "split_pruning_summary.json, split_rctd_weights.tsv.gz, "
    "split_rctd_entropy_metrics.tsv) with workflow/scripts/get_cell_level_metric.py "
    "(or a SPLIT-specific metric mode). Do NOT pass SPLIT to the transcript-level "
    "get_metric.py, which expects a transcript-level standardized parquet with exact "
    "per-molecule x/y/cell_id.\n")

# Coarse compartment keyword map for NSCLC mixture scores (heuristic).
_COMPARTMENT_KEYWORDS = {
    "tumor": ("epitheli", "tumor", "tumour", "carcinoma", "malignant", "alveolar",
              "club", "ciliated", "basal cell", "secretory"),
    "immune": ("t cell", "b cell", "nk", "natural killer", "macrophage", "monocyte",
               "dendritic", "mast", "myeloid", "lymphocyte", "plasma", "neutrophil",
               "immune", "leukocyte", "granulocyte", "kupffer"),
    "stromal": ("fibroblast", "endotheli", "smooth muscle", "pericyte", "stromal",
                "mesotheli", "muscle", "stellate", "adipocyte"),
}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    rc.add_shared_args(p)
    p.add_argument("--r-env", default="tracer_benchmark_r",
                   help="conda env with SPLIT + spacexr + Seurat.")
    p.add_argument("--common-inputs", default="results/tsu20_tools/common_inputs")
    p.add_argument("--split-raw-dir", type=Path, default=None,
                   help="Existing SPLIT output dir with purified_counts.mtx + "
                        "cell_meta.parquet (enables WRAP mode; no silent fallback).")
    p.add_argument("--features-tsv", type=Path, default=None,
                   help="Gene list matching purified_counts rows "
                        "(default: <common-inputs>/xenium_features.tsv).")
    p.add_argument("--cores", type=int, default=2)
    p.add_argument("--umi-min", type=int, default=10)
    p.add_argument("--counts-min", type=int, default=10)
    return p


# ===========================================================================
# Cell-level (purified) loading
# ===========================================================================
def _load_purified_h5ad(raw_dir: Path, features_tsv: Path, *, out_path: Path, log):
    """Build cells x genes AnnData from purified_counts.mtx (genes x cells)."""
    import anndata as ad
    import scipy.io as sio
    import scipy.sparse as sp
    mtx = sio.mmread(str(raw_dir / "purified_counts.mtx")).tocsr()  # genes x cells
    genes = [g.strip() for g in Path(features_tsv).read_text().splitlines() if g.strip()]
    cm = pd.read_parquet(raw_dir / "cell_meta.parquet") if (raw_dir / "cell_meta.parquet").exists() \
        else pd.read_csv(raw_dir / "cell_meta.csv", index_col=0)
    cell_ids = (cm["cell_id"].astype(str).to_numpy() if "cell_id" in cm.columns
                else cm.index.astype(str).to_numpy())
    n_g, n_c = mtx.shape
    if len(genes) != n_g:
        log.warning("features (%d) != purified rows (%d); using generic gene names.",
                    len(genes), n_g)
        genes = [f"gene_{i}" for i in range(n_g)]
    if len(cell_ids) != n_c:
        log.warning("cell_meta rows (%d) != purified cols (%d); using generic cell ids.",
                    len(cell_ids), n_c)
        cell_ids = np.array([f"cell_{i}" for i in range(n_c)])
    X = sp.csr_matrix(mtx.T)  # cells x genes (fractional expected counts)
    obs = cm.set_index(cm["cell_id"].astype(str)) if "cell_id" in cm.columns else cm.copy()
    obs.index = pd.Index(cell_ids, name="cell_id")
    # Sanitize obs for h5ad: coerce non-numeric / mixed object & boolean columns
    # to plain strings so the vlen-string writer doesn't choke (e.g. `same_class`).
    for c in obs.columns:
        if obs[c].dtype == bool or obs[c].dtype == object or str(obs[c].dtype) == "boolean":
            obs[c] = obs[c].astype(str)
    var = pd.DataFrame(index=pd.Index(genes, name="feature_name"))
    adata = ad.AnnData(X=X.astype(np.float32), obs=obs, var=var)
    adata.layers["counts"] = X.astype(np.float32).copy()
    adata.uns["counts_note"] = "SPLIT-purified EXPECTED (fractional) counts, not integers."
    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_path)
    log.info("Wrote SPLIT purified cell-by-gene h5ad: %s (%d cells x %d genes)",
             out_path, adata.n_obs, adata.n_vars)
    return adata, cm


def _build_original_cbg(df_tx: pd.DataFrame, *, out_path: Path, log):
    """Build the ORIGINAL assigned cells x genes integer-count AnnData from the
    standardized transcript table (assigned original cell_id only). Returns
    (adata, n_assigned, n_unassigned)."""
    import anndata as ad
    import scipy.sparse as sp
    cid = df_tx["cell_id"].astype(str)
    is_un = cid.isin(rc.UNASSIGNED_TOKENS)
    n_unassigned = int(is_un.sum())
    sub = df_tx.loc[~is_un, ["cell_id", "feature_name"]].copy()
    sub["cell_id"] = sub["cell_id"].astype(str)
    sub["feature_name"] = sub["feature_name"].astype(str)
    n_assigned = int(len(sub))
    cg = sub.groupby(["cell_id", "feature_name"], observed=True).size().rename("count").reset_index()
    cell_cat = pd.Categorical(cg["cell_id"])
    gene_cat = pd.Categorical(cg["feature_name"])
    X = sp.csr_matrix(
        (cg["count"].to_numpy(np.float32), (cell_cat.codes, gene_cat.codes)),
        shape=(len(cell_cat.categories), len(gene_cat.categories)))
    obs = pd.DataFrame(index=pd.Index(cell_cat.categories.astype(str), name="cell_id"))
    var = pd.DataFrame(index=pd.Index(gene_cat.categories.astype(str), name="feature_name"))
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.layers["counts"] = X.copy()
    adata.uns["counts_note"] = "ORIGINAL Xenium assigned integer counts (assigned cell_id only)."
    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_path)
    log.info("Wrote ORIGINAL cell-by-gene h5ad: %s (%d cells x %d genes, %d assigned / %d unassigned tx)",
             out_path, adata.n_obs, adata.n_vars, n_assigned, n_unassigned)
    return adata, n_assigned, n_unassigned


# ===========================================================================
# Count-level pruning estimate
# ===========================================================================
def _dense_block(adata, cells: list[str], genes: list[str]) -> np.ndarray:
    """Dense (len(cells) x len(genes)) float64 block from an AnnData, aligned to
    the given cell/gene name order."""
    sub = adata[cells, genes]
    X = sub.X
    arr = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    return np.asarray(arr, dtype=np.float64)


def compute_count_level_pruning(orig_ad, pur_ad, *, n_assigned_total: int,
                                n_unassigned_total: int, outputs_dir: Path, log) -> dict:
    """Compare original vs purified cell-by-gene over shared cells & genes and
    write the removed-count tables + pruning summary. Returns the summary dict."""
    orig_cells = set(map(str, orig_ad.obs_names))
    orig_genes = set(map(str, orig_ad.var_names))
    # Preserve purified order for determinism.
    shared_cells = [c for c in map(str, pur_ad.obs_names) if c in orig_cells]
    shared_genes = [g for g in map(str, pur_ad.var_names) if g in orig_genes]
    log.info("Pruning alignment: %d shared cells (orig=%d, purified=%d), "
             "%d shared genes (orig=%d, purified=%d)",
             len(shared_cells), orig_ad.n_obs, pur_ad.n_obs,
             len(shared_genes), orig_ad.n_vars, pur_ad.n_vars)

    orig_block = _dense_block(orig_ad, shared_cells, shared_genes)
    pur_block = _dense_block(pur_ad, shared_cells, shared_genes)
    diff = orig_block - pur_block  # removed (may be fractional / slightly negative)

    # Negatives BEFORE clamping.
    neg_mask = diff < -_NEG_EPS
    n_neg = int(neg_mask.sum())
    neg_mag = float(-diff[neg_mask].sum()) if n_neg else 0.0

    removed_clamped = np.clip(diff, 0.0, None)
    purified_are_fractional = bool(np.any(np.abs(pur_block - np.round(pur_block)) > 1e-6))

    original_assigned_count = float(orig_block.sum())          # over aligned cells/genes
    purified_retained_count = float(pur_block.sum())
    estimated_removed_count = float(removed_clamped.sum())
    frac_removed = (estimated_removed_count / original_assigned_count
                    if original_assigned_count > 0 else 0.0)
    pseudo_unassigned = float(n_unassigned_total) + estimated_removed_count

    # --- per-(cell,gene) long table (nonzero original OR purified) ----------
    keep = (orig_block > 0) | (pur_block > 1e-6)
    ci, gi = np.nonzero(keep)
    cg_path = outputs_dir / "split_removed_counts_by_cell_gene.tsv.gz"
    cells_arr = np.asarray(shared_cells)
    genes_arr = np.asarray(shared_genes)
    long_df = pd.DataFrame({
        "cell_id": cells_arr[ci],
        "feature_name": genes_arr[gi],
        "original_count": orig_block[ci, gi],
        "purified_count": pur_block[ci, gi],
        "removed_count": diff[ci, gi],  # raw (unrounded, may be negative)
    })
    with gzip.open(cg_path, "wt") as f:
        long_df.to_csv(f, sep="\t", index=False, float_format="%.6g")
    log.info("Wrote %s (%d nonzero cell-gene rows)", cg_path, len(long_df))

    # --- per-gene summary ---------------------------------------------------
    gene_df = pd.DataFrame({
        "feature_name": genes_arr,
        "original_count": orig_block.sum(axis=0),
        "purified_count": pur_block.sum(axis=0),
        "removed_count_raw": diff.sum(axis=0),
        "removed_count_clamped": removed_clamped.sum(axis=0),
    }).sort_values("removed_count_clamped", ascending=False)
    gene_df.to_csv(outputs_dir / "split_removed_counts_by_gene.tsv", sep="\t",
                   index=False, float_format="%.6g")

    # --- per-cell summary ---------------------------------------------------
    cell_df = pd.DataFrame({
        "cell_id": cells_arr,
        "original_count": orig_block.sum(axis=1),
        "purified_count": pur_block.sum(axis=1),
        "removed_count_raw": diff.sum(axis=1),
        "removed_count_clamped": removed_clamped.sum(axis=1),
    })
    cell_df.to_csv(outputs_dir / "split_removed_counts_by_cell.tsv", sep="\t",
                   index=False, float_format="%.6g")

    summary = {
        "original_assigned_count": original_assigned_count,
        "purified_retained_count": purified_retained_count,
        "estimated_removed_count": estimated_removed_count,
        "original_unassigned_count": int(n_unassigned_total),
        "pseudo_post_cleaning_unassigned_count": pseudo_unassigned,
        "fraction_removed_of_original_assigned": frac_removed,
        "n_cells_original": int(orig_ad.n_obs),
        "n_cells_purified": int(pur_ad.n_obs),
        "n_cells_aligned": int(len(shared_cells)),
        "n_genes_original": int(orig_ad.n_vars),
        "n_genes_purified": int(pur_ad.n_vars),
        "n_genes_aligned": int(len(shared_genes)),
        "purified_counts_are_fractional": purified_are_fractional,
        "n_negative_differences_before_clamping": n_neg,
        "total_negative_magnitude_before_clamping": neg_mag,
        "exact_transcript_coordinates_available": False,
        "transcript_level_output_emitted": False,
        # Transparency extras (not in the required list, but informative):
        "original_assigned_count_all_cells": int(n_assigned_total),
        "n_cells_original_dropped_by_split": int(orig_ad.n_obs - len(shared_cells)),
        "note": ("Count-level pseudo-unassigned estimate. Pruning is computed over "
                 "cells & genes SHARED by the original and purified matrices; cells "
                 "SPLIT did not process (e.g. below --umi-min) are reflected by "
                 "n_cells_original vs n_cells_aligned, not in estimated_removed_count. "
                 "These are count-level estimates, NOT exact transcript reassignments."),
    }
    (outputs_dir / "split_pruning_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log.info("Pruning: original_assigned=%.0f purified_retained=%.1f estimated_removed=%.1f "
             "(%.2f%% of aligned original); pseudo_unassigned=%.1f; neg=%d (mag=%.3g); fractional=%s",
             original_assigned_count, purified_retained_count, estimated_removed_count,
             100 * frac_removed, pseudo_unassigned, n_neg, neg_mag, purified_are_fractional)
    return summary


# ===========================================================================
# RCTD weights + entropy / mixture metrics
# ===========================================================================
def _write_rctd_weights(cm: pd.DataFrame, out_path: Path, log) -> bool:
    """Emit a tidy first/second-type mixture-weights table from cell_meta."""
    if "weight_first_type" not in cm.columns:
        return False
    cid = cm["cell_id"].astype(str) if "cell_id" in cm.columns else cm.index.astype(str)
    w = pd.DataFrame({
        "cell_id": cid,
        "spot_class": cm.get("spot_class"),
        "first_type": cm.get("first_type"),
        "weight_first_type": cm.get("weight_first_type"),
        "second_type": cm.get("second_type"),
        "weight_second_type": cm.get("weight_second_type"),
        "rctd_weights_entropy": cm.get("rctd_weights_entropy"),
    })
    with gzip.open(out_path, "wt") as f:
        w.to_csv(f, sep="\t", index=False)
    log.info("Wrote SPLIT RCTD mixture weights: %s (%d cells)", out_path, len(w))
    return True


def _compartment(celltype: str) -> str:
    t = str(celltype).lower()
    for comp, kws in _COMPARTMENT_KEYWORDS.items():
        if any(k in t for k in kws):
            return comp
    return "other"


def _write_rctd_entropy_metrics(cm: pd.DataFrame, out_path: Path, log) -> bool:
    """Per-cell RCTD entropy / doublet / coarse-compartment mixture metrics."""
    if "weight_first_type" not in cm.columns:
        return False
    cid = (cm["cell_id"].astype(str) if "cell_id" in cm.columns
           else cm.index.astype(str)).to_numpy()
    w1 = pd.to_numeric(cm.get("weight_first_type"), errors="coerce").fillna(0.0).to_numpy()
    w2 = pd.to_numeric(cm.get("weight_second_type"), errors="coerce").fillna(0.0).to_numpy()
    first = cm.get("first_type").astype(str).to_numpy() if "first_type" in cm.columns else np.array([""] * len(cm))
    second = cm.get("second_type").astype(str).to_numpy() if "second_type" in cm.columns else np.array([""] * len(cm))
    spot = cm.get("spot_class").astype(str).to_numpy() if "spot_class" in cm.columns else np.array([""] * len(cm))
    max_weight = np.maximum(w1, w2)

    # Entropy: prefer RCTD's own value; else compute from the two normalized weights.
    if "rctd_weights_entropy" in cm.columns:
        entropy = pd.to_numeric(cm["rctd_weights_entropy"], errors="coerce").to_numpy()
    else:
        s = np.clip(w1 + w2, 1e-12, None)
        p1, p2 = w1 / s, w2 / s
        entropy = -(np.where(p1 > 0, p1 * np.log(p1), 0.0) + np.where(p2 > 0, p2 * np.log(p2), 0.0))

    comp1 = np.array([_compartment(t) for t in first])
    comp2 = np.array([_compartment(t) for t in second])
    minor = np.minimum(w1, w2)  # degree of the doublet's minor component

    def _mix(a: str, b: str) -> np.ndarray:
        pair_ab = (comp1 == a) & (comp2 == b)
        pair_ba = (comp1 == b) & (comp2 == a)
        return np.where(pair_ab | pair_ba, minor, 0.0)

    is_doublet = np.isin(np.char.lower(spot.astype(str)), ["doublet", "doublet_certain", "doublet_uncertain"])
    out = pd.DataFrame({
        "cell_id": cid,
        "spot_class": spot,
        "is_doublet_or_mixed": is_doublet,
        "first_type": first,
        "second_type": second,
        "weight_first_type": w1,
        "weight_second_type": w2,
        "max_weight": max_weight,
        "entropy": entropy,
        "compartment_first": comp1,
        "compartment_second": comp2,
        "tumor_immune_mixture_score": _mix("tumor", "immune"),
        "tumor_stromal_mixture_score": _mix("tumor", "stromal"),
        "immune_stromal_mixture_score": _mix("immune", "stromal"),
    })
    out.to_csv(out_path, sep="\t", index=False, float_format="%.6g")
    log.info("Wrote SPLIT RCTD entropy/mixture metrics: %s (%d cells; %d doublet/mixed)",
             out_path, len(out), int(is_doublet.sum()))
    return True


def main() -> int:
    args = build_argparser().parse_args()
    sentinel = args.outdir / "outputs" / "split_cell_by_gene.h5ad"
    rc.prepare_outdir(args.outdir, sentinel, args.overwrite)
    log = rc.setup_logging(args.outdir, "run_split")
    log.info("=== run_split.py === sample=%s seed=%d", args.sample_name, args.seed)

    outputs_dir = args.outdir / "outputs"
    raw_dir = outputs_dir / "split_raw_output"
    raw_dir.mkdir(parents=True, exist_ok=True)
    features_tsv = args.features_tsv or (Path(args.common_inputs) / "xenium_features.tsv")
    timer = rc.StageTimer(log)
    notes = ["SPLIT/RCTD is evaluated as a cell-level profile purification / "
             "deconvolution method (see not_transcript_level_reason.txt).",
             "Removed transcripts are estimated at the COUNT level (original minus "
             "purified cell-by-gene) and treated as count-level pseudo-unassigned; "
             "exact transcript coordinates of removed molecules are NOT available."]

    wrap_dir = args.split_raw_dir
    wrap = wrap_dir is not None and (Path(wrap_dir) / "purified_counts.mtx").exists()

    split_version = spacexr_version = "unknown"
    ext_rss = None

    # --- load_inputs: read original transcript table (needed for pruning) ---
    with timer.time("load_inputs"):
        df_tx = rc.read_parquet_robust(args.transcripts, log=log)
        ren = {}
        if "feature_name" not in df_tx.columns and "gene" in df_tx.columns:
            ren["gene"] = "feature_name"
        if ren:
            df_tx = df_tx.rename(columns=ren)
        for c in ("feature_name", "cell_id"):
            if c not in df_tx.columns:
                raise SystemExit(f"--transcripts missing required column {c!r}; "
                                 f"have {sorted(df_tx.columns)}")
        log.info("Loaded original transcripts: %s (%d rows)", args.transcripts, len(df_tx))

    with timer.time("convert_inputs"):
        pass  # SPLIT consumes precomputed common_inputs mtx files directly.

    # --- run_method ---------------------------------------------------------
    if wrap:
        notes.append(f"WRAP MODE: SPLIT was NOT re-run; standardized existing output "
                     f"at {wrap_dir}. Runtime/memory here do NOT reflect a SPLIT run.")
        log.info("WRAP mode — using existing SPLIT output: %s", wrap_dir)
        with timer.time("run_method"):
            pass
        src_dir = Path(wrap_dir)
    else:
        if not R_SCRIPT.exists():
            raise SystemExit(f"SPLIT R script not found: {R_SCRIPT}")
        log.info("RUN mode — invoking SPLIT/RCTD via conda env '%s'", args.r_env)
        with timer.time("run_method"):
            cmd = [
                "conda", "run", "-n", args.r_env, "Rscript", str(R_SCRIPT),
                "--xenium-dir", "dataset/lung_cancer_xenium_10x/TSU-20",
                "--scrna-h5ad", str(args.reference_h5ad or ""),
                "--celltype-column", args.reference_celltype_col,
                "--outdir", str(raw_dir),
                "--common-inputs", args.common_inputs,
                "--cores", str(args.cores),
                "--umi-min", str(args.umi_min),
                "--counts-min", str(args.counts_min),
            ]
            rc_code, ext_rss = rc.run_subprocess(cmd, log=log, outdir=args.outdir)
            if rc_code != 0:
                raise SystemExit(
                    f"SPLIT R script failed (exit {rc_code}); see run.log / "
                    f"{raw_dir}/method_info.json for the missing dependency or error.")
        # Attach R subprocess peak RSS AFTER the stage closes (record_external
        # scans completed stages, so it must run outside the `with` block).
        timer.record_external("run_method", ext_rss)
        src_dir = raw_dir

    mi = src_dir / "method_info.json"
    if mi.exists():
        d = json.load(open(mi))
        split_version = d.get("split_version", d.get("SPLIT_version", "unknown"))
        spacexr_version = d.get("spacexr_version", "unknown")

    if not (src_dir / "purified_counts.mtx").exists():
        raise SystemExit(f"SPLIT purified_counts.mtx not found in {src_dir}")

    # --- convert_outputs (cell-level) ---------------------------------------
    h5ad_path = outputs_dir / "split_cell_by_gene.h5ad"
    orig_h5ad_path = outputs_dir / "split_original_cell_by_gene.h5ad"
    meta_path = outputs_dir / "split_cell_metadata.tsv"
    weights_path = outputs_dir / "split_rctd_weights.tsv.gz"
    entropy_path = outputs_dir / "split_rctd_entropy_metrics.tsv"
    reason_path = outputs_dir / "not_transcript_level_reason.txt"
    with timer.time("convert_outputs"):
        pur_ad, cm = _load_purified_h5ad(src_dir, features_tsv, out_path=h5ad_path, log=log)
        cm.to_csv(meta_path, sep="\t", index=True)
        have_weights = _write_rctd_weights(cm, weights_path, log)
        have_entropy = _write_rctd_entropy_metrics(cm, entropy_path, log) if have_weights else False
        reason_path.write_text(NOT_TX_REASON)
        log.info("Wrote not_transcript_level_reason.txt")

    # --- compute_count_level_pruning ----------------------------------------
    with timer.time("compute_count_level_pruning"):
        orig_ad, n_assigned, n_unassigned = _build_original_cbg(
            df_tx, out_path=orig_h5ad_path, log=log)
        pruning = compute_count_level_pruning(
            orig_ad, pur_ad, n_assigned_total=n_assigned,
            n_unassigned_total=n_unassigned, outputs_dir=outputs_dir, log=log)

    # --- validate_schema (cell-level report) --------------------------------
    with timer.time("validate_schema"):
        report = {
            "method": METHOD,
            "level": "cell_level_only",
            "evaluation": "SPLIT/RCTD is evaluated as cell-level profile "
                          "purification / deconvolution.",
            "transcript_level_standardized_output": False,
            "transcript_level_output_emitted": False,
            "exact_transcript_coordinates_available": False,
            "exact_removed_molecule_coordinates_available": False,
            "count_level_pruning_method": (
                "Removed transcript counts estimated as original cell-by-gene minus "
                "purified cell-by-gene over shared cells & genes."),
            "removed_counts_may_be_fractional": True,
            "do_not_pass_to_transcript_level_get_metric": True,
            "evaluate_with": "workflow/scripts/get_cell_level_metric.py (cell-level metric mode)",
            "cell_by_gene_h5ad": str(h5ad_path),
            "original_cell_by_gene_h5ad": str(orig_h5ad_path),
            "n_cells_purified": int(pur_ad.n_obs),
            "n_genes_purified": int(pur_ad.n_vars),
            "purified_counts_are_integer": not pruning["purified_counts_are_fractional"],
            "rctd_weights_emitted": bool(have_weights),
            "rctd_entropy_metrics_emitted": bool(have_entropy),
            "split_version": split_version,
            "spacexr_version": spacexr_version,
            "mode": "wrap" if wrap else "run",
            "pruning_summary": pruning,
        }
        (args.outdir / "schema_validation_report.json").write_text(
            json.dumps(report, indent=2, default=str))
        log.info("SPLIT cell-level: %d cells x %d genes; pruning summary written.",
                 pur_ad.n_obs, pur_ad.n_vars)

    # --- write_outputs ------------------------------------------------------
    outs = [str(h5ad_path), str(orig_h5ad_path), str(meta_path), str(reason_path),
            str(outputs_dir / "split_removed_counts_by_cell_gene.tsv.gz"),
            str(outputs_dir / "split_removed_counts_by_gene.tsv"),
            str(outputs_dir / "split_removed_counts_by_cell.tsv"),
            str(outputs_dir / "split_pruning_summary.json")]
    if have_weights:
        outs.append(str(weights_path))
    if have_entropy:
        outs.append(str(entropy_path))
    with timer.time("write_outputs"):
        rc.write_provenance(
            outdir=args.outdir, method=METHOD, sample_name=args.sample_name,
            args=args, timer=timer, repo_root=_REPO_ROOT,
            inputs={"transcripts": str(args.transcripts),
                    "reference_h5ad": str(args.reference_h5ad or ""),
                    "common_inputs": str(args.common_inputs)},
            outputs=outs, method_version=split_version, runner_kind="R",
            extra_config={"spacexr_version": spacexr_version,
                          "reference_celltype_col": args.reference_celltype_col,
                          "mode": "wrap" if wrap else "run",
                          "level": "cell_level_only",
                          "cores": args.cores, "umi_min": args.umi_min,
                          "counts_min": args.counts_min},
            runtime_extra={
                "mode": "wrap" if wrap else "run",
                "level": "cell_level_only",
                "runtime_valid_for_benchmark": not wrap,
                "transcript_level_output_emitted": False,
                "estimated_removed_count": pruning["estimated_removed_count"],
                "pseudo_post_cleaning_unassigned_count":
                    pruning["pseudo_post_cleaning_unassigned_count"],
                "n_cells_purified": pruning["n_cells_purified"],
                "n_cells_aligned": pruning["n_cells_aligned"],
                "split_version": split_version,
                "spacexr_version": spacexr_version,
            },
            log=log, summary_extra_lines=notes)

    log.info("DONE. Total wall: %.1fs", timer.total_seconds)
    return 0


if __name__ == "__main__":
    sys.exit(main())

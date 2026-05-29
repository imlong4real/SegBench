#!/usr/bin/env python3
"""Standalone SPLIT runner for the TSU20 / NSCLC segmentation benchmark.

SPLIT (built on RCTD/spacexr) is a cell-profile **cleaning / deconvolution**
method. It does NOT emit transcript-level assignments: ``SPLIT::purify`` returns
a *purified cell-by-gene matrix of expected (fractional) counts*. Because those
purified counts are fractional expectations, the exact identity of which
individual transcripts were removed from each cell CANNOT be deterministically
reconstructed. Per the benchmark spec, SPLIT is therefore evaluated in
**cell-level-only mode** — this runner does not fabricate per-transcript
assignments. A ``not_transcript_level_reason.txt`` documents this.

This Python runner wraps the existing R implementation
``workflow/scripts/_count_correction/run_split_tsu20_real.R`` (run via the
`tracer_benchmark_r` conda env, which has SPLIT + spacexr + Seurat), records
runtime/memory/provenance like run_tracer.py, and standardizes the cell-level
output.

Modes:
  * RUN mode  — invoke the R SPLIT script (default).
  * WRAP mode — if --split-raw-dir already has purified_counts.mtx + cell_meta,
                skip the (slow) SPLIT/RCTD compute and just standardize them.

Always-emitted outputs:
    outputs/split_cell_by_gene.h5ad        (purified, fractional expected counts)
    outputs/split_cell_metadata.tsv
    outputs/split_rctd_weights.tsv.gz       (when derivable)
    outputs/not_transcript_level_reason.txt
A transcript-level standardized parquet is intentionally NOT produced.
Use a cell-level metric script for SPLIT; do not pass it to get_metric.py.
"""
from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _runner_common as rc  # noqa: E402

METHOD = "split"
R_SCRIPT = _REPO_ROOT / "workflow" / "scripts" / "_count_correction" / "run_split_tsu20_real.R"
NOT_TX_REASON = (
    "SPLIT is evaluated in CELL-LEVEL-ONLY mode.\n\n"
    "SPLIT::purify returns a purified cell-by-gene matrix of *expected (fractional) "
    "counts*, not integer transcript removals. Because the purified counts are "
    "fractional expectations produced by RCTD doublet decomposition + singlet "
    "purification, the identity of which individual transcripts were removed from "
    "each original cell cannot be deterministically reconstructed. Per the benchmark "
    "spec, no per-transcript cleaned-to-unassigned table is fabricated.\n\n"
    "Use the cell-level outputs (split_cell_by_gene.h5ad, split_cell_metadata.tsv, "
    "split_rctd_weights.tsv.gz) with a cell-level metric routine. Do NOT pass SPLIT "
    "to get_metric.py, which expects a transcript-level standardized parquet.\n")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    rc.add_shared_args(p)
    p.add_argument("--r-env", default="tracer_benchmark_r",
                   help="conda env with SPLIT + spacexr + Seurat.")
    p.add_argument("--common-inputs", default="results/tsu20_tools/common_inputs")
    p.add_argument("--split-raw-dir", type=Path, default=None,
                   help="Existing SPLIT output dir with purified_counts.mtx + "
                        "cell_meta.parquet (enables WRAP mode).")
    p.add_argument("--features-tsv", type=Path, default=None,
                   help="Gene list matching purified_counts rows "
                        "(default: <common-inputs>/xenium_features.tsv).")
    p.add_argument("--cores", type=int, default=2)
    p.add_argument("--umi-min", type=int, default=10)
    p.add_argument("--counts-min", type=int, default=10)
    return p


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
    log.info("Wrote SPLIT cell-by-gene h5ad: %s (%d cells x %d genes)",
             out_path, adata.n_obs, adata.n_vars)
    return adata, cm


def _write_rctd_weights(cm: pd.DataFrame, out_path: Path, log) -> bool:
    """Emit a tidy first/second-type mixture-weights table derived from cell_meta.
    Full RCTD weights live in post_processed_RCTD.rds."""
    cols = {"cell_id", "first_type", "second_type", "weight_first_type",
            "weight_second_type", "spot_class"}
    have = cols & set(cm.columns)
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
    })
    with gzip.open(out_path, "wt") as f:
        w.to_csv(f, sep="\t", index=False)
    log.info("Wrote SPLIT RCTD mixture weights: %s (%d cells)", out_path, len(w))
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
    notes = ["SPLIT is evaluated in cell-level-only mode (see not_transcript_level_reason.txt)."]

    wrap_dir = args.split_raw_dir
    wrap = wrap_dir is not None and (Path(wrap_dir) / "purified_counts.mtx").exists()

    split_version = spacexr_version = "unknown"
    with timer.time("load_inputs"):
        pass
    with timer.time("convert_inputs"):
        pass  # SPLIT consumes precomputed common_inputs mtx files directly.

    # --- run_method ---------------------------------------------------------
    if wrap:
        notes.append(f"SPLIT was NOT re-run; standardized existing output at {wrap_dir}.")
        log.info("WRAP mode — using existing SPLIT output: %s", wrap_dir)
        with timer.time("run_method"):
            pass
        src_dir = Path(wrap_dir)
        mi = src_dir / "method_info.json"
        if mi.exists():
            import json
            d = json.load(open(mi))
            split_version = d.get("split_version", "unknown")
            spacexr_version = d.get("spacexr_version", "unknown")
    else:
        if not R_SCRIPT.exists():
            raise SystemExit(f"SPLIT R script not found: {R_SCRIPT}")
        log.info("RUN mode — invoking SPLIT via conda env '%s'", args.r_env)
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
            timer.record_external("run_method", ext_rss)
            if rc_code != 0:
                raise SystemExit(
                    f"SPLIT R script failed (exit {rc_code}); see run.log / "
                    f"{raw_dir}/method_info.json for the missing dependency or error.")
        src_dir = raw_dir
        mi = raw_dir / "method_info.json"
        if mi.exists():
            import json
            d = json.load(open(mi))
            split_version = d.get("split_version", "unknown")
            spacexr_version = d.get("spacexr_version", "unknown")

    if not (src_dir / "purified_counts.mtx").exists():
        raise SystemExit(f"SPLIT purified_counts.mtx not found in {src_dir}")

    # --- convert_outputs (cell-level only) ----------------------------------
    h5ad_path = outputs_dir / "split_cell_by_gene.h5ad"
    meta_path = outputs_dir / "split_cell_metadata.tsv"
    weights_path = outputs_dir / "split_rctd_weights.tsv.gz"
    reason_path = outputs_dir / "not_transcript_level_reason.txt"
    with timer.time("convert_outputs"):
        adata, cm = _load_purified_h5ad(src_dir, features_tsv, out_path=h5ad_path, log=log)
        cm.to_csv(meta_path, sep="\t", index=True)
        have_weights = _write_rctd_weights(cm, weights_path, log)
        reason_path.write_text(NOT_TX_REASON)
        log.info("Wrote not_transcript_level_reason.txt")

    # --- validate_schema (cell-level report) --------------------------------
    with timer.time("validate_schema"):
        import json
        report = {
            "method": METHOD,
            "level": "cell_level_only",
            "transcript_level_standardized_output": False,
            "reason": "SPLIT purified counts are fractional expected counts; "
                      "transcript identity not deterministically reconstructable.",
            "cell_by_gene_h5ad": str(h5ad_path),
            "n_cells": int(adata.n_obs),
            "n_genes": int(adata.n_vars),
            "counts_are_integer": False,
            "rctd_weights_emitted": bool(have_weights),
            "split_version": split_version,
            "spacexr_version": spacexr_version,
        }
        (args.outdir / "schema_validation_report.json").write_text(
            json.dumps(report, indent=2, default=str))
        log.info("SPLIT cell-level: %d cells x %d genes", adata.n_obs, adata.n_vars)

    # --- write_outputs ------------------------------------------------------
    outs = [str(h5ad_path), str(meta_path), str(reason_path)]
    if have_weights:
        outs.insert(2, str(weights_path))
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
                          "level": "cell_level_only"},
            log=log, summary_extra_lines=notes)

    log.info("DONE. Total wall: %.1fs", timer.total_seconds)
    return 0


if __name__ == "__main__":
    sys.exit(main())

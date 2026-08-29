#!/usr/bin/env python3
"""Standalone cellAdmix runner for the TSU20 / NSCLC segmentation benchmark.

cellAdmix is a cell-profile **cleaning / admixture-correction** method, NOT a
de-novo transcript-level segmentation method. It operates on the *original*
Xenium cell assignments and flags contaminating transcripts for removal. This
runner therefore evaluates cellAdmix as a *cleaned/retained-transcript* method
on the original Xenium cell IDs (it does not redraw segmentation boundaries).

Because cellAdmix tracks a per-molecule id (`mol_id`) and emits explicit
``cleaned_transcripts`` / ``removed_transcripts`` tables, a faithful
transcript-level standardized output IS available:
    * retained transcripts keep their original cell_id    (cleaned_status=retained)
    * removed transcripts  -> cell_id = "UNASSIGNED"       (cleaned_status=cleaned_to_unassigned)
    * original_cell_id is preserved on every row.

This Python runner wraps the existing R implementation
``workflow/scripts/_count_correction/run_celladmix_tsu20_real.R`` (run via the
`tracer_benchmark_r` conda env, which has the cellAdmix package), records
runtime/memory/provenance like run_tracer.py, and standardizes the output.

Modes:
  * RUN mode  — invoke the R cellAdmix script (default).
  * WRAP mode — if --celladmix-raw-dir already has cleaned/removed parquets,
                skip the (slow) cellAdmix compute and just standardize them.

Outputs:
    outputs/celladmix_transcripts_standardized.parquet
    outputs/celladmix_cell_by_gene.h5ad
Metrics are computed later by get_metric.py.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# This file lives at <repo>/workflow/scripts/run_celladmix.py, so the repo root
# is parents[2] (parents[1] is the `workflow` dir). R_SCRIPT and provenance
# paths are built relative to the true repo root.
from .. import REPO_ROOT as _REPO_ROOT
from .. import common as rc
from .. import stats as stx
from . import _base

METHOD = "celladmix"
R_SCRIPT = _REPO_ROOT / "workflow" / "scripts" / "_count_correction" / "run_celladmix_tsu20_real.R"
CONCEPT_NOTE = (
    "cellAdmix is evaluated as a cleaned/retained-transcript method on original "
    "Xenium cell IDs, not as a new segmentation boundary method.")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _base.add_common_args(p, method=METHOD)
    _base.add_transcript_input_args(p)
    p.add_argument("--rscript", default=None,
                   help="Rscript to use. Defaults to the one configured in "
                        "configs/environments.yaml; falls back to `conda run "
                        "-n <--r-env>` only when nothing is configured.")
    p.add_argument("--r-env", default="tracer_benchmark_r",
                   help="conda env with the cellAdmix R package.")
    p.add_argument("--xenium-dir", default="dataset/lung_cancer_xenium_10x/TSU-20")
    p.add_argument("--clusters",
                   default="dataset/lung_cancer_xenium_10x/TSU-20/analysis/clustering/"
                           "gene_expression_graphclust/clusters.csv")
    p.add_argument("--common-inputs", default="results/tsu20_tools/common_inputs")
    p.add_argument("--celladmix-raw-dir", type=Path, default=None,
                   help="Existing cellAdmix output dir with cleaned/removed "
                        "transcript parquets (enables WRAP mode).")
    p.add_argument("--num-factors", type=int, default=10)
    p.add_argument("--nmol-dsamp", type=int, default=10000)
    p.add_argument("--n-cells-nmf", type=int, default=2000)
    p.add_argument("--bridge-cells", type=int, default=200)
    p.add_argument("--cores", type=int, default=None,
                   help="R worker cores (default: --threads).")
    return p


def _standardize_celladmix(cleaned: pd.DataFrame, removed: pd.DataFrame, *, log):
    """cleaned/removed cellAdmix tables (x,y,z,gene,cell,celltype,mol_id,factor)
    → standardized transcript contract with cleaned_status + original_cell_id."""
    def _prep(df: pd.DataFrame, status: str, to_unassigned: bool) -> pd.DataFrame:
        out = pd.DataFrame({
            "x": df["x"], "y": df["y"],
            "z": df["z"] if "z" in df.columns else 0.0,
            "feature_name": df["gene"].astype(str),
            "original_cell_id": df["cell"].astype(str),
            "transcript_id": df["mol_id"] if "mol_id" in df.columns else np.arange(len(df)),
        })
        out["cell_id"] = "UNASSIGNED" if to_unassigned else df["cell"].astype(str)
        out["cleaned_status"] = status
        return out

    frames = []
    if cleaned is not None and len(cleaned):
        frames.append(_prep(cleaned, "retained", to_unassigned=False))
    if removed is not None and len(removed):
        frames.append(_prep(removed, "cleaned_to_unassigned", to_unassigned=True))
    allt = pd.concat(frames, ignore_index=True)
    return rc.standardize_transcripts(allt, method=METHOD, log=log)


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    _base.resolve_config(args, method=METHOD)
    if args.dry_run:
        print(f"[dry-run] {METHOD}: transcripts={args.transcripts} "
              f"sample={args.sample_name} threads={args.threads} -> {args.outdir}")
        return 0
    sentinel = args.outdir / "outputs" / f"{METHOD}_transcripts_standardized.parquet"
    rc.prepare_outdir(args.outdir, sentinel, args.overwrite)
    log = rc.setup_logging(args.outdir, "run_celladmix")
    log.info("=== run_celladmix.py === sample=%s seed=%d", args.sample_name, args.seed)

    outputs_dir = args.outdir / "outputs"
    raw_dir = outputs_dir / "celladmix_raw_output"
    raw_dir.mkdir(parents=True, exist_ok=True)
    timer = rc.StageTimer(log)
    notes = [CONCEPT_NOTE]

    wrap_dir = args.celladmix_raw_dir
    wrap = wrap_dir is not None and (Path(wrap_dir) / "cleaned_transcripts.parquet").exists()

    celladmix_version = "unknown"
    with timer.time("load_inputs"):
        # Provenance reference to the standardized input (cellAdmix itself reads
        # the precomputed common_inputs molecule table).
        try:
            n_in = pd.read_parquet(args.transcripts, columns=["feature_name"]).shape[0]
            log.info("Reference input transcripts: %s (%d rows)", args.transcripts, n_in)
        except Exception as e:
            log.warning("Could not read --transcripts for provenance: %s", e)

    with timer.time("convert_inputs"):
        pass  # cellAdmix consumes common_inputs directly; nothing to convert.

    # --- run_method ---------------------------------------------------------
    if wrap:
        notes.append(f"cellAdmix was NOT re-run; standardized existing output at {wrap_dir}.")
        log.info("WRAP mode — using existing cellAdmix output: %s", wrap_dir)
        with timer.time("run_method"):
            pass
        clean_p = Path(wrap_dir) / "cleaned_transcripts.parquet"
        rem_p = Path(wrap_dir) / "removed_transcripts.parquet"
        mi = Path(wrap_dir) / "method_info.json"
        if mi.exists():
            import json
            celladmix_version = json.load(open(mi)).get("celladmix_version", "unknown")
    else:
        if not R_SCRIPT.exists():
            raise SystemExit(f"cellAdmix R script not found: {R_SCRIPT}")
        log.info("RUN mode — invoking cellAdmix via conda env '%s'", args.r_env)
        with timer.time("run_method"):
            launcher = ([args.rscript] if args.rscript
                        else ["conda", "run", "-n", args.r_env, "Rscript"])
            log.info("invoking cellAdmix via %s", " ".join(launcher))
            cmd = launcher + [
                str(R_SCRIPT),
                "--xenium-dir", str(args.xenium_dir),
                "--clusters", args.clusters,
                "--outdir", str(raw_dir),
                "--common-inputs", args.common_inputs,
                "--num-factors", str(args.num_factors),
                "--nmol-dsamp", str(args.nmol_dsamp),
                "--n-cells-nmf", str(args.n_cells_nmf),
                "--bridge-cells", str(args.bridge_cells),
                "--cores", str(args.cores),
            ]
            rc_code, ext_rss = rc.run_subprocess(cmd, log=log, outdir=args.outdir)
            if rc_code != 0:
                raise SystemExit(
                    f"cellAdmix R script failed (exit {rc_code}); see run.log / "
                    f"{raw_dir}/method_info.json for the missing dependency or error.")
        # Attach the R subprocess peak RSS AFTER the stage is recorded
        # (record_external scans completed stages, so it must run outside `with`).
        timer.record_external("run_method", ext_rss)
        clean_p = raw_dir / "cleaned_transcripts.parquet"
        rem_p = raw_dir / "removed_transcripts.parquet"
        mi = raw_dir / "method_info.json"
        if mi.exists():
            import json
            celladmix_version = json.load(open(mi)).get("celladmix_version", "unknown")

    if not clean_p.exists():
        raise SystemExit(f"cellAdmix cleaned transcripts not found: {clean_p}")

    # --- convert_outputs ----------------------------------------------------
    std_path = outputs_dir / f"{METHOD}_transcripts_standardized.parquet"
    h5ad_path = outputs_dir / f"{METHOD}_cell_by_gene.h5ad"
    with timer.time("convert_outputs"):
        cleaned = pd.read_parquet(clean_p)
        removed = pd.read_parquet(rem_p) if Path(rem_p).exists() else pd.DataFrame()
        log.info("cellAdmix: %d retained, %d removed transcripts",
                 len(cleaned), len(removed))
        std = _standardize_celladmix(cleaned, removed, log=log)
        std.to_parquet(std_path, index=False, compression="snappy")
        log.info("Wrote standardized transcripts: %s", std_path)
        # cell-by-gene built from RETAINED (cleaned) transcripts only. This is a
        # secondary artifact; never let a broken anndata/dask install sink the run
        # after the standardized parquet (the deliverable) is written.
        h5ad_ok = False
        try:
            rc.build_cell_by_gene_h5ad(std, out_path=h5ad_path, log=log)
            h5ad_ok = True
        except Exception as e:
            log.warning("Skipping cell-by-gene h5ad (%s: %s). The standardized "
                        "parquet was written; this does not affect benchmarking.",
                        type(e).__name__, str(e)[:160])

    # --- validate_schema ----------------------------------------------------
    with timer.time("validate_schema"):
        rc.validate_schema(
            std, method=METHOD, out_path=std_path, in_path=args.transcripts,
            report_path=args.outdir / "schema_validation_report.json", log=log,
            extra={"celladmix_version": celladmix_version,
                   "n_retained": int(len(cleaned)), "n_removed": int(len(removed)),
                   "evaluation_mode": "cleaned_to_unassigned_on_original_cell_ids"})

    # --- write_outputs ------------------------------------------------------
    with timer.time("write_outputs"):
        rc.write_provenance(
            outdir=args.outdir, method=METHOD, sample_name=args.sample_name,
            args=args, timer=timer, repo_root=_REPO_ROOT,
            inputs={"transcripts": str(args.transcripts),
                    "common_inputs": str(args.common_inputs),
                    "clusters": str(args.clusters)},
            outputs=[str(std_path)] + ([str(h5ad_path)] if h5ad_ok else [])
                    + [str(clean_p), str(rem_p)],
            method_version=celladmix_version, runner_kind="R",
            extra_config={"num_factors": args.num_factors, "nmol_dsamp": args.nmol_dsamp,
                          "n_cells_nmf": args.n_cells_nmf, "bridge_cells": args.bridge_cells,
                          "mode": "wrap" if wrap else "run"},
            log=log, summary_extra_lines=notes)

    log.info("DONE. Total wall: %.1fs", timer.total_seconds)
    stx.write_benchmark_stats(
        outdir=args.outdir, method=METHOD, modality="imaging",
        sample_name=args.sample_name, timer=timer, dataset=args.dataset,
        transcripts=stx.transcript_accounting(std),
        entities=stx.entity_accounting(std),
        qc={"n_removed_transcripts": int((std["cleaned_status"] == "cleaned_to_unassigned").sum())
                if "cleaned_status" in std.columns else None,
            "n_retained_transcripts": int((std["cleaned_status"] == "retained").sum())
                if "cleaned_status" in std.columns else None,
            "num_factors": int(args.num_factors)},
        outputs=[str(std_path)],
        notes=CONCEPT_NOTE)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Standalone proseg runner for the TSU20 / NSCLC segmentation benchmark.

Wraps the ``proseg`` binary (Rust; https://github.com/dmcable/proseg — installed
locally at ~/.cargo/bin/proseg, also packaged in reproducibility/proseg.def). The
invocation mirrors ``workflow/rules/_segmentation/proseg.smk`` but is driven from
a standardized transcripts parquet instead of a raw Xenium bundle, and produces
the benchmark's standardized transcript contract.

Pipeline stages (timed individually, like run_tracer.py):
    load_inputs    → read standardized transcripts parquet
    convert_inputs → write a proseg-compatible transcript CSV
    run_method     → run `proseg` (external; wrapped with /usr/bin/time -v when available)
    convert_outputs→ proseg transcript-metadata.csv.gz → standardized parquet (+ h5ad)
    validate_schema→ schema_validation_report.json
    write_outputs  → runtime_memory.json, runtime_by_stage.tsv, config_receipt.json, run_summary.md

This runner does NOT compute benchmark metrics — use get_metric.py for that.

EXAMPLE
=======
    python workflow/scripts/run_proseg.py \\
      --transcripts dataset/lung_cancer_xenium_10x/TSU-20/filtered_df_standardized.parquet \\
      --outdir results/benchmark_runs/tsu20/proseg \\
      --sample-name TSU20 --seed 1 --overwrite
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .. import REPO_ROOT as _REPO_ROOT
from .. import common as rc
from .. import stats as stx
from . import _base

METHOD = "proseg"
EXCLUDE_GENES = r"^(BLANK_|NegControl|antisense_|UnassignedCodeword|Codeword|DeprecatedCodeword)"


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _base.add_common_args(p, method=METHOD)
    _base.add_transcript_input_args(p)
    p.add_argument("--proseg-bin", default=shutil.which("proseg") or "proseg",
                   help="Path to the proseg binary (default: from PATH).")
    p.add_argument("--nthreads", type=int, default=None,
                   help="proseg threads (default: --threads).")
    p.add_argument("--voxel-layers", type=int, default=4,
                   help="proseg --voxel-layers (z-axis voxel layers).")
    p.add_argument("--extra-proseg-args", default="",
                   help="Extra raw args appended to the proseg command line.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    _base.resolve_config(args, method=METHOD)
    if args.dry_run:
        print(f"[dry-run] {METHOD}: transcripts={args.transcripts} "
              f"sample={args.sample_name} threads={args.threads} -> {args.outdir}")
        return 0
    sentinel = args.outdir / "outputs" / f"{METHOD}_transcripts_standardized.parquet"
    rc.prepare_outdir(args.outdir, sentinel, args.overwrite)
    log = rc.setup_logging(args.outdir, "run_proseg")
    log.info("=== run_proseg.py === sample=%s seed=%d", args.sample_name, args.seed)
    np.random.seed(args.seed)

    outputs_dir = args.outdir / "outputs"
    raw_dir = outputs_dir / "proseg_raw_output"
    raw_dir.mkdir(parents=True, exist_ok=True)
    timer = rc.StageTimer(log)

    # --- proseg availability check (clear error, no silent skip) ------------
    proseg_bin = args.proseg_bin
    if shutil.which(proseg_bin) is None and not Path(proseg_bin).exists():
        raise SystemExit(
            f"proseg binary not found ('{proseg_bin}'). Install proseg "
            f"(cargo install proseg) or build reproducibility/proseg.def, "
            f"or pass --proseg-bin."
        )
    proseg_version = "unknown"
    try:
        import subprocess
        proseg_version = subprocess.check_output(
            [proseg_bin, "--version"], text=True).strip()
    except Exception:
        pass
    log.info("proseg version: %s", proseg_version)

    # --- load_inputs --------------------------------------------------------
    with timer.time("load_inputs"):
        df_in = rc.load_input_transcripts(
            args.transcripts, log=log,
            max_transcripts=args.max_transcripts, seed=args.seed)
        for c in ("x", "y", "feature_name", "cell_id"):
            if c not in df_in.columns:
                raise SystemExit(f"input transcripts missing required column {c!r}; "
                                 f"have {sorted(df_in.columns)}")
        if "transcript_id" not in df_in.columns:
            df_in["transcript_id"] = np.arange(len(df_in), dtype=np.int64)
        if "z" not in df_in.columns:
            df_in["z"] = np.float32(0.0)
        if "qv" not in df_in.columns:
            df_in["qv"] = np.float32(40.0)
        df_in["cell_id"] = df_in["cell_id"].astype(str)

    # --- convert_inputs: proseg transcript CSV ------------------------------
    proseg_csv = raw_dir / "proseg_input_transcripts.csv"
    with timer.time("convert_inputs"):
        cols = ["transcript_id", "x", "y", "z", "feature_name", "cell_id", "qv"]
        df_in[cols].to_csv(proseg_csv, index=False)
        log.info("Wrote proseg input CSV: %s (%d transcripts)", proseg_csv, len(df_in))

    # --- run_method ---------------------------------------------------------
    tm_path = raw_dir / "transcript-metadata.csv.gz"
    cm_path = raw_dir / "cell-metadata.csv.gz"
    with timer.time("run_method"):
        # proseg refuses to overwrite an existing spatialdata zarr; clear it.
        stale_zarr = raw_dir / "proseg-output.zarr"
        if stale_zarr.exists():
            shutil.rmtree(stale_zarr, ignore_errors=True)
        cmd = [
            proseg_bin, str(proseg_csv.resolve()),
            "--overwrite",
            "--nthreads", str(args.nthreads),
            "--voxel-layers", str(args.voxel_layers),
            "-x", "x", "-y", "y", "-z", "z",
            "--gene-column", "feature_name",
            "--transcript-id-column", "transcript_id",
            "--cell-id-column", "cell_id",
            "--cell-id-unassigned", "UNASSIGNED",
            "--qv-column", "qv",
            "--excluded-genes", EXCLUDE_GENES,
            "--output-transcript-metadata", str(tm_path.resolve()),
            "--output-cell-metadata", str(cm_path.resolve()),
            "--output-counts", str((raw_dir / "counts.mtx.gz").resolve()),
        ]
        if args.extra_proseg_args:
            cmd += args.extra_proseg_args.split()
        rc_code, ext_rss = rc.run_subprocess(cmd, log=log, outdir=args.outdir, cwd=raw_dir)
        if rc_code != 0:
            raise SystemExit(f"proseg failed with exit code {rc_code}; see run.log.")
        if not tm_path.exists():
            raise SystemExit(f"proseg did not produce {tm_path}; see run.log.")
    # Attach proseg-only peak RSS AFTER the stage is recorded (record_external
    # scans completed stages, so it must run outside the `with` block).
    timer.record_external("run_method", ext_rss)

    # --- convert_outputs ----------------------------------------------------
    std_path = outputs_dir / f"{METHOD}_transcripts_standardized.parquet"
    h5ad_path = outputs_dir / f"{METHOD}_cell_by_gene.h5ad"
    with timer.time("convert_outputs"):
        tm = pd.read_csv(tm_path)
        log.info("proseg transcript-metadata columns: %s", list(tm.columns))
        cm = pd.read_csv(cm_path)
        # Map proseg integer 'cell' -> original input cell_id for provenance.
        cell_to_orig = {}
        if "cell" in cm.columns and "original_cell_id" in cm.columns:
            cell_to_orig = dict(zip(cm["cell"].astype(str), cm["original_cell_id"].astype(str)))

        # Unassigned: empty assignment or background==true.
        assignment = tm["assignment"].astype("string")
        is_bg = (tm["background"].astype(str).str.lower() == "true") if "background" in tm.columns else False
        cell_id = assignment.copy()
        cell_id[assignment.isna() | (assignment.astype(str) == "")] = "UNASSIGNED"
        if np.any(is_bg):
            cell_id[is_bg] = "UNASSIGNED"
        tm["cell_id"] = cell_id.astype(str)
        tm["original_cell_id"] = tm["cell_id"].map(lambda c: cell_to_orig.get(str(c), pd.NA))

        # Prefer observed (original) transcript coordinates. proseg emits BOTH
        # denoised x/y/z and observed_x/observed_y/observed_z — drop the denoised
        # ones first so the rename does not create duplicate x/y/z columns.
        ren = {"gene": "feature_name"}
        if "observed_x" in tm.columns:
            tm = tm.drop(columns=[c for c in ("x", "y", "z") if c in tm.columns])
            ren.update({"observed_x": "x", "observed_y": "y", "observed_z": "z"})
        std = rc.standardize_transcripts(tm, method=METHOD, rename=ren, log=log)
        # Re-attach qv from input by transcript_id (proseg drops it).
        if "qv" not in std.columns and "transcript_id" in std.columns:
            std = std.merge(df_in[["transcript_id", "qv"]], on="transcript_id", how="left")
        std.to_parquet(std_path, index=False, compression="snappy")
        log.info("Wrote standardized transcripts: %s", std_path)
        # The cell-by-gene h5ad is a secondary artifact. anndata's import/writer
        # can fail on a broken anndata/xarray/dask install; never let that sink a
        # 20-minute run after proseg already produced the standardized parquet
        # (the benchmark deliverable). Best-effort only.
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
            extra={"proseg_version": proseg_version, "n_proseg_cells": int(len(cm))})

    # --- write_outputs ------------------------------------------------------
    with timer.time("write_outputs"):
        rc.write_provenance(
            outdir=args.outdir, method=METHOD, sample_name=args.sample_name,
            args=args, timer=timer, repo_root=_REPO_ROOT,
            inputs={"transcripts": str(args.transcripts)},
            outputs=[str(std_path)] + ([str(h5ad_path)] if h5ad_ok else [])
                    + [str(tm_path), str(cm_path)],
            method_version=proseg_version, runner_kind="binary",
            extra_config={"nthreads": args.nthreads, "voxel_layers": args.voxel_layers,
                          "excluded_genes": EXCLUDE_GENES},
            log=log,
            summary_extra_lines=[
                "proseg is a de-novo transcript-level segmentation method; output is "
                "directly transcript-level and safe to pass to get_metric.py."])

    log.info("DONE. Total wall: %.1fs", timer.total_seconds)
    stx.write_benchmark_stats(
        outdir=args.outdir, method=METHOD, modality="imaging",
        sample_name=args.sample_name, timer=timer, dataset=args.dataset,
        transcripts=stx.transcript_accounting(std, n_input=len(df_in)),
        entities=stx.entity_accounting(std, n_entities=int(len(cm))),
        qc={"n_proseg_cells": int(len(cm)),
            "voxel_layers": int(args.voxel_layers),
            "excluded_genes_regex": EXCLUDE_GENES},
        method_version=proseg_version,
        outputs=[str(std_path)] + ([str(h5ad_path)] if h5ad_ok else []),
        notes="De-novo transcript-level segmentation; output is transcript-level.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

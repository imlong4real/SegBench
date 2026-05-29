#!/usr/bin/env python3
"""Standalone Baysor runner for the TSU20 / NSCLC segmentation benchmark.

Wraps the ``baysor`` CLI (Julia; https://github.com/kharchenkolab/Baysor),
mirroring ``workflow/rules/_segmentation/baysor.smk`` + ``workflow/configs/
baysor_xenium.toml``, but driven from a standardized transcripts parquet and
emitting the benchmark's standardized transcript contract.

Two modes (the runner picks the first that is available):
  1. RUN mode      — if a `baysor` binary is on PATH (or --baysor-bin / a
                     container is supplied), run Baysor end-to-end.
  2. WRAP-RAW mode — if `--raw-segmentation <segmentation.csv>` is given, skip
                     running Baysor and just standardize an existing raw Baysor
                     output. Used when Baysor is only available in a container
                     (Baysor is NOT installed locally on this machine).
If neither is possible, the runner exits with a clear error (no silent skip).

Stages: load_inputs → convert_inputs → run_method → convert_outputs →
validate_schema → write_outputs. Metrics are computed later by get_metric.py.

EXAMPLE (run mode, where baysor is installed)
=============================================
    python workflow/scripts/run_baysor.py \\
      --transcripts dataset/lung_cancer_xenium_10x/TSU-20/filtered_df_standardized.parquet \\
      --outdir results/benchmark_runs/tsu20/Baysor \\
      --sample-name TSU20 --seed 1 --overwrite

EXAMPLE (wrap an existing raw segmentation.csv)
===============================================
    python workflow/scripts/run_baysor.py \\
      --transcripts dataset/lung_cancer_xenium_10x/TSU-20/filtered_df_standardized.parquet \\
      --raw-segmentation results/tsu20_tools/baysor/raw/segmentation.csv \\
      --outdir results/benchmark_runs/tsu20/Baysor --overwrite
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
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

METHOD = "baysor"
DEFAULT_CONFIG = _REPO_ROOT / "workflow" / "configs" / "baysor_xenium.toml"


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    rc.add_shared_args(p)
    p.add_argument("--baysor-bin", default=shutil.which("baysor") or "baysor",
                   help="Path to the baysor binary (default: from PATH).")
    p.add_argument("--baysor-config", type=Path, default=DEFAULT_CONFIG,
                   help="Baysor TOML config (default: workflow/configs/baysor_xenium.toml).")
    p.add_argument("--raw-segmentation", type=Path, default=None,
                   help="Standardize this existing Baysor segmentation.csv instead "
                        "of running Baysor (WRAP-RAW mode).")
    p.add_argument("--nthreads", type=int, default=4)
    p.add_argument("--use-prior", action="store_true",
                   help="Pass the input cell_id as a Baysor prior (':cell_id').")
    return p


def standardize_baysor_segmentation(
    seg_csv: Path, df_in: pd.DataFrame, *, log
) -> pd.DataFrame:
    """Baysor segmentation.csv → standardized transcript table.

    Baysor columns: transcript_id, cell_id(prior), gene, x, y, z, qv,
    cell (baysor assignment), assignment_confidence, is_noise, molecule_id.
    """
    seg = pd.read_csv(seg_csv)
    log.info("Baysor segmentation columns: %s", list(seg.columns))
    # Baysor assignment is the `cell` column; the prior is `cell_id`.
    assign_col = "cell" if "cell" in seg.columns else "cell_id"
    seg["_orig"] = seg["cell_id"].astype(str) if "cell_id" in seg.columns else pd.NA
    seg["_assign"] = seg[assign_col].astype("string")
    # Noise / empty / zero → UNASSIGNED.
    is_noise = (seg["is_noise"].astype(str).str.lower() == "true") if "is_noise" in seg.columns else False
    blank = seg["_assign"].isna() | seg["_assign"].astype(str).isin({"", "0", "nan", "<NA>"})
    seg["cell_final"] = seg["_assign"].astype(str)
    seg.loc[blank.fillna(True).to_numpy(), "cell_final"] = "UNASSIGNED"
    if np.any(is_noise):
        seg.loc[is_noise.to_numpy(), "cell_final"] = "UNASSIGNED"

    keep = seg.rename(columns={
        "gene": "feature_name", "cell_final": "cell_id_std",
        "assignment_confidence": "assignment_confidence",
    })
    keep["original_cell_id"] = seg["_orig"]
    # Build a clean frame for standardize_transcripts.
    frame = pd.DataFrame({
        "x": seg["x"], "y": seg["y"],
        "z": seg["z"] if "z" in seg.columns else 0.0,
        "feature_name": seg["gene"] if "gene" in seg.columns else seg.get("feature_name"),
        "cell_id": seg["cell_final"],
        "original_cell_id": seg["_orig"],
    })
    if "transcript_id" in seg.columns:
        frame["transcript_id"] = seg["transcript_id"]
    if "qv" in seg.columns:
        frame["qv"] = seg["qv"]
    if "overlaps_nucleus" in seg.columns:
        frame["overlaps_nucleus"] = seg["overlaps_nucleus"]
    if "assignment_confidence" in seg.columns:
        frame["assignment_confidence"] = seg["assignment_confidence"]
    return rc.standardize_transcripts(frame, method=METHOD, log=log)


def main() -> int:
    args = build_argparser().parse_args()
    sentinel = args.outdir / "outputs" / f"{METHOD}_transcripts_standardized.parquet"
    rc.prepare_outdir(args.outdir, sentinel, args.overwrite)
    log = rc.setup_logging(args.outdir, "run_baysor")
    log.info("=== run_baysor.py === sample=%s seed=%d", args.sample_name, args.seed)
    np.random.seed(args.seed)

    outputs_dir = args.outdir / "outputs"
    raw_dir = outputs_dir / "baysor_raw_output"
    raw_dir.mkdir(parents=True, exist_ok=True)
    timer = rc.StageTimer(log)

    # Decide mode.
    have_bin = shutil.which(args.baysor_bin) is not None or Path(args.baysor_bin).exists()
    wrap_raw = args.raw_segmentation is not None
    if not have_bin and not wrap_raw:
        raise SystemExit(
            "Baysor is not runnable here: no `baysor` binary on PATH and no "
            "--raw-segmentation provided. Baysor is container-only on this machine "
            "(reproducibility/baysor.def); either install/point --baysor-bin to a "
            "baysor binary, or pass --raw-segmentation <segmentation.csv> to "
            "standardize an existing raw Baysor output.")

    baysor_version = "unknown"
    notes: list[str] = []

    # --- load_inputs --------------------------------------------------------
    with timer.time("load_inputs"):
        df_in = rc.load_input_transcripts(
            args.transcripts, log=log,
            max_transcripts=args.max_transcripts, seed=args.seed)
        if "transcript_id" not in df_in.columns:
            df_in["transcript_id"] = np.arange(len(df_in), dtype=np.int64)
        df_in["cell_id"] = df_in["cell_id"].astype(str)

    seg_csv: Path
    if have_bin and not wrap_raw:
        try:
            baysor_version = subprocess.check_output(
                [args.baysor_bin, "--version"], text=True).strip()
        except Exception:
            pass
        log.info("RUN mode — baysor version: %s", baysor_version)
        # --- convert_inputs: baysor parquet (x_location/y_location/...) ------
        baysor_in = raw_dir / "baysor_input.parquet"
        with timer.time("convert_inputs"):
            bdf = pd.DataFrame({
                "x_location": df_in["x"].astype(np.float32),
                "y_location": df_in["y"].astype(np.float32),
                "z_location": (df_in["z"] if "z" in df_in.columns else 0.0),
                "feature_name": df_in["feature_name"].astype(str),
                "cell_id": df_in["cell_id"].astype(str),
                "transcript_id": df_in["transcript_id"],
            })
            bdf.to_parquet(baysor_in, index=False)
        # --- run_method ------------------------------------------------------
        with timer.time("run_method"):
            cmd = [args.baysor_bin, "run", "-c", str(args.baysor_config.resolve()),
                   "-o", str(raw_dir.resolve()) + "/", str(baysor_in.resolve())]
            if args.use_prior:
                cmd.append(":cell_id")
            import os as _os
            env = dict(_os.environ, JULIA_NUM_THREADS=str(args.nthreads))
            rc_code, ext_rss = rc.run_subprocess(cmd, log=log, outdir=args.outdir,
                                                 env=env, cwd=raw_dir)
            timer.record_external("run_method", ext_rss)
            if rc_code != 0:
                raise SystemExit(f"baysor failed (exit {rc_code}); see run.log.")
        seg_csv = raw_dir / "segmentation.csv"
        if not seg_csv.exists():
            raise SystemExit(f"baysor did not produce {seg_csv}; see run.log.")
    else:
        # WRAP-RAW mode.
        notes.append(
            "Baysor was NOT re-run (no local binary/container). Standardized an "
            f"existing raw Baysor segmentation: {args.raw_segmentation}.")
        log.info("WRAP-RAW mode — standardizing %s", args.raw_segmentation)
        for _stage in ("convert_inputs", "run_method"):
            with timer.time(_stage):
                pass
        seg_csv = Path(args.raw_segmentation)
        # Copy/link the raw output into the run dir for provenance.
        try:
            shutil.copy2(seg_csv, raw_dir / "segmentation.csv")
            seg_csv = raw_dir / "segmentation.csv"
        except Exception as e:
            log.warning("Could not copy raw segmentation into run dir: %s", e)

    # --- convert_outputs ----------------------------------------------------
    std_path = outputs_dir / f"{METHOD}_transcripts_standardized.parquet"
    h5ad_path = outputs_dir / f"{METHOD}_cell_by_gene.h5ad"
    with timer.time("convert_outputs"):
        std = standardize_baysor_segmentation(seg_csv, df_in, log=log)
        std.to_parquet(std_path, index=False, compression="snappy")
        log.info("Wrote standardized transcripts: %s", std_path)
        rc.build_cell_by_gene_h5ad(std, out_path=h5ad_path, log=log)

    # --- validate_schema ----------------------------------------------------
    with timer.time("validate_schema"):
        rc.validate_schema(
            std, method=METHOD, out_path=std_path, in_path=args.transcripts,
            report_path=args.outdir / "schema_validation_report.json", log=log,
            extra={"baysor_version": baysor_version,
                   "mode": "wrap_raw" if (wrap_raw or not have_bin) else "run"})

    # --- write_outputs ------------------------------------------------------
    with timer.time("write_outputs"):
        rc.write_provenance(
            outdir=args.outdir, method=METHOD, sample_name=args.sample_name,
            args=args, timer=timer, repo_root=_REPO_ROOT,
            inputs={"transcripts": str(args.transcripts),
                    "baysor_config": str(args.baysor_config),
                    "raw_segmentation": str(args.raw_segmentation or "")},
            outputs=[str(std_path), str(h5ad_path), str(seg_csv)],
            method_version=baysor_version, runner_kind="binary",
            extra_config={"mode": "wrap_raw" if (wrap_raw or not have_bin) else "run",
                          "config": str(args.baysor_config)},
            log=log,
            summary_extra_lines=(notes or [
                "Baysor is a de-novo transcript-level segmentation method; output is "
                "transcript-level and safe to pass to get_metric.py."]))

    log.info("DONE. Total wall: %.1fs", timer.total_seconds)
    return 0


if __name__ == "__main__":
    sys.exit(main())

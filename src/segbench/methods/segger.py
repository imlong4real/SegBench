#!/usr/bin/env python3
"""Segger runner — GNN transcript-to-nucleus assignment.

Segger (https://github.com/EliHei2/segger_dev) learns a graph neural network
over a transcript/nucleus graph and assigns each molecule to a nucleus. It
ships three CLI scripts that must run in order:

    create_dataset_fast.py   tile the sample into PyG graphs (train/val/test)
    train_model.py           train the GNN                      (GPU)
    predict_fast.py          assign transcripts -> segger_cell_id

This wrapper drives all three, then standardizes ``segger_transcripts.parquet``
into the benchmark transcript contract. It supersedes the hand-rolled
``scripts/slurm/run_segger_*.sbatch`` jobs: those hard-coded one cluster's
paths, while this reads them from config/CLI.

MODES (``--mode``)
==================
  * ``run``  (default) — execute preprocess + train + predict for real. Only
        this mode produces benchmark-valid runtime/memory numbers.
  * ``wrap`` — standardize an existing ``--segger-transcripts`` parquet without
        re-running Segger. Useful for re-validating an old cluster run; the
        emitted runtime is labelled NOT benchmark-valid.

EXECUTION BACKEND
=================
Segger needs CUDA + torch-geometric, which is usually easiest inside a
container. Either works:

  native      --segger-cli-dir /path/to/segger_dev/src/segger/cli
  container   --container containers/python_cuda.sif \
              --segger-cli-dir /opt/segger_dev/src/segger/cli

With ``--container`` every step runs via ``apptainer exec --nv``; paths passed
to Segger are the container-visible ones, so pass ``--bind`` as needed.

INPUT
=====
Segger needs a Xenium-style bundle (``transcripts.parquet`` +
``nucleus_boundaries.parquet`` + ``experiment.xenium``), not a flat transcript
table. If you pass ``--transcripts`` (the benchmark's standardized parquet),
this wrapper builds that bundle for you via
``workflow/scripts/_segmentation/prepare_roi_segger_bundle.py``. If you already
have a real Xenium output directory, pass ``--xenium-dir`` and it is used
as-is, which is always preferable — real nucleus boundaries beat derived ones.

EXAMPLE
=======
    segbench run segger \\
      --xenium-dir  dataset/lung_cancer_xenium_10x/TSU-20 \\
      --outdir      benchmark_output/tsu20/segger \\
      --container   containers/python_cuda.sif \\
      --sample-name TSU20 --max-epochs 3 --overwrite
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .. import REPO_ROOT as _REPO_ROOT
from .. import common as rc
from .. import stats as stx
from . import _base

METHOD = "segger"
BUNDLE_SCRIPT = (_REPO_ROOT / "workflow" / "scripts" / "_segmentation"
                 / "prepare_roi_segger_bundle.py")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _base.add_common_args(p, method=METHOD)
    _base.add_transcript_input_args(p)
    p.add_argument("--mode", choices=("run", "wrap"), default="run",
                   help="run: execute Segger. wrap: standardize existing output.")
    p.add_argument("--xenium-dir", type=Path, default=None,
                   help="Existing Xenium bundle dir (preferred over --transcripts).")
    p.add_argument("--segger-transcripts", type=Path, default=None,
                   help="[wrap mode] existing segger_transcripts.parquet.")
    # --- execution backend ---
    p.add_argument("--container", type=Path, default=None,
                   help="Apptainer/Singularity .sif to run Segger inside.")
    p.add_argument("--container-exec", default=None,
                   help="apptainer|singularity (default: whichever is on PATH).")
    p.add_argument("--bind", action="append", default=[],
                   help="Extra bind mount for the container (repeatable), src:dst.")
    p.add_argument("--segger-cli-dir", default=None,
                   help="Directory holding create_dataset_fast.py etc. Omit to "
                        "invoke the installed `segger` console scripts.")
    p.add_argument("--python-bin", default=None,
                   help="Python used to launch the Segger CLI scripts.")
    # --- Segger hyper-parameters ---
    p.add_argument("--tile-width", type=int, default=200)
    p.add_argument("--tile-height", type=int, default=200)
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-tx-tokens", type=int, default=500)
    p.add_argument("--accelerator", default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--knn-method", default="kd_tree")
    p.add_argument("--model-version", type=int, default=0)
    p.add_argument("--use-cc", default="False",
                   help="Segger --use_cc (connected components) flag.")
    p.add_argument("--skip-preprocess", action="store_true",
                   help="Reuse preprocessed tiles already in the output dir.")
    return p


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------
def _container_exec(args) -> str | None:
    if args.container is None:
        return None
    if args.container_exec:
        return args.container_exec
    for cand in ("apptainer", "singularity"):
        if shutil.which(cand):
            return cand
    raise SystemExit(
        "--container was given but neither `apptainer` nor `singularity` is on "
        "PATH. Load the module, or drop --container to run Segger natively.")


def _wrap_cmd(args, inner: list[str]) -> list[str]:
    """Wrap a Segger command so it runs natively or inside the container."""
    py = args.python_bin or ("python" if args.container else sys.executable)
    cmd = [py] + inner
    exe = _container_exec(args)
    if exe is None:
        return cmd
    pre = [exe, "exec"]
    if args.accelerator == "cuda":
        pre.append("--nv")
    for b in args.bind:
        pre += ["--bind", b]
    return pre + [str(args.container)] + cmd


def _cli_script(args, name: str) -> list[str]:
    """Path/argv prefix for one Segger CLI entry point."""
    if args.segger_cli_dir:
        return [str(Path(args.segger_cli_dir) / name)]
    # Installed package: run the module rather than a loose script path.
    return ["-m", f"segger.cli.{name[:-3]}"]


def _run_step(args, inner: list[str], *, log, outdir: Path, label: str):
    cmd = _wrap_cmd(args, inner)
    code, ext_rss = rc.run_subprocess(
        cmd, log=log, outdir=outdir, external_time_name=f"external_time_{label}.txt")
    if code != 0:
        raise SystemExit(
            f"Segger {label} failed (exit {code}). See run.log. "
            f"Command: {' '.join(str(c) for c in cmd)}")
    return ext_rss


def _build_bundle(args, *, log, dest: Path) -> Path:
    """Derive a Xenium-style bundle from a standardized transcripts parquet."""
    tx = _base.require_input(args, "transcripts", "--transcripts")
    log.info("No --xenium-dir given; deriving a Xenium bundle from %s", tx)
    log.warning("Nucleus boundaries will be DERIVED from overlaps_nucleus "
                "transcripts. Pass --xenium-dir with real boundaries when you "
                "have them — Segger's accuracy depends on them.")
    cmd = [sys.executable, str(BUNDLE_SCRIPT), "--input", str(tx),
           "--outdir", str(dest), "--dataset", args.sample_name]
    code, _ = rc.run_subprocess(cmd, log=log, outdir=args.outdir,
                                external_time_name="external_time_bundle.txt")
    if code != 0:
        raise SystemExit(f"Building the Segger input bundle failed (exit {code}).")
    return dest / args.sample_name


def _standardize(df: pd.DataFrame, *, log) -> pd.DataFrame:
    """Segger predict output -> benchmark transcript contract.

    Column mapping mirrors ``workflow/scripts/_segmentation/
    standardize_segger_output.py``: the assignment lives in ``segger_cell_id``
    while ``cell_id`` still holds the vendor's original assignment, which we
    keep as ``original_cell_id`` for pre/post comparison.
    """
    ren = {"x_location": "x", "y_location": "y", "z_location": "z"}
    if "segger_cell_id" not in df.columns:
        raise SystemExit(
            f"Segger output has no 'segger_cell_id' column; got "
            f"{sorted(df.columns)}. Was --cell_id_col changed at predict time?")
    if "cell_id" in df.columns:
        df = df.rename(columns={"cell_id": "original_cell_id"})
    ren["segger_cell_id"] = "cell_id"
    keep = [c for c in ("x_location", "y_location", "z_location", "feature_name",
                        "segger_cell_id", "original_cell_id", "transcript_id",
                        "qv", "overlaps_nucleus", "score", "bound")
            if c in df.columns]
    return rc.standardize_transcripts(df[keep], method="Segger", rename=ren, log=log)


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    _base.resolve_config(args, method=METHOD)
    if args.max_epochs is None:
        args.max_epochs = 3
    if args.batch_size is None:
        args.batch_size = 4

    outputs_dir = args.outdir / "outputs"
    std_path = outputs_dir / f"{METHOD}_transcripts_standardized.parquet"
    rc.prepare_outdir(args.outdir, std_path, args.overwrite)
    log = rc.setup_logging(args.outdir, "segbench.segger")
    log.info("=== segger === mode=%s sample=%s seed=%d",
             args.mode, args.sample_name, args.seed)
    np.random.seed(args.seed)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    timer = rc.StageTimer(log)

    work = args.outdir / "work"
    prep_dir = work / "preprocessed_data"
    model_dir = work / "trained_model"
    pred_dir = work / "segger_output"

    if args.dry_run:
        log.info("[dry-run] mode=%s xenium_dir=%s transcripts=%s container=%s",
                 args.mode, args.xenium_dir, args.transcripts, args.container)
        log.info("[dry-run] would write %s", std_path)
        return 0

    ext_rss_max: float | None = None
    seg_tx_path: Path | None = None
    n_input: int | None = None

    # --- load_inputs --------------------------------------------------------
    with timer.time("load_inputs"):
        if args.mode == "wrap":
            seg_tx_path = _base.require_input(
                args, "segger_transcripts", "--segger-transcripts")
        else:
            if args.xenium_dir:
                bundle = Path(args.xenium_dir)
                if not bundle.exists():
                    raise SystemExit(f"--xenium-dir not found: {bundle}")
            else:
                bundle = _build_bundle(args, log=log, dest=work / "xenium_bundle")
            tx_file = bundle / "transcripts.parquet"
            if not tx_file.exists():
                raise SystemExit(
                    f"Segger input bundle has no transcripts.parquet: {tx_file}")
            try:
                n_input = int(len(rc.read_parquet_robust(tx_file, log=log)))
            except SystemExit:
                n_input = None
            log.info("Segger input bundle: %s (%s transcripts)", bundle, n_input)

    # --- run_method (preprocess + train + predict) --------------------------
    if args.mode == "run":
        with timer.time("run_method"):
            for d in (prep_dir, model_dir, pred_dir):
                d.mkdir(parents=True, exist_ok=True)

            has_tiles = any(prep_dir.rglob("*.pt"))
            if args.skip_preprocess and has_tiles:
                log.info("Reusing %d preprocessed tiles.", len(list(prep_dir.rglob('*.pt'))))
            else:
                r = _run_step(args, _cli_script(args, "create_dataset_fast.py") + [
                    "--base_dir", str(bundle),
                    "--data_dir", str(prep_dir),
                    "--sample_type", "xenium",
                    "--tile_width", str(args.tile_width),
                    "--tile_height", str(args.tile_height),
                    "--n_workers", str(args.threads),
                ], log=log, outdir=args.outdir, label="preprocess")
                ext_rss_max = max(filter(None, [ext_rss_max, r]), default=None)

            r = _run_step(args, _cli_script(args, "train_model.py") + [
                "--dataset_dir", str(prep_dir),
                "--models_dir", str(model_dir),
                "--sample_tag", str(args.sample_name),
                "--num_tx_tokens", str(args.num_tx_tokens),
                "--accelerator", args.accelerator,
                "--devices", str(args.devices),
                "--max_epochs", str(args.max_epochs),
                "--batch_size", str(args.batch_size),
                "--num_workers", str(args.threads),
            ], log=log, outdir=args.outdir, label="train")
            ext_rss_max = max(filter(None, [ext_rss_max, r]), default=None)

            r = _run_step(args, _cli_script(args, "predict_fast.py") + [
                "--segger_data_dir", str(prep_dir),
                "--models_dir", str(model_dir),
                "--benchmarks_dir", str(pred_dir),
                "--transcripts_file", str(bundle / "transcripts.parquet"),
                "--batch_size", str(args.batch_size),
                "--num_workers", str(args.threads),
                "--model_version", str(args.model_version),
                "--knn_method", args.knn_method,
                "--cell_id_col", "segger_cell_id",
                "--use_cc", str(args.use_cc),
            ], log=log, outdir=args.outdir, label="predict")
            ext_rss_max = max(filter(None, [ext_rss_max, r]), default=None)

            found = sorted(pred_dir.rglob("segger_transcripts.parquet"))
            if not found:
                raise SystemExit(
                    f"Segger predict produced no segger_transcripts.parquet under "
                    f"{pred_dir}. See run.log.")
            seg_tx_path = found[0]
        timer.record_external("run_method", ext_rss_max)

    # --- convert_outputs ----------------------------------------------------
    with timer.time("convert_outputs"):
        log.info("Standardizing Segger output: %s", seg_tx_path)
        std = _standardize(rc.read_parquet_robust(seg_tx_path, log=log), log=log)
        std.to_parquet(std_path, index=False, compression="snappy")
        log.info("Wrote %s", std_path)
        h5ad_path = outputs_dir / f"{METHOD}_cell_by_gene.h5ad"
        h5ad_ok = False
        try:
            rc.build_cell_by_gene_h5ad(std, out_path=h5ad_path, log=log)
            h5ad_ok = True
        except Exception as e:  # secondary artifact; never sink a GPU run
            log.warning("Skipping cell-by-gene h5ad (%s: %s).", type(e).__name__,
                        str(e)[:160])

    # --- validate_schema ----------------------------------------------------
    with timer.time("validate_schema"):
        rc.validate_schema(
            std, method=METHOD, out_path=std_path, in_path=str(seg_tx_path),
            report_path=args.outdir / "schema_validation_report.json", log=log,
            extra={"mode": args.mode, "max_epochs": args.max_epochs})

    # --- write_outputs ------------------------------------------------------
    with timer.time("write_outputs"):
        outs = [str(std_path)] + ([str(h5ad_path)] if h5ad_ok else [])
        rc.write_provenance(
            outdir=args.outdir, method=METHOD, sample_name=args.sample_name,
            args=args, timer=timer, repo_root=_REPO_ROOT,
            inputs={"xenium_dir": str(args.xenium_dir or ""),
                    "transcripts": str(args.transcripts or ""),
                    "segger_transcripts": str(seg_tx_path)},
            outputs=outs, runner_kind="python", log=log,
            extra_config={"mode": args.mode, "tile": [args.tile_width, args.tile_height],
                          "max_epochs": args.max_epochs, "batch_size": args.batch_size,
                          "accelerator": args.accelerator, "container": str(args.container or "")},
            summary_extra_lines=[
                "Segger is a de-novo transcript-level segmentation method; its "
                "output is transcript-level and safe to pass to get_metric.py."
                if args.mode == "run" else
                "WRAP mode: runtime/memory are NOT benchmark-valid."])

    n_bound = int(std["bound"].astype(str).str.lower().eq("true").sum()) \
        if "bound" in std.columns else None
    stx.write_benchmark_stats(
        outdir=args.outdir, method=METHOD, modality="imaging",
        sample_name=args.sample_name, timer=timer, dataset=args.dataset,
        transcripts=stx.transcript_accounting(std, n_input=n_input),
        entities=stx.entity_accounting(std),
        qc={"mode": args.mode,
            "max_epochs": int(args.max_epochs),
            "accelerator": args.accelerator,
            "n_bound_transcripts": n_bound,
            "mean_assignment_score": float(std["score"].mean())
                if "score" in std.columns else None,
            "runtime_valid_for_benchmark": args.mode == "run"},
        outputs=outs,
        notes=("Real Segger execution." if args.mode == "run" else
               "WRAP mode: runtime/memory NOT benchmark-valid."))

    log.info("DONE (mode=%s). Total wall: %.1fs", args.mode, timer.total_seconds)
    return 0


if __name__ == "__main__":
    sys.exit(main())

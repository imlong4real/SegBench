#!/usr/bin/env python3
"""Standalone Baysor runner for the TSU20 / NSCLC segmentation benchmark.

Drives the ``baysor`` CLI (Julia; https://github.com/kharchenkolab/Baysor),
mirroring ``workflow/rules/_segmentation/baysor.smk`` + ``workflow/configs/
baysor_xenium.toml``, but driven from a standardized transcripts parquet and
emitting the benchmark's standardized transcript contract.

TWO EXPLICIT MODES (``--mode``)
===============================
  * ``--mode run``  (DEFAULT, publication-grade)
        Execute Baysor end-to-end from the input transcripts: convert the
        standardized parquet into a Baysor CSV, (re)generate a Baysor config,
        invoke the real ``baysor`` binary, and parse its output. This is the
        ONLY mode whose runtime/memory numbers are valid for benchmarking,
        because Baysor actually runs here.

  * ``--mode wrap``
        Standardize a pre-existing raw ``segmentation.csv`` (``--raw-segmentation``)
        WITHOUT running Baysor. Useful for re-validating old outputs, but the
        emitted runtime/memory files are NOT valid for benchmarking and every
        report is labelled accordingly.

Fail-fast: in ``--mode run`` the runner NEVER silently falls back to wrap mode.
If the Baysor binary cannot be found it exits with a clear, actionable error.

Stages (each timed individually): load_inputs → convert_inputs → run_method →
convert_outputs → validate_schema → write_outputs. The run_method stage holds
the Baysor-only wall time and (via /usr/bin/time) the Baysor-only peak RSS.
Benchmark metrics are computed separately by get_metric.py.

OUTPUT LAYOUT (under --outdir)
==============================
    inputs/baysor_input.csv                          Baysor CSV input
    inputs/baysor_config.toml                        config actually used
    raw_output/                                      raw Baysor output dir
    outputs/baysor_transcripts_standardized.parquet  standardized contract
    outputs/baysor_cell_by_gene.h5ad                 cells x genes counts
    runtime_memory.json                              timing + memory + counts
    runtime_by_stage.tsv                             one row per stage
    external_time.txt                                /usr/bin/time capture
    schema_validation_report.json                    schema check
    run_summary.md                                   human-readable summary
    run.log                                          full log + tee'd Baysor I/O

SMOKE TEST (true RUN mode on a TSU20 subset)
============================================
    python workflow/scripts/run_baysor.py \\
      --mode run \\
      --transcripts dataset/lung_cancer_xenium_10x/TSU-20/filtered_df_standardized.parquet \\
      --outdir results/benchmark_runs/tsu20_smoke/Baysor \\
      --sample-name TSU20_smoke \\
      --n-transcripts-smoke 100000 \\
      --baysor-bin baysor --n-threads 4 --overwrite

FULL PUBLICATION RUN
====================
    python workflow/scripts/run_baysor.py \\
      --mode run \\
      --transcripts dataset/lung_cancer_xenium_10x/TSU-20/filtered_df_standardized.parquet \\
      --outdir results/benchmark_runs/tsu20/Baysor \\
      --sample-name TSU20 \\
      --baysor-bin baysor --n-threads 16 --overwrite

WRAP (re-standardize an existing raw segmentation.csv — NOT benchmarkable)
==========================================================================
    python workflow/scripts/run_baysor.py \\
      --mode wrap \\
      --transcripts dataset/lung_cancer_xenium_10x/TSU-20/filtered_df_standardized.parquet \\
      --raw-segmentation results/tsu20_tools/baysor/raw/segmentation.csv \\
      --outdir results/benchmark_runs/tsu20/Baysor --overwrite
"""
from __future__ import annotations

import argparse
import os
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
# Baysor 0.7 requires either a prior segmentation OR an explicit scale (approx
# cell radius). When neither is supplied we default to a Xenium-appropriate
# cell radius so RUN mode is self-sufficient.
DEFAULT_SCALE = 8.0
# Common locations to probe when the requested binary isn't directly resolvable.
_BAYSOR_FALLBACKS = (Path.home() / ".julia" / "bin" / "baysor",)
# Baysor input column names (kept distinct from the OUTPUT `cell` column).
_BX, _BY, _BZ, _BGENE, _BPRIOR = "x", "y", "z", "gene", "prior_cell"


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    rc.add_shared_args(p)
    p.add_argument("--mode", choices=("run", "wrap"), default="run",
                   help="run: execute Baysor from transcripts (publication-grade, "
                        "DEFAULT). wrap: standardize an existing segmentation.csv "
                        "WITHOUT running Baysor (not valid for runtime benchmarking).")
    p.add_argument("--baysor-bin", default=shutil.which("baysor") or "baysor",
                   help="Path to the baysor binary (default: from PATH; "
                        "~/.julia/bin/baysor is also probed).")
    p.add_argument("--config", "--baysor-config", dest="config", type=Path, default=None,
                   help="Baysor TOML config to use as-is. If omitted, a config is "
                        "generated under inputs/baysor_config.toml from the CLI "
                        "options (template: workflow/configs/baysor_xenium.toml).")
    p.add_argument("--scale", type=float, default=None,
                   help="Baysor -s/--scale (approx cell radius, same units as x/y). "
                        "If omitted, Baysor infers it from --min-molecules-per-cell.")
    p.add_argument("--prior-segmentation-confidence", type=float, default=0.5,
                   help="Confidence in the prior segmentation in [0,1] (only used "
                        "with --use-prior). Default 0.5.")
    p.add_argument("--min-molecules-per-cell", type=int, default=50,
                   help="Baysor -m/--min-molecules-per-cell. Default 50.")
    p.add_argument("--n-threads", "--nthreads", dest="n_threads", type=int, default=4,
                   help="JULIA_NUM_THREADS for Baysor. Default 4.")
    p.add_argument("--tempdir", type=Path, default=None,
                   help="Directory for Baysor/Julia temp files (sets TMPDIR).")
    p.add_argument("--use-prior", action="store_true",
                   help="Pass the input cell_id as a Baysor prior segmentation.")
    p.add_argument("--n-transcripts-smoke", type=int, default=None,
                   help="Smoke-test helper: subsample input to at most N transcripts "
                        "(seeded). Alias/override for --max-transcripts.")
    p.add_argument("--raw-segmentation", type=Path, default=None,
                   help="(--mode wrap only) Standardize this existing Baysor "
                        "segmentation.csv instead of running Baysor.")
    return p


def resolve_baysor_bin(requested: str) -> str | None:
    """Resolve the Baysor binary path, or None if it cannot be found.

    Resolution order: the requested path as-is (file) → on PATH. Known fallback
    locations (~/.julia/bin/baysor) are only probed when ``requested`` is a bare
    command name (no path separator) — an explicit path that does not exist
    fails fast rather than silently switching to a different binary. Never falls
    back to wrap mode.
    """
    cand = Path(requested).expanduser()
    if cand.exists():
        return str(cand)
    on_path = shutil.which(requested)
    if on_path:
        return on_path
    is_bare_name = os.sep not in requested and (
        os.altsep is None or os.altsep not in requested)
    if is_bare_name:
        for fb in _BAYSOR_FALLBACKS:
            if fb.exists():
                return str(fb)
    return None


def baysor_version(baysor_bin: str) -> str:
    try:
        return subprocess.check_output(
            [baysor_bin, "--version"], text=True,
            stderr=subprocess.DEVNULL).strip().splitlines()[-1].strip()
    except Exception:
        return "unknown"


def resolve_scale(args: argparse.Namespace, *, log) -> float | None:
    """Effective Baysor scale. None means 'let the prior drive segmentation'.

    Baysor 0.7 requires a prior OR a scale; if neither is supplied we fall back
    to DEFAULT_SCALE so RUN mode never fails for a missing scale.
    """
    if args.scale is not None:
        return float(args.scale)
    if args.use_prior:
        return None  # prior satisfies Baysor's requirement
    log.info("No --scale and no --use-prior; defaulting scale to %.1f "
             "(approx Xenium cell radius).", DEFAULT_SCALE)
    return DEFAULT_SCALE


def write_baysor_config(dest: Path, args: argparse.Namespace, scale: float | None,
                        *, log) -> Path:
    """Use --config as-is (copied for provenance) or generate one under inputs/."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if args.config is not None:
        shutil.copy2(args.config, dest)
        log.info("Using supplied Baysor config: %s (copied to %s)", args.config, dest)
        return dest
    # exclude_genes is "" so every input molecule survives into segmentation.csv,
    # keeping the molecule_id <-> input-row mapping bijective (the standardized
    # benchmark input already has control probes removed).
    seg_lines = [f"prior_segmentation_confidence = {float(args.prior_segmentation_confidence)}"]
    if scale is not None:
        seg_lines.insert(0, f"scale = {scale}")
    toml = (
        "# Auto-generated by run_baysor.py (--mode run).\n"
        "[data]\n"
        f'x = "{_BX}"\n'
        f'y = "{_BY}"\n'
        f'z = "{_BZ}"\n'
        f'gene = "{_BGENE}"\n'
        "min_molecules_per_gene = 0\n"
        f"min_molecules_per_cell = {int(args.min_molecules_per_cell)}\n"
        'exclude_genes = ""\n'
        "\n"
        "[segmentation]\n"
        + "\n".join(seg_lines) + "\n"
    )
    dest.write_text(toml)
    log.info("Generated Baysor config: %s", dest)
    return dest


def standardize_baysor_segmentation(
    seg_csv: Path, df_in: pd.DataFrame, *, log
) -> pd.DataFrame:
    """Baysor segmentation.csv → standardized transcript table.

    Baysor 0.7 columns: x, y, z, gene, cell (assignment), molecule_id (1-based
    input row index), confidence, cluster, assignment_confidence, is_noise.
    Empty `cell` or is_noise == True → UNASSIGNED. transcript_id / qv /
    overlaps_nucleus / original (prior) cell_id are recovered from df_in via
    molecule_id when the mapping is bijective.
    """
    seg = pd.read_csv(seg_csv)
    log.info("Baysor segmentation columns: %s", list(seg.columns))
    assign_col = "cell" if "cell" in seg.columns else "cell_id"
    cell = seg[assign_col].astype("string")
    is_noise = (seg["is_noise"].astype(str).str.lower() == "true") \
        if "is_noise" in seg.columns else pd.Series(False, index=seg.index)
    blank = cell.isna() | cell.astype(str).isin({"", "0", "nan", "<NA>", "NA"})
    cell_final = cell.astype(str)
    cell_final[blank.fillna(True).to_numpy()] = "UNASSIGNED"
    cell_final[is_noise.fillna(False).to_numpy()] = "UNASSIGNED"

    frame = pd.DataFrame({
        "x": seg["x"],
        "y": seg["y"],
        "z": seg["z"] if "z" in seg.columns else 0.0,
        "feature_name": seg["gene"] if "gene" in seg.columns else seg.get("feature_name"),
        "cell_id": cell_final.to_numpy(),
    })
    if "assignment_confidence" in seg.columns:
        frame["assignment_confidence"] = seg["assignment_confidence"]

    # Recover transcript-level provenance from the input via molecule_id.
    if "molecule_id" in seg.columns:
        mid = pd.to_numeric(seg["molecule_id"], errors="coerce")
        pos = (mid - 1).astype("Int64")
        valid = pos.notna() & (pos >= 0) & (pos < len(df_in))
        if bool(valid.all()) and seg["molecule_id"].is_unique:
            ipos = pos.astype(int).to_numpy()
            for col, std_col in (("transcript_id", "transcript_id"),
                                 ("qv", "qv"),
                                 ("overlaps_nucleus", "overlaps_nucleus"),
                                 ("cell_id", "original_cell_id")):
                if col in df_in.columns:
                    frame[std_col] = df_in[col].to_numpy()[ipos]
            log.info("Mapped %d molecules back to input rows via molecule_id.", len(frame))
        else:
            log.warning("molecule_id is not a clean 1..N input index "
                        "(unique=%s, all-in-range=%s); skipping provenance join.",
                        seg["molecule_id"].is_unique, bool(valid.all()))
    return rc.standardize_transcripts(frame, method=METHOD, log=log)


def main() -> int:
    args = build_argparser().parse_args()
    if args.n_transcripts_smoke is not None:
        args.max_transcripts = args.n_transcripts_smoke

    sentinel = args.outdir / "outputs" / f"{METHOD}_transcripts_standardized.parquet"
    rc.prepare_outdir(args.outdir, sentinel, args.overwrite)
    log = rc.setup_logging(args.outdir, "run_baysor")
    log.info("=== run_baysor.py === mode=%s sample=%s seed=%d",
             args.mode, args.sample_name, args.seed)
    np.random.seed(args.seed)

    inputs_dir = args.outdir / "inputs"
    raw_dir = args.outdir / "raw_output"
    outputs_dir = args.outdir / "outputs"
    for d in (inputs_dir, raw_dir, outputs_dir):
        d.mkdir(parents=True, exist_ok=True)
    timer = rc.StageTimer(log)
    notes: list[str] = []

    # --- mode gating + fail-fast --------------------------------------------
    baysor_bin: str | None = None
    if args.mode == "run":
        baysor_bin = resolve_baysor_bin(args.baysor_bin)
        if baysor_bin is None:
            raise SystemExit(
                "Baysor binary not found. Install Baysor or provide "
                "--baysor-bin /path/to/baysor. Do not fall back to wrap mode "
                "automatically.")
        bversion = baysor_version(baysor_bin)
        log.info("RUN mode — baysor binary: %s (version %s)", baysor_bin, bversion)
    else:  # wrap
        if args.raw_segmentation is None:
            raise SystemExit("--mode wrap requires --raw-segmentation <segmentation.csv>.")
        bversion = "n/a (wrap mode — Baysor not executed)"
        notes.append(
            "WRAP MODE: Baysor was NOT executed; an existing raw segmentation.csv "
            f"({args.raw_segmentation}) was standardized. The runtime/memory files "
            "in this directory are NOT valid for runtime/memory benchmarking.")
        log.warning(notes[-1])

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
        df_in["cell_id"] = df_in["cell_id"].astype(str)
        n_input = int(len(df_in))

    seg_csv: Path
    config_used = inputs_dir / "baysor_config.toml"
    baysor_in = inputs_dir / "baysor_input.csv"

    eff_scale = resolve_scale(args, log=log) if args.mode == "run" else None
    if args.mode == "run":
        # --- convert_inputs: Baysor CSV + config -----------------------------
        with timer.time("convert_inputs"):
            cols = {_BX: df_in["x"].astype(np.float32),
                    _BY: df_in["y"].astype(np.float32),
                    _BZ: df_in["z"].astype(np.float32),
                    _BGENE: df_in["feature_name"].astype(str)}
            if args.use_prior:
                cols[_BPRIOR] = df_in["cell_id"].astype(str)
            pd.DataFrame(cols).to_csv(baysor_in, index=False)
            log.info("Wrote Baysor input CSV: %s (%d transcripts)", baysor_in, n_input)
            write_baysor_config(config_used, args, eff_scale, log=log)

        # --- run_method: invoke the real Baysor binary -----------------------
        with timer.time("run_method"):
            cmd = [baysor_bin, "run",
                   "-c", str(config_used.resolve()),
                   "-o", str(raw_dir.resolve()) + "/",
                   "-m", str(args.min_molecules_per_cell)]
            if eff_scale is not None:
                cmd += ["-s", str(eff_scale)]
            cmd.append(str(baysor_in.resolve()))
            if args.use_prior:
                cmd += [f":{_BPRIOR}",
                        "--prior-segmentation-confidence",
                        str(args.prior_segmentation_confidence)]
            env = dict(os.environ, JULIA_NUM_THREADS=str(args.n_threads))
            if args.tempdir is not None:
                args.tempdir.mkdir(parents=True, exist_ok=True)
                env["TMPDIR"] = str(args.tempdir.resolve())
            rc_code, ext_rss = rc.run_subprocess(
                cmd, log=log, outdir=args.outdir, env=env, cwd=raw_dir)
            if rc_code != 0:
                raise SystemExit(f"Baysor failed (exit {rc_code}); see run.log.")
        # Attach Baysor-only peak RSS AFTER the stage is recorded (record_external
        # scans completed stages, so it must run outside the `with` block).
        timer.record_external("run_method", ext_rss)
        seg_csv = raw_dir / "segmentation.csv"
        if not seg_csv.exists():
            raise SystemExit(f"Baysor did not produce {seg_csv}; see run.log.")
    else:
        # WRAP mode: no input conversion / no Baysor execution.
        for _stage in ("convert_inputs", "run_method"):
            with timer.time(_stage):
                pass
        seg_csv = Path(args.raw_segmentation)
        try:
            shutil.copy2(seg_csv, raw_dir / "segmentation.csv")
            seg_csv = raw_dir / "segmentation.csv"
        except Exception as e:
            log.warning("Could not copy raw segmentation into run dir: %s", e)

    # --- convert_outputs ----------------------------------------------------
    std_path = outputs_dir / f"{METHOD}_transcripts_standardized.parquet"
    h5ad_path = outputs_dir / f"{METHOD}_cell_by_gene.h5ad"
    h5ad_ok = False
    with timer.time("convert_outputs"):
        std = standardize_baysor_segmentation(seg_csv, df_in, log=log)
        std.to_parquet(std_path, index=False, compression="snappy")
        log.info("Wrote standardized transcripts: %s", std_path)
        # The cell-by-gene h5ad is a secondary artifact (not a required RUN-mode
        # output). anndata's h5ad writer can fail on a broken anndata/xarray/dask
        # install; never let that sink the run — the standardized parquet is the
        # benchmark deliverable.
        try:
            rc.build_cell_by_gene_h5ad(std, out_path=h5ad_path, log=log)
            h5ad_ok = True
        except Exception as e:
            log.warning("Skipping cell-by-gene h5ad (%s: %s). The standardized "
                        "parquet was written; this does not affect benchmarking.",
                        type(e).__name__, str(e)[:160])

    n_output = int(len(std))
    n_baysor_cells = int(std.loc[std["cell_id"] != "UNASSIGNED", "cell_id"].nunique())

    # --- validate_schema ----------------------------------------------------
    with timer.time("validate_schema"):
        rc.validate_schema(
            std, method=METHOD, out_path=std_path, in_path=args.transcripts,
            report_path=args.outdir / "schema_validation_report.json", log=log,
            extra={"baysor_version": bversion, "mode": args.mode,
                   "runtime_valid_for_benchmark": args.mode == "run",
                   "n_input_transcripts": n_input,
                   "n_output_transcripts": n_output,
                   "n_baysor_cells": n_baysor_cells})

    # --- write_outputs ------------------------------------------------------
    run_stage = next((s for s in timer.stages if s.name == "run_method"), None)
    baysor_wall = run_stage.seconds if run_stage else None
    baysor_rss = run_stage.external_max_rss_gb if run_stage else None
    with timer.time("write_outputs"):
        rc.write_provenance(
            outdir=args.outdir, method=METHOD, sample_name=args.sample_name,
            args=args, timer=timer, repo_root=_REPO_ROOT,
            inputs={"transcripts": str(args.transcripts),
                    "baysor_config": str(config_used),
                    "raw_segmentation": str(args.raw_segmentation or "")},
            outputs=[str(std_path)] + ([str(h5ad_path)] if h5ad_ok else []) + [str(seg_csv)],
            method_version=bversion, runner_kind="binary",
            extra_config={"mode": args.mode,
                          "config": str(config_used),
                          "scale": eff_scale,
                          "min_molecules_per_cell": args.min_molecules_per_cell,
                          "prior_segmentation_confidence": args.prior_segmentation_confidence,
                          "use_prior": args.use_prior,
                          "n_threads": args.n_threads},
            runtime_extra={
                "mode": args.mode,
                "runtime_valid_for_benchmark": args.mode == "run",
                "baysor_only_seconds": baysor_wall,
                "baysor_only_peak_rss_gb": baysor_rss,
                "n_threads": args.n_threads,
                "baysor_version": bversion,
                "n_input_transcripts": n_input,
                "n_output_transcripts": n_output,
                "n_baysor_cells": n_baysor_cells,
            },
            log=log,
            summary_extra_lines=(notes or [
                "Baysor is a de-novo transcript-level segmentation method; output is "
                "transcript-level and safe to pass to get_metric.py.",
                f"Baysor-only wall time: {baysor_wall:.1f}s; "
                f"input transcripts: {n_input}; output transcripts: {n_output}; "
                f"Baysor cells: {n_baysor_cells}." if baysor_wall is not None else
                f"input transcripts: {n_input}; Baysor cells: {n_baysor_cells}."]))

    log.info("DONE (mode=%s). Total wall: %.1fs | Baysor-only: %s | cells: %d",
             args.mode, timer.total_seconds,
             f"{baysor_wall:.1f}s" if baysor_wall is not None else "n/a",
             n_baysor_cells)
    return 0


if __name__ == "__main__":
    sys.exit(main())

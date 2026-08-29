#!/usr/bin/env python3
"""Slim TRACER runner: standardized transcripts + NPMI panel → refined transcripts.

This is the production entry point introduced in the 2026-05-27 refactor.
Replaces the previous kitchen-sink ``run_tracer.py`` (kept as
``scripts/run_tracer_legacy.py`` for reference). The new script does ONE thing:
load inputs, run the canonical TRACER pipeline, write the standard outputs.

What this script DOES:
    - Load a standardized transcripts parquet (produced by preprocess_xenium.py).
    - Load an NPMI panel csv(.gz) (produced by build_pmi_from_scrna.py).
    - Load + override platform config (tracer.config.load_config).
    - Run ``tracer.pipeline.run_segmented_pipeline``.
    - Compute per-cell purity/conflict via tracer.metrics.
    - Emit:
        outputs/transcripts_tracer_refined.parquet
        outputs/cell_by_gene_tracer.h5ad
        outputs/cell_scores.tsv.gz
        run_summary.md
        runtime_memory.json
        config_receipt.json

What this script DOES NOT DO:
    - Compute NPMI (use scripts/build_pmi_from_scrna.py).
    - Run ovrlpy (use scripts/run_ovrlpy.py).
    - Run RCTD (use scripts/run_rctd.R).
    - Label transfer (use scripts/label_transfer_spatial.py).
    - Compute benchmark metrics (use scripts/get_metric.py).

These were the responsibilities of the legacy script. Each is now a
separate, composable module.

EXAMPLE
=======
::

    python scripts/run_tracer.py \\
      --transcripts datasets/lung_cancer_xenium_10x/filtered_df.parquet \\
      --pmi results/reference_pmi/lung_cancer_pmi.csv.gz \\
      --pmi-threshold 0.2 \\
      --platform xenium \\
      --outdir results/tracer/lung_xenium \\
      --sample-name lung_xenium --seed 1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import resource
import socket
import subprocess
import sys
import time
import platform as _platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path bootstrap — let the script run from any cwd.
# ---------------------------------------------------------------------------
from .. import REPO_ROOT as _REPO_ROOT
from .. import common as rc
from .. import stats as stx
from . import _base

METHOD = "tracer"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--transcripts", type=Path, default=None,
                   help="Standardized transcripts parquet from preprocess_xenium.py.")
    p.add_argument("--pmi", "--npmi", dest="pmi", type=Path, default=None,
                   help="cPMI/NPMI panel csv(.gz) from build_pmi_from_scrna.py. "
                        "(--npmi is a deprecated alias for --pmi.)")
    p.add_argument("--pmi-threshold", type=float, default=None,
                   help="Override the in-pipeline PMI threshold (default: from "
                        "platform/user config).")
    p.add_argument("--platform", default="xenium",
                   help="Platform preset name (matches src/tracer/configs/platforms/<name>.toml).")
    p.add_argument("--defaults-config", type=Path, default=None,
                   help="Documentation/provenance — currently informational.")
    p.add_argument("--platform-config", type=Path, default=None,
                   help="Documentation/provenance — currently informational.")
    p.add_argument("--user-config", type=Path, default=None,
                   help="Optional user-override TOML on top of defaults+platform.")
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--sample-name", default=None)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--min-tx-per-cell-for-scores", type=int, default=5,
                   help="Min transcripts/cell for cell-level purity/conflict scoring.")
    p.add_argument("--tau", type=float, default=0.05,
                   help="NPMI threshold for purity/conflict relu (default 0.05).")
    p.add_argument("--visiumhd-matrix", type=Path, default=None,
                   help="[tracer_seq] spaceranger bin matrix dir (no-seg mode).")
    p.add_argument("--spatial-dir", type=Path, default=None,
                   help="[tracer_seq] the matching spatial/ dir.")
    p.add_argument("--bin-size-um", type=float, default=2.0,
                   help="[tracer_seq] bin pitch in microns.")
    p.add_argument("--config", default=None,
                   help="User config YAML (highest-precedence layer).")
    p.add_argument("--dataset", default=None,
                   help="Dataset name (configs/datasets/<name>.yaml) or a path.")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--max-transcripts", type=int, default=None,
                   help="Smoke-test helper: subsample the input to N transcripts.")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate inputs and config, then stop.")
    p.add_argument("--overwrite", action="store_true",
                   help="If outdir exists, overwrite contents.")
    return p


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(outdir: Path) -> logging.Logger:
    outdir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("run_tracer")
    log.setLevel(logging.INFO)
    log.propagate = False
    if log.handlers:
        return log
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    fh = logging.FileHandler(outdir / "run.log", mode="a"); fh.setFormatter(fmt)
    log.addHandler(sh); log.addHandler(fh)
    return log


def file_sha1(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Runtime accounting
# ---------------------------------------------------------------------------
@dataclass
class StageTime:
    name: str
    seconds: float
    peak_rss_gb: float


class Timer:
    def __init__(self, log: logging.Logger):
        self.log = log
        self.stages: list[StageTime] = []

    def time(self, name: str):
        return _StageCtx(name, self.log, self)


class _StageCtx:
    def __init__(self, name: str, log: logging.Logger, timer: Timer):
        self.name = name; self.log = log; self.timer = timer; self.t0 = 0.0
    def __enter__(self):
        self.t0 = time.perf_counter()
        self.log.info("[stage start] %s", self.name)
        return self
    def __exit__(self, *exc):
        secs = time.perf_counter() - self.t0
        rss = _rss_gb()
        self.timer.stages.append(StageTime(self.name, secs, rss))
        self.log.info("[stage done]  %s — %.2fs  peak_rss=%.2f GB",
                      self.name, secs, rss)


def _rss_gb() -> float:
    try:
        import psutil
        return float(psutil.Process().memory_info().rss) / (1024 ** 3)
    except Exception:
        try:
            r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return (r if sys.platform == "darwin" else r * 1024) / (1024 ** 3)
        except Exception:
            return float("nan")


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def load_transcripts(path: Path, log: logging.Logger) -> pd.DataFrame:
    log.info("Loading transcripts: %s", path)
    df = pd.read_parquet(path)
    required = {"x", "y", "feature_name", "cell_id"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(
            f"Transcripts parquet missing required columns {missing}. "
            f"Use scripts/preprocess_xenium.py to standardize first. "
            f"Present columns: {sorted(df.columns)}"
        )
    if "z" not in df.columns:
        df["z"] = np.float32(0.0)
    if "overlaps_nucleus" not in df.columns:
        df["overlaps_nucleus"] = np.uint8(0)
    if "transcript_id" not in df.columns:
        df["transcript_id"] = np.arange(len(df), dtype=np.int64)
    df["cell_id"] = df["cell_id"].astype(str)
    for c in ("x", "y", "z"):
        df[c] = df[c].astype(np.float32)
    df["feature_name"] = df["feature_name"].astype(str)
    df["overlaps_nucleus"] = df["overlaps_nucleus"].astype(np.uint8)
    n_assigned = int((df["cell_id"] != "-1").sum())
    log.info("Loaded: %d rows, %d genes, assigned=%d, unassigned=%d",
             len(df), df["feature_name"].nunique(),
             n_assigned, len(df) - n_assigned)
    return df


def load_npmi_panel(path: Path, log: logging.Logger) -> pd.DataFrame:
    log.info("Loading NPMI panel: %s", path)
    df = pd.read_csv(path)
    if not {"gene_i", "gene_j"}.issubset(df.columns):
        raise SystemExit(
            f"NPMI panel missing gene_i/gene_j; columns: {list(df.columns)}"
        )
    # Pipeline expects long-format with both directions per pair.
    if (df.duplicated(["gene_i", "gene_j"]).any()):
        log.warning("NPMI panel has duplicate pairs — keeping first occurrence.")
        df = df.drop_duplicates(["gene_i", "gene_j"], keep="first")
    # Emit symmetric form (i, j) and (j, i) so downstream lookups work.
    rev = df.copy()
    rev["gene_i"], rev["gene_j"] = df["gene_j"].values, df["gene_i"].values
    panel = pd.concat([df, rev], ignore_index=True)
    # Drop self-pairs if present (i == j).
    panel = panel.loc[panel["gene_i"] != panel["gene_j"]].reset_index(drop=True)
    log.info("NPMI panel: %d rows after symmetric expansion; PMI: %s, NPMI: %s",
             len(panel),
             "yes" if "PMI" in panel.columns else "no",
             "yes" if "NPMI" in panel.columns else "no")
    return panel


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_tracer(df: pd.DataFrame, panel: pd.DataFrame, *,
               platform_name: str, user_config: Path | None,
               pmi_threshold_override: float | None,
               log: logging.Logger):
    """Apply config + invoke the canonical SEG pipeline."""
    from tracer.config import load_config
    import tracer.pipeline as pipeline

    cfg = load_config(path=user_config, platform=platform_name)
    if pmi_threshold_override is not None:
        # PMI threshold lives at the module level (`pipeline.PMI_THR`) for
        # legacy reasons; mirror it onto the config so the receipt reflects
        # the override.
        pipeline.PMI_THR = float(pmi_threshold_override)
        log.info("PMI threshold override: pipeline.PMI_THR = %.4f",
                 pipeline.PMI_THR)
    log.info("Calling run_segmented_pipeline (df=%d rows, panel=%d pairs)",
             len(df), len(panel))
    os.environ.setdefault("TRACER_STAGE_VERBOSE", "1")
    df_out, progression = pipeline.run_segmented_pipeline(
        df=df, npmi_panel=panel, cfg=cfg,
    )
    log.info("TRACER done — %d final stages; output rows=%d",
             len(progression), len(df_out))
    return df_out, progression, cfg


# ---------------------------------------------------------------------------
# Per-cell scores + cell-by-gene
# ---------------------------------------------------------------------------
UNASSIGNED_TOKENS = frozenset({
    "UNASSIGNED", "Unassigned", "unassigned",
    "DROP", "nan", "None", "", "0", "-1", "NA",
})


#: Candidate names for TRACER's final per-transcript label, newest first.
#: run_segmented_pipeline canonicalizes to ``tracer_id``; ``stitched`` is the
#: legacy name kept so older outputs still score.
LABEL_CANDIDATES = ("tracer_id", "stitched", "cell_id")


def resolve_label_col(df, requested: str | None = None) -> str:
    """Pick TRACER's final label column, tolerating the rename."""
    if requested and requested in df.columns:
        return requested
    for c in LABEL_CANDIDATES:
        if c in df.columns:
            return c
    raise SystemExit(
        f"TRACER output has none of {LABEL_CANDIDATES}; columns: {list(df.columns)}")


def build_outputs(
    df_post: pd.DataFrame, *,
    npmi_panel: pd.DataFrame, log: logging.Logger,
    label_col: str = "stitched", min_tx: int = 5, tau: float = 0.05,
) -> tuple[pd.DataFrame, "anndata.AnnData"]:
    """Compute per-cell purity/conflict + build cell-by-gene AnnData."""
    import anndata as ad
    import scipy.sparse as sp
    from tracer.metrics import (
        build_cell_gene_matrix, build_pmi_matrix,
        compute_cell_purity_relu, compute_cell_conflict_relu,
    )

    if "_etype" in df_post.columns:
        keep_mask = df_post["_etype"].astype(str).isin({"cell", "partial", "component"})
    else:
        keep_mask = ~df_post[label_col].astype(str).isin(UNASSIGNED_TOKENS)
    work = df_post.loc[keep_mask, [label_col, "feature_name", "x", "y", "z"]].copy()
    work = work.rename(columns={label_col: "cell_id"})

    cell_ids, _genes_cell, M, col_idx = build_cell_gene_matrix(
        work, min_transcripts=min_tx, genes_npm=npmi_panel,
        cell_col="cell_id", exclude_ids=set(UNASSIGNED_TOKENS),
    )
    npmi_mat, _gix = build_pmi_matrix(npmi_panel)
    _, _, _, pur_df = compute_cell_purity_relu(
        M=M, col_idx=col_idx, npmi_mat=npmi_mat, tau=tau, cell_ids=cell_ids,
    )
    _, _, _, conf_df = compute_cell_conflict_relu(
        M=M, col_idx=col_idx, npmi_mat=npmi_mat, tau=tau, cell_ids=cell_ids,
    )
    scores = (
        pur_df.rename(columns={"cell_purity_relu": "purity_score"})
              [["cell_id", "purity_score", "signal_strength",
                "relative_purity", "relative_conflict"]]
        .merge(
            conf_df.rename(columns={"cell_conflict_relu": "conflict_score"})
                   [["cell_id", "conflict_score"]],
            on="cell_id", how="outer",
        )
    )
    log.info("Per-cell scores: %d cells with purity, %d cells total in cell-by-gene",
             int(scores["purity_score"].notna().sum()), len(cell_ids))

    # Cell-by-gene AnnData (counts layer + score obs).
    cg = (
        work.groupby(["cell_id", "feature_name"], observed=True).size()
            .rename("count").reset_index()
    )
    cell_cat = pd.Categorical(cg["cell_id"])
    gene_cat = pd.Categorical(cg["feature_name"])
    X = sp.csr_matrix(
        (cg["count"].to_numpy(dtype=np.int32),
         (cell_cat.codes, gene_cat.codes)),
        shape=(len(cell_cat.categories), len(gene_cat.categories)),
    )
    obs = scores.set_index("cell_id").reindex(cell_cat.categories.astype(str))
    var = pd.DataFrame(index=pd.Index(gene_cat.categories.astype(str),
                                       name="feature_name"))
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.layers["counts"] = X.copy()
    return scores, adata


# ---------------------------------------------------------------------------
# Output dump
# ---------------------------------------------------------------------------
def write_outputs(
    df_post: pd.DataFrame, scores: pd.DataFrame, adata, *,
    outdir: Path, sample_name: str, args, cfg, panel_path: Path,
    transcripts_path: Path, progression: list[dict[str, Any]],
    timer: Timer, log: logging.Logger,
) -> None:
    outputs = outdir / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    df_post.to_parquet(outputs / "transcripts_tracer_refined.parquet",
                       index=False, compression="snappy")
    adata.write_h5ad(outputs / "cell_by_gene_tracer.h5ad")
    scores.to_csv(outputs / "cell_scores.tsv.gz", sep="\t", index=False,
                  compression="gzip")
    log.info("Wrote outputs to %s/outputs/", outdir)

    # config_receipt.json
    from tracer.config import to_dict as cfg_to_dict
    receipt = {
        "command": " ".join(sys.argv),
        "args": {k: str(v) if isinstance(v, Path) else v
                 for k, v in vars(args).items()},
        "sample_name": sample_name,
        "platform_name": args.platform,
        "config": cfg_to_dict(cfg),
        "inputs": {
            "transcripts": str(transcripts_path),
            "transcripts_sha1": file_sha1(transcripts_path),
            "transcripts_rows": int(len(df_post)),
            "pmi": str(panel_path),
            "pmi_sha1": file_sha1(panel_path),
        },
        "host": {
            "hostname": socket.gethostname(),
            "python": sys.version.split()[0],
            "platform": _platform.platform(),
            "executable": sys.executable,
        },
        "git_commit": git_commit_hash(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(outdir / "config_receipt.json", "w") as f:
        json.dump(receipt, f, indent=2, default=str)

    # runtime_memory.json
    rm = {
        "sample_name": sample_name,
        "stages": [asdict(s) for s in timer.stages],
        "total_seconds": float(sum(s.seconds for s in timer.stages)),
        "peak_rss_gb_observed": float(max((s.peak_rss_gb for s in timer.stages), default=0.0)),
    }
    with open(outdir / "runtime_memory.json", "w") as f:
        json.dump(rm, f, indent=2)

    # run_summary.md
    md_lines = [
        f"# TRACER run summary — {sample_name}",
        "",
        f"- Date (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- Platform preset: `{args.platform}`",
        f"- Git commit: `{git_commit_hash()}`",
        f"- Seed: {args.seed}",
        f"- PMI threshold override: {args.pmi_threshold}",
        f"- Transcripts: `{transcripts_path}` ({len(df_post):,} final rows)",
        f"- cPMI panel: `{panel_path}`",
        "",
        "## Stage progression",
        "",
        "| Stage | n_cells | n_partials | n_components | n_unassigned_tx | seconds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for s in progression:
        md_lines.append(
            f"| {s.get('stage', '')} | "
            f"{s.get('n_cells', 0):,} | {s.get('n_partials', 0):,} | "
            f"{s.get('n_components', 0):,} | {s.get('n_unassigned_tx', 0):,} | "
            f"{(s.get('stage_seconds') or 0):.2f} |"
        )
    md_lines += [
        "",
        "## Top-level outputs",
        "",
        "- `outputs/transcripts_tracer_refined.parquet`",
        "- `outputs/cell_by_gene_tracer.h5ad`",
        "- `outputs/cell_scores.tsv.gz`",
        "",
        "## Runtime",
        "",
        f"- Total wall time: {rm['total_seconds']:.1f} s",
        f"- Peak RSS observed: {rm['peak_rss_gb_observed']:.2f} GB",
    ]
    with open(outdir / "run_summary.md", "w") as f:
        f.write("\n".join(md_lines) + "\n")
    log.info("Wrote run_summary.md, config_receipt.json, runtime_memory.json")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None, method: str | None = None) -> int:
    args = build_argparser().parse_args(argv)
    method = method or METHOD
    _base.resolve_config(args, method=method, section="tracer")
    if method.endswith("_seq") and not args.dry_run:
        return _run_noseg(args, method=method)

    if args.dry_run:
        print(f"[dry-run] tracer: transcripts={args.transcripts} pmi={args.pmi} "
              f"platform={args.platform} -> {args.outdir}")
        return 0
    if args.outdir is None:
        raise SystemExit("--outdir is required (flag or config).")
    args.outdir = Path(args.outdir)
    args.outdir.mkdir(parents=True, exist_ok=True)
    if any(args.outdir.iterdir()) and not args.overwrite:
        # Allow re-runs: only block when the canonical outputs already exist.
        sentinel = args.outdir / "outputs" / "transcripts_tracer_refined.parquet"
        if sentinel.exists():
            raise SystemExit(
                f"outdir {args.outdir} already contains a TRACER run "
                f"({sentinel}). Pass --overwrite to replace it."
            )

    log = setup_logging(args.outdir)
    log.info("=== run_tracer.py ===")
    log.info("Sample: %s; platform: %s; seed: %d",
             args.sample_name, args.platform, args.seed)

    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    timer = Timer(log)

    with timer.time("load_transcripts"):
        df = load_transcripts(args.transcripts, log)
    with timer.time("load_pmi"):
        panel = load_npmi_panel(args.pmi, log)
    with timer.time("run_method"):
        df_post, progression, cfg = run_tracer(
            df, panel,
            platform_name=args.platform,
            user_config=args.user_config,
            pmi_threshold_override=args.pmi_threshold,
            log=log,
        )
    with timer.time("build_outputs"):
        scores, adata = build_outputs(
            df_post, npmi_panel=panel, log=log,
            label_col=resolve_label_col(df_post),
            min_tx=args.min_tx_per_cell_for_scores, tau=args.tau,
        )
    with timer.time("write_outputs"):
        write_outputs(
            df_post, scores, adata,
            outdir=args.outdir, sample_name=args.sample_name,
            args=args, cfg=cfg,
            panel_path=args.pmi, transcripts_path=args.transcripts,
            progression=progression, timer=timer, log=log,
        )

    n_cells = int(adata.n_obs) if adata is not None else None
    stx.write_benchmark_stats(
        outdir=args.outdir, method=method,
        modality="sequencing" if method.endswith("_seq") else "imaging",
        sample_name=args.sample_name, timer=timer, dataset=args.dataset,
        transcripts=_tracer_transcript_accounting(df_post, n_input=int(len(df))),
        entities=_tracer_entity_accounting(
            df_post, entity_kind="bin" if method.endswith("_seq") else "cell"),
        qc={"platform": args.platform,
            "pmi_threshold": args.pmi_threshold,
            "tau": args.tau,
            "pmi_panel": str(args.pmi)},
        outputs=[str(args.outdir / "outputs" / "transcripts_tracer_refined.parquet")],
        notes="NPMI-guided transcript refinement.")

    log.info("DONE. Total wall: %.1fs",
             sum(s.seconds for s in timer.stages))
    return 0


# ---------------------------------------------------------------------------
# TRACER-specific accounting
#
# TRACER marks every transcript with `_etype`: "cell" (a whole cell),
# "partial" (a fragment it could not stitch) or "unknown" (unassigned), and
# uses cell_id="-1" for unassigned. Counting on cell_id would therefore report
# 100% of transcripts assigned and treat "-1" as a real entity, so `_etype` is
# the authoritative signal.
# ---------------------------------------------------------------------------
_ASSIGNED_ETYPES = ("cell", "partial")


def _tracer_transcript_accounting(df, *, n_input=None):
    if "_etype" not in df.columns:
        return stx.transcript_accounting(df, cell_col=resolve_label_col(df),
                                         n_input=n_input)
    et = df["_etype"].astype(str)
    n_total = int(len(et))
    n_assigned = int(et.isin(_ASSIGNED_ETYPES).sum())
    return {
        "n_total": n_total,
        "n_assigned": n_assigned,
        "n_unassigned": n_total - n_assigned,
        "frac_assigned": (n_assigned / n_total) if n_total else None,
        "n_input": n_input,
        "delta_vs_input": (n_total - n_input) if n_input is not None else None,
    }


def _tracer_entity_accounting(df, *, entity_kind="cell"):
    """Whole and partial entities counted separately, never pooled.

    Pooling them would make mean-transcripts-per-profile incomparable with
    methods that emit only whole cells.
    """
    out = {"entity_kind": entity_kind}
    if "_etype" not in df.columns or "cell_id" not in df.columns:
        return stx.entity_accounting(df, cell_col=resolve_label_col(df),
                                     entity_kind=entity_kind)
    et = df["_etype"].astype(str)
    cid = df["cell_id"].astype(str)
    whole, part = cid[et == "cell"], cid[et == "partial"]
    assigned = cid[et.isin(_ASSIGNED_ETYPES)]
    out["n_entities"] = int(assigned.nunique())
    out["n_whole_cells"] = int(whole.nunique())
    out["n_partial_cells"] = int(part.nunique())
    if "feature_name" in df.columns:
        out["n_genes"] = int(df["feature_name"].astype(str).nunique())
    if len(assigned):
        out["median_transcripts_per_entity"] = float(assigned.value_counts().median())
    if whole.nunique():
        out["mean_transcripts_per_whole_cell"] = float(len(whole)) / whole.nunique()
    if part.nunique():
        out["mean_transcripts_per_partial_cell"] = float(len(part)) / part.nunique()
    return out


def _run_noseg(args, *, method: str) -> int:
    """Drive TRACER's no-seg pipeline for `tracer_seq`.

    Segmented mode consumes a transcript table; no-seg mode consumes the binned
    matrix directly and reconstructs pseudocell profiles, so it is a separate
    entry point rather than a flag on the same one. Entities here are
    reconstructed profiles over bins, which is why the registry gives
    tracer_seq entity_kind="bin".
    """
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(outdir)
    matrix = _base.require_input(args, "visiumhd_matrix", "--visiumhd-matrix")
    spatial = _base.require_input(args, "spatial_dir", "--spatial-dir")
    pmi = _base.require_input(args, "pmi", "--pmi")

    tracer_src = Path(os.environ.get(
        "TRACER_HOME", Path(_REPO_ROOT).parent / "TRACER")) / "src"
    platform_cfg = tracer_src / "tracer" / "configs" / "platforms" / "noseg.toml"

    timer = Timer(log)
    cmd = [sys.executable, "-m", "tracer.noseg_pipeline",
           "--visiumhd-matrix", str(matrix),
           "--spatial-dir", str(spatial),
           "--pmi", str(pmi),
           "--platform-config", str(platform_cfg),
           "--outdir", str(outdir),
           "--sample-name", str(args.sample_name),
           "--bin-size-um", str(args.bin_size_um),
           "--seed", str(args.seed),
           "--n-jobs", str(args.threads or 8)]
    if args.overwrite:
        cmd.append("--overwrite")
    if args.max_transcripts:
        cmd += ["--max-transcripts", str(args.max_transcripts)]

    env = dict(os.environ)
    env["PYTHONPATH"] = str(tracer_src) + os.pathsep + env.get("PYTHONPATH", "")
    with timer.time("run_method"):
        code, ext_rss = rc.run_subprocess(cmd, log=log, outdir=outdir, env=env)
    # Attach the externally measured peak RSS after the stage closes.
    timer.record_external("run_method", ext_rss)
    if code != 0:
        raise SystemExit(f"tracer no-seg failed (exit {code}); see run.log.")

    assign = outdir / "outputs" / "bin_to_profile_assignment.parquet"
    entities: dict = {"entity_kind": "bin"}
    transcripts: dict = {}
    if assign.exists():
        a = pd.read_parquet(assign)
        col = next((c for c in ("reconstructed_profile_id", "profile_id",
                                "pseudocell_id", "cell_id")
                    if c in a.columns), None)
        if col:
            pid = a[col].astype(str)
            assigned = pid[~pid.isin(("-1", "UNASSIGNED", "nan", ""))]
            entities["n_entities"] = int(assigned.nunique())
            if len(assigned):
                entities["median_transcripts_per_entity"] = float(
                    assigned.value_counts().median())
            transcripts = {
                "n_total": int(len(pid)),
                "n_assigned": int(len(assigned)),
                "n_unassigned": int(len(pid) - len(assigned)),
                "frac_assigned": (float(len(assigned)) / len(pid)) if len(pid) else None,
                "n_input": int(len(pid)),
                "delta_vs_input": 0,
            }

    stx.write_benchmark_stats(
        outdir=outdir, method=method, modality="sequencing",
        sample_name=args.sample_name, timer=timer, dataset=args.dataset,
        transcripts=transcripts, entities=entities,
        qc={"mode": "noseg", "bin_size_um": float(args.bin_size_um),
            "pmi_panel": str(pmi)},
        outputs=[str(x) for x in sorted((outdir / "outputs").glob("*.parquet"))],
        notes="TRACER no-seg mode: entities are reconstructed profiles over "
              "bins, not segmented cells.")
    log.info("DONE (no-seg).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

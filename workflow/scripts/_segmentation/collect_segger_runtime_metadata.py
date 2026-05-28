#!/usr/bin/env python3
"""Collect Segger training convergence and runtime metadata for benchmarking.

Reads:
  - lightning_logs/version_0/metrics.csv   (per-epoch training stats)
  - lightning_logs/version_0/hparams.yaml  (architecture hyperparameters)
  - a /usr/bin/time -v output file         (wall-clock + peak RSS, optional)
  - nvidia-smi dmon log                    (GPU memory, optional)
  - a stage-timing TSV                     (per-stage wall times, optional)

Writes:
  - benchmark/runtime_memory.json
  - benchmark/runtime_by_stage.tsv
  - benchmark/training_convergence.tsv
  - benchmark/training_summary.md
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models-dir", required=True, type=Path,
                   help="Segger models_dir (contains lightning_logs/)")
    p.add_argument("--outdir", required=True, type=Path,
                   help="Output directory (benchmark/)")
    p.add_argument("--model-version", type=int, default=0,
                   help="Lightning log version number (default: 0)")
    p.add_argument("--time-v-log", type=Path, default=None,
                   help="/usr/bin/time -v output file for training stage")
    p.add_argument("--predict-time-v-log", type=Path, default=None,
                   help="/usr/bin/time -v output file for prediction stage")
    p.add_argument("--preprocess-time-v-log", type=Path, default=None,
                   help="/usr/bin/time -v output file for preprocess stage")
    p.add_argument("--gpu-dmon-log", type=Path, default=None,
                   help="nvidia-smi dmon -s mu output file")
    p.add_argument("--stage-timing-tsv", type=Path, default=None,
                   help="TSV with columns stage,start_epoch,end_epoch,wall_seconds")
    p.add_argument("--gpu-model", type=str, default=None,
                   help="GPU model string (e.g. NVIDIA A100 80GB)")
    p.add_argument("--slurm-job-id", type=str, default=None,
                   help="SLURM job ID for the final run")
    p.add_argument("--container-sif", type=Path, default=None,
                   help="Path to the Apptainer .sif used")
    p.add_argument("--dataset", type=str, default="TSU-20",
                   help="Dataset name tag")
    p.add_argument("--run-label", type=str, default="final_convergence",
                   help="Label for this run (e.g. final_convergence, smoke_test)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# /usr/bin/time -v parser
# ---------------------------------------------------------------------------
_TIME_V_FIELDS = {
    "wall_clock_seconds": re.compile(
        r"Elapsed \(wall clock\) time.*?:\s+(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)"
    ),
    "peak_rss_kb": re.compile(r"Maximum resident set size \(kbytes\):\s+(\d+)"),
    "user_seconds": re.compile(r"User time \(seconds\):\s+([\d.]+)"),
    "system_seconds": re.compile(r"System time \(seconds\):\s+([\d.]+)"),
    "voluntary_context_switches": re.compile(r"Voluntary context switches:\s+(\d+)"),
    "page_faults": re.compile(r"Major \(requiring I/O\) page faults:\s+(\d+)"),
}


def parse_time_v(path: Path) -> dict:
    if path is None or not path.exists():
        return {}
    text = path.read_text()
    result = {}
    for key, pattern in _TIME_V_FIELDS.items():
        m = pattern.search(text)
        if not m:
            continue
        if key == "wall_clock_seconds":
            groups = m.groups()
            # groups = (hours_or_None, minutes, seconds)
            h = float(groups[0]) if groups[0] is not None else 0.0
            mi = float(groups[1])
            s = float(groups[2])
            result[key] = h * 3600 + mi * 60 + s
        else:
            result[key] = float(m.group(1))
    if "peak_rss_kb" in result:
        result["peak_rss_gb"] = round(result["peak_rss_kb"] / 1024 / 1024, 3)
    return result


# ---------------------------------------------------------------------------
# nvidia-smi dmon parser
# ---------------------------------------------------------------------------
def parse_gpu_dmon(path: Path) -> dict:
    """Parse nvidia-smi dmon -d 5 -s mu output for peak GPU memory (MiB)."""
    if path is None or not path.exists():
        return {}
    peak_mib = 0
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        # Format: timestamp gpu_idx power_draw temp sm_util mem_util fb_used
        # With -s mu: timestamp gpu mem_util fb_used
        # Exact columns depend on flags; we look for the largest integer value
        # that looks like MiB memory (typically > 100 and < 200000).
        for p in parts:
            try:
                val = int(p)
                if 100 < val < 200_000:
                    peak_mib = max(peak_mib, val)
            except ValueError:
                pass
    return {"peak_gpu_memory_mib": peak_mib,
            "peak_gpu_memory_gb": round(peak_mib / 1024, 2)} if peak_mib else {}


# ---------------------------------------------------------------------------
# metrics.csv parser
# ---------------------------------------------------------------------------
def parse_metrics_csv(metrics_path: Path) -> tuple[pd.DataFrame, dict]:
    """Return (convergence_df, best_epoch_info)."""
    if not metrics_path.exists():
        return pd.DataFrame(), {}

    df = pd.read_csv(metrics_path)

    # Separate epoch-level rows from step-level rows.
    # Validation rows have validation_loss; training rows have train_loss.
    val_rows = df.dropna(subset=["validation_loss"]).copy() if "validation_loss" in df.columns else pd.DataFrame()
    train_rows = df.dropna(subset=["train_loss"]).copy() if "train_loss" in df.columns else pd.DataFrame()

    # Build per-epoch convergence table (join on epoch).
    if not val_rows.empty and not train_rows.empty:
        conv = val_rows[["epoch", "validation_loss", "validation_auroc", "validation_f1"]].merge(
            train_rows[["epoch", "train_loss"]], on="epoch", how="outer"
        ).sort_values("epoch").reset_index(drop=True)
    elif not val_rows.empty:
        conv = val_rows[["epoch", "validation_loss",
                          "validation_auroc", "validation_f1"]].sort_values("epoch").reset_index(drop=True)
    else:
        conv = df.copy()

    # Best epoch = epoch with minimum validation_loss.
    best_info: dict = {}
    if "validation_loss" in conv.columns and not conv["validation_loss"].isna().all():
        best_idx = int(conv["validation_loss"].idxmin())
        best_row = conv.loc[best_idx]
        best_info = {
            "best_epoch": int(best_row["epoch"]),
            "best_val_loss": float(best_row["validation_loss"]),
            "best_val_auroc": float(best_row["validation_auroc"])
                if "validation_auroc" in best_row and pd.notna(best_row["validation_auroc"]) else None,
            "best_val_f1": float(best_row["validation_f1"])
                if "validation_f1" in best_row and pd.notna(best_row["validation_f1"]) else None,
            "total_epochs_trained": int(conv["epoch"].max()) + 1,
        }

    return conv, best_info


# ---------------------------------------------------------------------------
# hparams.yaml reader
# ---------------------------------------------------------------------------
def read_hparams(hparams_path: Path) -> dict:
    if not hparams_path.exists():
        return {}
    try:
        import yaml  # type: ignore
        with open(hparams_path) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        # Fallback: naive key: value parsing
        result = {}
        for line in hparams_path.read_text().splitlines():
            line = line.strip()
            if ":" in line and not line.startswith("-"):
                k, _, v = line.partition(":")
                result[k.strip()] = v.strip()
        return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    version_dir = args.models_dir / "lightning_logs" / f"version_{args.model_version}"
    metrics_path = version_dir / "metrics.csv"
    hparams_path = version_dir / "hparams.yaml"

    # ── Parse convergence ─────────────────────────────────────────────────────
    conv_df, best_info = parse_metrics_csv(metrics_path)
    hparams = read_hparams(hparams_path)

    # ── Parse timing files ────────────────────────────────────────────────────
    train_time = parse_time_v(args.time_v_log)
    predict_time = parse_time_v(args.predict_time_v_log)
    preprocess_time = parse_time_v(args.preprocess_time_v_log)
    gpu_stats = parse_gpu_dmon(args.gpu_dmon_log)

    total_wall = sum(
        t.get("wall_clock_seconds", 0.0)
        for t in (preprocess_time, train_time, predict_time)
    )
    peak_rss_gb = max(
        (t.get("peak_rss_gb", 0.0) for t in (preprocess_time, train_time, predict_time)),
        default=0.0,
    )

    # ── Find best checkpoint path ─────────────────────────────────────────────
    ckpt_dir = version_dir / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("*.ckpt")) if ckpt_dir.exists() else []
    best_ckpt = str(ckpts[-1]) if ckpts else None

    # ── Write runtime_memory.json ─────────────────────────────────────────────
    runtime_json = {
        "method": "Segger",
        "dataset": args.dataset,
        "run_label": args.run_label,
        "slurm_job_id": args.slurm_job_id,
        "container_sif": str(args.container_sif) if args.container_sif else None,
        "gpu_model": args.gpu_model,
        "runtime_seconds": total_wall if total_wall > 0 else None,
        "peak_memory_gb": peak_rss_gb if peak_rss_gb > 0 else None,
        "peak_gpu_memory_gb": gpu_stats.get("peak_gpu_memory_gb"),
        "peak_gpu_memory_mib": gpu_stats.get("peak_gpu_memory_mib"),
        "stages": {
            "preprocess": {
                "wall_clock_seconds": preprocess_time.get("wall_clock_seconds"),
                "peak_rss_gb": preprocess_time.get("peak_rss_gb"),
            },
            "train": {
                "wall_clock_seconds": train_time.get("wall_clock_seconds"),
                "peak_rss_gb": train_time.get("peak_rss_gb"),
            },
            "predict": {
                "wall_clock_seconds": predict_time.get("wall_clock_seconds"),
                "peak_rss_gb": predict_time.get("peak_rss_gb"),
            },
        },
        "best_checkpoint": best_ckpt,
        **best_info,
        "hparams": hparams,
        "generated_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    runtime_out = args.outdir / "runtime_memory.json"
    with open(runtime_out, "w") as f:
        json.dump(runtime_json, f, indent=2)
    print(f"[metadata] Wrote: {runtime_out}")

    # ── Write runtime_by_stage.tsv ────────────────────────────────────────────
    stage_rows = []
    for stage, tdict in [("preprocess", preprocess_time),
                          ("train", train_time),
                          ("predict", predict_time)]:
        stage_rows.append({
            "stage": stage,
            "wall_clock_seconds": tdict.get("wall_clock_seconds"),
            "peak_rss_gb": tdict.get("peak_rss_gb"),
            "user_seconds": tdict.get("user_seconds"),
            "system_seconds": tdict.get("system_seconds"),
        })
    stage_df = pd.DataFrame(stage_rows)
    stage_out = args.outdir / "runtime_by_stage.tsv"
    stage_df.to_csv(stage_out, sep="\t", index=False)
    print(f"[metadata] Wrote: {stage_out}")

    # ── Write training_convergence.tsv ────────────────────────────────────────
    conv_out = args.outdir / "training_convergence.tsv"
    if not conv_df.empty:
        conv_df.to_csv(conv_out, sep="\t", index=False)
        print(f"[metadata] Wrote: {conv_out}  ({len(conv_df)} epochs)")
    else:
        pd.DataFrame(columns=["epoch", "train_loss", "validation_loss",
                               "validation_auroc", "validation_f1"]).to_csv(
            conv_out, sep="\t", index=False)
        print(f"[metadata] Wrote empty convergence table: {conv_out}")

    # ── Write training_summary.md ─────────────────────────────────────────────
    def _fmt(v, fmt=".4f"):
        return f"{v:{fmt}}" if v is not None and v == v else "N/A"

    n_epochs = best_info.get("total_epochs_trained", "N/A")
    best_ep = best_info.get("best_epoch", "N/A")
    best_loss = _fmt(best_info.get("best_val_loss"))
    best_auroc = _fmt(best_info.get("best_val_auroc"))
    best_f1 = _fmt(best_info.get("best_val_f1"))
    run_tag = f"[{args.run_label}]" if args.run_label else ""

    train_wall = _fmt(train_time.get("wall_clock_seconds"), ".1f") if train_time else "N/A"
    predict_wall = _fmt(predict_time.get("wall_clock_seconds"), ".1f") if predict_time else "N/A"
    preproc_wall = _fmt(preprocess_time.get("wall_clock_seconds"), ".1f") if preprocess_time else "N/A"
    peak_ram = _fmt(peak_rss_gb, ".2f") if peak_rss_gb > 0 else "N/A"
    peak_gpu = _fmt(gpu_stats.get("peak_gpu_memory_gb"), ".1f") if gpu_stats else "N/A"
    gpu_mib = str(gpu_stats.get("peak_gpu_memory_mib", "N/A"))

    md_lines = [
        f"# Segger Training Summary — TSU-20 {run_tag}",
        "",
        f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}  ",
        f"**Dataset:** {args.dataset}  ",
        f"**Run label:** `{args.run_label}`  ",
    ]
    if args.slurm_job_id:
        md_lines.append(f"**SLURM job ID:** `{args.slurm_job_id}`  ")
    if args.gpu_model:
        md_lines.append(f"**GPU:** {args.gpu_model}  ")
    if args.container_sif:
        md_lines.append(f"**Container:** `{args.container_sif.name}`  ")

    md_lines += [
        "",
        "## Convergence",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total epochs trained | {n_epochs} |",
        f"| Best epoch (min val_loss) | {best_ep} |",
        f"| Best validation loss | {best_loss} |",
        f"| Best validation AUROC | {best_auroc} |",
        f"| Best validation F1 | {best_f1} |",
        f"| Best checkpoint | `{best_ckpt or 'N/A'}` |",
        "",
        "## Runtime",
        "",
        f"| Stage | Wall time (s) |",
        f"|-------|---------------|",
        f"| Preprocess | {preproc_wall} |",
        f"| Train | {train_wall} |",
        f"| Predict | {predict_wall} |",
        "",
        "## Memory",
        "",
        f"| Resource | Peak |",
        f"|----------|------|",
        f"| CPU RAM (GB) | {peak_ram} |",
        f"| GPU memory (GB) | {peak_gpu} |",
        f"| GPU memory (MiB) | {gpu_mib} |",
        "",
        "## Model Architecture (hparams)",
        "",
        "```yaml",
    ]
    for k, v in sorted(hparams.items()):
        md_lines.append(f"{k}: {v}")
    md_lines += [
        "```",
        "",
        "## Final Checkpoint",
        "",
        f"```",
        best_ckpt or "N/A",
        f"```",
    ]
    if not conv_df.empty:
        md_lines += [
            "",
            "## Training Convergence Table",
            "",
            conv_df.to_markdown(index=False) if hasattr(conv_df, "to_markdown") else conv_df.to_string(),
        ]

    summary_out = args.outdir / "training_summary.md"
    summary_out.write_text("\n".join(md_lines) + "\n")
    print(f"[metadata] Wrote: {summary_out}")

    print(f"[metadata] Done. Best epoch={best_ep}, val_loss={best_loss}, "
          f"total_epochs={n_epochs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

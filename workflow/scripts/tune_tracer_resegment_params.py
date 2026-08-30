#!/usr/bin/env python3
"""Parameter sweep + scoring for TRACER **resegment** mode on the four Fig-3 ROIs.

This is an *additional* "best TRACER resegmentation" analysis — it does NOT
replace the production resegment default unless a config clearly improves marker
specificity and Kendall correlation without pathological fragmentation / collapse.

Pipeline per (dataset, combo)
-----------------------------
1. Materialize a per-combo override TOML (deep-merged onto defaults+platform by
   ``tracer.config.load_config``).
2. Run ``scripts/run_tracer.py --mode resegment`` (TRACER repo) producing the
   standard outputs (refined parquet, cell-by-gene h5ad, runtime_memory.json,
   config_receipt.json).
3. Score the run with the **canonical** benchmark code path
   (``_roi_summary_heatmaps.build_matrices_and_metrics.compute_block_b`` +
   ``get_metric``): marker specificity log2FC, Kendall tau vs matched scRNA,
   NPMI relative purity / conflict — computed on the *combined* TRACER matrix
   (``_etype in {cell,partial,component}``), plus refined-only / reconstructed-
   only as secondary. Records cells, median tx/profile, unassigned fraction,
   runtime, peak RSS.
4. Write ``metrics.json`` into the run dir (resumable: existing runs are skipped
   unless --force).

Then ``--summarize`` aggregates every metrics.json into:
  results/tracer_resegment_tuning/tuning_summary.tsv
  results/tracer_resegment_tuning/best_config_per_dataset.tsv
  results/tracer_resegment_tuning/best_config_per_platform.toml
  results/tracer_resegment_tuning/tuning_summary.md
  results/tracer_resegment_tuning/plots/*.png

Ranking
-------
primary score = mean rank of (marker_log2fc, kendall_tau) within a dataset
(rank 1 = best). Penalties are then ADDED (rank-units) for:
  * extreme combined cell-count inflation / collapse vs the per-dataset median
  * low refined median transcripts / profile (fragmentation into sparse cells)
  * high unassigned transcript fraction
  * excessive runtime / peak memory vs the per-dataset median
  * suspicious near-perfect purity carried by sparse profiles
Final ranking is by ``penalized_score`` ascending. Every component is kept in
the summary TSV so the penalty is transparent and auditable.

Run with the ``spatial`` conda env (scanpy + tracer importable):
  ~/anaconda3/envs/spatial/bin/python workflow/scripts/tune_tracer_resegment_params.py run
  ~/anaconda3/envs/spatial/bin/python workflow/scripts/tune_tracer_resegment_params.py summarize
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "workflow" / "scripts"
HEATMAPS = SCRIPTS / "_roi_summary_heatmaps"
TRACER_REPO = Path("/Users/lyuan13/Desktop/TRACER")
RUN_TRACER = TRACER_REPO / "scripts" / "run_tracer.py"

OUT = REPO / "results" / "tracer_resegment_tuning"
RUNS = OUT / "runs"
CONFIGS = OUT / "configs"
PLOTS = OUT / "plots"

sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(HEATMAPS))

log = logging.getLogger("tune_reseg")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s :: %(message)s",
                    datefmt="%H:%M:%S")

ASSIGNED_ETYPES = {"cell", "partial", "component"}

# ---------------------------------------------------------------------------
# Per-dataset RUN inputs (transcripts + NPMI fed to TRACER).
# These mirror the EXACT inputs the production resegment default used
# (read from each dataset's tracer_resegment/config_receipt.json on 2026-06-09)
# so a tuned run is directly comparable to the current default.
# Scoring references (scRNA + coherence NPMI) come from the heatmap registry.
# ---------------------------------------------------------------------------
DATASET_RUN_INPUTS = {
    "atera_cervical": dict(
        transcripts="dataset/atera_cervical/tracer_seg/input_transcripts.parquet",
        npmi="dataset/atera_cervical/tracer_seg/npmi_panel.csv.gz", platform="xenium"),
    "xenium5k_cervical": dict(
        transcripts="dataset/xenium5k_cervical/tracer_seg/input_transcripts.parquet",
        npmi="dataset/xenium5k_cervical/tracer_seg/npmi_panel.csv.gz", platform="xenium"),
    "cosmx_nsclc": dict(
        transcripts="dataset/cosmx_nsclc/tracer_seg/input_transcripts.parquet",
        npmi="dataset/cosmx_nsclc/tracer_seg/npmi_panel.csv.gz", platform="xenium"),
    "merfish_mouse_ileum": dict(
        transcripts="dataset/merfish_mouse_ileum/tracer_seg/input_transcripts_um.parquet",
        npmi="dataset/merfish_mouse_ileum/tracer_seg/npmi_panel_selfref.csv.gz",
        platform="xenium"),
}
DATASET_ORDER = ["atera_cervical", "xenium5k_cervical", "cosmx_nsclc", "merfish_mouse_ileum"]
PLATFORM_OF = {"atera_cervical": "Xenium", "xenium5k_cervical": "Xenium5K",
               "cosmx_nsclc": "CosMx", "merfish_mouse_ileum": "MERFISH"}

# ---------------------------------------------------------------------------
# Parameter combinations.
#
# Each combo = (name, override-dict, optional pmi-threshold CLI override, note).
# The override-dict is a {section: {key: value}} mapping serialized to a user
# TOML and deep-merged onto defaults+platform. Knobs were chosen for their
# leverage on over-MERGING (cell collapse → kendall loss in the current
# default) and over-FRAGMENTATION (sparse profiles).
#
# Disabling sentinels:
#   c_union_bypass = 1.0           (config caps at [0,1]; only PERFECT-coherence
#                                   unions bypass ⇒ effectively off)
#   mahalanobis_d_rescue = 0.01    (config requires >0; D ≤ 0.01 ⇒ rescue-merge
#                                   effectively never fires)
#   maha_remerge_d = 0.01          (same, Phase-1 remerge)
# ---------------------------------------------------------------------------
COMBOS = [
    ("default", {}, None,
     "Current production resegment default (platform=xenium, no overrides)."),

    # --- anti-over-merge: preserve cells the default fuses (kendall recovery) ---
    ("stitch_dC0.10", {"stitch": {"deltaC_min": 0.10}}, None,
     "Raise stitch ΔC accept gate 0.03→0.10 (fewer cross-entity merges)."),
    ("stitch_nobypass", {"stitch": {"c_union_bypass": 1.0}}, None,
     "Effectively disable C(union) bypass for ΔC-failing pairs (ΔC-only gate)."),
    ("stitch_dist3", {"stitch": {"dist_threshold_um": 3.0}}, None,
     "Tighten stitch spatial gate 5→3 µm (only near-touching fragments merge)."),
    ("stitch_maha_off",
     {"stitch": {"mahalanobis_d_rescue": 0.01}, "phase1": {"maha_remerge_d": 0.01}}, None,
     "Disable Mahalanobis rescue-merges (Phase-1 remerge + stitch)."),
    ("antimerge_combo",
     {"stitch": {"deltaC_min": 0.08, "c_union_bypass": 1.0,
                 "dist_threshold_um": 3.0, "mahalanobis_d_rescue": 0.01},
      "phase1": {"maha_remerge_d": 0.01}}, None,
     "Strong anti-merge preset (high ΔC + no bypass + tight dist + no maha)."),

    # --- PMI / negative-veto specificity (marker specificity) ---
    ("pmi0.3",
     {"phase1": {"pmi_threshold": 0.3}, "group": {"cascade_pmi_threshold": 0.3}},
     0.3, "Stricter compatible-pair PMI 0.2→0.3 (phase1+cascade+stitch global)."),
    ("pmi0.1",
     {"phase1": {"pmi_threshold": 0.1}, "group": {"cascade_pmi_threshold": 0.1}},
     0.1, "Looser compatible-pair PMI 0.2→0.1 (more admissive)."),
    ("neg_strict",
     {"phase1": {"neg_npmi_threshold": -0.1}, "rescue": {"neg_threshold": -0.1}}, None,
     "Stronger negative veto -0.2→-0.1 (reject weak anti-correlated admits)."),

    # --- demote / fragmentation floor ---
    ("demote3", {"demote": {"min_size": 3}}, None,
     "Lower demote floor 5→3 (keep more small entities; risk fragmentation)."),
    ("demote10", {"demote": {"min_size": 10}}, None,
     "Raise demote floor 5→10 (drop sparse fragments; risk cell loss)."),

    # --- rescue admission ---
    ("rescue_strict",
     {"rescue": {"mean_admit_threshold": 0.7, "witness_min_admit": 4}}, None,
     "Stricter rescue admission (mean 0.5→0.7, witness 3→4)."),
    ("rescue_passes5",
     {"rescue": {"post_group_passes": 5}, "final_rescue": {"max_passes": 5}}, None,
     "noseg-like recall: 5 post-group + 5 final-rescue passes."),

    # --- combined presets ---
    ("specificity_preset",
     {"phase1": {"pmi_threshold": 0.3, "neg_npmi_threshold": -0.1},
      "group": {"cascade_pmi_threshold": 0.3},
      "rescue": {"mean_admit_threshold": 0.7, "neg_threshold": -0.1}},
     0.3, "Specificity preset: strict PMI + strict neg veto + strict rescue."),
    ("balanced_preset",
     {"stitch": {"deltaC_min": 0.08, "c_union_bypass": 1.0,
                 "dist_threshold_um": 3.0, "mahalanobis_d_rescue": 0.01},
      "phase1": {"maha_remerge_d": 0.01},
      "demote": {"min_size": 8}},
     None, "Balanced preset: anti-merge + demote 8 (preserve cells, drop dust)."),
]
COMBO_NAMES = [c[0] for c in COMBOS]


# ---------------------------------------------------------------------------
# Minimal TOML serializer for {section: {key: scalar}} override dicts.
# ---------------------------------------------------------------------------
def _toml_scalar(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    return f'"{v}"'


def write_override_toml(override: dict, path: Path) -> None:
    lines = ["# Auto-generated TRACER resegment tuning override.",
             "# Deep-merged onto defaults.toml + platform preset.", ""]
    for section, kv in override.items():
        lines.append(f"[{section}]")
        for k, v in kv.items():
            lines.append(f"{k} = {_toml_scalar(v)}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Run one TRACER resegment job.
# ---------------------------------------------------------------------------
def run_one(dataset: str, combo_name: str, override: dict,
            pmi: float | None, force: bool) -> Path:
    rin = DATASET_RUN_INPUTS[dataset]
    outdir = RUNS / dataset / combo_name
    sentinel = outdir / "outputs" / "transcripts_tracer_refined.parquet"
    cfg_toml = CONFIGS / f"{combo_name}.toml"
    write_override_toml(override, cfg_toml)
    if sentinel.exists() and not force:
        log.info("[%s/%s] run cached -> skip", dataset, combo_name)
        return outdir
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(RUN_TRACER),
           "--transcripts", str(REPO / rin["transcripts"]),
           "--pmi", str(REPO / rin["npmi"]),
           "--platform", rin["platform"],
           "--mode", "resegment",
           "--outdir", str(outdir),
           "--sample-name", f"{dataset}_{combo_name}",
           "--seed", "1", "--overwrite"]
    if override:  # only attach a user-config when there's something to override
        cmd += ["--user-config", str(cfg_toml)]
    if pmi is not None:
        cmd += ["--pmi-threshold", str(pmi)]
    log.info("[%s/%s] RUN: %s", dataset, combo_name, " ".join(cmd))
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=str(TRACER_REPO),
                       capture_output=True, text=True)
    dt = time.perf_counter() - t0
    (outdir / "tune_run.log").write_text(r.stdout + "\n===STDERR===\n" + r.stderr)
    if r.returncode != 0:
        log.error("[%s/%s] run_tracer FAILED (rc=%d, %.0fs) — see tune_run.log\n%s",
                  dataset, combo_name, r.returncode, dt, r.stderr[-2000:])
        raise RuntimeError(f"run_tracer failed for {dataset}/{combo_name}")
    log.info("[%s/%s] run_tracer done in %.0fs", dataset, combo_name, dt)
    return outdir


# ---------------------------------------------------------------------------
# Scoring (canonical benchmark code path).
# ---------------------------------------------------------------------------
_REF_CACHE: dict = {}
_NPMI_CACHE: dict = {}


def _load_scoring_refs(dataset: str):
    """Load scRNA reference + coherence NPMI panel for one dataset (cached)."""
    import get_metric as gm
    import registry as R
    if dataset not in _REF_CACHE:
        cfg = R.DATASETS[dataset]
        _REF_CACHE[dataset] = gm.load_reference(
            Path(cfg["reference_h5ad"]), cfg["reference_celltype_col"], log)
        npmi_panel = None
        if Path(cfg["npmi"]).exists():
            nd = pd.read_csv(cfg["npmi"])
            rev = nd.copy()
            rev["gene_i"], rev["gene_j"] = nd["gene_j"].values, nd["gene_i"].values
            npmi_panel = pd.concat([nd, rev], ignore_index=True)
            npmi_panel = npmi_panel.loc[npmi_panel["gene_i"] != npmi_panel["gene_j"]]
        _NPMI_CACHE[dataset] = npmi_panel
    return _REF_CACHE[dataset], _NPMI_CACHE[dataset]


def _adata_from_parquet(df: pd.DataFrame, etype: str | None):
    """Build cells×genes AnnData from a refined transcript parquet.

    etype=None -> combined (cell+partial+component); else that single _etype.
    """
    import get_metric as gm
    df = df.copy()
    df["feature_name"] = df["feature_name"].astype(str)
    if etype is None:
        sub = df.loc[df["_etype"].astype(str).isin(ASSIGNED_ETYPES)]
    else:
        sub = df.loc[df["_etype"].astype(str) == etype]
    if len(sub) == 0:
        return None
    return gm.build_cellxgene(sub, "stitched", keep_ids=None, log=log)


def _matrix_stats(adata):
    if adata is None or adata.n_obs == 0:
        return dict(n_cells=0, median_tx=float("nan"))
    counts = np.asarray(adata.X.sum(axis=1)).ravel()
    return dict(n_cells=int(adata.n_obs), median_tx=float(np.median(counts)))


def score_one(dataset: str, combo_name: str, outdir: Path, force: bool) -> dict:
    import build_matrices_and_metrics as BM
    metrics_path = outdir / "metrics.json"
    if metrics_path.exists() and not force:
        log.info("[%s/%s] metrics cached -> skip", dataset, combo_name)
        return json.loads(metrics_path.read_text())

    ref, npmi_panel = _load_scoring_refs(dataset)
    pq = outdir / "outputs" / "transcripts_tracer_refined.parquet"
    df = pd.read_parquet(pq)

    et = df["_etype"].astype(str)
    n_total = len(df)
    n_assigned = int(et.isin(ASSIGNED_ETYPES).sum())
    unassigned_frac = float(1.0 - n_assigned / max(n_total, 1))

    rec = dict(dataset=dataset, platform=PLATFORM_OF[dataset], combo=combo_name,
               n_transcripts_total=n_total, unassigned_frac=unassigned_frac)

    # Runtime / memory from run_tracer's own accounting.
    try:
        rm = json.loads((outdir / "runtime_memory.json").read_text())
        rec["runtime_seconds"] = float(rm.get("total_seconds", float("nan")))
        rec["peak_memory_gb"] = float(rm.get("peak_rss_gb_observed", float("nan")))
    except Exception:
        rec["runtime_seconds"] = float("nan")
        rec["peak_memory_gb"] = float("nan")

    # Primary scoring on the combined matrix; secondary on refined / reconstructed.
    for tag, etype in (("combined", None), ("refined", "cell"),
                       ("reconstructed", "partial")):
        adata = _adata_from_parquet(df, etype)
        st = _matrix_stats(adata)
        rec[f"{tag}_n_cells"] = st["n_cells"]
        rec[f"{tag}_median_tx"] = st["median_tx"]
        if adata is None or adata.n_obs < 5:
            for m in ("marker_log2fc", "kendall_tau", "pearson_r",
                      "relative_purity", "relative_conflict"):
                rec[f"{tag}_{m}"] = float("nan")
            continue
        sdir = outdir / "score" / tag
        b = BM.compute_block_b(dataset, f"{combo_name}_{tag}", adata, ref,
                               npmi_panel, sdir)
        for m in ("marker_log2fc", "kendall_tau", "pearson_r",
                  "relative_purity", "relative_conflict"):
            rec[f"{tag}_{m}"] = float(b.get(m, float("nan")))

    metrics_path.write_text(json.dumps(rec, indent=2))
    log.info("[%s/%s] scored: combined cells=%d tx=%.0f log2fc=%.3f kendall=%.3f "
             "purity=%.3f unassigned=%.3f",
             dataset, combo_name, rec["combined_n_cells"], rec["combined_median_tx"],
             rec["combined_marker_log2fc"], rec["combined_kendall_tau"],
             rec["combined_relative_purity"], unassigned_frac)
    return rec


# ---------------------------------------------------------------------------
# Refine-in-place baseline (mode=refine_in_place) — scored identically.
# ---------------------------------------------------------------------------
def run_refine_in_place(dataset: str, force: bool) -> Path:
    rin = DATASET_RUN_INPUTS[dataset]
    outdir = RUNS / dataset / "__refine_in_place"
    sentinel = outdir / "outputs" / "transcripts_tracer_refined.parquet"
    if sentinel.exists() and not force:
        log.info("[%s/refine_in_place] run cached -> skip", dataset)
        return outdir
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(RUN_TRACER),
           "--transcripts", str(REPO / rin["transcripts"]),
           "--npmi", str(REPO / rin["npmi"]),
           "--platform", rin["platform"],
           "--mode", "refine_in_place",
           "--prune-metric", "npmi", "--prune-threshold", "0.2",
           "--outdir", str(outdir),
           "--sample-name", f"{dataset}_refine_in_place",
           "--seed", "1", "--overwrite"]
    log.info("[%s/refine_in_place] RUN: %s", dataset, " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(TRACER_REPO), capture_output=True, text=True)
    (outdir / "tune_run.log").write_text(r.stdout + "\n===STDERR===\n" + r.stderr)
    if r.returncode != 0:
        log.error("[%s/refine_in_place] FAILED\n%s", dataset, r.stderr[-2000:])
        raise RuntimeError(f"refine_in_place failed for {dataset}")
    return outdir


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------
def cmd_run(args):
    datasets = args.datasets or DATASET_ORDER
    combos = [c for c in COMBOS if (not args.combos or c[0] in args.combos)]
    for dataset in datasets:
        for (name, override, pmi, _note) in combos:
            try:
                outdir = run_one(dataset, name, override, pmi, args.force_run)
                score_one(dataset, name, outdir, args.force_score)
            except Exception as e:
                log.error("[%s/%s] FAILED: %s", dataset, name, e)
                if args.strict:
                    raise
    log.info("Sweep done for datasets=%s combos=%d", datasets, len(combos))


# ---------------------------------------------------------------------------
# Summarize / rank
# ---------------------------------------------------------------------------
def _collect() -> pd.DataFrame:
    rows = []
    for dataset in DATASET_ORDER:
        for name in COMBO_NAMES:
            mp = RUNS / dataset / name / "metrics.json"
            if mp.exists():
                rows.append(json.loads(mp.read_text()))
    return pd.DataFrame(rows)


def _rank_and_penalize(df: pd.DataFrame) -> pd.DataFrame:
    """Per-dataset: mean rank of (marker_log2fc, kendall_tau) + penalties."""
    out = []
    for dataset, g in df.groupby("dataset"):
        g = g.copy()
        # primary: rank 1 = best (higher is better for both)
        g["rank_marker"] = g["combined_marker_log2fc"].rank(ascending=False, method="min")
        g["rank_kendall"] = g["combined_kendall_tau"].rank(ascending=False, method="min")
        g["mean_rank"] = g[["rank_marker", "rank_kendall"]].mean(axis=1)

        med_cells = g["combined_n_cells"].median()
        med_rt = g["runtime_seconds"].median()
        med_mem = g["peak_memory_gb"].median()

        def pen(row):
            p = 0.0
            flags = []
            # extreme cell-count inflation / collapse vs per-dataset median
            if med_cells and med_cells > 0:
                ratio = row["combined_n_cells"] / med_cells
                if ratio < 0.4 or ratio > 2.5:
                    p += 2.0
                    flags.append(f"cell_ratio={ratio:.2f}")
                elif ratio < 0.6 or ratio > 1.7:
                    p += 1.0
                    flags.append(f"cell_ratio={ratio:.2f}")
            # fragmentation: sparse refined profiles
            rmt = row["refined_median_tx"]
            if rmt == rmt and rmt < 20:
                p += 1.5
                flags.append(f"refined_tx={rmt:.0f}")
            # high unassigned fraction
            uf = row["unassigned_frac"]
            if uf == uf and uf > 0.25:
                p += 1.5
                flags.append(f"unassigned={uf:.2f}")
            elif uf == uf and uf > 0.15:
                p += 0.5
                flags.append(f"unassigned={uf:.2f}")
            # suspicious near-perfect purity carried by sparse profiles
            rp = row["combined_relative_purity"]
            if (rp == rp and rp > 0.999 and rmt == rmt and rmt < 25):
                p += 1.0
                flags.append("suspicious_purity")
            # excessive runtime / memory vs median
            if med_rt and med_rt > 0 and row["runtime_seconds"] / med_rt > 2.0:
                p += 0.5
                flags.append("slow")
            if med_mem and med_mem > 0 and row["peak_memory_gb"] / med_mem > 2.0:
                p += 0.5
                flags.append("high_mem")
            return pd.Series({"penalty": p, "penalty_flags": ";".join(flags)})

        pens = g.apply(pen, axis=1)
        g = pd.concat([g, pens], axis=1)
        g["penalized_score"] = g["mean_rank"] + g["penalty"]
        g["final_rank"] = g["penalized_score"].rank(method="min")
        out.append(g)
    return pd.concat(out, ignore_index=True)


# columns surfaced (in order) in the summary TSV
SUMMARY_COLS = [
    "dataset", "platform", "combo", "final_rank", "penalized_score", "mean_rank",
    "rank_marker", "rank_kendall", "penalty", "penalty_flags",
    "combined_marker_log2fc", "combined_kendall_tau", "combined_pearson_r",
    "combined_relative_purity", "combined_relative_conflict",
    "combined_n_cells", "combined_median_tx",
    "refined_marker_log2fc", "refined_kendall_tau", "refined_n_cells", "refined_median_tx",
    "reconstructed_marker_log2fc", "reconstructed_kendall_tau",
    "reconstructed_n_cells", "reconstructed_median_tx",
    "unassigned_frac", "runtime_seconds", "peak_memory_gb",
]


def _combo_note(name: str) -> str:
    for n, _o, _p, note in COMBOS:
        if n == name:
            return note
    return ""


def _override_for(name: str) -> dict:
    for n, o, _p, _note in COMBOS:
        if n == name:
            return o
    return {}


def cmd_summarize(args):
    df = _collect()
    if df.empty:
        raise SystemExit("No metrics.json found under runs/. Run the sweep first.")
    ranked = _rank_and_penalize(df)
    OUT.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)

    cols = [c for c in SUMMARY_COLS if c in ranked.columns]
    summary = ranked[cols].sort_values(["dataset", "final_rank"]).reset_index(drop=True)
    summary.to_csv(OUT / "tuning_summary.tsv", sep="\t", index=False)
    log.info("WROTE %s (%d rows)", OUT / "tuning_summary.tsv", len(summary))

    # best per dataset
    best = (ranked.sort_values(["dataset", "final_rank"])
                  .groupby("dataset", as_index=False).first())
    best_cols = ["dataset", "platform", "combo", "penalized_score", "mean_rank",
                 "penalty", "penalty_flags", "combined_marker_log2fc",
                 "combined_kendall_tau", "combined_relative_purity",
                 "combined_relative_conflict", "combined_n_cells",
                 "combined_median_tx", "unassigned_frac",
                 "runtime_seconds", "peak_memory_gb"]
    best_cols = [c for c in best_cols if c in best.columns]
    best = best[best_cols]
    best["override"] = best["combo"].map(lambda c: json.dumps(_override_for(c)))
    best["note"] = best["combo"].map(_combo_note)
    best.to_csv(OUT / "best_config_per_dataset.tsv", sep="\t", index=False)
    log.info("WROTE %s", OUT / "best_config_per_dataset.tsv")

    _write_best_platform_toml(best)
    _write_markdown(summary, best, df)
    _make_plots(ranked)
    log.info("Summarize complete -> %s", OUT)


def _write_best_platform_toml(best: pd.DataFrame):
    lines = ["# Best TRACER resegment override per platform (tuning result).",
             "# Each block is the deep-merge override on defaults.toml + platform",
             "# preset. 'default' means no override won — keep the production default.",
             "# Treat as an ADDITIONAL 'best resegmentation', not a drop-in default;",
             "# see tuning_summary.md for overfitting-risk caveats.", ""]
    for _, row in best.iterrows():
        ov = _override_for(row["combo"])
        lines.append(f"# === {row['platform']} ({row['dataset']}): combo={row['combo']} ===")
        lines.append(f"#   marker_log2fc={row['combined_marker_log2fc']:.3f} "
                     f"kendall={row['combined_kendall_tau']:.3f} "
                     f"cells={int(row['combined_n_cells'])} "
                     f"penalty={row['penalty']:.1f} [{row['penalty_flags']}]")
        slug = f"{row['platform']}.{row['combo']}"
        if not ov:
            lines.append(f"[\"{slug}\"]  # no override (production default won)")
            lines.append("")
            continue
        for section, kv in ov.items():
            lines.append(f"[\"{slug}\".{section}]")
            for k, v in kv.items():
                lines.append(f"{k} = {_toml_scalar(v)}")
        lines.append("")
    (OUT / "best_config_per_platform.toml").write_text("\n".join(lines))
    log.info("WROTE %s", OUT / "best_config_per_platform.toml")


def _write_markdown(summary, best, raw):
    md = ["# TRACER resegment parameter tuning — summary", "",
          "Additional **best TRACER resegmentation** analysis across four Fig-3 ROIs.",
          "Primary objective: marker specificity (log2FC) **and** Kendall τ vs matched",
          "scRNA, scored on the *combined* TRACER matrix via the canonical benchmark",
          "code path. Ranking = mean rank of (marker, kendall) + transparent penalties.",
          "", "## Best configuration per dataset", "",
          "| Dataset | Platform | Best combo | marker log2FC | Kendall τ | cells | "
          "rel. purity | penalty | flags |",
          "|---|---|---|---:|---:|---:|---:|---:|---|"]
    for _, r in best.iterrows():
        md.append(f"| {r['dataset']} | {r['platform']} | `{r['combo']}` | "
                  f"{r['combined_marker_log2fc']:.3f} | {r['combined_kendall_tau']:.3f} | "
                  f"{int(r['combined_n_cells'])} | {r['combined_relative_purity']:.3f} | "
                  f"{r['penalty']:.1f} | {r['penalty_flags']} |")
    md += ["", "## Full ranking (top 5 per dataset)", ""]
    for dataset in DATASET_ORDER:
        sub = summary[summary["dataset"] == dataset].head(5)
        if sub.empty:
            continue
        md += [f"### {dataset} ({PLATFORM_OF[dataset]})", "",
               "| rank | combo | marker | kendall | cells | tx/prof | penalty | flags |",
               "|---:|---|---:|---:|---:|---:|---:|---|"]
        for _, r in sub.iterrows():
            md.append(f"| {int(r['final_rank'])} | `{r['combo']}` | "
                      f"{r['combined_marker_log2fc']:.3f} | {r['combined_kendall_tau']:.3f} | "
                      f"{int(r['combined_n_cells'])} | {r['combined_median_tx']:.0f} | "
                      f"{r['penalty']:.1f} | {r['penalty_flags']} |")
        md.append("")
    (OUT / "tuning_summary.md").write_text("\n".join(md) + "\n")
    log.info("WROTE %s", OUT / "tuning_summary.md")


def _make_plots(ranked):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1) marker & kendall vs combo, faceted by dataset
    for metric in ("combined_marker_log2fc", "combined_kendall_tau"):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=False)
        for ax, dataset in zip(axes, DATASET_ORDER):
            g = ranked[ranked["dataset"] == dataset].set_index("combo").reindex(COMBO_NAMES)
            vals = g[metric].values
            colors = ["tab:green" if c == "default" else "tab:blue" for c in COMBO_NAMES]
            ax.barh(range(len(COMBO_NAMES)), vals, color=colors)
            ax.set_yticks(range(len(COMBO_NAMES)))
            ax.set_yticklabels(COMBO_NAMES, fontsize=7)
            ax.invert_yaxis()
            ax.set_title(f"{dataset}", fontsize=9)
            ax.set_xlabel(metric.replace("combined_", ""))
            ax.axvline(0, color="k", lw=0.5)
        fig.suptitle(metric.replace("combined_", "combined "))
        fig.tight_layout()
        fig.savefig(PLOTS / f"{metric}_by_combo.png", dpi=130)
        plt.close(fig)

    # 2) marker vs kendall scatter (over-merge tradeoff), per dataset, sized by cells
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
    for ax, dataset in zip(axes, DATASET_ORDER):
        g = ranked[ranked["dataset"] == dataset]
        sizes = 20 + 120 * (g["combined_n_cells"] / max(g["combined_n_cells"].max(), 1))
        ax.scatter(g["combined_marker_log2fc"], g["combined_kendall_tau"],
                   s=sizes, alpha=0.7)
        for _, r in g.iterrows():
            ax.annotate(r["combo"], (r["combined_marker_log2fc"], r["combined_kendall_tau"]),
                        fontsize=6)
        d = g[g["combo"] == "default"]
        if not d.empty:
            ax.scatter(d["combined_marker_log2fc"], d["combined_kendall_tau"],
                       s=80, facecolors="none", edgecolors="red", linewidths=1.5,
                       label="default")
        ax.set_xlabel("marker log2FC"); ax.set_ylabel("Kendall τ")
        ax.set_title(dataset, fontsize=9); ax.legend(fontsize=7)
    fig.suptitle("Marker specificity vs Kendall τ (point size ∝ combined cells; red ring = default)")
    fig.tight_layout()
    fig.savefig(PLOTS / "marker_vs_kendall_scatter.png", dpi=130)
    plt.close(fig)

    # 3) cells & tx/profile vs combo
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, dataset in zip(axes, DATASET_ORDER):
        g = ranked[ranked["dataset"] == dataset].set_index("combo").reindex(COMBO_NAMES)
        ax.barh(range(len(COMBO_NAMES)), g["combined_n_cells"].values, color="tab:purple")
        ax.set_yticks(range(len(COMBO_NAMES)))
        ax.set_yticklabels(COMBO_NAMES, fontsize=7); ax.invert_yaxis()
        ax.set_title(dataset, fontsize=9); ax.set_xlabel("combined cells")
    fig.suptitle("Combined cell count by combo (collapse/inflation diagnostic)")
    fig.tight_layout()
    fig.savefig(PLOTS / "combined_cells_by_combo.png", dpi=130)
    plt.close(fig)
    log.info("WROTE plots -> %s", PLOTS)


# ---------------------------------------------------------------------------
# Comparison vs baselines (original / refine-in-place / resegment default / best)
# ---------------------------------------------------------------------------
# IMPORTANT: use all_metrics_long.tsv (the CONSOLIDATED scoring == compute_block_b:
# data-driven top-30 markers on each method's own panel), NOT block_ab_long.tsv.
# block_ab_long was overwritten by revise_biological.py with a DIFFERENT marker
# methodology (top-3 curated markers on a completed full panel), which is ~2.5×
# higher and NOT comparable to the tuning runs scored here. Kendall is identical
# across both sources; only marker_log2fc differs.
CACHED_BLOCK_AB = (REPO / "results" / "fig3_cross_platform_roi_benchmark"
                   / "summary_heatmaps" / "all_metrics_long.tsv")
CACHED_METHODS = ["original", "baysor", "proseg", "celladmix", "split", "segger"]


def _cached_method_metrics() -> pd.DataFrame:
    """marker_log2fc / kendall_tau / cells for cached methods, scored with the
    same compute_block_b methodology as the tuning runs (all_metrics_long.tsv)."""
    if not CACHED_BLOCK_AB.exists():
        log.warning("cached all_metrics_long.tsv not found at %s", CACHED_BLOCK_AB)
        return pd.DataFrame()
    df = pd.read_csv(CACHED_BLOCK_AB, sep="\t")
    keep = df[df["entity"].isin(CACHED_METHODS)
              & df["metric"].isin(["marker_log2fc", "kendall_tau", "total_cells",
                                    "relative_purity"])]
    piv = keep.pivot_table(index=["dataset", "entity"], columns="metric",
                           values="value").reset_index()
    piv = piv.rename(columns={"entity": "method",
                              "total_cells": "n_cells",
                              "marker_log2fc": "marker_log2fc",
                              "kendall_tau": "kendall_tau",
                              "relative_purity": "relative_purity"})
    piv["source"] = "cached_tracer_seg_benchmark"
    return piv


def cmd_compare(args):
    """Run refine-in-place, then build the cross-method comparison table."""
    # 1) refine-in-place for every dataset, scored identically to the sweep.
    datasets = args.datasets or DATASET_ORDER
    for dataset in datasets:
        try:
            od = run_refine_in_place(dataset, args.force_run)
            score_one(dataset, "__refine_in_place", od, args.force_score)
        except Exception as e:
            log.error("[%s] refine_in_place failed: %s", dataset, e)

    # 2) sweep results -> default + best resegment per dataset
    sweep = _collect()
    if sweep.empty:
        raise SystemExit("No sweep metrics.json yet; run the sweep first.")
    ranked = _rank_and_penalize(sweep)
    best = (ranked.sort_values(["dataset", "final_rank"])
                  .groupby("dataset", as_index=False).first())

    rows = []
    cached = _cached_method_metrics()
    for dataset in datasets:
        plat = PLATFORM_OF[dataset]

        # cached native + other methods
        if not cached.empty:
            for _, c in cached[cached["dataset"] == dataset].iterrows():
                rows.append(dict(dataset=dataset, platform=plat,
                                 method=f"{c['method']}", track=c["method"],
                                 marker_log2fc=c.get("marker_log2fc"),
                                 kendall_tau=c.get("kendall_tau"),
                                 n_cells=c.get("n_cells"),
                                 relative_purity=c.get("relative_purity"),
                                 source="cached_benchmark"))

        # TRACER refine-in-place (freshly scored, combined matrix)
        rip = RUNS / dataset / "__refine_in_place" / "metrics.json"
        if rip.exists():
            m = json.loads(rip.read_text())
            rows.append(dict(dataset=dataset, platform=plat,
                             method="TRACER_refine_in_place", track="tracer_refine_in_place",
                             marker_log2fc=m["combined_marker_log2fc"],
                             kendall_tau=m["combined_kendall_tau"],
                             n_cells=m["combined_n_cells"],
                             relative_purity=m["combined_relative_purity"],
                             source="this_analysis"))

        # TRACER resegment DEFAULT (current production default; from sweep)
        d = sweep[(sweep.dataset == dataset) & (sweep.combo == "default")]
        if not d.empty:
            d = d.iloc[0]
            rows.append(dict(dataset=dataset, platform=plat,
                             method="TRACER_resegment_default", track="tracer_reseg_default",
                             marker_log2fc=d["combined_marker_log2fc"],
                             kendall_tau=d["combined_kendall_tau"],
                             n_cells=d["combined_n_cells"],
                             relative_purity=d["combined_relative_purity"],
                             source="this_analysis"))

        # TRACER resegment BEST (tuned)
        b = best[best.dataset == dataset]
        if not b.empty:
            b = b.iloc[0]
            rows.append(dict(dataset=dataset, platform=plat,
                             method=f"TRACER_resegment_best[{b['combo']}]",
                             track="tracer_reseg_best",
                             marker_log2fc=b["combined_marker_log2fc"],
                             kendall_tau=b["combined_kendall_tau"],
                             n_cells=b["combined_n_cells"],
                             relative_purity=b["combined_relative_purity"],
                             source="this_analysis"))

    comp = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    comp.to_csv(OUT / "comparison_vs_baselines.tsv", sep="\t", index=False)
    log.info("WROTE %s (%d rows)", OUT / "comparison_vs_baselines.tsv", len(comp))

    # 3) markdown: per-dataset comparison + best-vs-default delta
    md = ["# TRACER resegment — best tuned vs baselines", "",
          "Marker specificity (log2FC) and Kendall τ scored on the **combined** TRACER",
          "matrix (resegment / refine-in-place) or the method's cell-by-gene (cached",
          "methods), all via the same canonical get_metric code path. Higher = better",
          "for both metrics.", ""]
    for dataset in datasets:
        sub = comp[comp.dataset == dataset].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("marker_log2fc", ascending=False)
        md += [f"## {dataset} ({PLATFORM_OF[dataset]})", "",
               "| method | marker log2FC | Kendall τ | cells | rel. purity | source |",
               "|---|---:|---:|---:|---:|---|"]
        for _, r in sub.iterrows():
            def f(v):
                return "—" if v != v or v is None else f"{v:.3f}"
            nc = "—" if (r["n_cells"] != r["n_cells"] or r["n_cells"] is None) else f"{int(r['n_cells'])}"
            md.append(f"| {r['method']} | {f(r['marker_log2fc'])} | {f(r['kendall_tau'])} | "
                      f"{nc} | {f(r['relative_purity'])} | {r['source']} |")
        # best-vs-default delta line
        dft = sub[sub.track == "tracer_reseg_default"]
        bst = sub[sub.track == "tracer_reseg_best"]
        if not dft.empty and not bst.empty:
            dm = bst.iloc[0]["marker_log2fc"] - dft.iloc[0]["marker_log2fc"]
            dk = bst.iloc[0]["kendall_tau"] - dft.iloc[0]["kendall_tau"]
            md += ["", f"**Best resegment − resegment default:** Δmarker = {dm:+.3f}, "
                   f"Δkendall = {dk:+.3f}.", ""]
    (OUT / "comparison_vs_baselines.md").write_text("\n".join(md) + "\n")
    log.info("WROTE %s", OUT / "comparison_vs_baselines.md")


# ---------------------------------------------------------------------------
def build_argparser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="run + score the sweep")
    pr.add_argument("--datasets", nargs="*", choices=DATASET_ORDER, default=None)
    pr.add_argument("--combos", nargs="*", choices=COMBO_NAMES, default=None)
    pr.add_argument("--force-run", action="store_true", help="re-run even if cached")
    pr.add_argument("--force-score", action="store_true", help="re-score even if cached")
    pr.add_argument("--strict", action="store_true", help="abort on first failure")
    pr.set_defaults(func=cmd_run)

    ps = sub.add_parser("summarize", help="aggregate + rank + plots")
    ps.set_defaults(func=cmd_summarize)

    pc = sub.add_parser("compare", help="refine-in-place + comparison vs baselines")
    pc.add_argument("--datasets", nargs="*", choices=DATASET_ORDER, default=None)
    pc.add_argument("--force-run", action="store_true")
    pc.add_argument("--force-score", action="store_true")
    pc.set_defaults(func=cmd_compare)
    return p


def main():
    args = build_argparser().parse_args()
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())

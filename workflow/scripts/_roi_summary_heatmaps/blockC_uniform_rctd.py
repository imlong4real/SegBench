#!/usr/bin/env python3
"""Block C — uniform RCTD entropy / max-weight across ALL methods + datasets.

Goal: a complete, FAIR RCTD comparison — same scRNA reference, same gene panel
(each method's own panel intersected with the reference, identically), same
profile QC (10 ≤ transcripts ≤ 900), same spacexr RCTD pipeline (doublet mode,
single-core to avoid the known parallel chooseSigma bug), and common abundant
cell types. SPLIT is run through the SAME RCTD pipeline (its internal RCTD is
NOT reused). The TRACER entities are derived from the BEST resegment config.

Entities (column keys):
  original, baysor, proseg, segger, celladmix, split,
  TRACER_resegment (best, combined), TRACER_refine (refine-in-place, combined),
  TRACER_refined (best, _etype==cell), TRACER_reconstructed (best, _etype==partial)

Subcommands:
  build      -> _work_blockC/<ds>/<ent>.h5ad   (float64, QC 10..900)
  rctd       -> metrics_blockC/<ds>/<ent>/rctd/ (calls run_rctd.R, max_cores=1)
  summarize  -> common-abundant-type tables + main/companion heatmaps + audit md

Run build + summarize in the `spatial` env; rctd uses tracer_benchmark_r (Rscript).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "workflow" / "scripts"))
sys.path.insert(0, str(HERE))
import registry as R  # noqa: E402

FIG3 = REPO / "results" / "fig3_cross_platform_roi_benchmark"
SHEAT = FIG3 / "summary_heatmaps"
TUNE = REPO / "results" / "tracer_resegment_tuning"
RUNS = TUNE / "runs"
WORKC = SHEAT / "_work_blockC"
METRICSC = SHEAT / "metrics_blockC"
RUN_RCTD = REPO / "workflow" / "scripts" / "run_rctd.R"
RSCRIPT = Path.home() / "anaconda3" / "envs" / "tracer_benchmark_r" / "bin" / "Rscript"

BEST_COMBO = {"atera_cervical": "specificity_preset", "xenium5k_cervical": "stitch_maha_off",
              "cosmx_nsclc": "stitch_dC0.10", "merfish_mouse_ileum": "specificity_preset"}

ENTITIES = ["original", "baysor", "proseg", "segger", "celladmix", "split",
            "TRACER_resegment", "TRACER_refine", "TRACER_refined", "TRACER_reconstructed"]
MAIN_METHODS = ["original", "baysor", "proseg", "segger", "celladmix", "split", "TRACER_resegment"]
COMPANION_METHODS = ["TRACER_resegment", "TRACER_refine", "TRACER_refined", "TRACER_reconstructed"]
LABELS = {
    "original": "Original /\nbaseline", "baysor": "Baysor", "proseg": "proseg",
    "segger": "Segger*", "celladmix": "cellAdmix", "split": "SPLIT",
    "TRACER_resegment": "TRACER\nResegment", "TRACER_refine": "TRACER\nRefine-in-Place",
    "TRACER_refined": "TRACER\nRefined", "TRACER_reconstructed": "TRACER\nReconstructed",
}
PLATFORM_DISPLAY = {"atera_cervical": "Atera", "xenium5k_cervical": "Xenium5K",
                    "cosmx_nsclc": "CosMx", "merfish_mouse_ileum": "MERFISH"}
LO, HI = 10, 900
ABUND_MIN = 30          # common-abundant: >= this many RCTD profiles in >= half methods
RECON_MIN_QC = 30       # min QC profiles to even run RCTD on reconstructed partials
EXCLUDE_CELLTYPES = {"Unannotated", "unannotated", "UNANNOTATED"}


def platform_label(ds):
    return PLATFORM_DISPLAY[ds]


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def _qc_write(X, obs_names, var_names, obs_src, outp):
    import anndata as ad
    X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
    # spacexr's SpatialRNA requires INTEGER counts but the dgRMatrix must be of
    # type double — so round to nearest integer and store as float64. This is a
    # no-op for transcript-count methods; SPLIT / cellAdmix emit fractional
    # (probabilistic) counts which are integerized here so the SAME RCTD pipeline
    # can be run on every method (documented in the audit note).
    X = X.copy()
    X.data = np.rint(X.data)
    X.eliminate_zeros()
    tx = np.asarray(X.sum(1)).ravel()
    keep = (tx >= LO) & (tx <= HI)
    Xk = X[keep].astype(np.float64)
    names = np.asarray(obs_names).astype(str)[keep]
    o = ad.AnnData(X=Xk.copy(), obs=pd.DataFrame(index=names),
                   var=pd.DataFrame(index=np.asarray(var_names).astype(str)))
    o.layers["counts"] = Xk.copy()
    if obs_src is not None:
        for cx, cy in (("x_centroid", "y_centroid"), ("cell_centroid_x", "cell_centroid_y")):
            if cx in obs_src and cy in obs_src:
                o.obs["x_centroid"] = np.asarray(obs_src[cx], float)[keep]
                o.obs["y_centroid"] = np.asarray(obs_src[cy], float)[keep]
                break
    outp.parent.mkdir(parents=True, exist_ok=True)
    o.write_h5ad(outp)
    return int(keep.sum()), int(len(keep))


def _read_h5ad_counts(p):
    import anndata as ad
    a = ad.read_h5ad(p)
    X = a.layers["counts"] if "counts" in a.layers else a.X
    return X, a.obs_names, a.var_names, a.obs


def _build_tracer_split(ds, etype):
    import get_metric as gm
    import logging
    pq = RUNS / ds / BEST_COMBO[ds] / "outputs" / "transcripts_tracer_refined.parquet"
    df = pd.read_parquet(pq)
    df["feature_name"] = df["feature_name"].astype(str)
    sub = df[df["_etype"].astype(str) == etype]
    if len(sub) == 0:
        return None
    a = gm.build_cellxgene(sub, "stitched", keep_ids=None, log=logging.getLogger("blockC"))
    return a


def cmd_build(args):
    import anndata as ad
    rows = []
    for ds in R.DATASET_ORDER:
        for ent in ENTITIES:
            outp = WORKC / ds / f"{ent}.h5ad"
            if outp.exists() and not args.force:
                continue
            src = None
            if ent in ("original", "baysor", "proseg", "segger", "celladmix", "split"):
                p = R.work_h5ad(ds, ent)
                if not p.exists():
                    rows.append(dict(dataset=ds, entity=ent, status="absent_input"))
                    continue
                X, on, vn, obs = _read_h5ad_counts(p)
            elif ent == "TRACER_resegment":
                X, on, vn, obs = _read_h5ad_counts(
                    RUNS / ds / BEST_COMBO[ds] / "outputs" / "cell_by_gene_tracer.h5ad")
            elif ent == "TRACER_refine":
                X, on, vn, obs = _read_h5ad_counts(
                    RUNS / ds / "__refine_in_place" / "outputs" / "cell_by_gene_tracer.h5ad")
            elif ent in ("TRACER_refined", "TRACER_reconstructed"):
                et = "cell" if ent == "TRACER_refined" else "partial"
                a = _build_tracer_split(ds, et)
                if a is None:
                    rows.append(dict(dataset=ds, entity=ent, status="no_profiles"))
                    continue
                X, on, vn, obs = a.X, a.obs_names, a.var_names, None
            n_post, n_pre = _qc_write(X, on, vn, obs, outp)
            rows.append(dict(dataset=ds, entity=ent, status="built",
                             n_pre=n_pre, n_post=n_post))
            print(f"[build] {ds}/{ent}: {n_post}/{n_pre} pass QC")
    pd.DataFrame(rows).to_csv(WORKC / "build_log.tsv", sep="\t", index=False)
    print("build done ->", WORKC)


# ---------------------------------------------------------------------------
# rctd
# ---------------------------------------------------------------------------
def _ref_for(ds):
    cfg = R.DATASETS[ds]
    return str(REPO / cfg["reference_h5ad"]), cfg["reference_celltype_col"]


def cmd_rctd(args):
    datasets = args.datasets or R.DATASET_ORDER
    ents = args.entities or ENTITIES
    for ds in datasets:
        ref_h5ad, ref_col = _ref_for(ds)
        ref_min_umi = 10 if ds == "merfish_mouse_ileum" else 100
        exclude = ds in ("atera_cervical", "xenium5k_cervical")
        for ent in ents:
            spatial = WORKC / ds / f"{ent}.h5ad"
            if not spatial.exists():
                print(f"[skip] {ds}/{ent}: no QC matrix"); continue
            # reconstructed: only run if enough QC profiles
            if ent == "TRACER_reconstructed":
                import anndata as ad
                n = ad.read_h5ad(spatial, backed="r").n_obs
                if n < RECON_MIN_QC:
                    print(f"[skip] {ds}/{ent}: only {n} QC profiles (<{RECON_MIN_QC})"); continue
            outdir = METRICSC / ds / ent / "rctd"
            if (outdir / "rctd_entropy_metrics.tsv").exists() and not args.force:
                print(f"[done] {ds}/{ent}"); continue
            outdir.mkdir(parents=True, exist_ok=True)
            cmd = [str(RSCRIPT), str(RUN_RCTD),
                   "--spatial-h5ad", str(spatial),
                   "--reference-h5ad", ref_h5ad, "--reference-celltype-col", ref_col,
                   "--outdir", str(outdir), "--doublet-mode", "doublet",
                   "--umi-min", "10", "--umi-min-sigma", "20",
                   "--reference-min-umi", str(ref_min_umi), "--max-cores", "1", "--seed", "1"]
            if exclude:
                cmd += ["--exclude-celltypes", "Unannotated"]
            print(f"==== RCTD {ds}/{ent} ====")
            with open(outdir / "rctd.log", "w") as lg:
                rc = subprocess.run(cmd, stdout=lg, stderr=subprocess.STDOUT).returncode
            print("  OK" if rc == 0 else f"  FAILED (rc={rc}; see {outdir}/rctd.log)")


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def _per_cell(ds, ent):
    p = METRICSC / ds / ent / "rctd" / "rctd_cell_assignments_post.tsv"
    return pd.read_csv(p, sep="\t") if p.exists() else None


def _present_methods(ds, methods):
    return [m for m in methods if _per_cell(ds, m) is not None]


def cmd_summarize(args):
    import matplotlib
    matplotlib.use("Agg")
    # ---- gather per-cell, per-type counts ----
    pc = {}  # (ds,ent) -> df
    for ds in R.DATASET_ORDER:
        for ent in ENTITIES:
            d = _per_cell(ds, ent)
            if d is not None:
                pc[(ds, ent)] = d

    # ---- common abundant cell types per dataset (>=ABUND_MIN in >=half methods) ----
    filt_rows, abund_map = [], {}
    for ds in R.DATASET_ORDER:
        present = _present_methods(ds, MAIN_METHODS)   # define abundance on MAIN set
        counts = {m: pc[(ds, m)]["dominant_celltype"].astype(str).value_counts() for m in present}
        allcts = sorted({ct for c in counts.values() for ct in c.index} - EXCLUDE_CELLTYPES)
        need = int(np.ceil(len(present) / 2))
        ab = []
        for ct in allcts:
            n_meet = sum(int(counts[m].get(ct, 0)) >= ABUND_MIN for m in present)
            is_ab = n_meet >= need
            if is_ab:
                ab.append(ct)
            filt_rows.append(dict(dataset=ds, platform=platform_label(ds), cell_type=ct,
                                  n_methods_meeting=n_meet, n_methods_present=len(present),
                                  threshold=ABUND_MIN, included=is_ab,
                                  reason=("abundant in ≥half methods" if is_ab
                                          else f"<{ABUND_MIN} profiles in ≥half methods"),
                                  **{f"n_{m}": int(counts[m].get(ct, 0)) for m in present}))
        abund_map[ds] = ab
    filt = pd.DataFrame(filt_rows)
    filt.to_csv(SHEAT / "rctd_common_celltype_filtering_summary.tsv", sep="\t", index=False)

    # ---- per-(ds,ent) aggregates over abundant types + per-cell-type tables ----
    agg_rows, ct_rows = [], []
    for ds in R.DATASET_ORDER:
        ab = set(abund_map[ds])
        for ent in ENTITIES:
            if (ds, ent) not in pc:
                continue
            d = pc[(ds, ent)].copy()
            d["dominant_celltype"] = d["dominant_celltype"].astype(str)
            d = d[~d["dominant_celltype"].isin(EXCLUDE_CELLTYPES)]
            d_ab = d[d["dominant_celltype"].isin(ab)]
            agg_rows.append(dict(
                dataset=ds, platform=platform_label(ds), entity=ent,
                median_entropy=float(d_ab["entropy"].median()) if len(d_ab) else np.nan,
                median_max_weight=float(d_ab["max_weight"].median()) if len(d_ab) else np.nan,
                n_profiles_used=int(len(d_ab)), n_profiles_total=int(len(d)),
                n_profiles_excluded=int(len(d) - len(d_ab)),
                n_common_abundant_types=len(ab)))
            for ct, g in d.groupby("dominant_celltype"):
                ct_rows.append(dict(dataset=ds, platform=platform_label(ds), entity=ent,
                                    cell_type=ct, n_profiles=int(len(g)),
                                    abundant=ct in ab,
                                    median_entropy=float(g["entropy"].median()),
                                    median_max_weight=float(g["max_weight"].median())))
    agg = pd.DataFrame(agg_rows)
    ct_long = pd.DataFrame(ct_rows)
    agg.to_csv(SHEAT / "source_data_blockC_common_celltypes.tsv", sep="\t", index=False)
    ct_long[["dataset", "platform", "entity", "cell_type", "n_profiles", "abundant", "median_entropy"]].to_csv(
        SHEAT / "celltype_rctd_entropy.tsv", sep="\t", index=False)
    ct_long[["dataset", "platform", "entity", "cell_type", "n_profiles", "abundant", "median_max_weight"]].to_csv(
        SHEAT / "celltype_rctd_max_weight.tsv", sep="\t", index=False)

    _plot_main(agg)
    _plot_companion(agg)
    _write_audit(agg, filt, abund_map)
    print("summarize done ->", SHEAT)


def _matrix(agg, metric, methods):
    M = np.full((len(R.DATASET_ORDER), len(methods)), np.nan)
    for di, ds in enumerate(R.DATASET_ORDER):
        for ci, m in enumerate(methods):
            r = agg[(agg.dataset == ds) & (agg.entity == m)]
            if len(r):
                M[di, ci] = r[metric].iloc[0]
    return M


def _heatmap(metrics, methods, agg, out_base, title, subtitle, footnote):
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    try:
        import seaborn  # noqa
        CMAP = "mako_r"
    except Exception:
        CMAP = "viridis"
    n_ds, ncols, npan = len(R.DATASET_ORDER), len(methods), len(metrics)
    fig_h = npan * (n_ds * 0.46 + 0.7) + 2.8
    fig_w = max(8, ncols * 1.25 + 2.6)
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(npan, 1, hspace=0.85, left=0.20, right=0.86,
                          top=1 - 1.3 / fig_h, bottom=2.3 / fig_h)
    DIR = {"median_entropy": "lower", "median_max_weight": "higher"}
    MLAB = {"median_entropy": "RCTD entropy (median over abundant types)",
            "median_max_weight": "RCTD max weight (median over abundant types)"}
    for pi, metric in enumerate(metrics):
        ax = fig.add_subplot(gs[pi, 0])
        raw = _matrix(agg, metric, methods)
        direction = DIR[metric]
        g = np.full_like(raw, np.nan)
        for di in range(n_ds):
            vals = raw[di]; fin = np.isfinite(vals)
            if fin.sum() >= 1:
                vmin, vmax = np.nanmin(vals[fin]), np.nanmax(vals[fin])
                norm = (vals - vmin) / (vmax - vmin) if vmax > vmin else np.where(fin, 0.5, np.nan)
                g[di] = (1 - norm) if direction == "lower" else norm
        cmap = plt.get_cmap(CMAP).copy(); cmap.set_bad("#e8e8e8")
        im = ax.imshow(np.ma.masked_invalid(g), aspect="auto", cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(np.arange(-.5, ncols, 1), minor=True)
        ax.set_yticks(np.arange(-.5, n_ds, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.4); ax.tick_params(which="minor", length=0)
        for di in range(n_ds):
            for ci in range(ncols):
                v = raw[di, ci]
                if not np.isfinite(v):
                    ax.text(ci, di, "NA", ha="center", va="center", fontsize=6.3,
                            color="#9a9a9a", style="italic"); continue
                gg = g[di, ci]; r, gr, b, _ = cmap(gg if np.isfinite(gg) else 0.5)
                lum = 0.2126 * r + 0.7152 * gr + 0.0722 * b
                ax.text(ci, di, f"{v:.2f}", ha="center", va="center", fontsize=6.9,
                        color="white" if lum < 0.55 else "#15161a")
        ax.set_yticks(range(n_ds)); ax.set_yticklabels([platform_label(d) for d in R.DATASET_ORDER], fontsize=8)
        ax.set_xticks(range(ncols))
        if pi == npan - 1:
            ax.set_xticklabels([LABELS[m] for m in methods], fontsize=7.2, rotation=40, ha="right")
        else:
            ax.set_xticklabels([])
        for sp_ in ax.spines.values():
            sp_.set_visible(False)
        ax.set_title(f"{MLAB[metric]}   ·   {'lower = better' if direction=='lower' else 'higher = better'}",
                     fontsize=9, loc="left", pad=4, fontweight="bold")
        cax = ax.inset_axes([1.012, 0, 0.02, 1.0]); cb = fig.colorbar(im, cax=cax)
        cb.set_ticks([0, 1]); cb.set_ticklabels(["worse", "better"], fontsize=6); cb.outline.set_linewidth(0.4)
    fig.suptitle(title, x=0.20, y=1 - 0.35 / fig_h, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.20, 1 - 0.78 / fig_h, subtitle, fontsize=7.2, color="#333", va="top")
    fig.text(0.20, 1.5 / fig_h, footnote, fontsize=6.3, color="#444", va="top", wrap=True)
    for ext in ("png", "svg"):
        fig.savefig(f"{out_base}.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_base + ".png/.svg")


def _plot_main(agg):
    foot = ("Same scRNA reference + spacexr RCTD (doublet mode, single-core) + QC "
            "(10≤tx≤900) for ALL methods; SPLIT run through the SAME pipeline (internal "
            "RCTD not reused). Median over common abundant cell types (≥30 RCTD profiles "
            "in ≥half the methods; 'Unannotated' excluded). Original/baseline: Atera/"
            "Xenium5K=10x · CosMx=CosMx SMI · MERFISH=Baysor (standalone Baysor NA on "
            "MERFISH). * Segger = GPU. TRACER Resegment = best per-platform tuned config.")
    _heatmap(["median_entropy", "median_max_weight"], MAIN_METHODS, agg,
             str(SHEAT / "blockC_rctd_common_celltypes"),
             "Block C — RCTD deconvolution purity (common abundant cell types)",
             "Whole-cell / profile benchmark. Single TRACER Resegment column; "
             "refined/reconstructed in companion.", foot)


def _plot_companion(agg):
    foot = ("TRACER entities only, same RCTD pipeline. Resegment = best combined; "
            "Refine-in-Place = refinement-only (original partition kept); Refined = "
            "whole-cell subset (_etype=cell); Reconstructed = partial subset "
            "(_etype=partial). Median over common abundant cell types.")
    present = [m for m in COMPANION_METHODS
               if agg[agg.entity == m]["median_entropy"].notna().any()]
    _heatmap(["median_entropy", "median_max_weight"], present, agg,
             str(SHEAT / "blockC_tracer_refined_reconstructed_rctd"),
             "Block C companion — TRACER entities (RCTD)",
             "TRACER Resegment vs Refine-in-Place vs Refined vs Reconstructed.", foot)


def _write_audit(agg, filt, abund_map):
    md = ["# Block C RCTD update — audit & summary", "",
          "## RCTD implementation",
          "- **spacexr RCTD** (R), `spacexr::run.RCTD` **doublet mode** via "
          "`workflow/scripts/run_rctd.R`. NOT a Python Poisson-EM reimplementation.",
          "- Single-core (`--max-cores 1`) to avoid the known spacexr parallel "
          "`chooseSigma` failure.",
          "- Per-cell Shannon entropy of normalized RCTD weights (lower = purer) and "
          "max weight (higher = purer); cells grouped by RCTD `dominant_celltype`.",
          "- Args: `--umi-min 10 --umi-min-sigma 20 --doublet-mode doublet`, "
          "`--reference-min-umi 100` (10 for MERFISH's sparse ~236-gene panel), "
          "`--exclude-celltypes Unannotated` on the cervical datasets.", "",
          "## Reference & gene panel",
          "- scRNA reference per dataset (identical across methods): "
          "cervical (Atera+Xenium5K), lung_cancer_50k (CosMx), gut GSE92332 (MERFISH).",
          "- Gene panel = each method's own panel intersected with the reference by "
          "spacexr (identical procedure for every method, including SPLIT).", "",
          "## QC threshold (uniform)",
          f"- Profiles with **{LO} ≤ transcripts ≤ {HI}** kept, applied identically to "
          "every method/dataset before RCTD (float64 matrices in `_work_blockC/`).", "",
          "## Common abundant cell-type criteria",
          f"- A cell type is **included** if it has **≥ {ABUND_MIN} RCTD-assigned "
          "profiles in at least half** of the main methods present for that dataset; "
          "'Unannotated' is always excluded. Per-dataset included sets:",
          ]
    for ds in R.DATASET_ORDER:
        md.append(f"  - **{platform_label(ds)}**: {', '.join(abund_map[ds]) or '(none)'}")
    md += ["- Full inclusion/exclusion table with per-method counts: "
           "`rctd_common_celltype_filtering_summary.tsv`.", "",
           "## SPLIT handling",
           "- **SPLIT was rerun uniformly** through `run_rctd.R` on its QC'd cell-by-gene "
           "matrix (its internal RCTD is NOT used here), so it is directly comparable.", "",
           "## Whole-cell vs partial split",
           "- Main heatmap shows a single **TRACER Resegment** column (best combined).",
           "- **TRACER Refined** (whole-cell) and **TRACER Reconstructed** (partial) are "
           "in the companion (`blockC_tracer_refined_reconstructed_rctd`).",
           "", "## Median entropy / max-weight (over abundant types)", "",
           "| dataset | method | median entropy | median max wt | profiles used | excluded | #types |",
           "|---|---|---:|---:|---:|---:|---:|"]
    for ds in R.DATASET_ORDER:
        for m in ENTITIES:
            r = agg[(agg.dataset == ds) & (agg.entity == m)]
            if not len(r):
                continue
            r = r.iloc[0]
            md.append(f"| {platform_label(ds)} | {m} | {r['median_entropy']:.3f} | "
                      f"{r['median_max_weight']:.3f} | {int(r['n_profiles_used'])} | "
                      f"{int(r['n_profiles_excluded'])} | {int(r['n_common_abundant_types'])} |")
    # missing-value report
    md += ["", "## Missing values & reasons", ""]
    miss = []
    for ds in R.DATASET_ORDER:
        for m in MAIN_METHODS + ["TRACER_refine", "TRACER_refined", "TRACER_reconstructed"]:
            r = agg[(agg.dataset == ds) & (agg.entity == m)]
            if not len(r) or not np.isfinite(r["median_entropy"].iloc[0]):
                reason = ("standalone Baysor = baseline on MERFISH" if (m == "baysor" and ds == "merfish_mouse_ileum")
                          else "too few QC profiles for RCTD" if m == "TRACER_reconstructed"
                          else "no RCTD result (see metrics_blockC/<ds>/<ent>/rctd/rctd.log)")
                miss.append(f"- {platform_label(ds)} / {m}: {reason}")
    md += miss or ["- None — all main cells populated."]
    md += ["", "## Caveats", "",
           "- **Integer counts for spacexr.** spacexr's `SpatialRNA` requires integer "
           "counts, but **SPLIT and cellAdmix emit fractional (probabilistic) counts** "
           "(this is the original reason SPLIT shipped its own internal RCTD). To run the "
           "identical pipeline on every method, counts were rounded to the nearest "
           "integer (a no-op for transcript-count methods). Rounding zeroes low-"
           "probability genes, which can modestly **amplify** SPLIT's apparent purity, so "
           "SPLIT's strong entropy/max-weight should be read with that in mind. It is, "
           "however, now a uniform and reproducible comparison rather than a "
           "provenance-mixed one.",
           "- **Grouping is by RCTD `dominant_celltype`** (RCTD's own argmax), so the "
           "metrics reflect RCTD's assignment confidence per profile; cell-type "
           "abundance is defined on RCTD output, consistent across methods.",
           "- TRACER Reconstructed (partials) has few profiles on the Xenium panels "
           "(Atera 36, Xenium5K 39 in abundant types) — interpret its companion values "
           "as indicative only.",
           "", "## Output files", "",
           "- `blockC_rctd_common_celltypes.{png,svg}` — main (7 whole-cell methods + TRACER Resegment)",
           "- `blockC_tracer_refined_reconstructed_rctd.{png,svg}` — companion (TRACER entities)",
           "- `source_data_blockC_common_celltypes.tsv` — per-method aggregates + profile counts",
           "- `celltype_rctd_entropy.tsv`, `celltype_rctd_max_weight.tsv` — per-cell-type tables",
           "- `rctd_common_celltype_filtering_summary.tsv` — inclusion/exclusion + per-method counts"]
    (SHEAT / "blockC_rctd_update_summary.md").write_text("\n".join(md) + "\n")
    print("wrote blockC_rctd_update_summary.md")


# ---------------------------------------------------------------------------
def build_argparser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    pb = sub.add_parser("build"); pb.add_argument("--force", action="store_true"); pb.set_defaults(func=cmd_build)
    pr = sub.add_parser("rctd")
    pr.add_argument("--datasets", nargs="*"); pr.add_argument("--entities", nargs="*")
    pr.add_argument("--force", action="store_true"); pr.set_defaults(func=cmd_rctd)
    ps = sub.add_parser("summarize"); ps.set_defaults(func=cmd_summarize)
    return p


def main():
    args = build_argparser().parse_args()
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())

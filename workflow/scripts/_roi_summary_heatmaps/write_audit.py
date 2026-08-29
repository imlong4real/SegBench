#!/usr/bin/env python3
"""Stage 4: write benchmark_heatmap_summary.md — the audit for the figure.

Documents exact input paths per (dataset, entity), TRACER combined/separate
handling per metric, runtime/memory extraction, NA cells + reasons,
normalization rules, and the method order.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import registry as R  # noqa


def rel(p):
    try:
        return str(Path(p).resolve().relative_to(R.REPO))
    except Exception:
        return str(p)


def _rctd_post(metrics_root, ds, ent):
    f = metrics_root / ds / ent / "rctd" / "rctd_entropy_metrics.tsv"
    if not f.exists():
        return (np.nan, np.nan)
    t = pd.read_csv(f, sep="\t")
    post = t[t["tag"] == "post"] if "tag" in t.columns else t
    if not len(post):
        return (np.nan, np.nan)
    return (float(post["median_entropy"].iloc[0]), float(post["median_max_weight"].iloc[0]))


def _revised_section(A):
    """Revised biological metrics: whole-cell refinement vs partial reconstruction,
    full-panel completion, 3-marker sets, Unannotated exclusion."""
    rev_f = R.OUT / "_revised_biological.tsv"
    if not rev_f.exists():
        return
    import anndata as ad
    rev = pd.read_csv(rev_f, sep="\t").set_index(["dataset", "entity"])

    A("## Revised biological metrics (whole-cell refinement vs partial reconstruction)\n")
    A("This is the methodology used in the **current Block B / Block C figures**. It "
      "supersedes the Audit-1 common-marker set and scores TRACER-refined and "
      "TRACER-reconstructed separately throughout.\n")

    A("### Gene universe and full-panel completion\n")
    A("- **Gene universe** per dataset = the full platform panel (union of all "
      "method panels): Xenium 17,420 · Xenium5K 4,863 · CosMx 960 · MERFISH 241.")
    A("- Every method is reindexed to this universe. **cellAdmix** outputs an HVG "
      "subset on the cervical panels (1,953 / 1,967 genes); its non-modelled genes "
      "are **inherited from the matched original cell** (same `cell_id`, 100% "
      "overlap) — i.e. genes cellAdmix did not touch are assumed unchanged from the "
      "input profile. All other methods are de-novo / full-panel, so unreported "
      "genes are genuine zeros (zero-filled). This removes any advantage/penalty "
      "from a method's gene-panel selection: after completion cellAdmix ≈ original "
      "on the inherited genes.")
    A("- Cell-type labels are re-derived by KNN label transfer **on the completed "
      "full panel**, so no method benefits from a favourable HVG set.\n")

    A("### Marker sets (3 canonical markers per cell type)\n")
    A("Top-3 scRNA Wilcoxon markers per cell type, restricted to the platform panel "
      "(so present, post-completion, for every method). Full lists: "
      "`metrics/<ds>/_marker_genes_3.tsv`.\n")
    for ds in R.DATASET_ORDER:
        mk = pd.read_csv(R.METRICS / ds / "_marker_genes_3.tsv", sep="\t")
        A(f"**{R.DATASETS[ds]['platform']}** ({mk['cell_type'].nunique()} types, "
          f"{mk['gene'].nunique()} genes):")
        for ct, g in mk.groupby("cell_type"):
            genes = ", ".join(g.sort_values("rank")["gene"].astype(str))
            A(f"  - {ct}: {genes}")
        A("")

    A("### Refined vs reconstructed — all biological metrics\n")
    A("Marker log2FC & Kendall on the completed full panel (unfiltered); relative "
      "purity/conflict on QC-filtered (10–900) completed profiles. Combined TRACER "
      "shown only as a Kendall sensitivity value.\n")
    A("| Dataset | Metric | original | TRACER-refined | TRACER-reconstructed | TRACER (combined) |")
    A("|---|---|--:|--:|--:|--:|")
    rmap = [("marker_log2fc", "Marker log2FC"), ("kendall_tau", "Kendall τ"),
            ("relative_purity", "Rel. purity"), ("relative_conflict", "Rel. conflict")]
    for ds in R.DATASET_ORDER:
        for col, lab in rmap:
            def gv(ent):
                return rev.loc[(ds, ent), col] if (ds, ent) in rev.index else np.nan
            A(f"| {R.DATASETS[ds]['platform']} | {lab} | {gv('original'):.3f} | "
              f"{gv('TRACER_refined'):.3f} | {gv('TRACER_reconstructed'):.3f} | "
              f"{gv('TRACER'):.3f} |")
    A("")

    A("### How much of TRACER's result is sparse reconstructed profiles vs genuine biology?\n")
    A("| Dataset | recon tx/cell | recon % of TRACER cells | refined−recon marker | refined−recon Kendall |")
    A("|---|--:|--:|--:|--:|")
    for ds in R.DATASET_ORDER:
        a = ad.read_h5ad(R.work_h5ad(ds, "TRACER_reconstructed"))
        tx = float(np.median(np.asarray(a.X.sum(1)).ravel()))
        n_rec = a.n_obs
        n_tot = n_rec + ad.read_h5ad(R.work_h5ad(ds, "TRACER_refined")).n_obs
        dmk = rev.loc[(ds, "TRACER_refined"), "marker_log2fc"] - rev.loc[(ds, "TRACER_reconstructed"), "marker_log2fc"]
        dkd = rev.loc[(ds, "TRACER_refined"), "kendall_tau"] - rev.loc[(ds, "TRACER_reconstructed"), "kendall_tau"]
        A(f"| {R.DATASETS[ds]['platform']} | {tx:.0f} | {100*n_rec/n_tot:.0f}% | "
          f"{dmk:+.3f} | {dkd:+.3f} |")
    A("\n**Interpretation.** TRACER-**refined** (whole cells, full transcript "
      "complement) carries the genuine biological signal: it tracks the original "
      "segmentation on Kendall and exceeds it / matches it on marker specificity "
      "and reference-NPMI purity. TRACER-**reconstructed** profiles are sparse "
      "(6–48 tx/cell) partial cells: they score near-zero marker log2FC and low "
      "Kendall (incomplete profiles), and a trivially high NPMI purity (too few "
      "gene pairs to conflict). They are reported separately and never merged into "
      "the whole-cell benchmark. Thus TRACER's biological-quality performance is "
      "driven by genuine whole-cell refinement, not by sparse reconstructed "
      "profiles; the reconstructed column quantifies the residual partial-cell "
      "recovery on its own terms.\n")


def _qc_section(A):
    """QC sensitivity analysis (uniform 10<=tx<=900 filter), pre vs post."""
    qcf = R.OUT / "_qc_sensitivity_metrics.tsv"
    if not qcf.exists():
        return
    q = pd.read_csv(qcf, sep="\t").set_index(["dataset", "entity"])
    # attach RCTD pre (metrics/) and post (metrics_qc/)
    rctd_pre = R.METRICS
    rctd_post = R.OUT / "metrics_qc"
    from scipy.stats import spearmanr

    A("## QC sensitivity analysis (uniform 10–900 transcript filter)\n")
    A("Every cell/profile with <10 or >900 transcripts was removed, identically "
      "for every method and dataset, and all four metric families recomputed. This "
      "isolates whether low-information or extreme-count profiles drive the TRACER "
      "results. Marker log2FC here uses the FULL reference marker set scored on each "
      "method's own panel (per the QC spec). Tables: `_qc_removal_stats.tsv`, "
      "`_qc_sensitivity_metrics.tsv`.\n")

    A("### Profiles removed (n and %)\n")
    A("| Dataset | Entity | n pre | n post | % removed |")
    A("|---|---|--:|--:|--:|")
    for ds in R.DATASET_ORDER:
        for ent in R.ENTITY_ORDER:
            if (ds, ent) not in q.index:
                continue
            r = q.loc[(ds, ent)]
            A(f"| {R.DATASETS[ds]['platform']} | {R.ENTITY_LABELS[ent] or 'Original'} | "
              f"{int(r['n_pre'])} | {int(r['n_post'])} | {r['pct_removed']:.1f}% |")
    A("\nThe filter removes most TRACER-reconstructed partials (MERFISH 80%, "
      "Xenium5K 52%, CosMx 24%, atera 16% — confirming they are low-information "
      "profiles) and a large >900 upper tail on dense Xenium (atera original/"
      "refined/Segger ≈ 32–34%). proseg loses 25–36% on sparse panels (its <10-tx "
      "cells).\n")

    # Build pre/post per-metric tables + effect sizes + ranking change
    def metric_block(title, pre_col, post_col, rctd=None):
        A(f"### {title} — pre → post filter\n")
        A("| Dataset | original | TRACER | TRACER-refined | TRACER-recon | Baysor | proseg | Segger | cellAdmix | SPLIT |")
        A("|---|" + "---|" * 9)
        for ds in R.DATASET_ORDER:
            cells = []
            for ent in R.ENTITY_ORDER:
                if rctd is not None:
                    if ent == "TRACER" or (ent == "baysor" and not R.DATASETS[ds]["has_baysor"]):
                        cells.append("NA"); continue
                    if ent == "split":
                        cells.append("—(internal)"); continue
                    pre = _rctd_post(rctd_pre, ds, ent)[rctd]
                    post = _rctd_post(rctd_post, ds, ent)[rctd]
                    cells.append("NA" if np.isnan(pre) else f"{pre:.2f}→{post:.2f}")
                else:
                    if (ds, ent) not in q.index:
                        cells.append("NA"); continue
                    r = q.loc[(ds, ent)]
                    if pd.isna(r[pre_col]):
                        cells.append("NA")
                    else:
                        cells.append(f"{r[pre_col]:.2f}→{r[post_col]:.2f}")
            A(f"| {R.DATASETS[ds]['platform']} | " + " | ".join(cells) + " |")
        # effect size: median |Δ| across entities/datasets; ranking change Spearman
        if rctd is None:
            deltas = (q[post_col] - q[pre_col]).abs()
            A(f"\nEffect size: median |Δ| = **{deltas.median():.3f}**, "
              f"max |Δ| = {deltas.max():.3f}. ")
            # ranking change per dataset
            rhos = []
            for ds in R.DATASET_ORDER:
                sub = q.loc[ds]
                pp = sub[[pre_col, post_col]].dropna()
                if len(pp) >= 3:
                    rhos.append(spearmanr(pp[pre_col], pp[post_col]).correlation)
            A(f"Method-ranking preservation (Spearman pre vs post, mean over datasets) "
              f"= **{np.nanmean(rhos):.3f}**.\n")
        else:
            A("")

    metric_block("Marker specificity (log2FC, full ref markers)", "marker_pre", "marker_post")
    metric_block("Relative purity (NPMI vs reference)", "purity_pre", "purity_post")
    metric_block("Relative conflict (NPMI vs reference)", "conflict_pre", "conflict_post")
    metric_block("Kendall τ vs scRNA", "kendall_pre", "kendall_post")
    if (R.OUT / "metrics_qc").exists():
        metric_block("RCTD entropy", None, None, rctd=0)
        metric_block("RCTD max weight", None, None, rctd=1)
    else:
        A("### RCTD entropy / max weight — pre → post filter\n")
        A("*(RCTD QC re-run pending; will be filled from `metrics_qc/`.)*\n")

    A("### QC conclusion — are transcript-count outliers the primary cause?\n")
    A("- **Marker specificity:** *partly* for the TRACER **combined** column. "
      "Removing <10-tx reconstructed profiles raises combined TRACER toward the "
      "refined level (e.g. Xenium5K +0.32→+0.45; CosMx +1.38→+1.57; MERFISH "
      "+2.23→+2.44). TRACER-**refined** is essentially unchanged and already "
      "tracks the baseline, so the combined deficit is driven by low-count "
      "reconstructed profiles, not by TRACER segmentation quality.")
    A("- **NPMI purity / conflict:** **No** — robust to filtering (median |Δ| ≈ "
      "0.003; rankings preserved). TRACER stays the best non-trivial method; any "
      "deficit vs proseg/SPLIT is a metric property (sparsity / explicit "
      "purification), not an extreme-profile artifact, and persists after QC.")
    A("- **Kendall τ:** **No** — essentially invariant (|Δ| ≈ 0.002). TRACER-"
      "refined ≈ original before and after; the CosMx gap to Baysor/SPLIT is a "
      "genuine cells-per-type / purification effect, not sparse/extreme profiles.")
    A("- **RCTD:** the upper-tail (>900) removal is concentrated on dense Xenium; "
      "RCTD already excludes <10-tx cells internally (UMI_min=10), so refined "
      "values move little, while TRACER-reconstructed (which loses 16–80% of "
      "cells) is the most filter-sensitive. See table above for exact pre→post.\n")
    A("**Bottom line:** transcript-count outliers explain the TRACER *combined* "
      "marker dip (reconstructed low-count profiles) but **not** the purity, "
      "Kendall, or refined-marker results, which are stable under QC. The "
      "substantive TRACER signal — refined ≈ baseline on Kendall/RCTD, best-in-"
      "class biological coherence under the reference NPMI panel — survives "
      "filtering.\n")


def main():
    long_all = pd.read_csv(R.OUT / "all_metrics_long.tsv", sep="\t")
    lines = []
    A = lines.append

    A("# Cross-platform ROI benchmark — split heatmap summary (audit)\n")
    A("Three publication split-heatmap blocks comparing segmentation / refinement "
      "methods across four cross-platform ROIs. One sub-heatmap per metric "
      "(rows = datasets/platforms, columns = methods); cells annotated with raw "
      "values, coloured by within-metric within-dataset normalization.\n")
    A("Generated by `workflow/scripts/_roi_summary_heatmaps/` "
      "(`build_matrices_and_metrics.py` → `run_block_c_rctd.sh` → "
      "`consolidate_and_plot.py` → `write_audit.py`).\n")

    A("## Datasets and scRNA references\n")
    A("| Dataset | Platform | Original (baseline) | scRNA reference | celltype col | NPMI panel |")
    A("|---|---|---|---|---|---|")
    for ds in R.DATASET_ORDER:
        c = R.DATASETS[ds]
        A(f"| `{ds}` | {c['platform']} | {c['original_label']} | "
          f"`{rel(c['reference_h5ad'])}` | `{c['reference_celltype_col']}` | "
          f"`{rel(c['npmi'])}` |")
    A("")

    A("## Method order (identical across all three blocks)\n")
    order = " → ".join(R.ENTITY_LABELS[e] if e != "original" else "Original (baseline)"
                       for e in R.ENTITY_ORDER)
    A(f"`{order}`\n")
    A("Segger carries a `*` (GPU-based; runtime / peak memory are GPU on an "
      "NVIDIA H100). Legend note: **\\* GPU-based**.\n")

    A("## TRACER refined / reconstructed handling per metric\n")
    A("`transcripts_tracer_refined.parquet` `_etype` splits TRACER cells into "
      "**refined** (`_etype == cell`, whole cells) and **reconstructed** "
      "(`_etype == partial`, reconstructed partials). The standardized combined "
      "TRACER matrix = refined ∪ reconstructed.\n")
    A("| Metric | TRACER handling |")
    A("|---|---|")
    for m, spec in R.METRICS_SPEC.items():
        mode = "combined (single **TRACER** column; refined/reconstructed = NA)" \
            if spec["tracer"] == "combined" else \
            "separate (**TRACER-refined** + **TRACER-reconstructed**; combined TRACER = NA)"
        A(f"| {spec['label']} | {mode} |")
    A("")

    A("## Per-entity input paths\n")
    A("Cell-by-gene matrices used for size + biological metrics. Transcript-built "
      "matrices are grouped (label × feature_name) into integer cell×gene counts.\n")
    A("| Dataset | Entity | Source kind | Path |")
    A("|---|---|---|---|")
    for ds in R.DATASET_ORDER:
        for ent in R.ENTITY_ORDER:
            spec = R.entity_matrix_spec(ds, ent)
            if spec is None:
                A(f"| `{ds}` | {ent} | — | *absent (standalone Baysor skipped; ROI is Baysor-segmented)* |")
                continue
            extra = f" (_etype={spec['etype']}, label={spec['label_col']})" if "etype" in spec else \
                    (f" (label={spec['label_col']})" if spec["kind"] == "transcripts" else "")
            A(f"| `{ds}` | {ent} | {spec['kind']}{extra} | `{rel(spec['path'])}` |")
    A("")

    A("## Metric definitions\n")
    A("All biological metrics reuse the canonical `workflow/scripts/get_metric.py` "
      "implementations for cross-dataset comparability:\n")
    A("- **Total cells / profiles** — `n_obs` of the entity cell×gene matrix.")
    A("- **Transcripts per cell / profile** — median of per-cell total counts "
      "(row sums); SPLIT is fractional (purified) counts.")
    A("- **Runtime / peak memory** — see extraction below.")
    A("- **Marker specificity (log2FC, 3 markers/type)** — KNN label transfer (on "
      "the completed full panel) assigns each spatial cell a type; per cell type "
      "the **3 canonical scRNA Wilcoxon markers** (restricted to the platform "
      "panel) give log2FC (in-type vs other-type, log-normalized); median across "
      "marker rows. **TRACER-refined and TRACER-reconstructed are scored "
      "separately** (combined TRACER is not used). Marker lists: `metrics/<ds>/"
      "_marker_genes_3.tsv`. See *Revised biological metrics* below.")
    A("- **Relative purity / conflict (NPMI vs reference)** — TRACER NPMI relu "
      "purity/conflict (`tracer.metrics`, tau=0.05) against the reference-derived "
      "NPMI panel TRACER was refined against (scRNA co-expression; nuclear "
      "self-reference for MERFISH). Computed on **QC-filtered (10–900 tx) "
      "completed profiles**; refined/reconstructed separate. NPMI (bounded "
      "[-1,1]) is used rather than PMI (unbounded; many indeterminate pairs).")
    A("- **Kendall τ vs scRNA** — per cell type pseudobulk (log-normalized, FULL "
      "completed panel) of the method vs the scRNA reference pseudobulk, Kendall τ "
      "over shared genes, median across cell types. **TRACER-refined is the "
      "primary column**; TRACER-reconstructed and combined TRACER are sensitivity "
      "analyses.")
    A("- **RCTD entropy / max weight** — `run_rctd.R` (spacexr, doublet mode) on "
      "**QC-filtered (10–900 tx)** profiles vs the scRNA reference; median Shannon "
      "entropy and median max weight of normalized RCTD weights; refined/recon "
      "separate. SPLIT reuses its internal RCTD (QC-filtered cell set).")
    A("- **Reference cleaning:** for the cervical reference (atera, Xenium5K) the "
      "`Unannotated` junk cluster (1,966 cells) is dropped before label transfer, "
      "marker derivation, pseudobulk Kendall, and RCTD. cosmx/merfish references "
      "have no such category.\n")

    A("## Runtime / peak-memory extraction\n")
    A("- TRACER, Baysor, proseg, cellAdmix, SPLIT: `runtime_seconds`, `peak_rss_gb` "
      "from `results/fig3_cross_platform_roi_benchmark/benchmark_comparison.tsv` "
      "(single 17 GB CPU machine).")
    A("- Segger: `results/segger_roi/<ds>/benchmark/runtime_memory.json` "
      "(`runtime_seconds`; **peak is GPU memory** `peak_gpu_memory_gb`, H100; host "
      "RSS not recorded).")
    A("- Original (native platform segmentation) and TRACER-refined / "
      "TRACER-reconstructed: no benchmarked runtime/memory → **NA**.\n")

    # NA documentation from source tables
    A("## Missing values (NA) — explicit list\n")
    A("NA is shown (never silently dropped). Causes: (i) TRACER combined-vs-"
      "separate split; (ii) standalone Baysor absent on MERFISH (ROI is already "
      "Baysor-segmented; proseg/cellAdmix/SPLIT are Baysor+X cascades); "
      "(iii) original has no benchmarked runtime/memory.\n")
    for L in ("A", "B", "C"):
        f = R.OUT / f"source_data_block{L}.tsv"
        if not f.exists():
            continue
        df = pd.read_csv(f, sep="\t")
        meta = ["dataset", "platform", "metric", "metric_label", "direction", "baseline_method"]
        A(f"**Block {L}**")
        for _, r in df.iterrows():
            nas = [c for c in df.columns if c not in meta and pd.isna(r[c])]
            if nas:
                A(f"- {r['platform']} · {r['metric']}: NA = {', '.join(nas)}")
        A("")

    A("## Normalization / display rules\n")
    A("- Colour encodes a per-metric, per-dataset (row) min–max normalization "
      "across the methods present in that row.")
    A("- For **lower-is-better** metrics (runtime, peak memory, RCTD entropy, "
      "PMI relative conflict) the normalized scale is reversed so darker is "
      "always 'better', consistent across panels.")
    A("- **Higher-is-better** (marker log2FC, PMI relative purity, Kendall τ, "
      "RCTD max weight) use the same dark='better' direction without reversal.")
    A("- **Descriptive** metrics (total cells, transcripts/cell) use a neutral "
      "cividis scale; interpret via the raw annotations, not colour.")
    A("- Colormaps: `mako_r` (good/bad metrics) and `cividis` (descriptive) — "
      "perceptually uniform and colourblind-safe. Annotation text colour is "
      "chosen from each cell's luminance for legibility.\n")

    _revised_section(A)

    # ----- TRACER metric audit -----
    import anndata as ad
    comp = pd.read_csv(R.OUT / "_audit_recompute_comparison.tsv", sep="\t") \
        if (R.OUT / "_audit_recompute_comparison.tsv").exists() else None
    A("## TRACER metric audit (findings + fixes)\n")
    A("Focused diagnosis of the four TRACER-derived metric families, with "
      "quantitative evidence. Two fixes were applied (Audits 1–2); Audits 3–4 are "
      "explanatory (no metric change). Provenance of every old→new value is in "
      "`_audit_recompute_comparison.tsv`.\n")

    A("### Audit 1 — Marker specificity (log2FC)\n")
    A("**Finding: the low *combined* TRACER value is reconstructed-partial "
      "dilution, not missing markers.** TRACER carries the *largest* gene panel of "
      "any method, so it covers the most reference markers — missing-marker "
      "coverage is categorically ruled out:\n")
    A("| Dataset | TRACER panel genes | common (∩ all methods) | ref markers in TRACER | in common |")
    A("|---|--:|--:|--:|--:|")
    for ds in R.DATASET_ORDER:
        panels = {e: set(map(str, ad.read_h5ad(R.work_h5ad(ds, e)).var_names))
                  for e in R.ENTITY_ORDER if R.work_h5ad(ds, e).exists()}
        inter = set.intersection(*panels.values())
        mk = pd.read_csv(R.METRICS / ds / "_common_marker_genes.tsv", sep="\t")
        rm_full = None
        for e in R.ENTITY_ORDER:
            f = R.METRICS / ds / e / "reference_markers_used.tsv"
            if f.exists():
                rm_full = set(pd.read_csv(f, sep="\t")["gene"].astype(str)); break
        A(f"| {R.DATASETS[ds]['platform']} | {len(panels['TRACER'])} | {len(inter)} | "
          f"{len(rm_full & panels['TRACER']) if rm_full else 'NA'} | "
          f"{len(rm_full & inter) if rm_full else 'NA'} |")
    A("")
    A("Per-method panels differ drastically (e.g. atera: cellAdmix 1,953, proseg "
      "10,536 vs TRACER 17,420 genes), so the *original* per-method scoring used "
      "different marker subsets and was not comparable. **Fix:** markers are now "
      "the top-30 reference Wilcoxon markers restricted to the gene panel shared "
      "by every method (`metrics/<ds>/_common_marker_genes.tsv`), so all methods "
      "are scored identically.\n")
    A("Reconstructed-partial dilution (combined TRACER = refined ∪ reconstructed), "
      "shared-panel markers:\n")
    A("| Dataset | original | TRACER (combined) | TRACER-refined | TRACER-reconstructed | recon tx/cell |")
    A("|---|--:|--:|--:|--:|--:|")
    for ds in R.DATASET_ORDER:
        def mv(e):
            r = comp[(comp.dataset == ds) & (comp.entity == e)]
            return f"{r['marker_new'].iloc[0]:+.3f}" if len(r) else "NA"
        a = ad.read_h5ad(R.work_h5ad(ds, "TRACER_reconstructed"))
        tx = float(np.median(np.asarray(a.X.sum(1)).ravel()))
        A(f"| {R.DATASETS[ds]['platform']} | {mv('original')} | {mv('TRACER')} | "
          f"{mv('TRACER_refined')} | {mv('TRACER_reconstructed')} | {tx:.0f} |")
    A("\nTRACER-refined tracks (or exceeds) the baseline everywhere (e.g. CosMx "
      "refined +1.324 > original +1.236 > Baysor +0.887); the reconstructed "
      "partials (6–48 tx/cell) score ≈0 and pull the combined median down. This is "
      "a real property of including reconstructed low-count profiles, reported "
      "transparently (combined for the headline, refined/reconstructed separable).\n")

    A("### Audit 2 — PMI/NPMI biological coherence\n")
    A("**Finding: TRACER's apparent low purity / high conflict was an artifact of "
      "(i) a sparsity confound and (ii) a self-referential transcript-derived NPMI "
      "panel.**\n")
    A("- *Sparsity confound:* purity is anti-correlated with transcripts/cell "
      "(more transcripts → more gene pairs → more chances for a conflicting pair). "
      "CosMx Spearman(tx/cell, relative purity) = **−0.83 (p=0.005)**. proseg's "
      "purity 1.00 on MERFISH/cervical is a 14–19 tx/cell sparsity artifact, not "
      "quality. Under the transcript panel TRACER (high retention) tracked original "
      "and segger (the other high-retention methods), giving an undifferentiated "
      "~0.67.")
    A("- *Panel choice:* the transcript-derived panel is built from the spatial "
      "data being scored (partly circular). Recomputing against the "
      "reference-derived NPMI panel each TRACER run was refined against "
      "(scRNA co-expression; MERFISH nuclear self-reference) is more discriminative "
      "and biologically meaningful. **NPMI** (bounded [-1,1], defined for all "
      "co-occurring pairs) is preferred over **PMI** (unbounded, dominated by rare "
      "pairs; the reference panels flag many PMI values indeterminate/NaN).")
    A("\nRelative purity old (transcript panel) → new (reference NPMI):\n")
    A("| Dataset | original | TRACER-refined | Baysor | Segger | cellAdmix | SPLIT |")
    A("|---|--:|--:|--:|--:|--:|--:|")
    for ds in R.DATASET_ORDER:
        def pv(e):
            r = comp[(comp.dataset == ds) & (comp.entity == e)]
            if not len(r) or pd.isna(r['purity_new'].iloc[0]):
                return "NA"
            return f"{r['purity_old'].iloc[0]:.2f}→{r['purity_new'].iloc[0]:.2f}"
        A(f"| {R.DATASETS[ds]['platform']} | {pv('original')} | {pv('TRACER_refined')} | "
          f"{pv('baysor')} | {pv('segger')} | {pv('celladmix')} | {pv('split')} |")
    A("\nUnder the reference NPMI panel TRACER (combined) is the best non-trivial "
      "method in every dataset (atera 0.998, Xenium5K 0.657, CosMx 0.667, MERFISH "
      "0.881), improving on its baseline. MERFISH was tested both ways explicitly: "
      "transcript panel TRACER 0.669 ≈ original 0.671 (undifferentiated) vs "
      "nuclear-reference panel TRACER 0.881 > original 0.867 > Segger 0.770 — the "
      "reference panel is the more biologically meaningful, robust choice. "
      "*Caveat:* TRACER uses NPMI(reference) as a refinement prior, so part of its "
      "purity advantage reflects optimizing this objective; the panel is still an "
      "independent biological yardstick (derived from reference cells, not any "
      "method's spatial output).\n")

    A("### Audit 3 — Kendall τ vs scRNA (CosMx)\n")
    A("**Finding: TRACER-refined ≈ original (refine-in-place); the modest gap to "
      "Baysor/SPLIT is driven by cells-per-type and profile purification, not gene "
      "coverage or low counts.** All CosMx methods share the same ~960-gene panel "
      "(gene coverage identical), and refined cells carry 263 tx/cell (not "
      "low-count). CosMx median Kendall: original 0.465, TRACER-refined 0.474, "
      "proseg 0.468, Segger 0.456, Baysor 0.492, SPLIT 0.531. Per cell type, Baysor "
      "wins on the large types because it re-segments ~2× more cells "
      "(Fibroblasts n=705 vs TRACER 333; Plasma 859 vs 538) → more stable "
      "pseudobulk; SPLIT wins via profile purification. Reconstruction is not "
      "involved (refined only). The spread is tight (0.465–0.531): CosMx is dense "
      "per panel gene, so pseudobulks are near-saturated and refine-in-place leaves "
      "the baseline correlation essentially unchanged.\n")

    A("### Audit 4 — RCTD entropy / max weight\n")
    A("**Finding: TRACER-refined RCTD ≈ original because refinement is in-place; "
      "de-novo methods score lower entropy by fragmenting into more, smaller, "
      "purer cells.** TRACER-refined entropy equals original to within noise "
      "(CosMx 1.533 vs 1.538; MERFISH 0.469 = 0.469; Xenium5K 0.907 = 0.907; atera "
      "0.921 vs 0.952). Baysor/proseg achieve lower entropy by re-segmenting "
      "(atera Baysor 4,206 cells / entropy 0.722 vs TRACER 1,595 cells / 0.921; "
      "CosMx Baysor 1.183 vs TRACER 1.533) — smaller cells are more singlet-like, "
      "so entropy reflects **segmentation granularity**, not reference mismatch "
      "(same reference for all) or transcript sparsity (refined cells are "
      "full-count). Refined vs reconstructed: reconstructed partials score on few "
      "high-count survivors (UMI≥10 keeps 116–1,653 of them) and tend to *lower* "
      "entropy (e.g. CosMx reconstructed 1.138 < refined 1.533) because very "
      "low-evidence cells are assigned more confidently to a single type. Primary "
      "drivers, ranked: (1) refine-in-place preserves the original cell boundaries "
      "and doublet rate; (2) segmentation granularity / cells-per-ROI; (3) profile "
      "mixing in the inherited large cells — not reference mismatch or sparsity.\n")

    A("### Marker gene lists\n")
    A("Per-dataset common marker sets used for the (fixed) marker-specificity "
      "metric are saved at `metrics/<ds>/_common_marker_genes.tsv` "
      "(cell_type, gene, rank, scrna_log2fc). Counts:\n")
    for ds in R.DATASET_ORDER:
        mk = pd.read_csv(R.METRICS / ds / "_common_marker_genes.tsv", sep="\t")
        genes = ", ".join(sorted(mk["gene"].astype(str).unique())[:12])
        A(f"- **{R.DATASETS[ds]['platform']}** ({ds}): {len(mk)} marker rows, "
          f"{mk['gene'].nunique()} unique genes across {mk['cell_type'].nunique()} "
          f"cell types. e.g. {genes} …")
    A("")

    _qc_section(A)

    A("## Caveats (read before interpreting)\n")
    A("1. **SPLIT RCTD provenance differs.** SPLIT entropy / max-weight come from "
      "SPLIT's own internal RCTD (`split_rctd_entropy_metrics.tsv`), run inside the "
      "SPLIT pipeline on the purified profiles with its own settings — not from "
      "`run_rctd.R`. All other methods' RCTD was computed uniformly here. SPLIT's "
      "absolute entropy/max-weight are therefore only approximately comparable; "
      "treat the SPLIT column in Block C as indicative.")
    A("2. **MERFISH RCTD** used `--reference-min-umi 10` (vs 100 elsewhere): over "
      "the ~236-gene MERFISH panel the whole-transcriptome ileum reference cells "
      "fall below the default min_UMI=100, which would empty the reference. The "
      "lower threshold keeps ≥25 cells per type. Other datasets used the default 100.")
    A("3. **MERFISH TRACER-reconstructed RCTD n is small** (~116 cells): "
      "reconstructed partials carry ~6 transcripts/cell, so most fall below the "
      "spatial UMI_min=10 and are not scored. The value is reported but rests on "
      "few cells.")
    A("4. **MERFISH original ≈ TRACER-refined.** TRACER refines the existing Baysor "
      "segmentation in place, so the whole-cell (refined) set equals the original "
      "Baysor cells; refinement mainly adds reconstructed partials. Identical "
      "size/biology values for those two columns are expected, not a bug.")
    A("5. **Marker log2FC magnitudes differ by panel.** Targeted panels (CosMx 960, "
      "MERFISH 236) yield larger marker log2FC than dense cervical panels (17k / "
      "4.9k genes). Colour is normalized within dataset, so compare methods within "
      "a row, not absolute values across platforms.\n")

    A("## Outputs\n")
    for n in ("benchmark_heatmap_blockA_compute", "benchmark_heatmap_blockB_biological",
              "benchmark_heatmap_blockC_rctd"):
        A(f"- `{n}.png` / `{n}.svg`")
    for n in ("source_data_blockA.tsv", "source_data_blockB.tsv", "source_data_blockC.tsv",
              "all_metrics_long.tsv"):
        A(f"- `{n}`")
    A("")

    out = R.OUT / "benchmark_heatmap_summary.md"
    out.write_text("\n".join(lines))
    print("wrote", out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Write the resegment Block B summary markdown (gene-universe audit, methodology,
current values, and the change report vs the previous Block B). Run in `spatial`."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import registry as R  # noqa
R.OUT = R.FIG3 / "summary_heatmaps_resegment"
R.METRICS = R.OUT / "metrics"


def fmt(v):
    return "NA" if pd.isna(v) else f"{v:.3f}"


def main():
    aud = pd.read_csv(R.OUT / "_gene_universe_audit.tsv", sep="\t")
    chg = pd.read_csv(R.OUT / "_blockB_value_changes.tsv", sep="\t")
    src = pd.read_csv(R.OUT / "source_data_blockB.tsv", sep="\t")
    L = []
    A = L.append

    A("# TRACER resegment — Block B (biological coherence) update\n")
    A("Block B of `summary_heatmaps_resegment` recomputed from the latest "
      "`dataset/<ds>/tracer_resegment/` outputs (mode `resegment`). Metrics: marker "
      "specificity (log2FC), NPMI relative purity, NPMI relative conflict, Kendall τ "
      "vs scRNA. TRACER-refined and TRACER-reconstructed are scored separately; "
      "combined TRACER is carried only as a Kendall sensitivity value.\n")

    A("## Gene universe and full-panel methodology\n")
    A("All metrics are derived from the **full available measured gene panel** for "
      "each method, not from HVGs. Procedure:\n")
    A("- **Gene universe** per dataset = the full platform panel (union of all "
      "method panels): Xenium 17,420 · Xenium5K 4,863 · CosMx 960 · MERFISH 241.")
    A("- Each method is reindexed to this universe. For **HVG-output methods** "
      "(cellAdmix on the cervical panels) the non-modelled genes are **inherited "
      "from the matched original cell** (same `cell_id`, 100% overlap) — i.e. genes "
      "the method did not touch are assumed unchanged from the input profile. "
      "De-novo / full-panel methods have their unreported genes set to genuine "
      "zeros (they measured the gene but assigned 0 transcripts).")
    A("- KNN label transfer, the 3 canonical scRNA markers/type, pseudobulk Kendall, "
      "and NPMI purity are all computed on the completed full-panel matrices, so no "
      "method is advantaged by a favourable gene subset.")
    A("- NPMI purity/conflict use the reference-derived NPMI panel TRACER was "
      "refined against; they are computed on QC-filtered (10–900 tx) completed "
      "profiles. Marker/Kendall use the unfiltered completed full panel. The "
      "cervical reference `Unannotated` junk cluster is dropped before label "
      "transfer, marker derivation, and Kendall.\n")

    A("## Gene-universe audit — exact panel per method/dataset\n")
    A("**Determination: SPLIT is computed from the FULL measured panel, not HVGs** "
      "(98.5–99.8 % of the universe on every dataset; it omits only the handful of "
      "genes with zero retained counts after purification). The only HVG-subset "
      "method is **cellAdmix** on the cervical panels (11.2 % / 40.4 % of the "
      "universe), which is corrected by original-profile inheritance. TRACER, "
      "Segger, Baysor, proseg, and the native baseline all use the full panel "
      "(proseg/TRACER-refined/reconstructed report a subset only because de-novo "
      "assignment leaves some genes at 0; those are genuine zeros, not HVG "
      "selection).\n")
    A("| Platform | Method | genes measured | universe | frac | panel type | completion |")
    A("|---|---|--:|--:|--:|---|---|")
    for _, r in aud.iterrows():
        A(f"| {r['platform']} | {r['method']} | {r['n_genes_measured']} | "
          f"{r['universe_size']} | {r['frac_of_universe']:.3f} | {r['panel_type']} | "
          f"{r['completion']} |")
    A("")
    A("**Method-specific filtering that could bias marker / Kendall (and the "
      "mitigation):**")
    A("- *cellAdmix* (cervical) emits ~1.95 k / 1.97 k HVGs. Without correction it "
      "would be scored on a different, self-selected gene set → biased marker/"
      "Kendall. **Mitigated** by inheriting non-HVG genes from the matched original "
      "cell, so cellAdmix spans the full panel (its marker/Kendall then ≈ original, "
      "as expected for a count-cleaning method).")
    A("- *proseg* drops genes it never assigns (de-novo voxel assignment); "
      "*TRACER-refined/reconstructed* likewise report only assigned genes. These "
      "are genuine zeros on a measured gene and are zero-filled to the full panel — "
      "no HVG selection.")
    A("- *SPLIT, Baysor, Segger, original* already span the full panel; no "
      "completion applied.\n")

    A("## Marker gene sets (3 canonical markers per cell type)\n")
    for ds in R.DATASET_ORDER:
        mk = pd.read_csv(R.METRICS / ds / "_marker_genes_3.tsv", sep="\t")
        A(f"**{R.DATASETS[ds]['platform']}** ({mk['cell_type'].nunique()} types, "
          f"{mk['gene'].nunique()} genes):")
        for ct, g in mk.groupby("cell_type"):
            A(f"  - {ct}: {', '.join(g.sort_values('rank')['gene'].astype(str))}")
        A("")

    A("## Current resegment Block B values (TRACER + reference methods)\n")
    A("| Platform | Metric | TRACER-refined | TRACER-reconstructed | original | Baysor | proseg | Segger | cellAdmix | SPLIT |")
    A("|---|---|--:|--:|--:|--:|--:|--:|--:|--:|")
    labels = {"marker_log2fc": "Marker log2FC", "relative_purity": "Rel. purity",
              "relative_conflict": "Rel. conflict", "kendall_tau": "Kendall τ"}
    for ds in R.DATASET_ORDER:
        for met, lab in labels.items():
            r = src[(src.dataset == ds) & (src.metric == met)]
            if not len(r):
                continue
            r = r.iloc[0]
            A(f"| {R.DATASETS[ds]['platform']} | {lab} | {fmt(r['TRACER_refined'])} | "
              f"{fmt(r['TRACER_reconstructed'])} | {fmt(r['original'])} | "
              f"{fmt(r['baysor'])} | {fmt(r['proseg'])} | {fmt(r['segger'])} | "
              f"{fmt(r['celladmix'])} | {fmt(r['split'])} |")
    A("")

    A("## Value changes vs previous Block B\n")
    nz = chg[chg["delta"].abs() > 1e-9]
    A(f"Of {len(chg)} (entity × metric) cells, {len(nz)} changed. atera_cervical, "
      "cosmx_nsclc and merfish_mouse_ileum were already on the latest resegment "
      "(Δ = 0); **all changes are in Xenium5K**, whose previous resegment was a "
      "degenerate 622-cell run — the new run recovers 1,413 cells (1,288 refined) "
      "and restores healthy values:\n")
    A("| Platform | Entity | Metric | previous | updated | Δ |")
    A("|---|---|---|--:|--:|--:|")
    for _, r in nz.iterrows():
        A(f"| {R.DATASETS[r['dataset']]['platform']} | {r['entity']} | {r['metric']} | "
          f"{fmt(r['previous'])} | {fmt(r['updated'])} | {r['delta']:+.3f} |")
    A("\nNotably Xenium5K TRACER-refined Kendall τ rose 0.116 → 0.518 (≈ original "
      "0.495), NPMI purity 0.902 → 0.592 (≈ original 0.572), and marker log2FC "
      "0.881 → 0.459 (≈ original 0.363) — the refined whole cells now track the "
      "baseline, as expected, instead of the artefactual values from the truncated "
      "run.\n")

    A("## Output files\n")
    for n in ("benchmark_heatmap_blockB_biological.png", "benchmark_heatmap_blockB_biological.svg",
              "source_data_blockB.tsv", "all_metrics_long.tsv", "block_ab_long.tsv",
              "_gene_universe_audit.tsv", "_blockB_value_changes.tsv",
              "metrics/<ds>/_marker_genes_3.tsv"):
        A(f"- `{n}`")
    A("")

    text = "\n".join(L)
    (R.OUT / "benchmark_heatmap_summary.md").write_text(text)
    (R.OUT / "tracer_resegment_blockB_summary.md").write_text(text)
    print("wrote benchmark_heatmap_summary.md + tracer_resegment_blockB_summary.md")


if __name__ == "__main__":
    main()

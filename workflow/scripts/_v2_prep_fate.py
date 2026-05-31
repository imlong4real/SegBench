#!/usr/bin/env python3
"""v2 source-data builder: Section 3 transcript fate + gene enrichment.

Fate panel shows ONLY unassigned and removed/cleaned/pseudo-unassigned transcripts
(never assigned). Gene panel ranks genes disproportionately present among the
unassigned/removed/cleaned transcripts of each method.

Method-class semantics:
  * de novo seg. (Baysor, proseg, Segger) + original + TRACER : unassigned = tx
    whose cell_id is a sentinel (UNASSIGNED / -1).
  * cellAdmix : cleaned-to-unassigned/removed transcripts (same sentinel).
  * SPLIT     : count-level pseudo-unassigned estimates (removed cell-by-gene
    counts; not exact transcript coordinates).

Outputs:
  source_data/transcript_fate_v2.tsv
  source_data/unassigned_removed_gene_enrichment_v2.tsv
"""
from __future__ import annotations
import sys, logging
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _v2_common as C

log = logging.getLogger("v2fate")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s",
                    datefmt="%H:%M:%S")

SEG_METHODS = {"original": "original", "Baysor": "Baysor", "proseg": "proseg",
               "segger": "Segger", "cellAdmix": "cellAdmix", "TRACER": "TRACER"}


def gene_unassigned_table(method_raw):
    """Per-gene total & unassigned/cleaned counts from a transcript table."""
    df = pd.read_parquet(C.TRANSCRIPTS[method_raw], columns=["cell_id", "feature_name"])
    df["cell_id"] = df["cell_id"].astype(str)
    df["feature_name"] = df["feature_name"].astype(str)
    df = df[~df["feature_name"].str.lower().str.startswith(
        ("blank", "negcontrol", "antisense", "deprecated", "control", "unassigned"))]
    total = df.groupby("feature_name").size().rename("total_count")
    un = df[df["cell_id"].isin(C.UNASSIGNED_TOKENS)].groupby("feature_name").size().rename("n_affected")
    out = pd.concat([total, un], axis=1).fillna(0.0)
    out["fraction_affected"] = out["n_affected"] / out["total_count"].clip(lower=1)
    return out.reset_index()


def split_removed_table():
    """SPLIT per-gene removed (pseudo-unassigned) counts vs original counts."""
    import anndata as ad
    orig = ad.read_h5ad(C.ROOT / "results/benchmark_runs/tsu20/SPLIT/outputs/split_original_cell_by_gene.h5ad")
    pur = ad.read_h5ad(C.SPLIT_PURIFIED_H5AD)
    genes = [g for g in map(str, pur.var_names) if g in set(map(str, orig.var_names))]
    o = orig[:, genes]; p = pur[:, genes]
    o_sum = np.asarray(o.X.sum(axis=0)).ravel()
    # align cells for a fair removed estimate (purified subset of original)
    shared = [c for c in map(str, pur.obs_names) if c in set(map(str, orig.obs_names))]
    o2 = orig[shared, genes]; p2 = pur[shared, genes]
    removed = np.asarray(o2.X.sum(axis=0)).ravel() - np.asarray(p2.X.sum(axis=0)).ravel()
    removed = np.clip(removed, 0, None)
    tot = np.asarray(o2.X.sum(axis=0)).ravel()
    out = pd.DataFrame({"feature_name": genes, "total_count": tot, "n_affected": removed})
    out["fraction_affected"] = out["n_affected"] / np.clip(out["total_count"], 1, None)
    return out


def main():
    C.ensure_dirs()
    ta = pd.read_csv(C.SUMMARY / "transcript_assignment_summary.tsv", sep="\t")
    ta = ta.set_index("method")

    # ---- fate bars ----
    fate_rows = []
    def add(method, fate_type, count, label, note=""):
        fate_rows.append(dict(method=method, fate_type=fate_type,
                              count=(float(count) if pd.notna(count) else np.nan),
                              fate_label=label, note=note))
    add("original", "unassigned", ta.loc["original", "unassigned_transcripts"],
        "unassigned", "baseline Xenium unassigned")
    add("Baysor", "unassigned", ta.loc["Baysor", "unassigned_transcripts"],
        "unassigned", "de novo segmentation")
    add("proseg", "unassigned", ta.loc["proseg", "unassigned_transcripts"],
        "unassigned", "de novo segmentation")
    add("Segger", "unassigned", ta.loc["Segger", "unassigned_transcripts"],
        "unassigned", "de novo segmentation (GPU)")
    add("cellAdmix", "removed_cleaned", ta.loc["cellAdmix", "removed_or_cleaned_transcripts"],
        "cleaned-to-unassigned/removed", "transcript cleaning on original Xenium cell IDs")
    add("SPLIT", "removed_cleaned", ta.loc["SPLIT", "removed_or_cleaned_transcripts"],
        "removed (count-level)", "cell-level count estimate; not exact transcript coords")
    add("SPLIT", "pseudo_unassigned", ta.loc["SPLIT", "pseudo_unassigned_transcripts"],
        "pseudo-unassigned (count-level estimate)",
        "count-level pseudo-unassigned; not exact transcript coords")
    add("TRACER-refined", "unassigned", ta.loc["TRACER", "unassigned_transcripts"],
        "unassigned", "TRACER run (whole+partial share one transcript pass)")
    add("TRACER-reconstructed", "unassigned", ta.loc["TRACER", "unassigned_transcripts"],
        "unassigned", "same TRACER transcript pass as TRACER-refined")
    fate = pd.DataFrame(fate_rows)
    fate["method"] = pd.Categorical(fate["method"], C.METHOD_ORDER, ordered=True)
    fate = fate.sort_values(["method", "fate_type"])
    C.save_source(fate, "transcript_fate_v2.tsv")

    # ---- gene enrichment ----
    enr = []
    for raw, disp in SEG_METHODS.items():
        if disp == "TRACER":
            disp = "TRACER"
        t = gene_unassigned_table(raw)
        t["method"] = ("TRACER-refined" if disp == "TRACER" else disp)
        enr.append(t)
    s = split_removed_table(); s["method"] = "SPLIT"
    enr.append(s)
    allg = pd.concat(enr, ignore_index=True)

    # top-50 per method by fraction (require some absolute support)
    top_rows = []
    for m, sub in allg.groupby("method"):
        sub = sub[sub["total_count"] >= 20]
        sub = sub.sort_values(["fraction_affected", "n_affected"], ascending=False).head(50).copy()
        sub["rank"] = range(1, len(sub) + 1)
        top_rows.append(sub)
    top = pd.concat(top_rows, ignore_index=True)

    # interpretable lineage markers (45) overlay flag
    lin = pd.read_csv(C.SRCDIR / "lineage_marker_genes_45_v2.tsv", sep="\t")
    marker_set = set(lin["gene"])
    allg["is_lineage_marker"] = allg["feature_name"].isin(marker_set)
    top["is_lineage_marker"] = top["feature_name"].isin(marker_set)

    # also export per-method fraction/count for the 45 interpretable markers (for the dotplot)
    dot = allg[allg["feature_name"].isin(marker_set)].copy()
    dot = dot.merge(lin[["gene", "cell_type"]].rename(columns={"gene": "feature_name",
                                                              "cell_type": "marker_lineage"}),
                    on="feature_name", how="left")

    out = pd.concat([
        top.assign(table="top50_per_method"),
        dot.assign(table="lineage_marker_overlay"),
    ], ignore_index=True)
    cols = ["table", "method", "feature_name", "marker_lineage", "n_affected",
            "total_count", "fraction_affected", "rank", "is_lineage_marker"]
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    C.save_source(out[cols], "unassigned_removed_gene_enrichment_v2.tsv")

    log.info("fate rows=%d; enrichment genes (top50 union)=%d; dot markers=%d",
             len(fate), top["feature_name"].nunique(), dot["feature_name"].nunique())
    (C.FIGDIR / "_v2_fate_receipt.txt").write_text(
        f"fate_rows={len(fate)} top50_methods={top['method'].nunique()} "
        f"dot_markers={dot['feature_name'].nunique()}")
    print("DONE fate")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""v2 source-data builder: cell-level panels.

Sections produced:
  2. runtime_memory_v2.tsv
  4. cell_count_raincloud_v2.tsv + transcripts_per_cell_distribution_v2.tsv
  5. npmi_purity_conflict_v2.tsv + npmi_purity_conflict_stats_v2.tsv
  9. rctd_entropy_maxweight_v2.tsv + rctd_entropy_maxweight_stats_v2.tsv

TRACER is split into 'TRACER-refined' (whole cells, is_partial==False) and
'TRACER-reconstructed' (partial cells, is_partial==True). The 10<n<900 transcript
filter is applied to per-cell purity/conflict (matches metrics recomputation spec).
"""
from __future__ import annotations
import json, sys, logging
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _v2_common as C

log = logging.getLogger("v2cells")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s",
                    datefmt="%H:%M:%S")

TRANSCRIPT_METHODS = ["original", "Baysor", "proseg", "segger", "cellAdmix", "TRACER"]
DISPLAY = {"original": "original", "Baysor": "Baysor", "proseg": "proseg",
           "segger": "Segger", "cellAdmix": "cellAdmix", "TRACER": "TRACER"}


# --------------------------------------------------------------------------- #
def _per_cell_tx(method):
    """Per-cell transcript counts (pre-filter) from the method transcript table.

    Returns DataFrame [cell_id, n_transcripts, group] where group is the display
    method name, except TRACER which is split into refined/reconstructed.
    """
    p = C.TRANSCRIPTS[method]
    cols = ["cell_id"]
    if method == "TRACER":
        cols = ["cell_id", "_etype"]
    df = pd.read_parquet(p, columns=cols)
    df["cell_id"] = df["cell_id"].astype(str)
    df = df[~df["cell_id"].isin(C.UNASSIGNED_TOKENS)]
    if method == "TRACER":
        df = df[df["_etype"].astype(str).isin(["cell", "partial"])]
        g = df.groupby("cell_id").agg(n_transcripts=("cell_id", "size"),
                                      etype=("_etype", "first")).reset_index()
        g["group"] = np.where(g["etype"].astype(str).eq("partial"),
                              "TRACER-reconstructed", "TRACER-refined")
        return g[["cell_id", "n_transcripts", "group"]]
    g = df.groupby("cell_id").size().rename("n_transcripts").reset_index()
    g["group"] = DISPLAY[method]
    return g


def build_counts_and_distribution():
    cc = pd.read_csv(C.SUMMARY / "cell_count_summary.tsv", sep="\t")
    cc = cc.set_index("method")
    dist_rows = []
    count_rows = []
    tracer_tx = None
    for m in TRANSCRIPT_METHODS:
        tx = _per_cell_tx(m)
        dist_rows.append(tx)
        if m == "TRACER":
            tracer_tx = tx
            for grp in ("TRACER-refined", "TRACER-reconstructed"):
                sub = tx[tx["group"] == grp]
                n_after = int(((sub.n_transcripts > C.TX_MIN) & (sub.n_transcripts < C.TX_MAX)).sum())
                count_rows.append(dict(method=grp, total_cells=int(len(sub)),
                                       n_after_filter=n_after))
        else:
            disp = DISPLAY[m]
            total = int(cc.loc[m, "n_cells_before_filter"]) if m in cc.index else int(len(tx))
            n_after = int(((tx.n_transcripts > C.TX_MIN) & (tx.n_transcripts < C.TX_MAX)).sum())
            count_rows.append(dict(method=disp, total_cells=total, n_after_filter=n_after))
    # SPLIT (cell-level): total = original segmentation cells; distribution from
    # purified (rounded) per-cell totals.
    import anndata as ad
    pur = ad.read_h5ad(C.SPLIT_PURIFIED_H5AD)
    sp_tot = np.asarray(pur.X.sum(axis=1)).ravel()
    sp_tx = pd.DataFrame({"cell_id": pur.obs_names.astype(str),
                          "n_transcripts": np.rint(sp_tot).astype(int),
                          "group": "SPLIT"})
    dist_rows.append(sp_tx)
    split_total = int(cc.loc["SPLIT", "n_cells_before_filter"]) if "SPLIT" in cc.index else int(pur.n_obs)
    n_after = int(((sp_tx.n_transcripts > C.TX_MIN) & (sp_tx.n_transcripts < C.TX_MAX)).sum())
    count_rows.append(dict(method="SPLIT", total_cells=split_total, n_after_filter=n_after))

    counts = pd.DataFrame(count_rows)
    # tidy whole/partial split annotation for TRACER
    counts["tx_min"] = C.TX_MIN
    counts["tx_max"] = C.TX_MAX
    counts["method"] = pd.Categorical(counts["method"], C.METHOD_ORDER, ordered=True)
    counts = counts.sort_values("method")
    C.save_source(counts, "cell_count_raincloud_v2.tsv")

    dist = pd.concat(dist_rows, ignore_index=True)
    C.save_source(dist, "transcripts_per_cell_distribution_v2.tsv")
    log.info("counts: %s", counts.to_dict("records"))
    return counts


# --------------------------------------------------------------------------- #
def _cellqc(method):
    df = pd.read_csv(C.CELL_QC[method], sep="\t")
    df["cell_id"] = df["cell_id"].astype(str)
    return df


def _split_per_cell_purity():
    """Recompute SPLIT per-cell relative purity/conflict on the purified matrix.

    SPLIT has no per-cell purity table; reuse get_metric._compute_purity_conflict
    (the same metrics.py relu code used for every other method)."""
    import anndata as ad, scipy.sparse as sp
    import get_cell_level_metric, get_metric
    pur = ad.read_h5ad(C.SPLIT_PURIFIED_H5AD)
    X = sp.csr_matrix(np.rint(np.asarray(pur.X.todense() if hasattr(pur.X, "todense")
                                         else pur.X)).astype(np.float64))
    a = ad.AnnData(X=X, obs=pd.DataFrame(index=pur.obs_names.astype(str)),
                   var=pd.DataFrame(index=pur.var_names.astype(str)))
    panel = get_cell_level_metric._load_npmi_panel(C.NPMI_PANEL)
    _, _, rel_pur, rel_conf, _ = get_metric._compute_purity_conflict(
        a, panel, tau=0.05, log=log)
    n_tx = np.rint(np.asarray(pur.X.sum(axis=1)).ravel()).astype(int)
    return pd.DataFrame({"cell_id": pur.obs_names.astype(str), "n_transcripts": n_tx,
                         "relative_purity": rel_pur, "relative_conflict": rel_conf,
                         "group": "SPLIT"})


def build_purity_conflict():
    rows = []
    for m in TRANSCRIPT_METHODS:
        q = _cellqc(m)[["cell_id", "n_transcripts", "relative_purity", "relative_conflict"]].copy()
        if m == "TRACER":
            groups = _per_cell_tx("TRACER").set_index("cell_id")["group"]
            q["group"] = q["cell_id"].map(groups)
            q = q[q["group"].isin(["TRACER-refined", "TRACER-reconstructed"])]
        else:
            q["group"] = DISPLAY[m]
        rows.append(q)
    rows.append(_split_per_cell_purity())
    pc = pd.concat(rows, ignore_index=True)
    # 10 < n < 900 filter
    pc = pc[(pc.n_transcripts > C.TX_MIN) & (pc.n_transcripts < C.TX_MAX)].copy()
    pc = pc.dropna(subset=["relative_purity", "relative_conflict"])
    C.save_source(pc[["group", "cell_id", "n_transcripts", "relative_purity",
                      "relative_conflict"]].rename(columns={"group": "method"}),
                  "npmi_purity_conflict_v2.tsv")

    # summary + stats vs TRACER-refined
    ref = pc[pc.group == "TRACER-refined"]
    stat_rows = []
    summ_rows = []
    for g in C.METHOD_ORDER:
        sub = pc[pc.group == g]
        if sub.empty:
            continue
        summ_rows.append(dict(method=g, n_cells=len(sub),
                              median_relative_purity=float(sub.relative_purity.median()),
                              mean_relative_purity=float(sub.relative_purity.mean()),
                              sem_relative_purity=float(sub.relative_purity.sem()),
                              median_relative_conflict=float(sub.relative_conflict.median()),
                              mean_relative_conflict=float(sub.relative_conflict.mean()),
                              sem_relative_conflict=float(sub.relative_conflict.sem())))
        if g == "TRACER-refined":
            continue
        # unpaired (independent populations, different cells) -> Mann-Whitney one-sided
        from scipy.stats import mannwhitneyu
        for metric, alt in [("relative_purity", "greater"), ("relative_conflict", "less")]:
            a = ref[metric].to_numpy(); b = sub[metric].to_numpy()
            try:
                U, p = mannwhitneyu(a, b, alternative=alt)
            except ValueError:
                U, p = np.nan, np.nan
            stat_rows.append(dict(metric=metric, comparison=f"TRACER-refined vs {g}",
                                  test="Mann-Whitney U one-sided (independent cells)",
                                  alternative=alt, n_ref=len(a), n_other=len(b),
                                  U=float(U) if U == U else np.nan, p_value=p,
                                  p_label=C.p_label(p), stars=C.p_to_stars(p)))
    C.save_source(pd.DataFrame(summ_rows), "npmi_purity_conflict_summary_v2.tsv")
    C.save_source(pd.DataFrame(stat_rows), "npmi_purity_conflict_stats_v2.tsv")
    log.info("purity/conflict cells per group: %s",
             pc.group.value_counts().to_dict())
    return summ_rows


# --------------------------------------------------------------------------- #
def build_rctd():
    def from_weights(weights_path: Path, disp: str) -> pd.DataFrame:
        w = pd.read_csv(weights_path, sep="\t")
        cell_id = w["cell_id"].astype(str) if "cell_id" in w.columns else w.index.astype(str)
        W = w.drop(columns=["cell_id"], errors="ignore").apply(pd.to_numeric, errors="coerce").fillna(0.0)
        arr = W.to_numpy(dtype=np.float64)
        arr = np.clip(arr, 0, None)
        denom = arr.sum(axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        p = arr / denom
        plogp = np.zeros_like(p)
        nz = p > 0
        plogp[nz] = p[nz] * np.log(p[nz])
        entropy = -plogp.sum(axis=1)
        return pd.DataFrame({"method": disp, "cell_id": cell_id,
                             "rctd_entropy": entropy,
                             "rctd_max_weight": p.max(axis=1)})

    rows = []
    order_map = {"original": "original", "Baysor": "Baysor", "proseg": "proseg",
                 "segger": "Segger", "cellAdmix": "cellAdmix", "SPLIT": "SPLIT",
                 "TRACER-refined": "TRACER-refined",
                 "TRACER-reconstructed": "TRACER-reconstructed"}
    for raw, disp in order_map.items():
        p = C.RCTD_ENTROPY[raw]
        if not Path(p).exists():
            log.warning("missing rctd entropy: %s", p); continue
        df = pd.read_csv(p, sep="\t")
        if "cell_id" not in df.columns or "rctd_weights_entropy" not in df.columns:
            weights = Path(p).with_name("rctd_weights_post.tsv.gz")
            if not weights.exists():
                log.warning("missing per-cell rctd weights: %s", weights); continue
            rows.append(from_weights(weights, disp))
            continue
        ent = pd.to_numeric(df["rctd_weights_entropy"], errors="coerce")
        if "max_weight" in df.columns:
            mw = pd.to_numeric(df["max_weight"], errors="coerce")
        else:
            w1 = pd.to_numeric(df["weight_first_type"], errors="coerce")
            w2 = pd.to_numeric(df["weight_second_type"], errors="coerce")
            mw = pd.concat([w1, w2], axis=1).max(axis=1)
        sub = pd.DataFrame({"method": disp, "cell_id": df["cell_id"].astype(str),
                            "rctd_entropy": ent, "rctd_max_weight": mw})
        rows.append(sub.dropna(subset=["rctd_entropy", "rctd_max_weight"]))
    rc = pd.concat(rows, ignore_index=True)
    C.save_source(rc, "rctd_entropy_maxweight_v2.tsv")

    ref = rc[rc.method == "TRACER-refined"]
    from scipy.stats import mannwhitneyu
    stat_rows, summ_rows = [], []
    for g in C.METHOD_ORDER:
        sub = rc[rc.method == g]
        if sub.empty:
            continue
        summ_rows.append(dict(method=g, n_cells=len(sub),
                              median_entropy=float(sub.rctd_entropy.median()),
                              mean_entropy=float(sub.rctd_entropy.mean()),
                              sem_entropy=float(sub.rctd_entropy.sem()),
                              median_max_weight=float(sub.rctd_max_weight.median()),
                              mean_max_weight=float(sub.rctd_max_weight.mean()),
                              sem_max_weight=float(sub.rctd_max_weight.sem())))
        if g == "TRACER-refined":
            continue
        for metric, alt in [("rctd_entropy", "less"), ("rctd_max_weight", "greater")]:
            a = ref[metric].to_numpy(); b = sub[metric].to_numpy()
            try:
                U, p = mannwhitneyu(a, b, alternative=alt)
            except ValueError:
                U, p = np.nan, np.nan
            stat_rows.append(dict(metric=metric, comparison=f"TRACER-refined vs {g}",
                                  test="Mann-Whitney U one-sided (independent cells)",
                                  alternative=alt, n_ref=len(a), n_other=len(b),
                                  U=float(U) if U == U else np.nan, p_value=p,
                                  p_label=C.p_label(p), stars=C.p_to_stars(p)))
    C.save_source(pd.DataFrame(summ_rows), "rctd_entropy_maxweight_summary_v2.tsv")
    C.save_source(pd.DataFrame(stat_rows), "rctd_entropy_maxweight_stats_v2.tsv")
    log.info("rctd cells per method: %s", rc.method.value_counts().to_dict())


# --------------------------------------------------------------------------- #
def build_runtime():
    rt = pd.read_csv(C.SUMMARY / "method_runtime_summary.tsv", sep="\t")
    disp = {"TRACER": "TRACER", "Baysor": "Baysor", "proseg": "proseg",
            "Segger": "Segger", "cellAdmix": "cellAdmix", "SPLIT": "SPLIT",
            "original": "original"}
    rows = []
    for _, r in rt.iterrows():
        m = disp.get(r["method"], r["method"])
        gpu = bool(str(r.get("gpu_used")) == "True")
        rtime = r.get("runtime_seconds")
        # TRACER row -> both refined & reconstructed share the same run
        targets = (["TRACER-refined", "TRACER-reconstructed"] if m == "TRACER" else [m])
        for t in targets:
            rows.append(dict(method=t,
                             runtime_seconds=(float(rtime) if pd.notna(rtime) else np.nan),
                             runtime_minutes=(float(rtime) / 60 if pd.notna(rtime) else np.nan),
                             peak_cpu_memory_gb=(float(r["peak_memory_gb"])
                                                 if pd.notna(r.get("peak_memory_gb")) else np.nan),
                             peak_gpu_memory_gb=(float(r["peak_gpu_memory_gb"])
                                                 if pd.notna(r.get("peak_gpu_memory_gb")) else np.nan),
                             gpu_based=gpu,
                             shared_tracer_run=(m == "TRACER")))
    out = pd.DataFrame(rows)
    out["method"] = pd.Categorical(out["method"], C.METHOD_ORDER, ordered=True)
    out = out.sort_values("method")
    C.save_source(out, "runtime_memory_v2.tsv")
    log.info("runtime rows: %d", len(out))


def main():
    C.ensure_dirs()
    build_runtime()
    counts = build_counts_and_distribution()
    summ = build_purity_conflict()
    build_rctd()
    (C.FIGDIR / "_v2_prep_cells_receipt.txt").write_text(
        "cells_prep_ok n_count_rows=%d" % len(counts))
    print("DONE cells prep")


if __name__ == "__main__":
    main()

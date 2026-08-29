#!/usr/bin/env python3
"""Unified cross-method evaluation.

Produces one comparison row per method run, with identical metric definitions
across methods wherever that is mathematically possible. Nothing here
reimplements a metric: every number comes from the same code path for every
method, which is the only way the columns are comparable.

    runtime / peak RSS      benchmark_stats.json          (segbench.stats)
    RCTD entropy, max wt    workflow/scripts/run_rctd.R   (spacexr doublet mode)
    Kendall vs scRNA        get_metric.py                 (pseudo-bulk per type)
    marker-specificity LFC  get_metric.py                 (top-N per type)
    conflict / purity       tracer.metrics                (cPMI relu)
    entity + transcript     benchmark_stats.json

WHAT IS AND IS NOT COMPARABLE
=============================
Some quantities genuinely do not mean the same thing across methods; those are
emitted as NaN with a reason in ``<column>_note`` rather than being coerced
into false equivalence:

  * SPLIT returns fractional expected counts, so it has no per-transcript
    assignment. Its transcript columns are NaN and its entity counts are
    cell-level only.
  * Bin2Cell entities are 2um BINS before cell calling and CELLS after; a bin
    count must never be compared against a cell count. ``entity_kind`` carries
    this, and the two are kept in separate columns.
  * TRACER splits entities into whole and partial cells; those are reported
    separately (``n_whole_cells`` / ``n_partial_cells`` and the matching
    mean-transcript columns) rather than pooled, because pooling them would
    make its mean-transcripts-per-profile incomparable with methods that emit
    only whole cells.
  * conflict/purity are cPMI-panel quantities and are only defined for runs
    scored against the same panel; a method scored against a different panel
    gets NaN, not a rescaled number.

RARE CELL TYPES
===============
RCTD and the marker metrics are restricted to reference cell types with at
least ``--min-reference-cells`` cells (default 50). Without this, a reference
population of a handful of cells produces an unstable pseudo-bulk profile that
dominates a median-over-celltypes summary. The dropped types are recorded in
``excluded_celltypes`` so the restriction is visible, not silent.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import REPO_ROOT

#: Reference cell types with fewer cells than this are dropped from RCTD and
#: the marker/Kendall metrics (see module docstring).
MIN_REFERENCE_CELLS = 50

RCTD_SCRIPT = REPO_ROOT / "workflow" / "scripts" / "run_rctd.R"


# ---------------------------------------------------------------------------
@dataclass
class EvalRow:
    """One method's row in the unified comparison table."""
    dataset: str
    method: str
    entity_kind: str = "cell"
    status: str = "ok"
    notes: dict[str, str] = field(default_factory=dict)
    values: dict[str, Any] = field(default_factory=dict)

    def set(self, key: str, value: Any, note: str | None = None) -> None:
        self.values[key] = value
        if note:
            self.notes[key] = note

    def na(self, key: str, why: str) -> None:
        """Mark a quantity as genuinely non-comparable for this method."""
        self.values[key] = np.nan
        self.notes[key] = why

    def to_dict(self) -> dict[str, Any]:
        out = {"dataset": self.dataset, "method": self.method,
               "entity_kind": self.entity_kind, "status": self.status}
        out.update(self.values)
        for k, v in self.notes.items():
            out[f"{k}_note"] = v
        return out


# ---------------------------------------------------------------------------
def common_celltypes(reference_h5ad: Path, celltype_col: str,
                     min_cells: int = MIN_REFERENCE_CELLS) -> tuple[list[str], list[str]]:
    """Reference cell types that are adequately represented.

    Returns (kept, dropped). Applied identically to every method so the
    RCTD/Kendall/marker comparison is over the same cell-type universe.
    """
    import anndata as ad
    a = ad.read_h5ad(reference_h5ad, backed="r")
    if celltype_col not in a.obs.columns:
        raise SystemExit(
            f"reference {reference_h5ad} has no obs column {celltype_col!r}; "
            f"available: {list(a.obs.columns)[:20]}")
    vc = a.obs[celltype_col].astype(str).value_counts()
    kept = sorted(vc[vc >= min_cells].index.tolist())
    dropped = sorted(vc[vc < min_cells].index.tolist())
    return kept, dropped


def read_stats(run_dir: Path) -> dict[str, Any]:
    p = Path(run_dir) / "benchmark_stats.json"
    return json.loads(p.read_text()) if p.exists() else {}


# ---------------------------------------------------------------------------
def entity_metrics(row: EvalRow, stats: dict, transcripts: Path | None) -> None:
    """Entity counts and mean transcripts per profile.

    TRACER reports whole and partial cells separately (see module docstring);
    every other method reports a single entity count.
    """
    ents, tx = stats.get("entities", {}), stats.get("transcripts", {})
    row.set("n_entities", ents.get("n_entities"))
    row.set("n_genes", ents.get("n_genes"))
    row.set("n_transcripts_total", tx.get("n_total"))
    row.set("n_transcripts_assigned", tx.get("n_assigned"))
    row.set("n_transcripts_unassigned", tx.get("n_unassigned"))
    row.set("frac_assigned", tx.get("frac_assigned"))

    n_assigned, n_ent = tx.get("n_assigned"), ents.get("n_entities")
    if n_assigned and n_ent:
        row.set("mean_transcripts_per_profile", float(n_assigned) / float(n_ent))
    else:
        row.set("mean_transcripts_per_profile", np.nan)

    if row.method.startswith("tracer") and transcripts and Path(transcripts).exists():
        _tracer_whole_partial(row, Path(transcripts))


def _tracer_whole_partial(row: EvalRow, transcripts: Path) -> None:
    """Split TRACER entities into whole vs partial cells.

    TRACER labels every transcript with `_etype` ("cell" / "partial" /
    "unknown"); that is the authoritative signal. Pooling whole and partial
    entities would make mean-transcripts-per-profile incomparable with methods
    that emit only whole cells, so they are reported separately.
    """
    try:
        df = pd.read_parquet(transcripts)
    except Exception as exc:
        row.notes["n_whole_cells"] = f"unreadable transcripts: {exc}"
        return
    if "_etype" not in df.columns or "cell_id" not in df.columns:
        row.notes["n_whole_cells"] = "no _etype column in TRACER output"
        return
    et = df["_etype"].astype(str)
    cid = df["cell_id"].astype(str)
    whole, part = cid[et == "cell"], cid[et == "partial"]
    n_whole, n_part = whole.nunique(), part.nunique()
    row.set("n_whole_cells", int(n_whole))
    row.set("n_partial_cells", int(n_part))
    row.set("mean_transcripts_per_whole_cell",
            float(len(whole)) / n_whole if n_whole else np.nan)
    row.set("mean_transcripts_per_partial_cell",
            float(len(part)) / n_part if n_part else np.nan)


def runtime_metrics(row: EvalRow, stats: dict) -> None:
    rt, mem = stats.get("runtime", {}), stats.get("memory", {})
    row.set("runtime_total_s", rt.get("total_seconds"))
    row.set("runtime_method_s", rt.get("method_seconds"))
    row.set("peak_rss_gb", mem.get("peak_rss_gb"))
    row.set("peak_rss_source", mem.get("source"))
    if mem.get("source") != "external_time":
        row.notes["peak_rss_gb"] = (
            "measured in-process (no external subprocess to wrap); "
            "an underestimate relative to /usr/bin/time figures")


def tracer_conflict_purity(row: EvalRow, run_dir: Path) -> None:
    """cPMI conflict/purity, when the method run emitted them."""
    for name in ("cell_scores.tsv.gz", "outputs/cell_scores.tsv.gz"):
        p = Path(run_dir) / name
        if p.exists():
            try:
                s = pd.read_csv(p, sep="\t")
                for src, dst in (("purity", "cpmi_purity"), ("conflict", "cpmi_conflict")):
                    if src in s.columns:
                        row.set(dst, float(pd.to_numeric(s[src], errors="coerce").median()))
                return
            except Exception:
                pass
    row.na("cpmi_purity", "method does not emit cPMI cell scores")
    row.na("cpmi_conflict", "method does not emit cPMI cell scores")


# ---------------------------------------------------------------------------
def run_rctd(*, cell_h5ad: Path, reference_h5ad: Path, celltype_col: str,
             outdir: Path, rscript: str, exclude_celltypes: list[str],
             cores: int = 4, reference_min_umi: int = 100,
             tag: str = "post") -> dict[str, float]:
    """Run spacexr RCTD and return median entropy / max weight.

    The same script, doublet mode and cell-type universe are used for every
    method, so entropy and max-weight are directly comparable.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [rscript, str(RCTD_SCRIPT),
           "--spatial-h5ad", str(cell_h5ad),
           "--reference-h5ad", str(reference_h5ad),
           "--reference-celltype-col", celltype_col,
           "--outdir", str(outdir),
           "--doublet-mode", "doublet",
           "--max-cores", str(cores),
           "--reference-min-umi", str(reference_min_umi)]
    if exclude_celltypes:
        cmd += ["--exclude-celltypes", ",".join(exclude_celltypes)]
    (outdir / "rctd_cmd.txt").write_text(" ".join(cmd) + "\n")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    (outdir / "rctd.log").write_text((proc.stdout or "") + "\n" + (proc.stderr or ""))
    if proc.returncode != 0:
        return {"rctd_status": f"failed(rc={proc.returncode})"}

    per_cell = outdir / f"rctd_cell_assignments_{tag}.tsv"
    if not per_cell.exists():
        cands = sorted(outdir.glob("rctd_cell_assignments_*.tsv"))
        if not cands:
            return {"rctd_status": "no per-cell output"}
        per_cell = cands[0]
    df = pd.read_csv(per_cell, sep="\t")
    return {
        "rctd_entropy_median": float(pd.to_numeric(df["entropy"], errors="coerce").median()),
        "rctd_max_weight_median": float(pd.to_numeric(df["max_weight"], errors="coerce").median()),
        "rctd_n_cells_scored": int(len(df)),
        "rctd_status": "ok",
    }


def build_table(rows: list[EvalRow]) -> pd.DataFrame:
    """Assemble the unified comparison table, ordered for reading."""
    df = pd.DataFrame([r.to_dict() for r in rows])
    lead = ["dataset", "method", "entity_kind", "status",
            "runtime_total_s", "runtime_method_s", "peak_rss_gb", "peak_rss_source",
            "n_entities", "n_whole_cells", "n_partial_cells",
            "mean_transcripts_per_profile",
            "mean_transcripts_per_whole_cell", "mean_transcripts_per_partial_cell",
            "n_transcripts_total", "n_transcripts_assigned",
            "n_transcripts_unassigned", "frac_assigned",
            "rctd_entropy_median", "rctd_max_weight_median",
            "kendall_tau_median", "marker_logfc_median",
            "cpmi_purity", "cpmi_conflict"]
    cols = [c for c in lead if c in df.columns] + \
           [c for c in df.columns if c not in lead and not c.endswith("_note")] + \
           [c for c in df.columns if c.endswith("_note")]
    return df[cols]


# ---------------------------------------------------------------------------
# Reference-consistency + marker specificity
# ---------------------------------------------------------------------------
# Both use RCTD's dominant_celltype as the spatial label. Using one label
# source for every method is what makes these columns comparable: an
# independently-tuned label transfer per method would confound the metric with
# the transfer, and RCTD is already being run for entropy/max-weight anyway.

def _pseudobulk(X, labels: pd.Series, genes: list[str],
                types: list[str]) -> pd.DataFrame:
    """Mean log1p-CPM profile per cell type (cells x genes -> types x genes)."""
    import numpy as _np
    import scipy.sparse as sp
    out = {}
    for t in types:
        idx = _np.flatnonzero((labels == t).to_numpy())
        if idx.size < 5:            # too few cells for a stable profile
            continue
        sub = X[idx]
        tot = sub.sum(axis=1)
        tot = _np.asarray(tot).ravel() if sp.issparse(sub) else _np.asarray(tot).ravel()
        tot[tot == 0] = 1.0
        cpm = (sub.multiply(1e4 / tot[:, None]) if sp.issparse(sub)
               else sub * (1e4 / tot[:, None]))
        m = _np.asarray(cpm.mean(axis=0)).ravel()
        out[t] = _np.log1p(m)
    return pd.DataFrame(out, index=genes).T


def reference_consistency(
    row: EvalRow, *, cell_h5ad: Path, rctd_per_cell: Path,
    reference_h5ad: Path, celltype_col: str, kept_types: list[str],
) -> None:
    """Kendall tau (and Pearson r) between spatial and reference pseudo-bulk.

    Computed per cell type over shared genes, then summarised by the median
    across types so one abundant type cannot dominate.
    """
    import anndata as ad
    from scipy.stats import kendalltau, pearsonr
    try:
        q = ad.read_h5ad(cell_h5ad)
        rc = pd.read_csv(rctd_per_cell, sep="\t")
        r = ad.read_h5ad(reference_h5ad)
    except Exception as exc:
        row.na("kendall_tau_median", f"inputs unreadable: {exc}")
        return

    lab = rc.set_index(rc.columns[0])["dominant_celltype"].astype(str)
    lab = lab.reindex(q.obs_names.astype(str))
    shared = [g for g in q.var_names.astype(str) if g in set(r.var_names.astype(str))]
    if len(shared) < 20:
        row.na("kendall_tau_median", f"only {len(shared)} shared genes")
        return

    types = [t for t in kept_types if (lab == t).sum() >= 5]
    if not types:
        row.na("kendall_tau_median", "no cell type reached 5 spatial cells")
        return

    qX = q[:, shared].X
    rX = r[:, shared].layers["counts"] if "counts" in r.layers else r[:, shared].X
    qb = _pseudobulk(qX, lab.reset_index(drop=True), shared, types)
    rb = _pseudobulk(rX, r.obs[celltype_col].astype(str).reset_index(drop=True),
                     shared, types)
    common = [t for t in qb.index if t in rb.index]
    if not common:
        row.na("kendall_tau_median", "no cell type present in both")
        return

    kt, pr = [], []
    for t in common:
        a, b = qb.loc[t].to_numpy(), rb.loc[t].to_numpy()
        if np.std(a) == 0 or np.std(b) == 0:
            continue
        kt.append(float(kendalltau(a, b)[0]))
        pr.append(float(pearsonr(a, b)[0]))
    if kt:
        row.set("kendall_tau_median", float(np.nanmedian(kt)))
        row.set("pearson_r_median", float(np.nanmedian(pr)))
        row.set("n_celltypes_scored", len(kt))
    else:
        row.na("kendall_tau_median", "all profiles constant")


def marker_specificity(
    row: EvalRow, *, cell_h5ad: Path, rctd_per_cell: Path,
    reference_h5ad: Path, celltype_col: str, kept_types: list[str],
    n_top: int = 30,
) -> None:
    """Median log2FC of each cell type's reference markers in the spatial data.

    Markers are chosen once from the reference (top-N by reference log2FC per
    type) and reused for every method, so the gene set is not re-tuned per
    method.
    """
    import anndata as ad
    try:
        q = ad.read_h5ad(cell_h5ad)
        rc = pd.read_csv(rctd_per_cell, sep="\t")
        r = ad.read_h5ad(reference_h5ad)
    except Exception as exc:
        row.na("marker_logfc_median", f"inputs unreadable: {exc}")
        return

    lab = pd.Series(rc.set_index(rc.columns[0])["dominant_celltype"].astype(str)) \
            .reindex(q.obs_names.astype(str)).reset_index(drop=True)
    shared = [g for g in q.var_names.astype(str) if g in set(r.var_names.astype(str))]
    types = [t for t in kept_types if (lab == t).sum() >= 5]
    if len(shared) < 20 or not types:
        row.na("marker_logfc_median", "insufficient shared genes or cell types")
        return

    rlab = r.obs[celltype_col].astype(str).reset_index(drop=True)
    rX = r[:, shared].layers["counts"] if "counts" in r.layers else r[:, shared].X
    rb = _pseudobulk(rX, rlab, shared, types)
    qb = _pseudobulk(q[:, shared].X, lab, shared, types)
    common = [t for t in qb.index if t in rb.index]
    if not common:
        row.na("marker_logfc_median", "no cell type present in both")
        return

    lfcs = []
    for t in common:
        rest = [u for u in common if u != t]
        if not rest:
            continue
        ref_lfc = (rb.loc[t] - rb.loc[rest].mean(axis=0)).sort_values(ascending=False)
        markers = [g for g in ref_lfc.index[:n_top]]
        if not markers:
            continue
        # Same markers, evaluated in the spatial data.
        spat = qb.loc[t, markers] - qb.loc[rest, markers].mean(axis=0)
        lfcs.append(float(np.nanmedian(spat.to_numpy())))
    if lfcs:
        row.set("marker_logfc_median", float(np.nanmedian(lfcs)))
        row.set("n_marker_celltypes", len(lfcs))
    else:
        row.na("marker_logfc_median", "no markers resolvable")

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

    TRACER marks fragments it could not stitch into a whole cell; pooling them
    with whole cells would make mean-transcripts-per-profile incomparable with
    methods that only emit whole cells.
    """
    try:
        df = pd.read_parquet(transcripts)
    except Exception as exc:
        row.notes["n_whole_cells"] = f"unreadable transcripts: {exc}"
        return
    col = next((c for c in ("stitched", "cell_id") if c in df.columns), None)
    if col is None:
        return
    cid = df[col].astype(str)
    assigned = cid[cid != "UNASSIGNED"]
    if not len(assigned):
        return
    try:
        import sys
        sys.path.insert(0, str(Path(REPO_ROOT).parent / "TRACER" / "src"))
        from tracer.plot import is_whole_cell_id
        whole_mask = is_whole_cell_id(assigned)
    except Exception:
        # TRACER's own predicate is authoritative; without it, fall back to the
        # documented id convention (partial fragments carry a suffix).
        whole_mask = ~assigned.str.contains(r"[_:\-](frag|part|p)\d+$", regex=True)

    whole_ids = assigned[whole_mask]
    part_ids = assigned[~whole_mask]
    n_whole, n_part = whole_ids.nunique(), part_ids.nunique()
    row.set("n_whole_cells", int(n_whole))
    row.set("n_partial_cells", int(n_part))
    row.set("mean_transcripts_per_whole_cell",
            float(len(whole_ids)) / n_whole if n_whole else np.nan)
    row.set("mean_transcripts_per_partial_cell",
            float(len(part_ids)) / n_part if n_part else np.nan)


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

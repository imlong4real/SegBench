"""Selection-bias and reference-circularity audit for the method comparison.

Three problems make the headline comparison table misleading on its own, and
this module measures all three rather than papering over them.

**Selection.** Methods do not all score the same cells. A purification method
that declines to process low-depth cells is measured on an easier population
than a method that keeps them, and every per-cell metric (RCTD entropy, max
weight, and the pseudo-bulk metrics downstream of the labels) improves simply
by dropping hard cells. `matched_metrics` recomputes each metric on the
intersection of cells that every comparable method actually scored, so the
population is held fixed.

**Circularity.** SPLIT purifies *using* RCTD against the scRNA reference, and
the benchmark then evaluates with RCTD against that same reference. Its RCTD
scores are therefore partly a measure of how well it optimised the metric it
was given, not of segmentation quality. Metrics of that family are labelled
reference-concordance rather than accuracy, and `heldout_reference` re-scores
the pseudo-bulk metrics against study-disjoint donors that the purification
step never saw.

**Identity.** Only methods that preserve the vendor cell-id space can be
matched cell-for-cell. Re-segmentation methods invent their own entities, so
`comparable_methods` reports which methods can join a matched comparison and
which genuinely cannot.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

#: Metrics computed against the same scRNA reference that purification-style
#: methods consume. These measure agreement with a reference, not correctness,
#: and a method that optimises against that reference is expected to lead.
REFERENCE_CONCORDANCE_METRICS = (
    "rctd_entropy_median",
    "rctd_max_weight_median",
    "kendall_tau_median",
    "marker_logfc_median",
)

#: Column holding the per-cell RCTD table each method was scored with.
RCTD_PER_CELL = Path("rctd") / "rctd_cell_assignments_post.tsv"


@dataclass
class Funnel:
    """Cell accounting from vendor segmentation to scored profiles."""

    method: str
    stages: list[tuple[str, int]] = field(default_factory=list)

    def add(self, label: str, n: int) -> None:
        self.stages.append((label, int(n)))

    def to_frame(self) -> pd.DataFrame:
        rows, prev = [], None
        for label, n in self.stages:
            rows.append({
                "method": self.method,
                "stage": label,
                "n_cells": n,
                "lost_vs_previous": None if prev is None else prev - n,
                "frac_of_first": n / self.stages[0][1] if self.stages else np.nan,
            })
            prev = n
        return pd.DataFrame(rows)


def read_per_cell(run_dir: Path) -> pd.DataFrame | None:
    """The cached RCTD per-cell table for one method run, or None."""
    p = Path(run_dir) / RCTD_PER_CELL
    if not p.exists():
        return None
    df = pd.read_csv(p, sep="\t")
    df[df.columns[0]] = df[df.columns[0]].astype(str)
    return df.set_index(df.columns[0])


def comparable_methods(run_root: Path, methods: Sequence[str],
                       id_reference: Iterable[str]) -> tuple[list[str], dict[str, str]]:
    """Split methods into those sharing the vendor id space and those that don't.

    A method qualifies when a non-trivial share of its scored ids are vendor
    ids. Re-segmentation methods mint their own ids and cannot be matched
    cell-for-cell at all -- that is a property of the method, not a gap in the
    data, so they are reported with a reason rather than silently dropped.
    """
    ref = set(map(str, id_reference))
    ok, why = [], {}
    for m in methods:
        pc = read_per_cell(Path(run_root) / m)
        if pc is None:
            why[m] = "no cached RCTD per-cell table"
            continue
        overlap = len(set(pc.index) & ref) / max(len(pc), 1)
        if overlap >= 0.5:
            ok.append(m)
        else:
            why[m] = (f"re-segmentation: only {overlap:.1%} of its cell ids are "
                      "vendor ids, so no cell-for-cell match exists")
    return ok, why


def matched_cell_set(run_root: Path, methods: Sequence[str]) -> list[str]:
    """Cells scored by RCTD for *every* listed method."""
    sets = []
    for m in methods:
        pc = read_per_cell(Path(run_root) / m)
        if pc is not None:
            sets.append(set(pc.index))
    if not sets:
        return []
    return sorted(set.intersection(*sets))


def per_cell_medians(run_root: Path, method: str,
                     restrict: Sequence[str] | None = None) -> dict[str, float]:
    """Median entropy / max weight, optionally over a fixed cell set."""
    pc = read_per_cell(Path(run_root) / method)
    if pc is None:
        return {}
    if restrict is not None:
        pc = pc.loc[pc.index.intersection(pd.Index(restrict))]
    if pc.empty:
        return {}
    out = {"n_cells_scored": int(len(pc))}
    for src, dst in (("entropy", "rctd_entropy_median"),
                     ("max_weight", "rctd_max_weight_median")):
        if src in pc.columns:
            out[dst] = float(pd.to_numeric(pc[src], errors="coerce").median())
    if "doublet_status" in pc.columns:
        frac = pc["doublet_status"].astype(str).value_counts(normalize=True)
        out["frac_singlet"] = float(frac.get("singlet", 0.0))
        out["frac_reject"] = float(frac.get("reject", 0.0))
    return out


def pseudobulk_metrics(*, cell_h5ad: Path, run_root: Path, method: str,
                       reference, celltype_col: str, kept_types: Sequence[str],
                       restrict: Sequence[str] | None = None,
                       n_top: int = 30) -> dict[str, float]:
    """Kendall tau and marker log2FC for one method against ``reference``.

    Identical arithmetic to :mod:`segbench.evaluate`; the additions are the
    optional cell restriction (for matched comparisons) and taking the
    reference as an already-loaded object, so a held-out subset can be passed
    in without touching disk again.
    """
    import anndata as ad
    from scipy.stats import kendalltau, pearsonr
    from .evaluate import _pseudobulk, _rctd_labels

    per_cell = Path(run_root) / method / RCTD_PER_CELL
    if not per_cell.exists():
        return {}
    q = ad.read_h5ad(cell_h5ad)
    lab = _rctd_labels(per_cell, q.obs_names).reset_index(drop=True)

    if restrict is not None:
        keep = pd.Index(q.obs_names.astype(str)).isin(set(map(str, restrict)))
        if keep.sum() == 0:
            return {}
        q, lab = q[keep], lab[keep].reset_index(drop=True)

    shared = [g for g in q.var_names.astype(str)
              if g in set(reference.var_names.astype(str))]
    if len(shared) < 20:
        return {}
    types = [t for t in kept_types if (lab == t).sum() >= 5]
    if not types:
        return {}

    rX = (reference[:, shared].layers["counts"] if "counts" in reference.layers
          else reference[:, shared].X)
    rlab = reference.obs[celltype_col].astype(str).reset_index(drop=True)
    rb = _pseudobulk(rX, rlab, shared, types)
    qb = _pseudobulk(q[:, shared].X, lab, shared, types)
    common = [t for t in qb.index if t in rb.index]
    if not common:
        return {}

    kt, pr = [], []
    for t in common:
        a, b = qb.loc[t].to_numpy(), rb.loc[t].to_numpy()
        if np.std(a) == 0 or np.std(b) == 0:
            continue
        kt.append(float(kendalltau(a, b)[0]))
        pr.append(float(pearsonr(a, b)[0]))

    lfcs = []
    for t in common:
        rest = [u for u in common if u != t]
        if not rest:
            continue
        ref_lfc = (rb.loc[t] - rb.loc[rest].mean(axis=0)).sort_values(ascending=False)
        markers = list(ref_lfc.index[:n_top])
        if not markers:
            continue
        lfcs.append(float(np.nanmedian(
            (qb.loc[t, markers] - qb.loc[rest, markers].mean(axis=0)).to_numpy())))

    out: dict[str, float] = {"n_celltypes_scored": len(kt)}
    if kt:
        out["kendall_tau_median"] = float(np.nanmedian(kt))
        out["pearson_r_median"] = float(np.nanmedian(pr))
    if lfcs:
        out["marker_logfc_median"] = float(np.nanmedian(lfcs))
        out["n_marker_celltypes"] = len(lfcs)
    return out


def heldout_reference(reference_h5ad: Path, *, split_col: str,
                      heldout_value: str, panel) -> tuple[object, dict]:
    """Load only the held-out slice of the reference, restricted to the panel.

    Returns the subset plus a description of what it contains, so the report
    can state exactly which donors or studies the evaluation used.
    """
    from .evaluate import _load_reference
    full = _load_reference(reference_h5ad, panel)
    if split_col not in full.obs.columns:
        raise KeyError(f"{split_col!r} not in reference obs")
    mask = full.obs[split_col].astype(str) == str(heldout_value)
    sub = full[mask.to_numpy()].copy()
    info = {"split_col": split_col, "heldout_value": heldout_value,
            "n_cells": int(sub.shape[0])}
    return sub, info


def describe_disjointness(reference_h5ad: Path, *, split_col: str,
                          heldout_value: str, cohort_col: str) -> dict:
    """Check the held-out slice really is cohort-disjoint from the rest.

    A held-out split that shares donors or studies with the training half is
    not held out in any useful sense, so this is verified rather than assumed.
    """
    import anndata as ad
    a = ad.read_h5ad(reference_h5ad, backed="r")
    obs = a.obs[[split_col, cohort_col]].astype(str)
    held = set(obs.loc[obs[split_col] == str(heldout_value), cohort_col])
    rest = set(obs.loc[obs[split_col] != str(heldout_value), cohort_col])
    return {"heldout_cohorts": sorted(held), "other_cohorts": sorted(rest),
            "shared_cohorts": sorted(held & rest),
            "is_disjoint": not (held & rest)}

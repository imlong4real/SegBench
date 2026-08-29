#!/usr/bin/env python3
"""The standardized benchmark statistics contract.

Every method wrapper in :mod:`segbench.methods` finishes by calling
:func:`write_benchmark_stats`, which emits **one** ``benchmark_stats.json``
plus a flat one-row ``benchmark_stats.tsv``. Those two files are the only
thing the aggregation layer (``segbench collect``) needs to read, so adding
a method never requires touching the aggregator.

Contract (``benchmark_stats.json``)
-----------------------------------
``schema_version``  str   — bumped when the contract changes.
``method``          str   — registry name (``baysor``, ``bin2cell``, ...).
``modality``        str   — ``imaging`` | ``sequencing``.
``entity_kind``     str   — what one output row *is*: ``cell`` for
                            segmentation methods, ``bin``/``spot`` for
                            binned sequencing data before cell calling.
``status``          str   — ``ok`` | ``failed``.
``runtime``         obj   — wall time, total and for the method proper.
``memory``          obj   — peak RSS, in-process and externally measured.
``entities``        obj   — how many cells/entities came out, plus genes.
``transcripts``     obj   — assigned / unassigned accounting.
``qc``              obj   — free-form, method-relevant QC (see each wrapper).
``provenance``      obj   — command, host, versions, git commit.

Two fields deserve care because they are the ones most often compared
across methods:

``runtime.method_seconds``
    Wall time of the ``run_method`` stage *only* — i.e. the external tool,
    excluding our own I/O conversion. This is the number to quote when
    comparing methods; ``runtime.total_seconds`` includes format shims that
    differ per method and would unfairly penalise tools needing conversion.

``memory.method_peak_rss_gb``
    Peak RSS of the external tool as measured by ``/usr/bin/time``. For
    in-process (pure Python) methods there is no subprocess to measure, so
    this falls back to the in-process peak and ``memory.source`` records
    which was used.
"""
from __future__ import annotations

import json
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCHEMA_VERSION = "1.0"

#: Stage name every wrapper uses for the external tool itself. ``runtime.
#: method_seconds`` and ``memory.method_peak_rss_gb`` are read from it.
METHOD_STAGE = "run_method"


def _f(x: Any) -> float | None:
    """Coerce to a JSON-safe float (NaN/inf -> None)."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if (np.isnan(v) or np.isinf(v)) else v


def transcript_accounting(
    df: pd.DataFrame | None,
    *,
    cell_col: str = "cell_id",
    unassigned_token: str = "UNASSIGNED",
    n_input: int | None = None,
) -> dict[str, Any]:
    """Assigned/unassigned accounting from a standardized transcript table.

    ``n_input`` is the transcript count *before* the method ran; when given,
    ``delta_vs_input`` shows whether the method dropped or duplicated rows
    (proseg, for instance, can emit more rows than it consumed because it
    splits transcripts across voxel layers).
    """
    if df is None or cell_col not in getattr(df, "columns", []):
        return {
            "n_total": None, "n_assigned": None, "n_unassigned": None,
            "frac_assigned": None, "n_input": n_input, "delta_vs_input": None,
        }
    cid = df[cell_col].astype(str)
    n_total = int(len(cid))
    n_assigned = int((cid != unassigned_token).sum())
    return {
        "n_total": n_total,
        "n_assigned": n_assigned,
        "n_unassigned": n_total - n_assigned,
        "frac_assigned": _f(n_assigned / n_total) if n_total else None,
        "n_input": n_input,
        "delta_vs_input": (n_total - n_input) if n_input is not None else None,
    }


def entity_accounting(
    df: pd.DataFrame | None,
    *,
    cell_col: str = "cell_id",
    gene_col: str = "feature_name",
    unassigned_token: str = "UNASSIGNED",
    entity_kind: str = "cell",
    n_entities: int | None = None,
    n_genes: int | None = None,
) -> dict[str, Any]:
    """Count output entities (cells/bins) and genes.

    Cell-level methods that never produce a transcript table (SPLIT) pass
    ``n_entities``/``n_genes`` directly instead of a dataframe.
    """
    out: dict[str, Any] = {"entity_kind": entity_kind,
                           "n_entities": n_entities, "n_genes": n_genes}
    if df is not None and cell_col in getattr(df, "columns", []):
        cid = df[cell_col].astype(str)
        assigned = cid[cid != unassigned_token]
        out["n_entities"] = int(assigned.nunique())
        if gene_col in df.columns:
            out["n_genes"] = int(df[gene_col].astype(str).nunique())
        # Median transcripts per entity is the cheapest useful size summary.
        if len(assigned):
            out["median_transcripts_per_entity"] = _f(
                assigned.value_counts().median())
    return out


def write_benchmark_stats(
    *,
    outdir: Path,
    method: str,
    modality: str,
    sample_name: str,
    timer: Any,
    transcripts: dict[str, Any] | None = None,
    entities: dict[str, Any] | None = None,
    qc: dict[str, Any] | None = None,
    status: str = "ok",
    method_version: str | None = None,
    dataset: str | None = None,
    outputs: list[str] | None = None,
    notes: str | None = None,
    extra_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write ``benchmark_stats.json`` + ``benchmark_stats.tsv`` into ``outdir``.

    ``timer`` is a :class:`segbench.common.StageTimer`; per-stage seconds and
    peak RSS are read off it, so wrappers never assemble timing by hand.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    stages = list(getattr(timer, "stages", []))
    by_stage = {s.name: _f(s.seconds) for s in stages}
    method_stage = next((s for s in reversed(stages) if s.name == METHOD_STAGE), None)

    # Peak memory: prefer the externally measured value for the tool proper,
    # and say which source we used so cross-method comparisons stay honest.
    ext = _f(getattr(method_stage, "external_max_rss_gb", None)) if method_stage else None
    inproc = _f(getattr(timer, "peak_rss_gb_observed", None))
    memory = {
        "peak_rss_gb": ext if ext is not None else inproc,
        "method_peak_rss_gb": ext if ext is not None else (
            _f(getattr(method_stage, "peak_rss_gb", None)) if method_stage else None),
        "inprocess_peak_rss_gb": inproc,
        "source": "external_time" if ext is not None else "psutil_inprocess",
    }

    stats: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "method": method,
        "modality": modality,
        "entity_kind": (entities or {}).get("entity_kind", "cell"),
        "status": status,
        "sample_name": sample_name,
        "dataset": dataset,
        "runtime": {
            "total_seconds": _f(getattr(timer, "total_seconds", None)),
            "method_seconds": _f(getattr(method_stage, "seconds", None)) if method_stage else None,
            "by_stage_seconds": by_stage,
        },
        "memory": memory,
        "entities": entities or {},
        "transcripts": transcripts or {},
        "qc": qc or {},
        "provenance": {
            "command": " ".join(sys.argv),
            "hostname": socket.gethostname(),
            "method_version": method_version,
            "python": sys.version.split()[0],
            "outputs": outputs or [],
            "notes": notes,
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **(extra_provenance or {}),
        },
    }
    (outdir / "benchmark_stats.json").write_text(json.dumps(stats, indent=2, default=str))
    flatten_stats_to_tsv(stats, outdir / "benchmark_stats.tsv")
    return stats


def flatten_stats(stats: dict[str, Any]) -> dict[str, Any]:
    """Flatten the nested stats dict into one comparable row."""
    rt, mem = stats.get("runtime", {}), stats.get("memory", {})
    ents, tx = stats.get("entities", {}), stats.get("transcripts", {})
    row = {
        "method": stats.get("method"),
        "modality": stats.get("modality"),
        "sample_name": stats.get("sample_name"),
        "dataset": stats.get("dataset"),
        "status": stats.get("status"),
        "entity_kind": stats.get("entity_kind"),
        "total_seconds": rt.get("total_seconds"),
        "method_seconds": rt.get("method_seconds"),
        "peak_rss_gb": mem.get("peak_rss_gb"),
        "method_peak_rss_gb": mem.get("method_peak_rss_gb"),
        "memory_source": mem.get("source"),
        "n_entities": ents.get("n_entities"),
        "n_genes": ents.get("n_genes"),
        "median_transcripts_per_entity": ents.get("median_transcripts_per_entity"),
        "n_transcripts_total": tx.get("n_total"),
        "n_transcripts_assigned": tx.get("n_assigned"),
        "n_transcripts_unassigned": tx.get("n_unassigned"),
        "frac_assigned": tx.get("frac_assigned"),
        "n_transcripts_input": tx.get("n_input"),
        "method_version": stats.get("provenance", {}).get("method_version"),
        "hostname": stats.get("provenance", {}).get("hostname"),
    }
    # Method-relevant QC is flattened with a qc_ prefix so every method can add
    # its own columns without colliding with the shared ones above.
    for k, v in (stats.get("qc") or {}).items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            row[f"qc_{k}"] = v
    return row


def flatten_stats_to_tsv(stats: dict[str, Any], path: Path) -> None:
    pd.DataFrame([flatten_stats(stats)]).to_csv(path, sep="\t", index=False)


def collect_stats(root: Path) -> pd.DataFrame:
    """Find every ``benchmark_stats.json`` under ``root`` and stack them."""
    rows = []
    for p in sorted(Path(root).rglob("benchmark_stats.json")):
        try:
            rows.append({**flatten_stats(json.loads(p.read_text())),
                         "run_dir": str(p.parent)})
        except Exception as exc:  # a corrupt run must not sink the summary
            rows.append({"method": None, "status": f"unreadable: {exc}",
                         "run_dir": str(p.parent)})
    return pd.DataFrame(rows)

#!/usr/bin/env python3
"""Run TRACER as a refinement layer on top of a standardized segmentation.

TRACER is expected to provide a Python module exposing one of the
following entry points (we probe them in order; first match wins):

    tracer.api.refine_transcript_assignment(...)
    tracer.refine.refine_transcript_assignment(...)
    tracer.refine_existing_segmentation(...)

Each call is expected to receive (at minimum) a DataFrame of standardized
transcripts and to return a DataFrame with an updated ``cell_id_method``
column (and optionally an ``assignment_source`` column whose values can be
"base", "tracer_seg_residual", "tracer_noseg_cascade", etc.).

If TRACER is not installed / not importable, this script falls back to a
**stub refinement** (no-op) so that the benchmark structure still passes
the smoke test. The stub clearly marks itself in ``method_info.json`` via
``extra.stub_refinement = true``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from _io_contract import (  # noqa: E402
    MethodInfo,
    load_standardized_transcripts,
    write_standardized_outputs,
    _git_commit,  # type: ignore
)


def _utcnow() -> str:
    return dt.datetime.utcnow().isoformat() + "Z"


def _maybe_import_tracer(repo_path: str | None):
    """Try to import TRACER, prepending an optional repo path to sys.path."""
    if repo_path:
        repo_path = str(Path(repo_path).resolve())
        for sub in ("", "src"):
            candidate = str(Path(repo_path) / sub) if sub else repo_path
            if os.path.isdir(candidate) and candidate not in sys.path:
                sys.path.insert(0, candidate)
    try:
        return importlib.import_module("tracer")
    except Exception as e:  # ImportError or anything raised on import
        print(f"[tracer-refine] TRACER unavailable: {type(e).__name__}: {e}", file=sys.stderr)
        return None


def _call_tracer(
    tracer_mod,
    transcripts: pd.DataFrame,
    *,
    base_method: str,
    mode: str,
    enable_seg_residual_cascade: bool,
    enable_noseg_cascade: bool,
    npmi_source: str,
    gene_panel: str | None,
    npmi_path: str | None,
    threads: int | None,
):
    """Probe TRACER for a compatible refinement entry point and call it."""
    entry_specs = [
        ("tracer.api", "refine_transcript_assignment"),
        ("tracer.refine", "refine_transcript_assignment"),
        ("tracer", "refine_existing_segmentation"),
        ("tracer", "refine_transcript_assignment"),
    ]
    fn = None
    chosen = None
    for mod_name, attr in entry_specs:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        if hasattr(mod, attr):
            fn = getattr(mod, attr)
            chosen = f"{mod_name}.{attr}"
            break
    if fn is None:
        raise RuntimeError(
            "Could not find a TRACER refinement entry point. "
            "Expected one of: tracer.api.refine_transcript_assignment, "
            "tracer.refine.refine_transcript_assignment, "
            "tracer.refine_existing_segmentation."
        )

    print(f"[tracer-refine] Using TRACER entry point: {chosen}")

    kwargs = dict(
        base_method=base_method,
        mode=mode,
        enable_seg_residual_cascade=enable_seg_residual_cascade,
        enable_noseg_cascade=enable_noseg_cascade,
        npmi_source=npmi_source,
        gene_panel=gene_panel,
        npmi_path=npmi_path,
        threads=threads,
    )
    # Be tolerant: only forward kwargs the function actually accepts.
    import inspect

    sig = inspect.signature(fn)
    accepted = set(sig.parameters.keys())
    forwarded = {k: v for k, v in kwargs.items() if k in accepted}
    return fn(transcripts, **forwarded), chosen


def _stub_refine(transcripts: pd.DataFrame) -> pd.DataFrame:
    """No-op refinement: keep the base assignment, mark assignment_source."""
    out = transcripts.copy()
    out["assignment_source"] = "stub_no_refinement"
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run TRACER refinement on a standardized segmentation output.")
    p.add_argument("--base-transcripts", required=True,
                   help="Path to standardized transcripts.parquet for the base method.")
    p.add_argument("--base-method", required=True,
                   help="Name of the base method (e.g., xenium_default, baysor, proseg, segger).")
    p.add_argument("--out-dir", required=True,
                   help="Output directory for the standardized TRACER-refined output.")
    p.add_argument("--tracer-repo-path", default=None)
    p.add_argument("--mode", default="refine_existing_segmentation")
    p.add_argument("--enable-seg-residual-cascade", action="store_true")
    p.add_argument("--enable-noseg-cascade", action="store_true")
    p.add_argument("--pmi-source", "--npmi-source", dest="pmi_source",
                   default="panel_or_reference")
    p.add_argument("--gene-panel", default=None)
    p.add_argument("--pmi-path", "--npmi-path", dest="pmi_path", default=None)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--log", default=None)
    p.add_argument("--allow-stub", action="store_true",
                   help="If TRACER is unavailable, write a no-op refined output instead of failing.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.log:
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(args.log, "w", encoding="utf-8")
        sys.stdout = log_fh
        sys.stderr = log_fh

    start = _utcnow()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    transcripts = load_standardized_transcripts(args.base_transcripts)
    # Preserve the original base assignment.
    if "cell_id_base" not in transcripts.columns:
        transcripts["cell_id_base"] = transcripts["cell_id_method"].copy()

    tracer_mod = _maybe_import_tracer(args.tracer_repo_path)
    used_stub = False
    chosen_entry: str | None = None
    if tracer_mod is None:
        if not args.allow_stub:
            raise SystemExit(
                "TRACER could not be imported and --allow-stub was not set. "
                "Set tracer.repo_path in your config to point to a TRACER checkout, "
                "or install tracer into the active environment."
            )
        print("[tracer-refine] Falling back to stub (no-op) refinement.")
        refined = _stub_refine(transcripts)
        used_stub = True
    else:
        try:
            refined, chosen_entry = _call_tracer(
                tracer_mod,
                transcripts,
                base_method=args.base_method,
                mode=args.mode,
                enable_seg_residual_cascade=args.enable_seg_residual_cascade,
                enable_noseg_cascade=args.enable_noseg_cascade,
                npmi_source=args.pmi_source,
                gene_panel=args.gene_panel,
                npmi_path=args.pmi_path,
                threads=args.threads,
            )
            if not isinstance(refined, pd.DataFrame):
                # Some TRACER builds return (df, summary_dict).
                if isinstance(refined, tuple) and len(refined) >= 1 and isinstance(refined[0], pd.DataFrame):
                    refined = refined[0]
                else:
                    raise RuntimeError(
                        "TRACER returned an unexpected object type; expected DataFrame."
                    )
        except Exception as e:
            if not args.allow_stub:
                raise
            print(f"[tracer-refine] TRACER call failed ({type(e).__name__}: {e}); using stub.")
            refined = _stub_refine(transcripts)
            used_stub = True

    # Make sure key columns are present.
    refined = refined.copy()
    if "cell_id_method" not in refined.columns:
        raise RuntimeError("TRACER output missing 'cell_id_method' column.")
    refined["method"] = f"tracer_from_{args.base_method}"
    if "assignment_source" not in refined.columns:
        refined["assignment_source"] = "tracer"

    info = MethodInfo(
        method_name=f"tracer_from_{args.base_method}",
        method_version=getattr(tracer_mod, "__version__", None) if tracer_mod else None,
        git_commit=_git_commit(args.tracer_repo_path),
        command=" ".join(sys.argv),
        input_files=[str(args.base_transcripts)],
        start_time=start,
        threads=args.threads,
        container_or_env=os.environ.get("CONDA_DEFAULT_ENV"),
        extra={
            "base_method": args.base_method,
            "mode": args.mode,
            "enable_seg_residual_cascade": args.enable_seg_residual_cascade,
            "enable_noseg_cascade": args.enable_noseg_cascade,
            "npmi_source": args.pmi_source,
            "gene_panel": args.gene_panel,
            "tracer_entry_point": chosen_entry,
            "stub_refinement": used_stub,
        },
    )

    result = write_standardized_outputs(
        args.out_dir,
        transcripts=refined,
        cells=None,
        method=f"tracer_from_{args.base_method}",
        method_info=info,
    )

    summary = {
        "base_method": args.base_method,
        "stub": used_stub,
        "entry_point": chosen_entry,
        **result,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

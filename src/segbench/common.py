#!/usr/bin/env python3
"""Shared infrastructure for the standalone method runners.

This module factors out the boilerplate that every ``run_<method>.py`` runner
needs so each runner stays thin and they all behave identically. It mirrors the
design of ``run_tracer.py`` (stage timing, peak RSS via psutil, command /
provenance receipt, runtime files, schema-validated standardized outputs) but
is shared across Baysor, proseg, cellAdmix, and SPLIT.

What lives here:
    * StageTimer        — stage wall time + peak RSS (psutil, with resource fallback)
    * file_sha1 / git_commit_hash / tool_versions
    * run_subprocess    — run an external command, tee to run.log, optionally
                          wrap with `/usr/bin/time -v` → external_time.txt and
                          parse Maximum-RSS into GB
    * standardize_transcripts — coerce any per-transcript table into the
                          benchmark transcript contract
    * validate_schema   — write schema_validation_report.json (raises on missing
                          required columns)
    * write_provenance  — runtime_memory.json, runtime_by_stage.tsv,
                          config_receipt.json, run_summary.md

Standardized transcript contract (what get_metric.py / run_ovrlpy.py expect):
    required:  x, y, feature_name, cell_id, method
    optional:  z, transcript_id, qv, overlaps_nucleus, original_cell_id,
               assignment_confidence, cleaned_status
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import resource
import shutil
import socket
import subprocess
import sys
import time
import platform as _platform
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Tokens that mean "this transcript is not assigned to a real cell".
UNASSIGNED_TOKENS: frozenset[str] = frozenset({
    "UNASSIGNED", "Unassigned", "unassigned",
    "DROP", "drop", "nan", "NaN", "None", "", "0", "-1", "NA", "<NA>",
})

REQUIRED_COLUMNS: tuple[str, ...] = ("x", "y", "feature_name", "cell_id", "method")
OPTIONAL_COLUMNS: tuple[str, ...] = (
    "z", "transcript_id", "qv", "overlaps_nucleus",
    "original_cell_id", "assignment_confidence", "cleaned_status",
)


# ===========================================================================
# Logging
# ===========================================================================
def setup_logging(outdir: Path, name: str) -> logging.Logger:
    outdir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    log.propagate = False
    if log.handlers:
        return log
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s :: %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    fh = logging.FileHandler(outdir / "run.log", mode="a"); fh.setFormatter(fmt)
    log.addHandler(sh); log.addHandler(fh)
    return log


# ===========================================================================
# Provenance helpers
# ===========================================================================
def file_sha1(path: Path, chunk: int = 1 << 20) -> str:
    path = Path(path)
    if not path.exists() or path.is_dir():
        return "n/a"
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def git_commit_hash(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def tool_versions(extra_python_pkgs: tuple[str, ...] = ()) -> dict[str, Any]:
    """Capture Python + selected package versions for the receipt."""
    import importlib.metadata as im
    pkgs: dict[str, str] = {}
    for p in ("numpy", "pandas", "pyarrow", "scipy", "anndata") + tuple(extra_python_pkgs):
        try:
            pkgs[p] = im.version(p)
        except Exception:
            pkgs[p] = "not_installed"
    return {
        "python": sys.version.split()[0],
        "platform": _platform.platform(),
        "executable": sys.executable,
        "packages": pkgs,
    }


# ===========================================================================
# Stage timing + peak RSS
# ===========================================================================
@dataclass
class StageTime:
    name: str
    seconds: float
    peak_rss_gb: float
    external_max_rss_gb: float | None = None


def _rss_gb() -> float:
    try:
        import psutil
        return float(psutil.Process().memory_info().rss) / (1024 ** 3)
    except Exception:
        try:
            r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # darwin reports bytes; linux reports kibibytes.
            return (r if sys.platform == "darwin" else r * 1024) / (1024 ** 3)
        except Exception:
            return float("nan")


class StageTimer:
    """Context-manager based stage timer; mirrors run_tracer.py's Timer."""

    def __init__(self, log: logging.Logger):
        self.log = log
        self.stages: list[StageTime] = []

    def time(self, name: str) -> "_StageCtx":
        return _StageCtx(name, self.log, self)

    def record_external(self, name: str, external_max_rss_gb: float | None) -> None:
        """Attach an externally-measured max RSS to the most recent stage of `name`."""
        for s in reversed(self.stages):
            if s.name == name:
                s.external_max_rss_gb = external_max_rss_gb
                return

    @property
    def total_seconds(self) -> float:
        return float(sum(s.seconds for s in self.stages))

    @property
    def peak_rss_gb_observed(self) -> float:
        vals = [s.peak_rss_gb for s in self.stages] + \
               [s.external_max_rss_gb for s in self.stages if s.external_max_rss_gb]
        vals = [v for v in vals if v is not None and not np.isnan(v)]
        return float(max(vals)) if vals else float("nan")


class _StageCtx:
    def __init__(self, name: str, log: logging.Logger, timer: StageTimer):
        self.name = name; self.log = log; self.timer = timer; self.t0 = 0.0
    def __enter__(self):
        self.t0 = time.perf_counter()
        self.log.info("[stage start] %s", self.name)
        return self
    def __exit__(self, *exc):
        secs = time.perf_counter() - self.t0
        rss = _rss_gb()
        self.timer.stages.append(StageTime(self.name, secs, rss))
        self.log.info("[stage done]  %s — %.2fs  peak_rss=%.2f GB", self.name, secs, rss)


# ===========================================================================
# External subprocess with /usr/bin/time -v wrapping
# ===========================================================================
# GNU `/usr/bin/time -v` reports kbytes; BSD/macOS `/usr/bin/time -l` reports bytes.
_TIME_RSS_RE_GNU = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")
_TIME_RSS_RE_BSD = re.compile(r"(\d+)\s+maximum resident set size")


def detect_external_time() -> tuple[str, str] | None:
    """Probe `/usr/bin/time` for a flavor that reports peak RSS.

    Returns ("gnu", "-v") on Linux GNU time, ("bsd", "-l") on macOS/BSD time,
    or None if neither is available. The bash builtin `time` is ignored — only
    the standalone `/usr/bin/time` binary reports max RSS.
    """
    gnu_time = "/usr/bin/time"
    if not Path(gnu_time).exists():
        return None
    try:
        probe = subprocess.run([gnu_time, "-v", "true"], capture_output=True, text=True)
        if "Maximum resident set size" in probe.stderr:
            return ("gnu", "-v")
    except Exception:
        pass
    try:
        probe = subprocess.run([gnu_time, "-l", "true"], capture_output=True, text=True)
        if "maximum resident set size" in probe.stderr:
            return ("bsd", "-l")
    except Exception:
        pass
    return None


def run_subprocess(
    cmd: list[str], *, log: logging.Logger, outdir: Path,
    external_time_name: str = "external_time.txt",
    env: dict[str, str] | None = None, cwd: Path | None = None,
) -> tuple[int, float | None]:
    """Run `cmd`, tee combined stdout/stderr into run.log, and (when a
    `/usr/bin/time` that reports peak RSS is available — GNU ``-v`` or BSD
    ``-l``) capture peak RSS into external_time.txt.

    Returns (returncode, external_max_rss_gb_or_None).
    """
    gnu_time = "/usr/bin/time"
    detected = detect_external_time()

    full_cmd = ([gnu_time, detected[1]] + cmd) if detected else cmd
    log.info("[exec] %s", " ".join(str(c) for c in full_cmd))
    proc = subprocess.run(full_cmd, capture_output=True, text=True,
                          env=env, cwd=str(cwd) if cwd else None)
    # Tee to run.log
    if proc.stdout:
        for line in proc.stdout.splitlines():
            log.info("[stdout] %s", line)
    if proc.stderr:
        for line in proc.stderr.splitlines():
            log.info("[stderr] %s", line)

    ext_rss_gb: float | None = None
    if detected:
        (outdir / external_time_name).write_text(proc.stderr)
        flavor = detected[0]
        if flavor == "gnu":
            m = _TIME_RSS_RE_GNU.search(proc.stderr)
            if m:
                ext_rss_gb = float(m.group(1)) / (1024 ** 2)  # kbytes → GB
        else:  # bsd
            m = _TIME_RSS_RE_BSD.search(proc.stderr)
            if m:
                ext_rss_gb = float(m.group(1)) / (1024 ** 3)  # bytes → GB
        if ext_rss_gb is not None:
            log.info("[exec] external max RSS = %.2f GB (%s time)", ext_rss_gb, flavor)
    return proc.returncode, ext_rss_gb


# ===========================================================================
# Standardization
# ===========================================================================
def standardize_transcripts(
    df: pd.DataFrame, *, method: str, rename: dict[str, str] | None = None,
    unassigned_extra: tuple[str, ...] = (), log: logging.Logger | None = None,
) -> pd.DataFrame:
    """Coerce an arbitrary per-transcript table into the standardized contract.

    `rename` maps source-column -> standard-column. After renaming, the table
    must contain x, y, feature_name, cell_id. Optional columns are carried
    through when present. cell_id is forced to string and unassigned tokens are
    normalized to the literal 'UNASSIGNED'.
    """
    out = df.rename(columns=dict(rename or {})).copy()
    missing = {"x", "y", "feature_name", "cell_id"} - set(out.columns)
    if missing:
        raise ValueError(
            f"standardize_transcripts: missing {sorted(missing)} after rename; "
            f"columns present: {sorted(out.columns)}"
        )

    out["method"] = method
    for c in ("x", "y"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(np.float32)
    if "z" in out.columns:
        out["z"] = pd.to_numeric(out["z"], errors="coerce").fillna(0.0).astype(np.float32)
    out["feature_name"] = out["feature_name"].astype(str)

    # Normalize cell_id to string + canonical UNASSIGNED.
    cid = out["cell_id"].astype("string")
    tokens = set(UNASSIGNED_TOKENS) | set(unassigned_extra)
    is_unassigned = cid.isna() | cid.isin(tokens)
    out["cell_id"] = cid.astype(str)
    out.loc[is_unassigned.fillna(True).to_numpy(), "cell_id"] = "UNASSIGNED"

    if "overlaps_nucleus" in out.columns:
        out["overlaps_nucleus"] = (
            pd.to_numeric(out["overlaps_nucleus"], errors="coerce")
            .fillna(0).astype(np.uint8)
        )
    if "qv" in out.columns:
        out["qv"] = pd.to_numeric(out["qv"], errors="coerce").astype(np.float32)
    if "original_cell_id" in out.columns:
        out["original_cell_id"] = out["original_cell_id"].astype("string").astype(str)

    # Order: required first, then any optional columns that are present.
    cols = [c for c in REQUIRED_COLUMNS] + \
           [c for c in OPTIONAL_COLUMNS if c in out.columns]
    cols += [c for c in out.columns if c not in cols]
    out = out[cols]
    if log is not None:
        n_assigned = int((out["cell_id"] != "UNASSIGNED").sum())
        log.info("standardized: %d transcripts, %d genes, assigned=%d (%.1f%%), "
                 "unique cells=%d",
                 len(out), out["feature_name"].nunique(), n_assigned,
                 100 * n_assigned / max(1, len(out)),
                 out.loc[out["cell_id"] != "UNASSIGNED", "cell_id"].nunique())
    return out


def validate_schema(
    df: pd.DataFrame, *, method: str, out_path: Path, in_path: Path | str,
    report_path: Path, log: logging.Logger, extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write schema_validation_report.json; raise SystemExit if invalid."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    cid = df["cell_id"].astype(str) if "cell_id" in df.columns else pd.Series([], dtype=str)
    n_total = int(len(df))
    n_assigned = int((cid != "UNASSIGNED").sum()) if n_total else 0
    report = {
        "method": method,
        "schema_valid": not missing,
        "output_path": str(out_path),
        "input_path": str(in_path),
        "required_columns": list(REQUIRED_COLUMNS),
        "optional_columns_present": [c for c in OPTIONAL_COLUMNS if c in df.columns],
        "columns_out": list(df.columns),
        "missing_columns": missing,
        "n_transcripts_total": n_total,
        "n_transcripts_assigned": n_assigned,
        "n_transcripts_unassigned": n_total - n_assigned,
        "frac_assigned": (n_assigned / n_total) if n_total else 0.0,
        "n_unique_cells": int(cid[cid != "UNASSIGNED"].nunique()) if n_total else 0,
        "cell_id_dtype": str(df["cell_id"].dtype) if "cell_id" in df.columns else "n/a",
        "feature_name_dtype": str(df["feature_name"].dtype) if "feature_name" in df.columns else "n/a",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if extra:
        report.update(extra)
    report_path.write_text(json.dumps(report, indent=2, default=str))
    if missing:
        log.error("Schema INVALID — missing required columns: %s", missing)
        raise SystemExit(f"Standardized output missing required columns: {missing}")
    log.info("Schema valid: %d transcripts, %d cells, %.1f%% assigned",
             n_total, report["n_unique_cells"], 100 * report["frac_assigned"])
    return report


# ===========================================================================
# Provenance / runtime writers
# ===========================================================================
def write_provenance(
    *, outdir: Path, method: str, sample_name: str, args: Any,
    timer: StageTimer, repo_root: Path,
    inputs: dict[str, str], outputs: list[str],
    extra_config: dict[str, Any] | None = None,
    method_version: str | None = None,
    versions: dict[str, Any] | None = None,
    runner_kind: str = "python",  # "python" | "R" | "julia/binary"
    log: logging.Logger,
    summary_extra_lines: list[str] | None = None,
    runtime_extra: dict[str, Any] | None = None,
) -> None:
    """Write runtime_memory.json, runtime_by_stage.tsv, config_receipt.json,
    run_summary.md. Shape is compatible with get_metric.py --runtime-json."""
    versions = versions or tool_versions()
    git = git_commit_hash(repo_root)
    host = socket.gethostname()
    command = " ".join(sys.argv)
    py_or_r = versions.get("python", sys.version.split()[0])

    # --- runtime_memory.json (matches run_tracer.py shape + extras) ----------
    rm = {
        "method": method,
        "sample_name": sample_name,
        "command": command,
        "hostname": host,
        "git_commit": git,
        "python_or_R_version": py_or_r,
        "method_version": method_version,
        "stages": [asdict(s) for s in timer.stages],
        "total_seconds": timer.total_seconds,
        "peak_rss_gb_observed": timer.peak_rss_gb_observed,
    }
    if runtime_extra:
        rm.update(runtime_extra)
    (outdir / "runtime_memory.json").write_text(json.dumps(rm, indent=2, default=str))

    # --- runtime_by_stage.tsv (one row per stage; §7 fields) ----------------
    rows = []
    for s in timer.stages:
        rows.append({
            "method": method,
            "sample_name": sample_name,
            "total_seconds": round(timer.total_seconds, 3),
            "peak_rss_gb_observed": round(timer.peak_rss_gb_observed, 4)
                if not np.isnan(timer.peak_rss_gb_observed) else np.nan,
            "stage": s.name,
            "stage_seconds": round(s.seconds, 3),
            "stage_peak_rss_gb": round(s.peak_rss_gb, 4) if not np.isnan(s.peak_rss_gb) else np.nan,
            "stage_external_max_rss_gb": s.external_max_rss_gb,
            "command": command,
            "hostname": host,
            "python_or_R_version": py_or_r,
            "git_commit": git,
        })
    pd.DataFrame(rows).to_csv(outdir / "runtime_by_stage.tsv", sep="\t", index=False)

    # --- config_receipt.json -------------------------------------------------
    receipt = {
        "method": method,
        "command": command,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "sample_name": sample_name,
        "runner_kind": runner_kind,
        "method_version": method_version,
        "inputs": inputs,
        "input_sha1": {k: file_sha1(Path(v)) for k, v in inputs.items() if v},
        "outputs": outputs,
        "output_sha1": {o: file_sha1(Path(o)) for o in outputs},
        "host": {"hostname": host, **versions},
        "git_commit": git,
        "config": extra_config or {},
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (outdir / "config_receipt.json").write_text(json.dumps(receipt, indent=2, default=str))

    # --- run_summary.md ------------------------------------------------------
    md = [
        f"# {method} run summary — {sample_name}",
        "",
        f"- Date (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- Method version: `{method_version}`",
        f"- Git commit: `{git}`",
        f"- Host: `{host}`  |  runner: `{runner_kind}`  |  {py_or_r}",
        f"- Command: `{command}`",
        "",
        "## Inputs",
        "",
    ]
    md += [f"- {k}: `{v}`" for k, v in inputs.items()]
    md += ["", "## Stage timing", "",
           "| stage | seconds | peak_rss_gb | external_max_rss_gb |",
           "|---|---:|---:|---:|"]
    for s in timer.stages:
        ext = "" if s.external_max_rss_gb is None else f"{s.external_max_rss_gb:.2f}"
        md.append(f"| {s.name} | {s.seconds:.2f} | {s.peak_rss_gb:.2f} | {ext} |")
    md += ["",
           f"- **Total wall time:** {timer.total_seconds:.1f} s",
           f"- **Peak RSS observed:** {timer.peak_rss_gb_observed:.2f} GB", ""]
    if summary_extra_lines:
        md += ["## Notes", ""] + summary_extra_lines + [""]
    md += ["## Outputs", ""] + [f"- `{o}`" for o in outputs]
    (outdir / "run_summary.md").write_text("\n".join(md) + "\n")
    log.info("Wrote runtime_memory.json, runtime_by_stage.tsv, config_receipt.json, run_summary.md")


# ===========================================================================
# Cell-by-gene AnnData
# ===========================================================================
def build_cell_by_gene_h5ad(
    df: pd.DataFrame, *, out_path: Path, log: logging.Logger,
    cell_col: str = "cell_id", gene_col: str = "feature_name",
    x_col: str = "x", y_col: str = "y",
):
    """Build a cells x genes counts AnnData from a standardized transcript table
    (assigned transcripts only) and write it to `out_path`. Returns the AnnData."""
    import anndata as ad
    import scipy.sparse as sp
    sub = df.loc[df[cell_col].astype(str) != "UNASSIGNED"].copy()
    sub = sub.loc[~sub[cell_col].astype(str).isin(UNASSIGNED_TOKENS)]
    cg = (sub.groupby([cell_col, gene_col], observed=True).size()
             .rename("count").reset_index())
    cell_cat = pd.Categorical(cg[cell_col].astype(str))
    gene_cat = pd.Categorical(cg[gene_col].astype(str))
    X = sp.csr_matrix(
        (cg["count"].to_numpy(np.float32), (cell_cat.codes, gene_cat.codes)),
        shape=(len(cell_cat.categories), len(gene_cat.categories)),
    )
    cells = cell_cat.categories.astype(str)
    # Per-cell centroids from transcript coordinates (for RCTD coords later).
    obs = pd.DataFrame(index=pd.Index(cells, name="cell_id"))
    if x_col in sub.columns and y_col in sub.columns:
        cen = sub.groupby(cell_col, observed=True)[[x_col, y_col]].mean().reindex(cells)
        obs["x_centroid"] = cen[x_col].to_numpy(np.float32)
        obs["y_centroid"] = cen[y_col].to_numpy(np.float32)
    var = pd.DataFrame(index=pd.Index(gene_cat.categories.astype(str), name="feature_name"))
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.layers["counts"] = X.copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_path)
    log.info("Wrote cell-by-gene h5ad: %s (%d cells x %d genes)",
             out_path, adata.n_obs, adata.n_vars)
    return adata


# ===========================================================================
# Shared CLI builder
# ===========================================================================
def add_shared_args(p) -> None:
    p.add_argument("--transcripts", required=True, type=Path,
                   help="Standardized input transcripts parquet "
                        "(x, y, feature_name, cell_id [, z, qv, ...]).")
    p.add_argument("--reference-h5ad", type=Path, default=None,
                   help="scRNA reference h5ad (used by methods that need it).")
    p.add_argument("--reference-celltype-col", default="cell_type",
                   help="obs column with reference cell-type labels.")
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--sample-name", default="TSU20")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max-transcripts", type=int, default=None,
                   help="Smoke-test helper: randomly subsample input to at most N "
                        "transcripts (seeded) before running the method.")


def prepare_outdir(outdir: Path, sentinel: Path, overwrite: bool) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    if sentinel.exists() and not overwrite:
        raise SystemExit(
            f"outdir {outdir} already contains a completed run ({sentinel}). "
            f"Pass --overwrite to replace it."
        )


def read_parquet_robust(path: Path, *, log: logging.Logger | None = None) -> pd.DataFrame:
    """Read a parquet file, falling back to the fastparquet engine.

    Files written by newer pyarrow (>=20) embed size-statistics / level
    histograms that older pyarrow readers (e.g. 19.x) fail to decode with
    "Repetition level histogram size mismatch". When pyarrow raises, we retry
    with the fastparquet engine, which is unaffected.
    """
    try:
        return pd.read_parquet(path)
    except Exception as e:  # pyarrow raises OSError on the histogram bug
        if log is not None:
            log.warning("pyarrow failed to read %s (%s: %s); retrying with "
                        "fastparquet engine.", path, type(e).__name__, str(e)[:120])
        try:
            return pd.read_parquet(path, engine="fastparquet")
        except Exception as e2:
            raise SystemExit(
                f"Could not read parquet {path}. pyarrow failed ({type(e).__name__}: "
                f"{str(e)[:120]}) and the fastparquet fallback also failed "
                f"({type(e2).__name__}: {str(e2)[:120]}). This file was likely "
                f"written by a newer pyarrow than your reader — upgrade pyarrow "
                f"(pip install -U 'pyarrow>=21') or install fastparquet."
            ) from e2


def load_input_transcripts(
    path: Path, *, log: logging.Logger, max_transcripts: int | None = None,
    seed: int = 1,
) -> pd.DataFrame:
    """Load the standardized input parquet, tolerating x_location/y_location."""
    log.info("Loading input transcripts: %s", path)
    df = read_parquet_robust(path, log=log)
    ren = {}
    if "x" not in df.columns and "x_location" in df.columns:
        ren["x_location"] = "x"
    if "y" not in df.columns and "y_location" in df.columns:
        ren["y_location"] = "y"
    if "z" not in df.columns and "z_location" in df.columns:
        ren["z_location"] = "z"
    if "feature_name" not in df.columns and "gene" in df.columns:
        ren["gene"] = "feature_name"
    if ren:
        df = df.rename(columns=ren)
    if max_transcripts is not None and len(df) > max_transcripts:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(df), size=max_transcripts, replace=False))
        df = df.iloc[idx].reset_index(drop=True)
        log.info("Subsampled input to %d transcripts (seed=%d).", len(df), seed)
    return df

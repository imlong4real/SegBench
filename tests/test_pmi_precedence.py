#!/usr/bin/env python3
"""An explicit --pmi must beat the dataset default, and be recorded as used.

A dataset config supplies a cPMI panel, and a caller may override it. If the
override is silently ignored, the run still succeeds and still writes a
receipt naming the requested file, so the substitution is invisible until
someone compares a log line against an output months later.

Run: python tests/test_pmi_precedence.py
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

PANEL_CSV = ("gene_i,gene_j,PMI,NPMI\n"
             "GENEA,GENEB,0.9,0.5\n"
             "GENEB,GENEC,-0.4,-0.3\n")


def _panel(tmp: Path, name: str) -> Path:
    p = tmp / name
    p.write_text(PANEL_CSV)
    return p


def _dataset_cfg(tmp: Path, pmi: Path) -> Path:
    cfg = tmp / "ds_precedence.yaml"
    cfg.write_text(
        "dataset:\n"
        "  name: ds_precedence\n"
        "  platform: xenium\n"
        "  modality: imaging\n"
        "sample_name: T\n"
        "inputs:\n"
        f"  transcripts: {tmp / 'tx.parquet'}\n"
        f"  pmi: {pmi}\n"
    )
    return cfg


def _dry_run_pmi(dataset_cfg: Path, outdir: Path, override: Path | None) -> str:
    """Return the pmi path the wrapper resolves, via --dry-run."""
    cmd = [sys.executable, "-m", "segbench", "run", "tracer",
           "--dataset", str(dataset_cfg), "--outdir", str(outdir), "--dry-run"]
    if override is not None:
        cmd += ["--pmi", str(override)]
    env = {"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin",
           "SEGBENCH_NO_REEXEC": "1", "HOME": str(outdir)}
    out = subprocess.run(cmd, capture_output=True, text=True, env=env,
                         cwd=str(REPO), timeout=180)
    if out.returncode != 0:
        raise AssertionError(f"dry-run failed: {out.stdout}\n{out.stderr}")
    for tok in out.stdout.split():
        if tok.startswith("pmi="):
            return tok[len("pmi="):]
    raise AssertionError(f"no pmi= in dry-run output: {out.stdout!r}")


def test_precedence() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        ds_panel = _panel(tmp, "dataset_panel.csv")
        cli_panel = _panel(tmp, "cli_panel.csv")
        cfg = _dataset_cfg(tmp, ds_panel)

        got = _dry_run_pmi(cfg, tmp / "a", None)
        assert Path(got) == ds_panel, f"dataset default not used: {got}"
        print(f"  dataset default resolves to {Path(got).name}")

        got = _dry_run_pmi(cfg, tmp / "b", cli_panel)
        assert Path(got) == cli_panel, (
            f"CLI --pmi was ignored: resolved {got}, expected {cli_panel}")
        print(f"  explicit --pmi overrides it -> {Path(got).name}")


def test_fingerprint_matches_file() -> None:
    """The receipt's digest must describe the bytes actually opened."""
    from segbench.methods.tracer import panel_fingerprint
    with tempfile.TemporaryDirectory() as td:
        p = _panel(Path(td), "panel.csv")
        fp = panel_fingerprint(p)
        assert fp["path"] == str(p.resolve())
        assert fp["sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()
        assert fp["bytes"] == p.stat().st_size
        print(f"  fingerprint sha256={fp['sha256'][:16]}... matches file")


if __name__ == "__main__":
    print("test_pmi_precedence")
    test_precedence()
    test_fingerprint_matches_file()
    print("PASS")

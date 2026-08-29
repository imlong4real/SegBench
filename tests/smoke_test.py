#!/usr/bin/env python3
"""Dependency-free smoke test for the segbench wiring.

Exercises everything that does NOT need an external tool installed:

  1. the package imports, and every registered method resolves to a module
  2. every wrapper builds its argument parser and honours --dry-run
  3. config layering (method defaults <- dataset <- user <- CLI flags)
  4. the standardized transcript contract + schema validation
  5. the benchmark_stats.json contract and the `collect` aggregation
  6. a missing external tool produces an actionable SystemExit, not a traceback

Run it with:

    python tests/smoke_test.py           # or: ./bin/segbench selftest
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FAILURES: list[str] = []
PASSES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASSES if cond else FAILURES).append(name if cond else f"{name}: {detail}")
    print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f"  — {detail}" if detail and not cond else ""))


# ---------------------------------------------------------------------------
def test_registry() -> None:
    print("\n[1] registry + module resolution")
    from segbench import registry
    expected = {"baysor", "proseg", "segger", "split", "celladmix",
                "tracer", "bin2cell", "tracer_seq"}
    check("all expected methods registered",
          expected <= set(registry.METHODS), f"missing {expected - set(registry.METHODS)}")
    for name, spec in registry.METHODS.items():
        try:
            mod = spec.load()
            check(f"{name} imports and defines main()", callable(getattr(mod, "main", None)))
        except Exception as exc:
            check(f"{name} imports", False, f"{type(exc).__name__}: {exc}")


def test_parsers() -> None:
    print("\n[2] argument parsers")
    from segbench import registry
    for name, spec in registry.METHODS.items():
        try:
            p = spec.load().build_argparser()
            opts = {a for act in p._actions for a in act.option_strings}
            missing = {"--outdir", "--config", "--dataset", "--sample-name",
                       "--overwrite", "--dry-run"} - opts
            check(f"{name} exposes the shared flags", not missing, f"missing {sorted(missing)}")
        except Exception as exc:
            check(f"{name} parser builds", False, f"{type(exc).__name__}: {exc}")


def test_config_layering(tmp: Path) -> None:
    print("\n[3] config layering")
    from segbench import config as cfgmod
    ds = tmp / "ds.yaml"
    ds.write_text("sample_name: FIXTURE\nthreads: 11\n"
                  f"inputs:\n  transcripts: {tmp / 'transcripts.parquet'}\n")
    cfg = cfgmod.build_config(method="proseg",
                              default_config="methods/proseg.yaml",
                              user_config=str(ds))
    check("method defaults loaded", cfg.get("proseg", {}).get("voxel_layers") == 4,
          f"got {cfg.get('proseg')}")
    check("user config overrides threads", cfg.get("threads") == 11, f"got {cfg.get('threads')}")

    from segbench.methods import proseg
    args = proseg.build_argparser().parse_args(
        ["--outdir", str(tmp / "out"), "--config", str(ds)])
    from segbench.methods import _base
    _base.resolve_config(args, method="proseg")
    check("config supplies sample_name", args.sample_name == "FIXTURE", f"got {args.sample_name}")
    check("--threads propagates to --nthreads", args.nthreads == 11, f"got {args.nthreads}")
    check("config supplies the input path",
          str(args.transcripts).endswith("transcripts.parquet"), f"got {args.transcripts}")

    # An explicit CLI flag must beat the config.
    args2 = proseg.build_argparser().parse_args(
        ["--outdir", str(tmp / "out"), "--config", str(ds), "--sample-name", "CLIWINS"])
    _base.resolve_config(args2, method="proseg")
    check("CLI flag beats config", args2.sample_name == "CLIWINS", f"got {args2.sample_name}")


def test_transcript_contract(tmp: Path) -> None:
    print("\n[4] standardized transcript contract")
    from segbench import common as rc
    import logging
    log = logging.getLogger("smoke"); log.addHandler(logging.NullHandler())

    raw = pd.DataFrame({
        "x_location": [1.0, 2.0, 3.0, 4.0],
        "y_location": [1.0, 2.0, 3.0, 4.0],
        "gene": ["A", "B", "A", "C"],
        "my_cell": ["c1", "", "c2", "NA"],   # blank + NA must become UNASSIGNED
    })
    std = rc.standardize_transcripts(
        raw, method="fixture",
        rename={"x_location": "x", "y_location": "y", "gene": "feature_name",
                "my_cell": "cell_id"}, log=log)
    for col in rc.REQUIRED_COLUMNS:
        check(f"required column {col!r} present", col in std.columns)
    check("unassigned tokens normalized",
          int((std.cell_id == "UNASSIGNED").sum()) == 2,
          f"got {list(std.cell_id)}")
    check("method column stamped", set(std.method) == {"fixture"})

    rep = tmp / "schema.json"
    rc.validate_schema(std, method="fixture", out_path=tmp / "o.parquet",
                       in_path="fixture", report_path=rep, log=log)
    r = json.loads(rep.read_text())
    check("schema report valid", r["schema_valid"] is True)
    check("schema report counts assignment",
          r["n_transcripts_assigned"] == 2 and r["n_transcripts_unassigned"] == 2,
          f"got {r['n_transcripts_assigned']}/{r['n_transcripts_unassigned']}")


def test_stats_contract(tmp: Path) -> None:
    print("\n[5] benchmark_stats contract + collect")
    from segbench import common as rc, stats as stx
    import logging
    log = logging.getLogger("smoke2"); log.addHandler(logging.NullHandler())

    df = pd.read_parquet(tmp / "transcripts.parquet")
    std = rc.standardize_transcripts(df, method="fixture", log=log)

    timer = rc.StageTimer(log)
    for stage in ("load_inputs", "run_method", "write_outputs"):
        with timer.time(stage):
            pass
    timer.record_external("run_method", 1.25)

    run_dir = tmp / "runs" / "fixture"
    s = stx.write_benchmark_stats(
        outdir=run_dir, method="fixture", modality="imaging",
        sample_name="FIXTURE", timer=timer, dataset="fixture_ds",
        transcripts=stx.transcript_accounting(std, n_input=len(df)),
        entities=stx.entity_accounting(std),
        qc={"example_metric": 0.5}, method_version="0.0-test")

    for key in ("schema_version", "method", "modality", "runtime", "memory",
                "entities", "transcripts", "qc", "provenance"):
        check(f"stats has {key!r}", key in s)
    check("runtime.method_seconds recorded",
          s["runtime"]["method_seconds"] is not None)
    check("external RSS preferred for peak memory",
          s["memory"]["peak_rss_gb"] == 1.25 and s["memory"]["source"] == "external_time",
          f"got {s['memory']}")
    check("entity count matches the fixture",
          s["entities"]["n_entities"] == int(
              std.loc[std.cell_id != "UNASSIGNED", "cell_id"].nunique()))
    check("assigned + unassigned == total",
          s["transcripts"]["n_assigned"] + s["transcripts"]["n_unassigned"]
          == s["transcripts"]["n_total"])
    check("benchmark_stats.json written", (run_dir / "benchmark_stats.json").exists())
    check("benchmark_stats.tsv written", (run_dir / "benchmark_stats.tsv").exists())

    # A second run so the aggregator has something to stack.
    stx.write_benchmark_stats(
        outdir=tmp / "runs" / "fixture2", method="fixture2", modality="sequencing",
        sample_name="FIXTURE", timer=timer,
        entities=stx.entity_accounting(None, entity_kind="bin", n_entities=7))
    agg = stx.collect_stats(tmp / "runs")
    check("collect finds both runs", len(agg) == 2, f"got {len(agg)}")
    check("collect flattens qc columns", "qc_example_metric" in agg.columns,
          f"cols={list(agg.columns)[:12]}")


def test_dry_runs(tmp: Path) -> None:
    print("\n[6] --dry-run through the CLI")
    env = {**__import__("os").environ, "PYTHONPATH": str(REPO / "src")}
    from segbench import registry
    for name in registry.METHODS:
        r = subprocess.run(
            [sys.executable, "-m", "segbench", "run", name, "--dry-run",
             "--outdir", str(tmp / "dry" / name)],
            capture_output=True, text=True, env=env, cwd=str(REPO))
        check(f"{name} --dry-run exits 0", r.returncode == 0,
              (r.stderr or r.stdout).strip().splitlines()[-1:] or ["no output"])


def test_missing_tool_message(tmp: Path) -> None:
    print("\n[7] missing external tool -> actionable error, not a traceback")
    env = {**__import__("os").environ, "PYTHONPATH": str(REPO / "src")}
    r = subprocess.run(
        [sys.executable, "-m", "segbench", "run", "proseg",
         "--transcripts", str(tmp / "transcripts.parquet"),
         "--outdir", str(tmp / "nodep"), "--proseg-bin", "definitely-not-installed",
         "--overwrite"],
        capture_output=True, text=True, env=env, cwd=str(REPO))
    out = r.stdout + r.stderr
    check("exits non-zero", r.returncode != 0, f"rc={r.returncode}")
    check("names the missing binary", "definitely-not-installed" in out)
    check("suggests how to fix it", "cargo install proseg" in out)
    check("no raw traceback", "Traceback (most recent call last)" not in out,
          out.strip()[-300:])


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="segbench_smoke_"))
    try:
        sys.path.insert(0, str(REPO / "tests"))
        from make_fixture import make_transcripts
        make_transcripts().to_parquet(tmp / "transcripts.parquet", index=False)

        test_registry()
        test_parsers()
        test_config_layering(tmp)
        test_transcript_contract(tmp)
        test_stats_contract(tmp)
        test_dry_runs(tmp)
        test_missing_tool_message(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{'=' * 62}")
    print(f"passed: {len(PASSES)}   failed: {len(FAILURES)}")
    for f in FAILURES:
        print(f"  FAIL {f}")
    print("=" * 62)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""``segbench`` — the single entry point for the benchmark suite.

    segbench list                     what methods exist, and can they run here
    segbench doctor [method ...]      check external dependencies in detail
    segbench run <method> [options]   run one method
    segbench suite <suite>            run a whole benchmark suite
    segbench collect <root>           aggregate every run under a directory

``run`` forwards all remaining options to the method wrapper, so
``segbench run baysor --help`` shows Baysor's own flags.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

import pandas as pd
from pathlib import Path

from . import REPO_ROOT, __version__
from . import config as cfgmod
from . import envs
from . import registry
from .registry import IMAGING, SEQUENCING


# ---------------------------------------------------------------------------
# Dependency probing
# ---------------------------------------------------------------------------
def _probe(spec: registry.MethodSpec) -> tuple[bool, list[str]]:
    """Is this method runnable here? Delegates to the environment config."""
    return envs.probe(spec.name, spec)


def _reexec_if_needed(spec: registry.MethodSpec, argv: list[str]) -> int | None:
    """Re-run this command under the method's own interpreter, if it has one.

    Methods like bin2cell and segger execute in-process inside environments the
    driver itself does not live in. Rather than make the user activate an env,
    we hand the same command to that interpreter and return its exit code.
    Returns None when no re-exec is needed.
    """
    py = envs.interpreter(spec.name)
    if py is None:
        return None
    env = dict(os.environ)
    src = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["SEGBENCH_NO_REEXEC"] = "1"          # the child must not bounce again
    cmd = [py, "-m", "segbench", "run", spec.name] + argv
    print(f"[segbench] running {spec.name} under {py}")
    return subprocess.call(cmd, env=env, cwd=str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------
def cmd_list(args: argparse.Namespace) -> int:
    rows = []
    for spec in registry.METHODS.values():
        if args.modality and spec.modality != args.modality:
            continue
        ok, missing = _probe(spec)
        rows.append((spec, ok, missing))

    for modality, title in ((IMAGING, "Imaging (molecule-resolved)"),
                            (SEQUENCING, "Sequencing (array / binned)")):
        sel = [r for r in rows if r[0].modality == modality]
        if not sel:
            continue
        print(f"\n{title}")
        print("-" * len(title))
        for spec, ok, missing in sel:
            mark = "ok " if ok else "MISS"
            print(f"  [{mark}] {spec.name:<12} {spec.kind:<13} {spec.summary}")
            if missing and args.verbose:
                print(f"{'':<21}missing: {', '.join(missing)}")
    print("\nRun one with:  segbench run <method> --help")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    names = args.methods or list(registry.METHODS)
    report = {}
    exit_code = 0
    print(f"segbench {__version__}  |  repo: {REPO_ROOT}")
    print(f"python: {sys.version.split()[0]}  ({sys.executable})\n")
    for name in names:
        spec = registry.get(name)
        ok, missing = _probe(spec)
        report[name] = {"runnable": ok, "missing": missing,
                        "external_deps": list(spec.external_deps)}
        status = "READY" if ok else "NOT READY"
        print(f"{name:<12} {status}")
        entry = envs.for_method(name)
        for key in ("python", "binary", "rscript"):
            if key in entry:
                got = envs.resolve(name, key)
                print(f"             {key:<8} {got or '(unresolved) ' + str(entry[key])}")
        if not ok:
            exit_code = 1
            for m in missing:
                print(f"             missing {m}")
            for d in spec.external_deps:
                print(f"             needs   {d}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"\nWrote {args.json}")
    return exit_code


def cmd_run(args: argparse.Namespace, rest: list[str]) -> int:
    spec = registry.get(args.method)
    if not os.environ.get("SEGBENCH_NO_REEXEC"):
        rc = _reexec_if_needed(spec, rest)
        if rc is not None:
            return rc
    mod = spec.load()
    if not hasattr(mod, "main"):
        raise SystemExit(f"Wrapper segbench.methods.{spec.module} defines no main()")
    return int(_invoke(mod, rest, spec) or 0)


def _invoke(mod, argv: list[str], spec: registry.MethodSpec):
    """Call a wrapper's main(), passing the registry name when it accepts one.

    One module can back several registry entries (``tracer`` and ``tracer_seq``
    share a wrapper), so those wrappers take a ``method=`` keyword to know which
    identity they are running under.
    """
    import inspect
    if "method" in inspect.signature(mod.main).parameters:
        return mod.main(argv, method=spec.name)
    return mod.main(argv)


def cmd_suite(args: argparse.Namespace) -> int:
    """Run every method in a suite config, then aggregate the results.

    A failing method does not abort the suite — it is recorded and the run
    continues, because a partial benchmark is far more useful than none.
    """
    suite_path = cfgmod.find_config(args.suite, "suites")
    if suite_path is None:
        raise SystemExit(f"Unknown suite {args.suite!r} (looked in configs/suites/)")
    suite = cfgmod.load_yaml(suite_path)

    outroot = Path(args.outdir) if args.outdir else cfgmod.resolve_path(
        suite.get("outdir", "benchmark_output/suite"))
    outroot.mkdir(parents=True, exist_ok=True)
    dataset = args.dataset or suite.get("dataset")

    entries = suite.get("methods") or []
    results: list[dict] = []
    for entry in entries:
        name = entry if isinstance(entry, str) else entry.get("name")
        opts = [] if isinstance(entry, str) else list(entry.get("args") or [])
        spec = registry.get(name)
        ok, missing = _probe(spec)
        run_dir = outroot / name

        if not ok and not args.force:
            print(f"[skip] {name}: missing {', '.join(missing)}")
            results.append({"method": name, "status": "skipped", "missing": missing})
            continue

        argv = ["--outdir", str(run_dir)]
        if dataset:
            argv += ["--dataset", str(dataset)]
        if args.sample_name:
            argv += ["--sample-name", args.sample_name]
        if args.overwrite:
            argv += ["--overwrite"]
        argv += opts + (args.extra or [])

        print(f"\n=== {name} ===")
        try:
            rc = _reexec_if_needed(spec, argv) if not os.environ.get("SEGBENCH_NO_REEXEC") else None
            if rc is None:
                rc = int(_invoke(spec.load(), argv, spec) or 0)
            status = "ok" if rc == 0 else f"failed(rc={rc})"
        except SystemExit as exc:
            rc = int(exc.code or 0) if isinstance(exc.code, int) else 1
            status = "ok" if rc == 0 else f"failed({exc})"
        except Exception as exc:
            rc, status = 1, f"error({type(exc).__name__}: {exc})"
            if args.traceback:
                import traceback
                traceback.print_exc()
        print(f"[{name}] {status}")
        results.append({"method": name, "status": status, "run_dir": str(run_dir)})

    (outroot / "suite_result.json").write_text(json.dumps(
        {"suite": str(suite_path), "dataset": dataset, "results": results}, indent=2))

    from .stats import collect_stats
    df = collect_stats(outroot)
    if len(df):
        df.to_csv(outroot / "benchmark_summary.tsv", sep="\t", index=False)
        print(f"\nSummary: {outroot / 'benchmark_summary.tsv'}  ({len(df)} runs)")
    n_failed = sum(1 for r in results if r["status"].startswith(("failed", "error")))
    print(f"Suite complete: {len(results)} entries, {n_failed} failed.")
    return 1 if n_failed else 0


def cmd_collect(args: argparse.Namespace) -> int:
    from .stats import collect_stats
    df = collect_stats(Path(args.root))
    if not len(df):
        print(f"No benchmark_stats.json found under {args.root}")
        return 1
    out = Path(args.out) if args.out else Path(args.root) / "benchmark_summary.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t", index=False)
    cols = [c for c in ("method", "status", "total_seconds", "method_seconds",
                        "peak_rss_gb", "n_entities", "n_transcripts_assigned",
                        "frac_assigned") if c in df.columns]
    print(df[cols].to_string(index=False))
    print(f"\nWrote {out}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Score every method run under a root and emit the comparison table + plots."""
    from . import evaluate as ev
    from . import report as rp
    from . import config as _c

    root = Path(args.root)
    run_dirs = sorted(p.parent for p in root.rglob("benchmark_stats.json"))
    if not run_dirs:
        raise SystemExit(f"No completed runs (benchmark_stats.json) under {root}")

    cfg = {}
    if args.dataset:
        ds = _c.find_config(args.dataset, "datasets")
        if ds:
            cfg = _c.load_yaml(ds)
    inputs = cfg.get("inputs", {}) if isinstance(cfg.get("inputs"), dict) else {}
    ref = args.reference_h5ad or inputs.get("reference_h5ad")
    ct_col = args.reference_celltype_col or inputs.get("reference_celltype_col")

    kept, dropped = [], []
    if ref and ct_col and Path(ref).exists():
        kept, dropped = ev.common_celltypes(Path(ref), ct_col, args.min_reference_cells)
        print(f"Reference cell types: keeping {len(kept)}, "
              f"excluding {len(dropped)} with < {args.min_reference_cells} cells")
        if dropped:
            print(f"  excluded: {', '.join(dropped)}")

    rows = []
    for d in run_dirs:
        stats = ev.read_stats(d)
        if not stats:
            continue
        row = ev.EvalRow(dataset=args.dataset or root.name,
                         method=stats.get("method", d.name),
                         entity_kind=stats.get("entity_kind", "cell"),
                         status=stats.get("status", "ok"))
        ev.runtime_metrics(row, stats)
        # TRACER writes transcripts_tracer_refined.parquet, bin2cell writes
        # *_bin_assignments.parquet; match any per-row table.
        tx = next((q for pat in ("outputs/*_standardized.parquet",
                                 "outputs/transcripts_*.parquet",
                                 "outputs/*_bin_assignments.parquet")
                   for q in sorted(d.glob(pat))), None)
        ev.entity_metrics(row, stats, tx)
        ev.tracer_conflict_purity(row, d)

        # --- reference-based metrics -----------------------------------
        # RCTD is the single label source: its dominant_celltype feeds both
        # the Kendall and the marker comparison, so no method gets its own
        # label transfer and the columns stay comparable.
        # TRACER names its file cell_by_gene_tracer.h5ad, so match the stem
        # anywhere rather than requiring it as a suffix.
        cell_h5ad = next(iter(sorted(d.glob("outputs/*cell_by_gene*.h5ad"))), None)
        rscript = envs.resolve("split", "rscript") or envs.resolve("celladmix", "rscript")
        if args.skip_rctd:
            row.na("rctd_entropy_median", "skipped (--skip-rctd)")
        elif not (ref and ct_col and cell_h5ad and rscript):
            missing = ("no cell_by_gene.h5ad" if not cell_h5ad else
                       "no Rscript configured" if not rscript else
                       "no reference/celltype column")
            row.na("rctd_entropy_median", missing)
            row.na("kendall_tau_median", missing)
            row.na("marker_logfc_median", missing)
        else:
            rdir = d / "rctd"
            per_cell = rdir / "rctd_cell_assignments_post.tsv"
            prep_json = rdir / "rctd_input_info.json"
            if not per_cell.exists():
                print(f"    running RCTD for {row.method} ...")
                # Normalise the matrix first: RCTD needs float64 integers, and
                # method outputs violate that in two different ways.
                try:
                    prep = ev.prepare_counts_for_rctd(
                        cell_h5ad, rdir / "rctd_input.h5ad", log=print)
                    prep_json.parent.mkdir(parents=True, exist_ok=True)
                    prep_json.write_text(json.dumps(prep, indent=2))
                except Exception as exc:
                    print(f"    (count normalisation failed: {exc})")
                    prep = {}
            else:
                # RCTD is cached. Re-read what it was actually given: the
                # downstream metrics must score the SAME matrix RCTD scored,
                # or a rounded run and a cached run disagree.
                prep = json.loads(prep_json.read_text()) if prep_json.exists() else {}
            for k, v in prep.items():
                row.set(k, v)
            if prep.get("rctd_input_h5ad") and Path(prep["rctd_input_h5ad"]).exists():
                cell_h5ad = Path(prep["rctd_input_h5ad"])
                res = ev.run_rctd(
                    cell_h5ad=cell_h5ad, reference_h5ad=Path(ref),
                    celltype_col=ct_col, outdir=rdir, rscript=rscript,
                    exclude_celltypes=dropped, cores=args.rctd_cores)
                for k, v in res.items():
                    row.set(k, v)
            if per_cell.exists():
                if "rctd_entropy_median" not in row.values:
                    df_rc = pd.read_csv(per_cell, sep="\t")
                    row.set("rctd_entropy_median",
                            float(pd.to_numeric(df_rc["entropy"], errors="coerce").median()))
                    row.set("rctd_max_weight_median",
                            float(pd.to_numeric(df_rc["max_weight"], errors="coerce").median()))
                ev.reference_consistency(
                    row, cell_h5ad=cell_h5ad, rctd_per_cell=per_cell,
                    reference_h5ad=Path(ref), celltype_col=ct_col, kept_types=kept)
                ev.marker_specificity(
                    row, cell_h5ad=cell_h5ad, rctd_per_cell=per_cell,
                    reference_h5ad=Path(ref), celltype_col=ct_col, kept_types=kept)
            else:
                for k in ("kendall_tau_median", "marker_logfc_median"):
                    row.na(k, "RCTD produced no per-cell table")

        row.set("run_dir", str(d))
        rows.append(row)
        print(f"  scored {row.method}")

    df = ev.build_table(rows)
    outdir = Path(args.outdir) if args.outdir else root / "summary"
    outdir.mkdir(parents=True, exist_ok=True)
    csv = outdir / "comparison_table.csv"
    df.to_csv(csv, index=False)
    print(f"\nWrote {csv}  ({len(df)} methods)")

    if not args.no_plots:
        try:
            f1 = rp.comparison_figure(df, outdir / "comparison",
                                      title=f"SegBench — {args.dataset or root.name}")
            print(f"Wrote {f1}")
            f2 = rp.runtime_memory_scatter(df, outdir / "cost_scatter")
            if f2:
                print(f"Wrote {f2}")
        except Exception as exc:
            print(f"[warn] plotting failed: {type(exc).__name__}: {exc}")
    md = rp.write_markdown_summary(df, outdir / "comparison.md",
                                   dataset=args.dataset or root.name,
                                   excluded_celltypes=dropped,
                                   min_reference_cells=args.min_reference_cells)
    print(f"Wrote {md}")
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    """Run the dependency-free wiring check in tests/smoke_test.py."""
    script = REPO_ROOT / "tests" / "smoke_test.py"
    if not script.exists():
        raise SystemExit(f"Self-test not found: {script}")
    return subprocess.call([sys.executable, str(script)], cwd=str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="segbench", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"segbench {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("list", help="list methods and whether they can run here")
    pl.add_argument("--modality", choices=(IMAGING, SEQUENCING))
    pl.add_argument("-v", "--verbose", action="store_true")
    pl.set_defaults(func=cmd_list)

    pd_ = sub.add_parser("doctor", help="check external dependencies")
    pd_.add_argument("methods", nargs="*")
    pd_.add_argument("--json", help="also write the report to this path")
    pd_.set_defaults(func=cmd_doctor)

    pr = sub.add_parser("run", help="run one method",
                        add_help=False)  # let the wrapper own --help
    pr.add_argument("method")
    pr.set_defaults(func=None)

    ps = sub.add_parser("suite", help="run a benchmark suite")
    ps.add_argument("suite", help="suite name (configs/suites/<name>.yaml) or a path")
    ps.add_argument("--outdir", default=None)
    ps.add_argument("--dataset", default=None)
    ps.add_argument("--sample-name", default=None)
    ps.add_argument("--overwrite", action="store_true")
    ps.add_argument("--force", action="store_true",
                    help="attempt methods whose dependencies look missing")
    ps.add_argument("--traceback", action="store_true")
    ps.add_argument("--extra", nargs=argparse.REMAINDER,
                    help="extra flags appended to every method invocation")
    ps.set_defaults(func=cmd_suite)

    pe = sub.add_parser("evaluate",
                        help="score all runs under a root -> comparison CSV + plots")
    pe.add_argument("root", help="directory containing method run dirs")
    pe.add_argument("--dataset", default=None, help="dataset config name")
    pe.add_argument("--outdir", default=None)
    pe.add_argument("--reference-h5ad", default=None)
    pe.add_argument("--reference-celltype-col", default=None)
    pe.add_argument("--min-reference-cells", type=int,
                    default=50, help="drop reference cell types below this "
                                       "count from RCTD/marker metrics")
    pe.add_argument("--skip-rctd", action="store_true",
                    help="skip RCTD (and the metrics that depend on its labels)")
    pe.add_argument("--rctd-cores", type=int, default=4)
    pe.add_argument("--no-plots", action="store_true")
    pe.set_defaults(func=cmd_evaluate)

    pt = sub.add_parser("selftest", help="run the dependency-free wiring check")
    pt.set_defaults(func=cmd_selftest)

    pc = sub.add_parser("collect", help="aggregate benchmark_stats.json files")
    pc.add_argument("root")
    pc.add_argument("--out", default=None)
    pc.set_defaults(func=cmd_collect)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # `run` passes everything after the method name through untouched.
    if argv and argv[0] == "run":
        if len(argv) < 2:
            raise SystemExit("segbench run: missing method name. "
                             "Try `segbench list`.")
        ns = argparse.Namespace(method=argv[1])
        return cmd_run(ns, argv[2:])
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())

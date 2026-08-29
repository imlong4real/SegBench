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
import shutil
import subprocess
import sys
from pathlib import Path

from . import REPO_ROOT, __version__
from . import config as cfgmod
from . import registry
from .registry import IMAGING, SEQUENCING


# ---------------------------------------------------------------------------
# Dependency probing
# ---------------------------------------------------------------------------
def _probe(spec: registry.MethodSpec) -> tuple[bool, list[str]]:
    """Return (runnable_here, missing) for one method.

    Probing is deliberately shallow — it imports nothing heavy. It answers
    "would this run crash immediately?", not "is every version correct?".
    """
    missing: list[str] = []
    mod_checks = {
        "baysor": [("bin", "baysor")],
        "proseg": [("bin", "proseg")],
        "segger": [("py", "torch"), ("py", "segger")],
        "split": [("bin", "Rscript")],
        "celladmix": [("bin", "Rscript")],
        "tracer": [("py", "tracer")],
        "tracer_seq": [("py", "tracer")],
        "bin2cell": [("py", "bin2cell")],
    }
    for kind, name in mod_checks.get(spec.name, []):
        if kind == "bin":
            if shutil.which(name) is None:
                missing.append(f"binary:{name}")
        else:
            try:
                __import__(name)
            except Exception:
                missing.append(f"python:{name}")
    return (not missing), missing


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

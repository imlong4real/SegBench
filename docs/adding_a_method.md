# Adding another method

Two files change: a wrapper module, and one registry entry. Nothing else in
the suite needs editing — the CLI, the suite runner, `segbench list/doctor`
and the aggregator all read the registry.

## 1. Write the wrapper

Create `src/segbench/methods/<name>.py`. The skeleton below is the whole
contract; copy `proseg.py` (the shortest real one) if you want a worked
example.

```python
#!/usr/bin/env python3
"""One-line summary, then: what the method does, what it needs, an example."""
from __future__ import annotations

import argparse
from pathlib import Path

from .. import REPO_ROOT as _REPO_ROOT
from .. import common as rc
from .. import stats as stx
from . import _base

METHOD = "mymethod"


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _base.add_common_args(p, method=METHOD)        # --outdir, --config, --dataset, ...
    _base.add_transcript_input_args(p)             # --transcripts, --reference-h5ad
    p.add_argument("--my-tuning-knob", type=float, default=0.5)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    _base.resolve_config(args, method=METHOD)      # layer in the YAML configs
    if args.dry_run:
        print(f"[dry-run] {METHOD} -> {args.outdir}")
        return 0

    outputs_dir = args.outdir / "outputs"
    std_path = outputs_dir / f"{METHOD}_transcripts_standardized.parquet"
    rc.prepare_outdir(args.outdir, std_path, args.overwrite)
    log = rc.setup_logging(args.outdir, f"segbench.{METHOD}")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    timer = rc.StageTimer(log)

    with timer.time("load_inputs"):
        df_in = rc.load_input_transcripts(
            args.transcripts, log=log,
            max_transcripts=args.max_transcripts, seed=args.seed)

    with timer.time("convert_inputs"):
        ...                                        # write the tool's input format

    with timer.time("run_method"):                 # MUST be named run_method
        code, ext_rss = rc.run_subprocess(cmd, log=log, outdir=args.outdir)
        if code != 0:
            raise SystemExit(f"mymethod failed (exit {code}); see run.log.")
    timer.record_external("run_method", ext_rss)   # outside the `with` block

    with timer.time("convert_outputs"):
        std = rc.standardize_transcripts(
            raw, method=METHOD, rename={"gene": "feature_name"}, log=log)
        std.to_parquet(std_path, index=False)

    with timer.time("validate_schema"):
        rc.validate_schema(std, method=METHOD, out_path=std_path,
                           in_path=args.transcripts, log=log,
                           report_path=args.outdir / "schema_validation_report.json")

    with timer.time("write_outputs"):
        rc.write_provenance(outdir=args.outdir, method=METHOD, args=args,
                            sample_name=args.sample_name, timer=timer,
                            repo_root=_REPO_ROOT, log=log,
                            inputs={"transcripts": str(args.transcripts)},
                            outputs=[str(std_path)])

    stx.write_benchmark_stats(
        outdir=args.outdir, method=METHOD, modality="imaging",
        sample_name=args.sample_name, timer=timer, dataset=args.dataset,
        transcripts=stx.transcript_accounting(std, n_input=len(df_in)),
        entities=stx.entity_accounting(std),
        qc={"my_tuning_knob": args.my_tuning_knob})
    return 0
```

### Five rules

1. **Name the tool's stage `run_method`.** `benchmark_stats.json` reads
   `runtime.method_seconds` and `memory.method_peak_rss_gb` from that stage
   name. Get it wrong and your method reports no comparable runtime.
2. **Call `timer.record_external(...)` *outside* the `with` block.** It scans
   completed stages, so inside the block there is nothing to attach to yet.
3. **Fail loudly.** A missing binary must raise `SystemExit` with a message
   naming the tool and how to install it — never fall back to a stub, and
   never emit fabricated outputs. `tests/smoke_test.py` asserts this.
4. **Standardize through `rc.standardize_transcripts`.** It normalises the
   unassigned token, coerces dtypes and orders columns, which is what makes
   `frac_assigned` comparable between methods.
5. **Put anything method-specific in `qc`.** Scalar keys are flattened to
   `qc_<key>` in the summary table automatically.

## 2. Register it

Add one entry to `METHODS` in `src/segbench/registry.py`:

```python
"mymethod": MethodSpec(
    name="mymethod", module="mymethod", modality=IMAGING, kind="segmentation",
    summary="One line shown by `segbench list`.",
    external_deps=("mymethod >= 1.0 (binary)",),
    default_config="methods/mymethod.yaml",
),
```

and add a probe in `_probe()` in `src/segbench/cli.py` so `segbench doctor`
can tell whether it is installed:

```python
"mymethod": [("bin", "mymethod")],       # or ("py", "mymethod") for a package
```

## 3. Add a default config

`configs/methods/mymethod.yaml`:

```yaml
method: mymethod
threads: 4
mymethod:
  my_tuning_knob: 0.5
```

## 4. Check it

```bash
segbench list                 # your method should appear
segbench doctor mymethod      # READY / NOT READY
segbench run mymethod --help
segbench selftest             # the shared-contract checks now cover it
```

`segbench selftest` iterates the registry, so it will automatically verify
that your module imports, builds a parser, exposes the shared flags and
honours `--dry-run`.

## Special cases

**Cell-level methods** (no per-transcript output, like SPLIT): set
`transcript_level=False` in the spec, pass explicit `n_entities`/`n_genes` to
`stx.entity_accounting(None, ...)`, and set `qc["transcript_level"] = False`.

**Sequencing-modality methods**: use `modality=SEQUENCING` and set
`entity_kind` to what a row actually is (`bin`, `spot`). Never let a `bin`
entity count be compared against a `cell` count — the docs and the summary
table both carry `entity_kind` so readers can tell them apart.

**One module, several registry entries** (as `tracer` and `tracer_seq` do):
give `main` a `method: str | None = None` keyword. The CLI detects it and
passes the registry name, so the module knows which identity it is running as.

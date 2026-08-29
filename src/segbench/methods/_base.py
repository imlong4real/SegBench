#!/usr/bin/env python3
"""Shared plumbing for method wrappers.

Every wrapper follows the same skeleton::

    def build_argparser(): ...        # method-specific flags
    def main(argv=None): ...          # load -> convert -> run -> standardize

:func:`add_common_args` gives them the identical set of shared flags, and
:func:`resolve_config` applies the layered YAML config before the wrapper
reads any value. Keeping both here is what makes the interface consistent:
a wrapper cannot accidentally spell ``--sample-name`` differently.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .. import config as cfgmod
from .. import envs
from .. import registry


def add_common_args(p: argparse.ArgumentParser, *, method: str) -> None:
    """Flags every method accepts, in the same spelling."""
    g = p.add_argument_group("common")
    g.add_argument("--outdir", type=Path, default=None,
                   help="Output directory for this run.")
    g.add_argument("--config", default=None,
                   help="User config YAML (highest-precedence layer).")
    g.add_argument("--dataset", default=None,
                   help="Dataset name (configs/datasets/<name>.yaml) or a path.")
    g.add_argument("--sample-name", default=None,
                   help="Sample label recorded in every output file.")
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--threads", type=int, default=None,
                   help="Thread/core budget passed to the underlying tool.")
    g.add_argument("--overwrite", action="store_true",
                   help="Replace an existing completed run in --outdir.")
    g.add_argument("--max-transcripts", type=int, default=None,
                   help="Smoke-test helper: subsample the input to N transcripts.")
    g.add_argument("--dry-run", action="store_true",
                   help="Validate inputs, config and dependencies, then stop "
                        "without running the method.")
    p.set_defaults(_method_name=method)


def add_transcript_input_args(p: argparse.ArgumentParser) -> None:
    """Inputs shared by the molecule-resolved (imaging) methods."""
    g = p.add_argument_group("inputs")
    g.add_argument("--transcripts", type=Path, default=None,
                   help="Standardized transcripts parquet "
                        "(x, y, feature_name, cell_id [, z, qv, ...]).")
    g.add_argument("--reference-h5ad", type=Path, default=None,
                   help="scRNA reference h5ad, for methods that need one.")
    g.add_argument("--reference-celltype-col", default=None,
                   help="obs column holding reference cell-type labels.")


def resolve_config(args: argparse.Namespace, *, method: str,
                   section: str | None = None) -> dict:
    """Build and apply the layered config, then normalise shared defaults."""
    spec = registry.METHODS.get(method)
    cfg = cfgmod.build_config(
        method=method,
        default_config=spec.default_config if spec else "",
        dataset=getattr(args, "dataset", None),
        user_config=getattr(args, "config", None),
    )
    cfgmod.apply_to_args(args, cfg, section=section or method)

    # Shared defaults are applied after the config so a config can set them.
    if getattr(args, "sample_name", None) in (None, ""):
        args.sample_name = cfg.get("sample_name") or "sample"
    if getattr(args, "threads", None) in (None, 0):
        args.threads = int(cfg.get("threads", 4))
    # Each tool spells its thread option differently; --threads is the one
    # canonical flag, and it fills whichever alias this tool uses.
    for alias in ("nthreads", "n_threads", "cores"):
        if getattr(args, alias, "absent") is None:
            setattr(args, alias, args.threads)

    for attr in ("transcripts", "reference_h5ad", "outdir"):
        val = getattr(args, attr, None)
        if isinstance(val, str):
            setattr(args, attr, cfgmod.resolve_path(val))
    # Point --baysor-bin / --proseg-bin / --rscript at the configured
    # environment unless the caller overrode them explicitly.
    envs.inject_tool_paths(args, method)

    if getattr(args, "outdir", None) is None:
        raise SystemExit(
            "--outdir is required (pass it on the command line or set "
            "`outdir:` in a config).")
    args.outdir = Path(args.outdir)
    return cfg


def require_input(args: argparse.Namespace, attr: str, flag: str) -> Path:
    """Fetch a required input path, failing with an actionable message."""
    val = getattr(args, attr, None)
    if val in (None, ""):
        raise SystemExit(
            f"{flag} is required for this method. Pass it on the command line, "
            f"or set it in a dataset config (configs/datasets/<name>.yaml).")
    p = Path(val)
    if not p.exists():
        raise SystemExit(f"{flag}: file not found: {p}")
    return p

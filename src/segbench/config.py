#!/usr/bin/env python3
"""YAML config loading, layering and path resolution.

Configuration is layered, lowest precedence first::

    1. the method's default config   (configs/methods/<method>.yaml)
    2. a dataset config              (--dataset / configs/datasets/<name>.yaml)
    3. a user config                 (--config)
    4. explicit command-line flags

so a run can be reproduced from a config alone, while any single value stays
overridable from the command line for quick experiments.

Paths inside a config may be written relative to the repo root, and may use
``${VAR}`` environment placeholders (e.g. ``${SEGBENCH_DATA}/xenium/TSU-20``).
That is how machine-specific locations stay out of the tracked files.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from . import REPO_ROOT

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

#: Root for datasets. Override per machine with the SEGBENCH_DATA env var so
#: that configs stay portable; defaults to <repo>/dataset.
DATA_ROOT_ENV = "SEGBENCH_DATA"


def data_root() -> Path:
    return Path(os.environ.get(DATA_ROOT_ENV, REPO_ROOT / "dataset")).expanduser()


def expand(value: Any) -> Any:
    """Recursively expand ``${VAR}`` / ``~`` in strings inside a config tree."""
    if isinstance(value, str):
        def _sub(m: re.Match) -> str:
            var = m.group(1)
            if var == DATA_ROOT_ENV:
                return str(data_root())
            got = os.environ.get(var)
            if got is None:
                raise SystemExit(
                    f"Config references ${{{var}}} but that environment variable "
                    f"is not set. Export it, or edit the config to use a literal "
                    f"path."
                )
            return got
        return os.path.expanduser(_ENV_RE.sub(_sub, value))
    if isinstance(value, dict):
        return {k: expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand(v) for v in value]
    return value


def resolve_path(value: str | Path | None) -> Path | None:
    """Resolve a config path: absolute stays put, relative is repo-root anchored."""
    if value in (None, ""):
        return None
    p = Path(str(value)).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p)


def deep_merge(base: dict, override: dict) -> dict:
    """Merge ``override`` onto ``base`` recursively (override wins)."""
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml(path: str | Path | None) -> dict:
    if path in (None, ""):
        return {}
    p = resolve_path(path)
    if p is None or not p.exists():
        raise SystemExit(f"Config file not found: {path}")
    import yaml
    with open(p) as fh:
        return expand(yaml.safe_load(fh) or {})


def find_config(spec_or_name: str, subdir: str) -> Path | None:
    """Resolve ``name`` to ``configs/<subdir>/<name>.yaml`` if it is a bare name."""
    p = Path(spec_or_name)
    if p.suffix in (".yaml", ".yml") or p.is_absolute() or p.exists():
        return resolve_path(spec_or_name)
    for ext in (".yaml", ".yml"):
        cand = REPO_ROOT / "configs" / subdir / f"{spec_or_name}{ext}"
        if cand.exists():
            return cand
    return None


def build_config(
    *, method: str, default_config: str = "", dataset: str | None = None,
    user_config: str | None = None, overrides: dict | None = None,
) -> dict:
    """Assemble the layered config for one method run."""
    cfg: dict = {}
    if default_config:
        default_path = REPO_ROOT / "configs" / default_config
        if default_path.exists():
            cfg = deep_merge(cfg, load_yaml(default_path))
    if dataset:
        ds_path = find_config(dataset, "datasets")
        if ds_path is None:
            raise SystemExit(
                f"Unknown dataset {dataset!r}: no configs/datasets/{dataset}.yaml "
                f"and no such file."
            )
        cfg = deep_merge(cfg, load_yaml(ds_path))
    if user_config:
        cfg = deep_merge(cfg, load_yaml(user_config))
    if overrides:
        cfg = deep_merge(cfg, overrides)
    cfg.setdefault("method", method)
    return cfg


def apply_to_args(args, cfg: dict, *, section: str | None = None) -> None:
    """Fill unset argparse attributes from a config dict.

    Command-line flags win: a key is only applied when the corresponding
    attribute is still at its parser default (``None``/unset). ``section``
    reads from ``cfg[section]`` (e.g. the ``baysor:`` block) falling back to
    the top level.
    """
    merged: dict = {}
    for key in ("dataset", "inputs", "run"):
        if isinstance(cfg.get(key), dict):
            merged.update(cfg[key])
    if section and isinstance(cfg.get(section), dict):
        merged.update(cfg[section])
    for key, value in merged.items():
        attr = key.replace("-", "_")
        if not hasattr(args, attr):
            continue
        if getattr(args, attr) in (None, ""):
            setattr(args, attr, value)

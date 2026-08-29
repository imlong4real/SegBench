#!/usr/bin/env python3
"""Per-method execution environments.

Different methods need genuinely different runtimes — a Julia binary, a Rust
binary, an R installation, a CUDA-enabled Python, a plain Python. This module
is what lets one command drive all of them: it resolves ``configs/
environments.yaml`` into concrete interpreter/binary paths, and lets the CLI
re-exec a wrapper under the interpreter that method needs.

Resolution order for the config file:
    1. ``$SEGBENCH_ENV_CONFIG``
    2. ``configs/environments.local.yaml``   (gitignored, per-machine)
    3. ``configs/environments.yaml``         (tracked, ``${VAR}``-templated)

A ``${VAR}`` that is unset resolves to ``None`` rather than raising, so a
partially-provisioned machine still reports cleanly through ``segbench doctor``
instead of crashing.
"""
from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

from . import REPO_ROOT

_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

#: Keys a method entry may define.
PYTHON, BINARY, RSCRIPT, ENV = "python", "binary", "rscript", "env"

#: Import probes run in a subprocess; keep them short so `doctor` stays
#: interactive even when an env is mid-install on a network filesystem.
IMPORT_PROBE_TIMEOUT_S = 25

#: Parsed config cache — resolve() is called many times per probe.
_CACHE: dict | None = None


def config_path() -> Path | None:
    explicit = os.environ.get("SEGBENCH_ENV_CONFIG")
    if explicit:
        return Path(explicit)
    for name in ("environments.local.yaml", "environments.yaml"):
        p = REPO_ROOT / "configs" / name
        if p.exists():
            return p
    return None


def _expand(value: Any) -> Any:
    """Expand ``${VAR}``; a string with an unset variable becomes None."""
    if isinstance(value, str):
        missing = [m.group(1) for m in _VAR.finditer(value)
                   if os.environ.get(m.group(1)) is None]
        if missing:
            return None
        return os.path.expanduser(_VAR.sub(lambda m: os.environ[m.group(1)], value))
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    return value


def load() -> dict[str, dict[str, Any]]:
    """Return ``{method: {python/binary/rscript/env: resolved}}`` (cached)."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    p = config_path()
    if p is None or not p.exists():
        _CACHE = {}
        return _CACHE
    try:
        import yaml
        raw = yaml.safe_load(p.read_text()) or {}
    except Exception:
        _CACHE = {}
        return _CACHE
    out: dict[str, dict[str, Any]] = {}
    for method, spec in (raw.get("methods") or {}).items():
        if isinstance(spec, dict):
            out[method] = {k: _expand(v) for k, v in spec.items()}
    _CACHE = out
    return _CACHE


def for_method(method: str) -> dict[str, Any]:
    return load().get(method, {})


def resolve(method: str, key: str) -> str | None:
    """One resolved path for a method, or None when unset/missing on disk."""
    val = for_method(method).get(key)
    if not val:
        return None
    p = Path(val)
    if p.exists():
        return str(p)
    # Allow a bare command name that happens to be on PATH.
    return shutil.which(val)


def interpreter(method: str) -> str | None:
    """The Python a method's wrapper must run under, if it differs from ours."""
    py = resolve(method, PYTHON)
    if not py:
        return None
    try:
        if Path(py).resolve() == Path(sys.executable).resolve():
            return None
    except OSError:
        pass
    return py


def probe(method: str, spec) -> tuple[bool, list[str]]:
    """Is `method` runnable here? Returns (ok, missing).

    Prefers the configured environment; falls back to PATH so a method
    installed system-wide still reports READY without any config.
    """
    entry = for_method(method)
    missing: list[str] = []

    # 1. Configured external binary / Rscript.
    for key, label in ((BINARY, "binary"), (RSCRIPT, "Rscript")):
        if key in entry:
            if resolve(method, key) is None:
                missing.append(f"{label}:{entry.get(key) or '<unset ${VAR}>'}")

    # 2. Configured interpreter must exist, and must import the method package.
    if PYTHON in entry:
        py = resolve(method, PYTHON)
        if py is None:
            missing.append(f"python:{entry.get(PYTHON) or '<unset ${VAR}>'}")
        else:
            pkg = {"segger": "segger", "bin2cell": "bin2cell",
                   "tracer": "tracer", "tracer_seq": "tracer"}.get(method)
            if pkg and not _can_import(py, pkg):
                missing.append(f"python-package:{pkg}")

    # 2b. R-backed methods need their packages, not just an Rscript.
    if RSCRIPT in entry and not missing:
        rs = resolve(method, RSCRIPT)
        if rs:
            for pkg in _r_packages_present(rs, R_REQUIREMENTS.get(method, ())):
                missing.append(f"r-package:{pkg}")

    # 3. Nothing configured for this method -> fall back to PATH probing.
    if not entry:
        return _probe_path(method)
    return (not missing), missing


def _can_import(python: str, module: str) -> bool:
    """Is `module` importable by `python`?

    Uses ``importlib.util.find_spec`` rather than a real import: answering
    "is it installed" must not pay the cost of pulling in torch/scanpy, which
    can take a minute and would make `doctor` look hung.
    """
    import subprocess
    code = (f"import importlib.util as u,sys;"
            f"sys.exit(0 if u.find_spec({module!r}) else 1)")
    try:
        return subprocess.run([python, "-c", code], capture_output=True,
                              timeout=IMPORT_PROBE_TIMEOUT_S).returncode == 0
    except Exception:
        return False


#: R packages each R-backed method needs present in the library tree.
R_REQUIREMENTS = {"split": ("spacexr", "SPLIT"), "celladmix": ("cellAdmix",)}


def _r_packages_present(rscript: str, packages: tuple[str, ...]) -> list[str]:
    """Missing R packages, found by looking in the env's library directory.

    A filesystem check rather than `Rscript -e requireNamespace(...)`: starting
    R costs seconds per package and `doctor` should stay fast.
    """
    lib = Path(rscript).resolve().parent.parent / "lib" / "R" / "library"
    if not lib.exists():
        return []          # unknown layout: do not claim a false negative
    return [p for p in packages if not (lib / p).exists()]


def _probe_path(method: str) -> tuple[bool, list[str]]:
    checks = {
        "baysor": [("bin", "baysor")], "proseg": [("bin", "proseg")],
        "segger": [("py", "torch"), ("py", "segger")],
        "split": [("bin", "Rscript")], "celladmix": [("bin", "Rscript")],
        "tracer": [("py", "tracer")], "tracer_seq": [("py", "tracer")],
        "bin2cell": [("py", "bin2cell")],
    }
    missing = []
    for kind, name in checks.get(method, []):
        if kind == "bin":
            if shutil.which(name) is None:
                missing.append(f"binary:{name}")
        else:
            try:
                __import__(name)
            except Exception:
                missing.append(f"python:{name}")
    return (not missing), missing


def inject_tool_paths(args, method: str) -> None:
    """Point a wrapper's tool flag at the configured binary when unset.

    Keeps ``--baysor-bin`` / ``--proseg-bin`` / ``--rscript`` overridable on the
    command line while making the configured environment the default.
    """
    binary = resolve(method, BINARY)
    if binary:
        for attr in ("baysor_bin", "proseg_bin"):
            if getattr(args, attr, None) in (None, "", "baysor", "proseg"):
                if hasattr(args, attr):
                    setattr(args, attr, binary)
    rscript = resolve(method, RSCRIPT)
    if rscript and hasattr(args, "rscript") and getattr(args, "rscript", None) in (None, ""):
        args.rscript = rscript

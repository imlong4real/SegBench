"""segbench — a reproducible benchmarking suite for spatial-transcriptomics
segmentation and transcript-refinement methods.

Every method is exposed through one interface::

    segbench run <method> --config <cfg.yaml> --outdir <dir>

and every method emits the same output contract (see :mod:`segbench.stats`),
so runtime, memory, entity counts and transcript assignment are directly
comparable across tools.
"""
from __future__ import annotations

from pathlib import Path

__version__ = "0.2.0"

#: Repository root, derived from this file's location
#: (``<repo>/src/segbench/__init__.py`` -> ``<repo>``). Everything that needs a
#: repo-relative path derives it from here rather than hard-coding a machine
#: path, so the tree can be cloned anywhere.
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

__all__ = ["REPO_ROOT", "__version__"]

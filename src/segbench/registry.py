#!/usr/bin/env python3
"""The method registry — the single place that knows what methods exist.

Adding a method means adding one :class:`MethodSpec` here and one module under
:mod:`segbench.methods`. Nothing else in the suite needs to change: the CLI,
the suite runner, the docs table and the aggregator all read this registry.

See ``docs/adding_a_method.md`` for the full walkthrough.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from types import ModuleType

#: Data modalities a method can consume.
IMAGING = "imaging"        # molecule-resolved: Xenium, CosMx, MERFISH
SEQUENCING = "sequencing"  # array/binned: Visium HD, Stereo-seq


@dataclass(frozen=True)
class MethodSpec:
    """Everything the suite knows about one method."""

    name: str
    module: str                       # dotted path under segbench.methods
    modality: str                     # IMAGING | SEQUENCING
    kind: str                         # segmentation | refinement | cell-calling
    entity_kind: str = "cell"         # what one output row is
    summary: str = ""
    #: External (non-pip) dependencies a user must install themselves.
    external_deps: tuple[str, ...] = ()
    #: Whether the method emits a per-transcript assignment table. SPLIT does
    #: not (it returns fractional expected counts), so it is scored per cell.
    transcript_level: bool = True
    #: Human-facing name for tables and plots. The ``name`` stays the stable
    #: key used by CLI arguments, directory names and joins; this is only what
    #: a reader sees. Empty means "use ``name``".
    display_name: str = ""
    #: Default config file, relative to ``configs/``.
    default_config: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def label(self) -> str:
        """What to print for this method."""
        return self.display_name or self.name

    def load(self) -> ModuleType:
        """Import the wrapper module (deferred: heavy deps stay unimported)."""
        return importlib.import_module(f"segbench.methods.{self.module}")


METHODS: dict[str, MethodSpec] = {
    # --- imaging / molecule-resolved -------------------------------------
    "baysor": MethodSpec(
        name="baysor", module="baysor", modality=IMAGING, kind="segmentation",
        summary="Bayesian de-novo segmentation from molecule positions (Julia).",
        external_deps=("baysor >= 0.7 (Julia binary)",),
        default_config="methods/baysor.yaml", tags=("de-novo",),
    ),
    "proseg": MethodSpec(
        name="proseg", module="proseg", modality=IMAGING, kind="segmentation",
        summary="Probabilistic voxel-based cell segmentation (Rust).",
        external_deps=("proseg >= 3.0 (cargo binary)",),
        default_config="methods/proseg.yaml", tags=("de-novo",),
    ),
    "segger": MethodSpec(
        name="segger", module="segger", modality=IMAGING, kind="segmentation",
        summary="Graph-neural-network transcript-to-nucleus assignment (GPU).",
        external_deps=("segger", "torch + torch-geometric (CUDA)"),
        default_config="methods/segger.yaml", tags=("de-novo", "gpu"),
    ),
    "split": MethodSpec(
        name="split", module="split", modality=IMAGING, kind="refinement",
        summary="RCTD-based purification of contaminated cell profiles (R).",
        external_deps=("R >= 4.3", "spacexr", "SPLIT"),
        transcript_level=False,   # returns fractional expected counts
        default_config="methods/split.yaml", tags=("cleanup", "cell-level"),
    ),
    "celladmix": MethodSpec(
        name="celladmix", module="celladmix", modality=IMAGING, kind="refinement",
        summary="NMF/CRF admixture correction; flags contaminating molecules (R).",
        external_deps=("R >= 4.3", "cellAdmix"),
        default_config="methods/celladmix.yaml", tags=("cleanup",),
    ),
    "tracer": MethodSpec(
        name="tracer", module="tracer", modality=IMAGING, kind="refinement",
        summary="cPMI-guided transcript reassignment / resegmentation.",
        external_deps=("tracer (python package)",),
        display_name="TRACER (Seg)",   # refines an existing segmentation
        default_config="methods/tracer.yaml", tags=("cleanup",),
    ),
    # --- sequencing / array-based ----------------------------------------
    "bin2cell": MethodSpec(
        name="bin2cell", module="bin2cell", modality=SEQUENCING,
        kind="cell-calling", entity_kind="cell",
        summary="Visium HD 2um bins -> cells via H&E/label-guided expansion.",
        external_deps=("bin2cell", "stardist or a precomputed label image"),
        default_config="methods/bin2cell.yaml", tags=("visium-hd",),
    ),
    "tracer_seq": MethodSpec(
        name="tracer_seq", module="tracer", modality=SEQUENCING,
        kind="refinement", entity_kind="bin",
        summary="TRACER refinement applied to binned sequencing data.",
        external_deps=("tracer (python package)",),
        display_name="TRACER (No-seg)",   # builds profiles without a segmentation
        default_config="methods/tracer_seq.yaml", tags=("cleanup", "visium-hd"),
    ),
}


def get(name: str) -> MethodSpec:
    """Look up a method, with a helpful error listing the valid names."""
    try:
        return METHODS[name]
    except KeyError:
        raise SystemExit(
            f"Unknown method {name!r}. Available: {', '.join(sorted(METHODS))}"
        ) from None


def by_modality(modality: str) -> list[MethodSpec]:
    return [m for m in METHODS.values() if m.modality == modality]

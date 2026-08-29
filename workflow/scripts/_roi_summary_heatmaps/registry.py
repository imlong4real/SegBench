#!/usr/bin/env python3
"""Shared registry for the cross-platform ROI benchmark summary heatmaps.

Defines the four datasets, the method-entities (columns), how each entity's
cell-by-gene matrix is sourced, the scRNA references, and the canonical
method order / display labels used across all three figure blocks.

This module is imported by every stage so paths and ordering stay identical.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
FIG3 = REPO / "results" / "fig3_cross_platform_roi_benchmark"
SEGGER = REPO / "results" / "segger_roi"

# Optional run namespace.  The default preserves the original tracer_seg
# benchmark; set ROI_TRACER_SOURCE=tracer_resegment and
# ROI_SUMMARY_NAME=summary_heatmaps_resegment for the resegment audit.
TRACER_SOURCE = os.environ.get("ROI_TRACER_SOURCE", "tracer_seg")
SUMMARY_NAME = os.environ.get("ROI_SUMMARY_NAME", "summary_heatmaps")
OUT = FIG3 / SUMMARY_NAME
WORK = OUT / "_work"                 # per-entity h5ads (RCTD inputs)
METRICS = OUT / "metrics"            # per-entity biological metric outputs

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
DATASETS = {
    "atera_cervical": {
        "platform": "Xenium",
        "original_label": "10x",
        "reference_h5ad": REPO / "dataset/cervical_scrna/h5ad/cervical_scrna_adc_scc_marker_annotated.h5ad",
        "reference_celltype_col": "cell_type",
        "reference_exclude_celltypes": ["Unannotated"],
        "npmi": REPO / "dataset/atera_cervical/npmi_panel.csv.gz",
        # Reference-derived NPMI panel TRACER was refined against (scRNA co-expression);
        # used for the biological-coherence purity/conflict metric (Audit 2).
        "npmi_reference": REPO / "dataset/atera_cervical/tracer_seg/npmi_panel.csv.gz",
        "has_baysor": True,
    },
    "xenium5k_cervical": {
        "platform": "Xenium5K",
        "original_label": "10x",
        "reference_h5ad": REPO / "dataset/cervical_scrna/h5ad/cervical_scrna_adc_scc_marker_annotated.h5ad",
        "reference_celltype_col": "cell_type",
        "reference_exclude_celltypes": ["Unannotated"],
        "npmi": REPO / "dataset/xenium5k_cervical/npmi_panel.csv.gz",
        "npmi_reference": REPO / "dataset/xenium5k_cervical/tracer_seg/npmi_panel.csv.gz",
        "has_baysor": True,
    },
    "cosmx_nsclc": {
        "platform": "CosMx",
        "original_label": "CosMx SMI",
        "reference_h5ad": REPO / "dataset/lung_cancer_scrna_10x/lung_cancer_50k.h5ad",
        "reference_celltype_col": "Cell_Cluster_level1",
        "reference_exclude_celltypes": [],
        "npmi": REPO / "dataset/cosmx_nsclc/npmi_panel.csv.gz",
        "npmi_reference": REPO / "dataset/cosmx_nsclc/tracer_seg/npmi_panel.csv.gz",
        "has_baysor": True,
    },
    "merfish_mouse_ileum": {
        "platform": "MERFISH",
        "original_label": "Baysor",  # ROI native segmentation is Baysor
        "reference_h5ad": REPO / "dataset/mouse_ileum_scrna/h5ad/gut_scrna_gse92332_ileum_annotated.h5ad",
        "reference_celltype_col": "cell_type",
        "reference_exclude_celltypes": [],
        "npmi": REPO / "dataset/merfish_mouse_ileum/npmi_panel.csv.gz",
        "npmi_reference": REPO / "dataset/merfish_mouse_ileum/tracer_seg/npmi_panel_selfref.csv.gz",
        "has_baysor": False,  # standalone Baysor intentionally skipped (it IS the baseline)
    },
}

# Nicely ordered platform display (for dataset row labels)
DATASET_ORDER = ["atera_cervical", "xenium5k_cervical", "cosmx_nsclc", "merfish_mouse_ileum"]

# ---------------------------------------------------------------------------
# Method-entities (figure columns)
# ---------------------------------------------------------------------------
# key -> display label. Segger carries a GPU asterisk.
ENTITY_LABELS = {
    "original": None,                       # filled per-dataset (10x / CosMx SMI / Baysor)
    "TRACER": "TRACER",
    "TRACER_refined": "TRACER-refined",
    "TRACER_reconstructed": "TRACER-reconstructed",
    "baysor": "Baysor",
    "proseg": "proseg",
    "segger": "Segger*",
    "celladmix": "cellAdmix",
    "split": "SPLIT",
}

# Canonical column order, identical across every heatmap.
ENTITY_ORDER = [
    "original",
    "TRACER",
    "TRACER_refined",
    "TRACER_reconstructed",
    "baysor",
    "proseg",
    "segger",
    "celladmix",
    "split",
]

GPU_ENTITIES = {"segger"}  # asterisk + GPU note in legend


def entity_label(entity: str, dataset: str) -> str:
    if entity == "original":
        return DATASETS[dataset]["original_label"]
    return ENTITY_LABELS[entity]


def std_dir(dataset: str, method: str) -> Path:
    return FIG3 / dataset / "_benchmark_standardized" / method


def entity_matrix_spec(dataset: str, entity: str) -> dict | None:
    """How to obtain a cells x genes raw-count AnnData for one entity.

    Returns a dict describing the source, or None if the entity does not
    exist for this dataset (e.g. standalone Baysor on MERFISH).
    """
    ds = DATASETS[dataset]
    if entity == "original":
        # tracer_resegment outputs do not carry a copy of the platform input
        # transcripts, so the immutable native baseline remains the tracer_seg
        # input parquet.
        input_path = REPO / f"dataset/{dataset}/{TRACER_SOURCE}/input_transcripts.parquet"
        if not input_path.exists():
            input_path = REPO / f"dataset/{dataset}/tracer_seg/input_transcripts.parquet"
        return {"kind": "transcripts",
                "path": input_path,
                "label_col": "cell_id"}
    if entity == "TRACER":
        if TRACER_SOURCE != "tracer_seg":
            return {"kind": "h5ad",
                    "path": REPO / f"dataset/{dataset}/{TRACER_SOURCE}/outputs/cell_by_gene_tracer.h5ad"}
        return {"kind": "h5ad", "path": std_dir(dataset, "TRACER") / "cell_by_gene.h5ad"}
    if entity == "TRACER_refined":
        return {"kind": "transcripts",
                "path": REPO / f"dataset/{dataset}/{TRACER_SOURCE}/outputs/transcripts_tracer_refined.parquet",
                "label_col": "stitched", "etype": "cell"}
    if entity == "TRACER_reconstructed":
        return {"kind": "transcripts",
                "path": REPO / f"dataset/{dataset}/{TRACER_SOURCE}/outputs/transcripts_tracer_refined.parquet",
                "label_col": "stitched", "etype": "partial"}
    if entity == "baysor":
        if not ds["has_baysor"]:
            return None
        return {"kind": "h5ad", "path": std_dir(dataset, "baysor") / "cell_by_gene.h5ad"}
    if entity == "proseg":
        return {"kind": "h5ad", "path": std_dir(dataset, "proseg") / "cell_by_gene.h5ad"}
    if entity == "celladmix":
        return {"kind": "h5ad", "path": std_dir(dataset, "cellAdmix") / "cell_by_gene.h5ad"}
    if entity == "split":
        return {"kind": "h5ad", "path": std_dir(dataset, "SPLIT") / "split_cell_by_gene.h5ad"}
    if entity == "segger":
        return {"kind": "h5ad", "path": SEGGER / dataset / "final" / "segger_adata.h5ad"}
    raise ValueError(entity)


def work_h5ad(dataset: str, entity: str) -> Path:
    return WORK / dataset / f"{entity}.h5ad"


# ---------------------------------------------------------------------------
# Metric definitions: name -> (block, direction, tracer_mode)
#   direction: 'higher' (better high) | 'lower' (better low) | 'neutral'
#   tracer_mode: 'combined' (use TRACER col, refined/recon NA)
#                'separate' (use refined+reconstructed, TRACER col NA)
# ---------------------------------------------------------------------------
METRICS_SPEC = {
    # Block A — size & compute
    "total_cells":          dict(block="A", direction="neutral",  tracer="combined", label="Total cells / profiles"),
    "transcripts_per_cell": dict(block="A", direction="neutral",  tracer="separate", label="Transcripts per cell / profile"),
    "runtime_seconds":      dict(block="A", direction="lower",    tracer="combined", label="Runtime (s)"),
    "peak_memory_gb":       dict(block="A", direction="lower",    tracer="combined", label="Peak memory (GB)"),
    # Block B — biological coherence
    "marker_log2fc":        dict(block="B", direction="higher",   tracer="separate", label="Marker specificity (log2FC, 3 markers/type)"),
    "relative_purity":      dict(block="B", direction="higher",   tracer="separate", label="Relative purity (NPMI vs reference)"),
    "relative_conflict":    dict(block="B", direction="lower",    tracer="separate", label="Relative conflict (NPMI vs reference)"),
    "kendall_tau":          dict(block="B", direction="higher",   tracer="separate", label="Kendall τ vs scRNA"),
    # Block C — RCTD purity
    "rctd_entropy":         dict(block="C", direction="lower",    tracer="separate", label="RCTD entropy"),
    "rctd_max_weight":      dict(block="C", direction="higher",   tracer="separate", label="RCTD max weight"),
}

# Which entities carry which tracer_mode value
TRACER_COMBINED_ENTITY = "TRACER"
TRACER_SEPARATE_ENTITIES = ("TRACER_refined", "TRACER_reconstructed")


def applicable(entity: str, metric: str, dataset: str) -> bool:
    """Whether a (entity, metric) cell should hold a value (else NA)."""
    spec = entity_matrix_spec(dataset, entity)
    if spec is None and entity != "original":
        # entity absent for this dataset
        if entity == "baysor":
            return False
    mode = METRICS_SPEC[metric]["tracer"]
    if entity == TRACER_COMBINED_ENTITY:
        return mode == "combined"
    if entity in TRACER_SEPARATE_ENTITIES:
        return mode == "separate"
    return True

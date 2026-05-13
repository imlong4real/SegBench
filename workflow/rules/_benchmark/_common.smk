# Common helpers for the benchmark layer.
# This file is included by Snakefile_benchmark; it expects `config` to be
# populated from a benchmark_*.yml file.

import os

# ---------------------------------------------------------------------------
# Config accessors
# ---------------------------------------------------------------------------

DATASET = config["dataset"]
DATASET_NAME = DATASET["name"]
XENIUM_DIR = DATASET["xenium_dir"]
OUTPUT_DIR = DATASET["output_dir"]
THREADS = int(DATASET.get("threads", 4))
QV_THRESHOLD = DATASET.get("qv_threshold")
MAX_TRANSCRIPTS = DATASET.get("max_transcripts")
ROI = DATASET.get("roi") or {}
CLUSTER_LABELS = DATASET.get("cluster_labels")

METHODS = config.get("methods", {})

OUTPUTS = config.get("outputs", {})
STAND_DIR = OUTPUTS.get("standardized_dir", os.path.join(OUTPUT_DIR, "standardized"))
METRICS_DIR = OUTPUTS.get("metrics_dir", os.path.join(OUTPUT_DIR, "metrics"))
FIGURES_DIR = OUTPUTS.get("figures_dir", os.path.join(OUTPUT_DIR, "figures"))
SUMMARY_DIR = OUTPUTS.get("summary_dir", os.path.join(OUTPUT_DIR, "summary"))

TRACER_CFG = config.get("tracer", {})
CELLADMIX_CFG = config.get("celladmix", {})
SPLIT_CFG = config.get("split", {})
OVRLPY_CFG = config.get("ovrlpy", {})
METRICS_CFG = config.get("metrics", {})

# ---------------------------------------------------------------------------
# Method selection
# ---------------------------------------------------------------------------

DIRECT_METHODS = [m for m in ("xenium_default", "baysor", "proseg", "segger") if METHODS.get(m)]
TRACER_BASES = [
    base
    for base in ("xenium_default", "baysor", "proseg", "segger")
    if METHODS.get(f"tracer_{base}")
]

CELLADMIX_BASES = []
if METHODS.get("celladmix_xenium_default"):
    CELLADMIX_BASES.append("xenium_default")
if METHODS.get("celladmix_tracer_xenium_default"):
    CELLADMIX_BASES.append("tracer_from_xenium_default")

SPLIT_BASES = []
if METHODS.get("split_xenium_default"):
    SPLIT_BASES.append("xenium_default")
if METHODS.get("split_tracer_xenium_default"):
    SPLIT_BASES.append("tracer_from_xenium_default")

OVRLPY_BASES = []
if METHODS.get("ovrlpy"):
    OVRLPY_BASES.append("xenium_default")
    if METHODS.get("tracer_xenium_default"):
        OVRLPY_BASES.append("tracer_from_xenium_default")


def standardized_dir(method: str) -> str:
    return os.path.join(STAND_DIR, method)


def standardized_transcripts(method: str) -> str:
    return os.path.join(standardized_dir(method), "transcripts.parquet")


def standardized_method_info(method: str) -> str:
    return os.path.join(standardized_dir(method), "method_info.json")


def standardized_marker_done(method: str) -> str:
    return os.path.join(standardized_dir(method), "cell_by_gene.mtx")


def npmi_summary(method: str) -> str:
    return os.path.join(METRICS_DIR, method, "npmi_summary.parquet")


def npmi_per_cell(method: str) -> str:
    return os.path.join(METRICS_DIR, method, "npmi_per_cell.parquet")


def marker_summary(method: str) -> str:
    return os.path.join(METRICS_DIR, method, "marker_summary.parquet")


def marker_per_cell(method: str) -> str:
    return os.path.join(METRICS_DIR, method, "marker_per_cell.parquet")


def all_benchmark_methods() -> list[str]:
    out: list[str] = []
    out.extend(DIRECT_METHODS)
    out.extend([f"tracer_from_{b}" for b in TRACER_BASES])
    # SPLIT / cellAdmix / ovrlpy produce *correction* artifacts rather than
    # standardized segmentation outputs, so they don't enter the metrics
    # table directly. They are emitted as sibling outputs.
    return out


def roi_args() -> list[str]:
    args = []
    if ROI.get("x_min") is not None:
        args += ["--roi-xmin", str(ROI["x_min"])]
    if ROI.get("x_max") is not None:
        args += ["--roi-xmax", str(ROI["x_max"])]
    if ROI.get("y_min") is not None:
        args += ["--roi-ymin", str(ROI["y_min"])]
    if ROI.get("y_max") is not None:
        args += ["--roi-ymax", str(ROI["y_max"])]
    return args

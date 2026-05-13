# Rules that produce standardized outputs from each direct method.
# These rules deliberately assume the raw method outputs already exist at
# conventional paths under `results/{dataset}/_raw/{method}/...`. The
# segmentation rules from the original SPLIT pipeline can be wired here, but
# for the first deliverable we only require xenium_default (which reads the
# raw Xenium bundle directly).

# ---------------------------------------------------------------------------
# xenium_default — reads the raw Xenium bundle directly.
# ---------------------------------------------------------------------------

rule standardize_xenium_default:
    input:
        transcripts=os.path.join(XENIUM_DIR, "transcripts.parquet"),
        cells=os.path.join(XENIUM_DIR, "cells.parquet"),
    output:
        method_info=os.path.join(STAND_DIR, "xenium_default", "method_info.json"),
        transcripts=os.path.join(STAND_DIR, "xenium_default", "transcripts.parquet"),
        cells=os.path.join(STAND_DIR, "xenium_default", "cells.parquet"),
        mtx=os.path.join(STAND_DIR, "xenium_default", "cell_by_gene.mtx"),
    log:
        os.path.join(OUTPUT_DIR, "logs", "standardize_xenium_default.log"),
    params:
        out_dir=os.path.join(STAND_DIR, "xenium_default"),
        roi_args=" ".join(roi_args()),
        qv_arg=(f"--qv-threshold {QV_THRESHOLD}" if QV_THRESHOLD is not None else ""),
        max_arg=(f"--max-transcripts {MAX_TRANSCRIPTS}" if MAX_TRANSCRIPTS else ""),
    threads: THREADS
    conda: "../../envs/benchmark.yml"
    shell:
        "python3 workflow/scripts/_benchmark/standardize_method_output.py "
        "--method xenium_default "
        "--xenium-dir {XENIUM_DIR} "
        "--out-dir {params.out_dir} "
        "--threads {threads} "
        "{params.qv_arg} {params.max_arg} {params.roi_args} "
        "--log {log}"


# ---------------------------------------------------------------------------
# baysor — wraps the existing runBaysor / adjustBaysorResults outputs.
# We expect the raw segmentation.csv to live at:
#   {OUTPUT_DIR}/_raw/baysor/segmentation.csv
# The user can either run the SPLIT pipeline's runBaysor rule and symlink,
# or place a pre-computed file there.
# ---------------------------------------------------------------------------

rule standardize_baysor:
    input:
        seg=os.path.join(OUTPUT_DIR, "_raw", "baysor", "segmentation.csv"),
    output:
        method_info=os.path.join(STAND_DIR, "baysor", "method_info.json"),
        transcripts=os.path.join(STAND_DIR, "baysor", "transcripts.parquet"),
        mtx=os.path.join(STAND_DIR, "baysor", "cell_by_gene.mtx"),
    log:
        os.path.join(OUTPUT_DIR, "logs", "standardize_baysor.log"),
    params:
        out_dir=os.path.join(STAND_DIR, "baysor"),
        roi_args=" ".join(roi_args()),
        qv_arg=(f"--qv-threshold {QV_THRESHOLD}" if QV_THRESHOLD is not None else ""),
        max_arg=(f"--max-transcripts {MAX_TRANSCRIPTS}" if MAX_TRANSCRIPTS else ""),
    threads: THREADS
    conda: "../../envs/benchmark.yml"
    shell:
        "python3 workflow/scripts/_benchmark/standardize_method_output.py "
        "--method baysor "
        "--baysor-segmentation-csv {input.seg} "
        "--xenium-dir {XENIUM_DIR} "
        "--out-dir {params.out_dir} "
        "--threads {threads} "
        "{params.qv_arg} {params.max_arg} {params.roi_args} "
        "--log {log}"


# ---------------------------------------------------------------------------
# proseg — expects transcript-metadata.csv.gz from proseg's raw output.
# ---------------------------------------------------------------------------

rule standardize_proseg:
    input:
        meta=os.path.join(OUTPUT_DIR, "_raw", "proseg", "transcript-metadata.csv.gz"),
    output:
        method_info=os.path.join(STAND_DIR, "proseg", "method_info.json"),
        transcripts=os.path.join(STAND_DIR, "proseg", "transcripts.parquet"),
        mtx=os.path.join(STAND_DIR, "proseg", "cell_by_gene.mtx"),
    log:
        os.path.join(OUTPUT_DIR, "logs", "standardize_proseg.log"),
    params:
        out_dir=os.path.join(STAND_DIR, "proseg"),
        roi_args=" ".join(roi_args()),
        qv_arg=(f"--qv-threshold {QV_THRESHOLD}" if QV_THRESHOLD is not None else ""),
        max_arg=(f"--max-transcripts {MAX_TRANSCRIPTS}" if MAX_TRANSCRIPTS else ""),
    threads: THREADS
    conda: "../../envs/benchmark.yml"
    shell:
        "python3 workflow/scripts/_benchmark/standardize_method_output.py "
        "--method proseg "
        "--proseg-transcript-metadata {input.meta} "
        "--out-dir {params.out_dir} "
        "--threads {threads} "
        "{params.qv_arg} {params.max_arg} {params.roi_args} "
        "--log {log}"


# ---------------------------------------------------------------------------
# segger — expects segger_transcripts.parquet from cleanSeggerPredictDir.
# ---------------------------------------------------------------------------

rule standardize_segger:
    input:
        transcripts=os.path.join(OUTPUT_DIR, "_raw", "segger", "segger_transcripts.parquet"),
    output:
        method_info=os.path.join(STAND_DIR, "segger", "method_info.json"),
        transcripts=os.path.join(STAND_DIR, "segger", "transcripts.parquet"),
        mtx=os.path.join(STAND_DIR, "segger", "cell_by_gene.mtx"),
    log:
        os.path.join(OUTPUT_DIR, "logs", "standardize_segger.log"),
    params:
        out_dir=os.path.join(STAND_DIR, "segger"),
        roi_args=" ".join(roi_args()),
        qv_arg=(f"--qv-threshold {QV_THRESHOLD}" if QV_THRESHOLD is not None else ""),
        max_arg=(f"--max-transcripts {MAX_TRANSCRIPTS}" if MAX_TRANSCRIPTS else ""),
    threads: THREADS
    conda: "../../envs/benchmark.yml"
    shell:
        "python3 workflow/scripts/_benchmark/standardize_method_output.py "
        "--method segger "
        "--segger-transcripts {input.transcripts} "
        "--out-dir {params.out_dir} "
        "--threads {threads} "
        "{params.qv_arg} {params.max_arg} {params.roi_args} "
        "--log {log}"

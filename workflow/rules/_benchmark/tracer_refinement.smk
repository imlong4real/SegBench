# TRACER refinement applied on top of a standardized base method.

def _tracer_repo_args() -> str:
    p = TRACER_CFG.get("repo_path")
    return f"--tracer-repo-path {p}" if p else ""


def _tracer_flag_args() -> str:
    flags = []
    if TRACER_CFG.get("enable_seg_residual_cascade"):
        flags.append("--enable-seg-residual-cascade")
    if TRACER_CFG.get("enable_noseg_cascade"):
        flags.append("--enable-noseg-cascade")
    mode = TRACER_CFG.get("mode") or "refine_existing_segmentation"
    flags += ["--mode", mode]
    npmi_source = TRACER_CFG.get("npmi_source")
    if npmi_source:
        flags += ["--npmi-source", npmi_source]
    gp = TRACER_CFG.get("gene_panel")
    if gp:
        flags += ["--gene-panel", gp]
    np_ = TRACER_CFG.get("npmi_path")
    if np_:
        flags += ["--npmi-path", np_]
    return " ".join(flags)


# Stubs are OFF by default. Only enabled when allow_stub: true is set
# *explicitly* in the config (used only by the smoke-test config).
def _tracer_allow_stub() -> str:
    return "--allow-stub" if TRACER_CFG.get("allow_stub", False) else ""


rule tracer_refine:
    input:
        transcripts=lambda wc: standardized_transcripts(wc.base),
        method_info=lambda wc: standardized_method_info(wc.base),
    output:
        method_info=os.path.join(STAND_DIR, "tracer_from_{base}", "method_info.json"),
        transcripts=os.path.join(STAND_DIR, "tracer_from_{base}", "transcripts.parquet"),
        mtx=os.path.join(STAND_DIR, "tracer_from_{base}", "cell_by_gene.mtx"),
    log:
        os.path.join(OUTPUT_DIR, "logs", "tracer_from_{base}.log"),
    params:
        out_dir=lambda wc: standardized_dir(f"tracer_from_{wc.base}"),
        tracer_args=_tracer_repo_args(),
        flag_args=_tracer_flag_args(),
        stub=_tracer_allow_stub(),
    wildcard_constraints:
        base="|".join(["xenium_default", "baysor", "proseg", "segger"]),
    threads: THREADS
    conda: "../../envs/benchmark.yml"
    shell:
        "python3 workflow/scripts/_benchmark/run_tracer_refine.py "
        "--base-transcripts {input.transcripts} "
        "--base-method {wildcards.base} "
        "--out-dir {params.out_dir} "
        "{params.tracer_args} {params.flag_args} {params.stub} "
        "--threads {threads} "
        "--log {log}"

# SPLIT benchmark targets — only meaningful when an RCTD post-processed RDS
# is available. Disabled by default in the example configs.

def _split_args() -> str:
    args = []
    rctd = SPLIT_CFG.get("rctd_post_processed_rds")
    if rctd:
        args += ["--rctd-rds", rctd]
    if SPLIT_CFG.get("allow_stub", False):
        args.append("--allow-stub")
    return " ".join(args)


rule split_correct:
    input:
        method_info=os.path.join(STAND_DIR, "{base}", "method_info.json"),
        mtx=os.path.join(STAND_DIR, "{base}", "cell_by_gene.mtx"),
    output:
        method_info=os.path.join(OUTPUT_DIR, "split_from_{base}", "method_info.json"),
        corrected=os.path.join(OUTPUT_DIR, "split_from_{base}", "corrected_counts.mtx"),
    log:
        os.path.join(OUTPUT_DIR, "logs", "split_from_{base}.log"),
    params:
        stand_dir=lambda wc: standardized_dir(wc.base),
        out_dir=lambda wc: os.path.join(OUTPUT_DIR, f"split_from_{wc.base}"),
        extra=_split_args(),
    threads: max(1, THREADS // 2)
    conda: "../../envs/celladmix.yml"  # reuses the R env
    shell:
        "Rscript workflow/scripts/_benchmark/run_split_benchmark.R "
        "--standardized-dir {params.stand_dir} "
        "--out-dir {params.out_dir} "
        "--base-method {wildcards.base} "
        "{params.extra} --log {log}"

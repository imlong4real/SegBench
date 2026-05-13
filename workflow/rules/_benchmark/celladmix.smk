# cellAdmix benchmark targets — applied on top of a standardized method.

def _celladmix_args() -> str:
    args = []
    if CELLADMIX_CFG.get("repo_path"):
        args += ["--repo-path", CELLADMIX_CFG["repo_path"]]
    args += ["--admixture-threshold", str(CELLADMIX_CFG.get("admixture_threshold", 0.1))]
    args += ["--min-cells-per-cluster", str(CELLADMIX_CFG.get("min_cells_per_cluster", 25))]
    if CELLADMIX_CFG.get("allow_stub", False):
        args.append("--allow-stub")
    return " ".join(args)


rule celladmix_correct:
    input:
        method_info=os.path.join(STAND_DIR, "{base}", "method_info.json"),
        mtx=os.path.join(STAND_DIR, "{base}", "cell_by_gene.mtx"),
        cluster_labels=CLUSTER_LABELS or [],
    output:
        method_info=os.path.join(OUTPUT_DIR, "celladmix_from_{base}", "method_info.json"),
        corrected=os.path.join(OUTPUT_DIR, "celladmix_from_{base}", "corrected_counts.mtx"),
    log:
        os.path.join(OUTPUT_DIR, "logs", "celladmix_from_{base}.log"),
    params:
        stand_dir=lambda wc: standardized_dir(wc.base),
        out_dir=lambda wc: os.path.join(OUTPUT_DIR, f"celladmix_from_{wc.base}"),
        clusters_arg=(f"--cluster-labels {CLUSTER_LABELS}" if CLUSTER_LABELS else ""),
        extra=_celladmix_args(),
    threads: max(1, THREADS // 2)
    conda: "../../envs/celladmix.yml"
    shell:
        "Rscript workflow/scripts/_benchmark/run_celladmix.R "
        "--standardized-dir {params.stand_dir} "
        "--out-dir {params.out_dir} "
        "--base-method {wildcards.base} "
        "{params.clusters_arg} {params.extra} "
        "--log {log}"

rule celladmix_xenium_default_real_tsu20:
    input:
        common_info="results/tsu20_tools/common_inputs/common_inputs_info.json",
    output:
        cleaned_transcripts="results/tsu20_tools/celladmix_xenium_default/cleaned_transcripts.parquet",
        corrected_counts="results/tsu20_tools/celladmix_xenium_default/corrected_counts.mtx",
        method_info="results/tsu20_tools/celladmix_xenium_default/method_info.json",
    params:
        xenium_dir=config["dataset"]["xenium_dir"],
        clusters=config["dataset"]["clusters"],
        outdir="results/tsu20_tools/celladmix_xenium_default",
        common_inputs=config.get("celladmix", {}).get("common_inputs", "results/tsu20_tools/common_inputs"),
        num_factors=config.get("celladmix", {}).get("num_factors", 10),
        nmol_dsamp=config.get("celladmix", {}).get("nmol_dsamp", 10000),
        n_cells_nmf=config.get("celladmix", {}).get("n_cells_nmf", 2000),
        cores=config.get("celladmix", {}).get("cores", 1),
    shell:
        "Rscript workflow/scripts/_count_correction/run_celladmix_tsu20_real.R "
        "--xenium-dir {params.xenium_dir} "
        "--clusters {params.clusters} "
        "--outdir {params.outdir} "
        "--common-inputs {params.common_inputs} "
        "--num-factors {params.num_factors} "
        "--nmol-dsamp {params.nmol_dsamp} "
        "--n-cells-nmf {params.n_cells_nmf} "
        "--cores {params.cores}"

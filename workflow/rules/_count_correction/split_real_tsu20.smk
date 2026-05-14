rule prepare_tsu20_common_inputs:
    input:
        h5ad=config["dataset"]["scrna_h5ad"],
        clusters=config["dataset"]["clusters"],
    output:
        xenium_counts="results/tsu20_tools/common_inputs/xenium_counts.mtx",
        scrna_counts="results/tsu20_tools/common_inputs/scrna_reference_counts.mtx",
        info="results/tsu20_tools/common_inputs/common_inputs_info.json",
    params:
        xenium_dir=config["dataset"]["xenium_dir"],
        outdir="results/tsu20_tools/common_inputs",
    shell:
        "python workflow/scripts/_count_correction/prepare_tsu20_common_inputs.py "
        "--xenium-dir {params.xenium_dir} "
        "--scrna-h5ad {input.h5ad} "
        "--clusters {input.clusters} "
        "--outdir {params.outdir}"


rule split_xenium_default_real_tsu20:
    input:
        common_info="results/tsu20_tools/common_inputs/common_inputs_info.json",
    output:
        split_result="results/tsu20_tools/split_xenium_default/split_result.rds",
        purified_counts="results/tsu20_tools/split_xenium_default/purified_counts.mtx",
        post_processed_rctd="results/tsu20_tools/split_xenium_default/post_processed_RCTD.rds",
        method_info="results/tsu20_tools/split_xenium_default/method_info.json",
    params:
        xenium_dir=config["dataset"]["xenium_dir"],
        scrna_h5ad=config["dataset"]["scrna_h5ad"],
        celltype_column=config.get("split", {}).get("reference_celltype_column", "auto"),
        outdir="results/tsu20_tools/split_xenium_default",
        common_inputs=config.get("split", {}).get("common_inputs", "results/tsu20_tools/common_inputs"),
    shell:
        "Rscript workflow/scripts/_count_correction/run_split_tsu20_real.R "
        "--xenium-dir {params.xenium_dir} "
        "--scrna-h5ad {params.scrna_h5ad} "
        "--celltype-column {params.celltype_column} "
        "--outdir {params.outdir} "
        "--common-inputs {params.common_inputs}"

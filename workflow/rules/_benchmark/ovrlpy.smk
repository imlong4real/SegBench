# ovrlpy benchmark targets — runs on a standardized transcripts table.

def _ovrlpy_args() -> str:
    args = []
    cell_d = OVRLPY_CFG.get("cell_diameter", 10)
    n_ct = OVRLPY_CFG.get("n_expected_celltypes", 30)
    args += ["--cell-diameter", str(cell_d), "--n-expected-celltypes", str(n_ct)]
    if OVRLPY_CFG.get("allow_stub", False):
        args.append("--allow-stub")
    return " ".join(args)


rule ovrlpy_run:
    input:
        method_info=os.path.join(STAND_DIR, "{base}", "method_info.json"),
        transcripts=os.path.join(STAND_DIR, "{base}", "transcripts.parquet"),
    output:
        method_info=os.path.join(OUTPUT_DIR, "ovrlpy_from_{base}", "method_info.json"),
        signal_integrity=os.path.join(OUTPUT_DIR, "ovrlpy_from_{base}", "signal_integrity.parquet"),
    log:
        os.path.join(OUTPUT_DIR, "logs", "ovrlpy_from_{base}.log"),
    params:
        stand_dir=lambda wc: standardized_dir(wc.base),
        out_dir=lambda wc: os.path.join(OUTPUT_DIR, f"ovrlpy_from_{wc.base}"),
        extra=_ovrlpy_args(),
    threads: THREADS
    conda: "../../envs/ovrlpy.yml"
    shell:
        "python3 workflow/scripts/_benchmark/run_ovrlpy_benchmark.py "
        "--standardized-dir {params.stand_dir} "
        "--out-dir {params.out_dir} "
        "{params.extra} --log {log}"

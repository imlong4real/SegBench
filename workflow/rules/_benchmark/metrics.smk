# NPMI + marker metrics + collection / report rules.

def _tracer_repo_args_for_metrics() -> str:
    p = TRACER_CFG.get("repo_path")
    return f"--tracer-repo-path {p}" if p else ""


rule npmi_metrics:
    input:
        method_info=os.path.join(STAND_DIR, "{method}", "method_info.json"),
        mtx=os.path.join(STAND_DIR, "{method}", "cell_by_gene.mtx"),
    output:
        per_cell=os.path.join(METRICS_DIR, "{method}", "npmi_per_cell.parquet"),
        summary=os.path.join(METRICS_DIR, "{method}", "npmi_summary.parquet"),
    log:
        os.path.join(OUTPUT_DIR, "logs", "npmi_{method}.log"),
    params:
        stand_dir=lambda wc: standardized_dir(wc.method),
        tracer_args=_tracer_repo_args_for_metrics(),
        min_mol=int(METRICS_CFG.get("npmi", {}).get("min_molecules_per_cell", 10)),
    threads: max(1, THREADS // 2)
    conda: "../../envs/benchmark.yml"
    shell:
        "python3 workflow/scripts/_benchmark/run_npmi_metrics.py "
        "--standardized-dir {params.stand_dir} "
        "--out-per-cell {output.per_cell} "
        "--out-summary {output.summary} "
        "{params.tracer_args} "
        "--min-molecules-per-cell {params.min_mol} "
        "--log {log}"


rule marker_specificity:
    input:
        method_info=os.path.join(STAND_DIR, "{method}", "method_info.json"),
        mtx=os.path.join(STAND_DIR, "{method}", "cell_by_gene.mtx"),
        marker_set=METRICS_CFG.get("marker_set", "resources/marker_sets/lung_cancer_markers.yml"),
    output:
        per_cell=os.path.join(METRICS_DIR, "{method}", "marker_per_cell.parquet"),
        summary=os.path.join(METRICS_DIR, "{method}", "marker_summary.parquet"),
    log:
        os.path.join(OUTPUT_DIR, "logs", "marker_{method}.log"),
    params:
        stand_dir=lambda wc: standardized_dir(wc.method),
    threads: max(1, THREADS // 2)
    conda: "../../envs/benchmark.yml"
    shell:
        "python3 workflow/scripts/_benchmark/run_marker_specificity.py "
        "--standardized-dir {params.stand_dir} "
        "--marker-set {input.marker_set} "
        "--out-per-cell {output.per_cell} "
        "--out-summary {output.summary} "
        "--log {log}"


def _collect_inputs():
    methods = all_benchmark_methods()
    files: dict[str, list[str]] = {
        "method_info": [standardized_method_info(m) for m in methods],
        "mtx": [standardized_marker_done(m) for m in methods],
        "npmi": [npmi_summary(m) for m in methods],
        "marker": [marker_summary(m) for m in methods],
    }
    return files


def _bundle_args() -> list[str]:
    args = []
    for m in all_benchmark_methods():
        args.append(
            f"--method-bundle method={m},"
            f"standardized={standardized_dir(m)},"
            f"npmi={npmi_summary(m)},"
            f"marker={marker_summary(m)}"
        )
    return args


rule collect_metrics:
    input:
        **_collect_inputs(),
    output:
        csv=os.path.join(SUMMARY_DIR, "metrics_all_methods.csv"),
        parquet=os.path.join(SUMMARY_DIR, "metrics_all_methods.parquet"),
        runtime_csv=os.path.join(SUMMARY_DIR, "method_runtime_summary.csv"),
    log:
        os.path.join(OUTPUT_DIR, "logs", "collect_metrics.log"),
    params:
        bundles=" ".join(_bundle_args()),
    conda: "../../envs/benchmark.yml"
    shell:
        "python3 workflow/scripts/_benchmark/collect_metrics.py "
        "{params.bundles} "
        "--out-csv {output.csv} "
        "--out-parquet {output.parquet} "
        "--out-runtime-csv {output.runtime_csv} "
        "--log {log}"


rule benchmark_report:
    input:
        csv=os.path.join(SUMMARY_DIR, "metrics_all_methods.csv"),
    output:
        report=os.path.join(SUMMARY_DIR, "benchmark_report.html"),
        fig1=os.path.join(FIGURES_DIR, "purity_conflict_paired.pdf"),
        fig2=os.path.join(FIGURES_DIR, "marker_specificity_heatmap.pdf"),
        fig3=os.path.join(FIGURES_DIR, "transcript_retention_vs_conflict.pdf"),
        fig4=os.path.join(FIGURES_DIR, "cell_count_transcript_count.pdf"),
        fig5=os.path.join(FIGURES_DIR, "runtime_summary.pdf"),
    log:
        os.path.join(OUTPUT_DIR, "logs", "benchmark_report.log"),
    conda: "../../envs/benchmark.yml"
    shell:
        "python3 workflow/scripts/_benchmark/plot_benchmark_summary.py "
        "--metrics-csv {input.csv} "
        "--figures-dir {FIGURES_DIR} "
        "--report-html {output.report} "
        "--log {log}"

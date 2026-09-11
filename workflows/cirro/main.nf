nextflow.enable.dsl = 2

def requiredParam(String name, value) {
    if (value == null || value.toString().trim().isEmpty()) error "Missing required parameter --${name}"
    value
}

def resolveDatasetPath(value, inputDir) {
    if (value == null || value.toString().trim().isEmpty()) return value
    def candidate = value.toString()
    if (candidate ==~ /^[A-Za-z][A-Za-z0-9+.-]*:\/\/.*/ || candidate.startsWith('/')) return candidate
    if (inputDir == null || inputDir.toString().trim().isEmpty()) return candidate
    def base = inputDir.toString().replaceFirst('/+$', '')
    if (candidate == '.' || candidate == './') return base
    def relative = candidate.replaceFirst('^/+', '')
    def leaf = base.tokenize('/').last()
    if (relative == leaf) relative = ''
    else if (relative.startsWith("${leaf}/")) relative = relative.substring(leaf.size() + 1)
    relative ? "${base}/${relative}" : base
}

process PREP_XENIUM {
    tag "${sample_name}-${run_scope}"
    label 'prep'
    input:
    path transcripts
    path xenium_dir
    path clusters
    path reference_train
    path exact_bundle_script
    path common_script
    val sample_name
    val run_scope
    val smoke_transcripts
    output:
    path 'xenium_prepared', emit: prepared
    script:
    def cap = run_scope == 'smoke' ? smoke_transcripts : 0
    """
    mkdir -p xenium_prepared/exact_xenium_bundle xenium_prepared/common_inputs
    cp '${transcripts}' xenium_prepared/filtered_transcripts.parquet
    python '${exact_bundle_script}' \
      --transcripts xenium_prepared/filtered_transcripts.parquet \
      --xenium-dir '${xenium_dir}' \
      --outdir xenium_prepared/exact_xenium_bundle \
      --qv-min 20 --max-transcripts '${cap}'
    cp xenium_prepared/exact_xenium_bundle/standardized_transcripts.parquet xenium_prepared/exact_transcripts.parquet
    python '${common_script}' \
      --xenium-dir xenium_prepared/exact_xenium_bundle \
      --scrna-h5ad '${reference_train}' \
      --clusters '${clusters}' \
      --outdir xenium_prepared/common_inputs \
      --celltype-column Cell_Cluster_level1
    cp xenium_prepared/exact_xenium_bundle/frozen_input_receipt.json xenium_prepared/frozen_input_receipt.json
    """
}

process PREP_BAYSOR {
    tag "${sample_name}"
    label 'prep'
    input:
    path prepared
    path prep_script
    val sample_name
    output:
    path 'baysor_input', emit: input
    script:
    """
    python '${prep_script}' --transcripts '${prepared}/exact_transcripts.parquet' \
      --outdir baysor_input --scale 8.0 --min-molecules 50
    """
}

process BAYSOR {
    tag "${sample_name}"
    label 'baysor'
    input:
    path baysor_input
    val sample_name
    output:
    path 'baysor_native', emit: native_output
    script:
    """
    mkdir -p baysor_native/raw
    start=\$(date +%s.%N)
    /usr/bin/time -v -o baysor_native/time.txt \
      baysor run -c '${baysor_input}/baysor_config.toml' \
      -o "\$PWD/baysor_native/raw/" -m 50 -s 8.0 \
      --polygon-format none '${baysor_input}/baysor_input.csv' \
      > baysor_native/run.log 2>&1
    end=\$(date +%s.%N)
    awk -v a="\$start" -v b="\$end" 'BEGIN {print b-a}' > baysor_native/wall_seconds.txt
    cp '${baysor_input}/baysor_config.toml' baysor_native/
    test -s baysor_native/raw/segmentation.csv
    """
}

process STANDARDIZE_BAYSOR {
    tag "${sample_name}"
    label 'standardize'
    input:
    path prepared
    path native_dir
    path standardizer
    val sample_name
    val seed
    output:
    path 'baysor', emit: results
    script:
    """
    wall=\$(cat '${native_dir}/wall_seconds.txt')
    python '${standardizer}' --transcripts '${prepared}/exact_transcripts.parquet' \
      --segmentation '${native_dir}/raw/segmentation.csv' --time-file '${native_dir}/time.txt' \
      --native-dir '${native_dir}/raw' --outdir baysor --sample-name '${sample_name}' \
      --seed '${seed}' --wall-seconds "\$wall" --requested-cpus '${task.cpus}' \
      --requested-memory-gb '${task.memory.toGiga()}'
    cp '${native_dir}/run.log' baysor/run.log
    """
}

process PROSEG {
    tag "${sample_name}"
    label 'proseg'
    input:
    path prepared
    path resource_runner
    val sample_name
    val seed
    output:
    path 'proseg', emit: results
    script:
    """
    mkdir -p proseg
    python '${resource_runner}' --output proseg/resource_usage.json --log proseg/container.log \
      --requested-cpus '${task.cpus}' --requested-memory-gb '${task.memory.toGiga()}' \
      --method proseg -- \
      python -m segbench run proseg --transcripts '${prepared}/exact_transcripts.parquet' \
      --outdir proseg --sample-name '${sample_name}' --seed '${seed}' \
      --threads '${task.cpus}' --voxel-layers 4 --overwrite
    """
}

process SEGGER {
    tag "${sample_name}"
    label 'segger_gpu'
    input:
    val cpu_method_gate
    path prepared
    path seeded_cli
    path resource_runner
    val sample_name
    val seed
    output:
    path 'segger_native', emit: native_output
    script:
    """
    mkdir -p segger_native/output
    python '${resource_runner}' --output segger_native/resource_usage.json \
      --log segger_native/run.log --requested-cpus '${task.cpus}' \
      --requested-memory-gb '${task.memory.toGiga()}' --requested-gpus 1 --method segger -- \
      python '${seeded_cli}' --seed '${seed}' -- segment \
      -i '${prepared}/exact_xenium_bundle' -o segger_native/output \
      --n-epochs 20 --save-anndata
    test -s segger_native/output/segger_segmentation.parquet
    """
}

process STANDARDIZE_SEGGER {
    tag "${sample_name}"
    label 'standardize'
    input:
    path prepared
    path native_dir
    path standardizer
    val sample_name
    val seed
    output:
    path 'segger', emit: results
    script:
    """
    python '${standardizer}' --transcripts '${prepared}/exact_transcripts.parquet' \
      --segger-output '${native_dir}/output' --resource-usage '${native_dir}/resource_usage.json' \
      --outdir segger --sample-name '${sample_name}' --seed '${seed}'
    cp '${native_dir}/run.log' segger/run.log
    """
}

process SPLIT {
    tag "${sample_name}"
    label 'r_method'
    input:
    path prepared
    path reference_train
    path resource_runner
    val sample_name
    val seed
    output:
    path 'split', emit: results
    script:
    """
    mkdir -p split
    python '${resource_runner}' --output split/resource_usage.json --log split/container.log \
      --requested-cpus '${task.cpus}' --requested-memory-gb '${task.memory.toGiga()}' --method split -- \
      python -m segbench run split --transcripts '${prepared}/exact_transcripts.parquet' \
      --xenium-dir '${prepared}/exact_xenium_bundle' --reference-h5ad '${reference_train}' \
      --reference-celltype-col Cell_Cluster_level1 --common-inputs '${prepared}/common_inputs' \
      --outdir split --sample-name '${sample_name}' --seed '${seed}' --threads '${task.cpus}' \
      --umi-min 10 --counts-min 10 --overwrite
    """
}

process CELLADMIX {
    tag "${sample_name}"
    label 'r_method'
    input:
    path prepared
    path clusters
    path resource_runner
    val sample_name
    val seed
    output:
    path 'celladmix', emit: results
    script:
    """
    mkdir -p celladmix
    python '${resource_runner}' --output celladmix/resource_usage.json --log celladmix/container.log \
      --requested-cpus '${task.cpus}' --requested-memory-gb '${task.memory.toGiga()}' --method celladmix -- \
      python -m segbench run celladmix --transcripts '${prepared}/exact_transcripts.parquet' \
      --xenium-dir '${prepared}/exact_xenium_bundle' --clusters '${clusters}' \
      --common-inputs '${prepared}/common_inputs' --outdir celladmix \
      --sample-name '${sample_name}' --seed '${seed}' --threads '${task.cpus}' \
      --num-factors 10 --nmol-dsamp 10000 --n-cells-nmf 2000 --bridge-cells 200 --overwrite
    """
}

process TRACER_SEG {
    tag "${sample_name}"
    label 'tracer'
    input:
    path transcripts
    path pmi
    path segbench_src
    path resource_runner
    val sample_name
    val seed
    output:
    path 'tracer_seg', emit: results
    script:
    """
    mkdir -p tracer_seg
    export PYTHONPATH="\$PWD/${segbench_src}:/app/src"
    export SEGBENCH_NO_REEXEC=1 TRACER_HOME=/app PYTHONHASHSEED='${seed}'
    python '${resource_runner}' --output tracer_seg/resource_usage.json --log tracer_seg/container.log \
      --requested-cpus '${task.cpus}' --requested-memory-gb '${task.memory.toGiga()}' --method tracer_seg -- \
      python -m segbench run tracer --transcripts '${transcripts}' --pmi '${pmi}' \
      --platform xenium --outdir tracer_seg --sample-name '${sample_name}' --seed '${seed}' \
      --threads '${task.cpus}' --tau 0.05 --min-tx-per-cell-for-scores 5 --overwrite
    """
}

process PREP_KIDNEY_SEG {
    tag "${sample_name}-${run_scope}"
    label 'prep'
    input:
    path seg_input
    path prep_script
    val sample_name
    val run_scope
    val smoke_rows
    output:
    path 'kidney_prepared', emit: prepared
    script:
    def cap = run_scope == 'smoke' ? smoke_rows : 0
    """
    mkdir -p kidney_prepared
    python '${prep_script}' --input '${seg_input}' --output kidney_prepared/kidney_seg_input.parquet \
      --max-rows '${cap}'
    """
}

process TRACER_NOSEG {
    tag "${sample_name}"
    label 'tracer'
    input:
    path matrix_dir
    path spatial_dir
    path pmi
    path segbench_src
    path resource_runner
    val sample_name
    val seed
    val run_scope
    val smoke_transcripts
    output:
    path 'tracer_noseg', emit: results
    script:
    def smokeArgs = run_scope == 'smoke' ? "--smoke --roi-size-um 500 --max-transcripts ${smoke_transcripts}" : ''
    """
    mkdir -p tracer_noseg
    export PYTHONPATH="\$PWD/${segbench_src}:/app/src"
    export SEGBENCH_NO_REEXEC=1 TRACER_HOME=/app PYTHONHASHSEED='${seed}'
    python '${resource_runner}' --output tracer_noseg/resource_usage.json --log tracer_noseg/container.log \
      --requested-cpus '${task.cpus}' --requested-memory-gb '${task.memory.toGiga()}' --method tracer_noseg -- \
      python -m segbench run tracer_seq --visiumhd-matrix '${matrix_dir}' --spatial-dir '${spatial_dir}' \
      --pmi '${pmi}' --bin-size-um 2 --outdir tracer_noseg --sample-name '${sample_name}' \
      --seed '${seed}' --threads '${task.cpus}' ${smokeArgs} --overwrite
    """
}

process BIN2CELL {
    tag "${sample_name}-${run_scope}"
    label 'bin2cell'
    input:
    path input_h5
    path spaceranger_dir
    path source_image
    path resource_runner
    path bin2cell_runner
    val sample_name
    val seed
    val run_scope
    val smoke_bins
    output:
    path 'bin2cell', emit: results
    script:
    def cap = run_scope == 'smoke' ? "--max-transcripts ${smoke_bins}" : ''
    def predictionMode = run_scope == 'smoke' ? 'direct' : 'tiled'
    def blockSize = run_scope == 'smoke' ? 512 : 4096
    """
    mkdir -p bin2cell
    python '${resource_runner}' --output bin2cell/resource_usage.json --log bin2cell/container.log \
      --requested-cpus '${task.cpus}' --requested-memory-gb '${task.memory.toGiga()}' --method bin2cell -- \
      python '${bin2cell_runner}' --prediction-mode '${predictionMode}' --block-size '${blockSize}' -- \
      --input-h5ad '${input_h5}' --spaceranger-dir '${spaceranger_dir}' \
      --source-image '${source_image}' --outdir bin2cell --sample-name '${sample_name}' \
      --seed '${seed}' --threads '${task.cpus}' ${cap} --mpp 0.5 --prob-thresh 0.01 \
      --nms-thresh 0.5 --stardist-model 2D_versatile_he --expand-algorithm max_bin_distance \
      --max-bin-distance 2 --expand-k 4 --min-counts 1 --overwrite
    """
}

process EVALUATE_XENIUM {
    tag "${sample_name}-${run_scope}"
    label 'evaluate'
    publishDir params.outdir, mode: 'copy', overwrite: true
    input:
    path method_results
    path reference_holdout
    path pmi
    path split_manifest
    path input_manifest
    path frozen_manifest
    path tidy_script
    path segbench_src
    path rctd_script
    val sample_name
    val run_scope
    val workflow_revision
    output:
    path 'benchmark_results', emit: results
    script:
    def methodDirs = method_results.collect { "'${it}'" }.join(' ')
    """
    mkdir -p benchmark_results/methods benchmark_results/evaluation
    for d in ${methodDirs}; do cp -r "\$d" benchmark_results/methods/; done
    export SEGBENCH_ENV_CONFIG=/opt/segbench/workflows/cirro/configs/environments.container.yaml
    export PYTHONPATH='${segbench_src}'
    export SEGBENCH_RCTD_SCRIPT='${rctd_script}'
    export RETICULATE_PYTHON=/opt/conda/bin/python
    /opt/conda/bin/Rscript -e 'cfg <- reticulate::py_config(); stopifnot(normalizePath(cfg\$python) == normalizePath(Sys.getenv("RETICULATE_PYTHON"))); cat(sprintf("reticulate_python=%s\\npython_version=%s\\n", cfg\$python, cfg\$version))' \
      > benchmark_results/evaluation/rctd_environment_receipt.txt
    python -m segbench evaluate benchmark_results/methods --dataset tsu20_lung \
      --reference-h5ad '${reference_holdout}' --reference-celltype-col Cell_Cluster_level1 \
      --min-reference-cells 50 --rctd-cores '${task.cpus}' --outdir benchmark_results/evaluation
    python -c 'from pathlib import Path; logs=sorted(Path("benchmark_results/methods").glob("*/rctd/rctd.log")); print("\\n".join("===== " + str(p) + " =====\\n" + p.read_text(errors="replace") for p in logs), flush=True)'
    python -c 'import pandas as pd; d=pd.read_csv("benchmark_results/evaluation/comparison_table.csv"); bad=d.loc[~d["rctd_status"].fillna("").astype(str).str.startswith("ok"), ["method", "rctd_status"]]; assert bad.empty, "RCTD gate failed:\\n" + bad.to_string(index=False)'
    python '${tidy_script}' --comparison benchmark_results/evaluation/comparison_table.csv \
      --methods-root benchmark_results/methods --dataset TSU-20 --platform Xenium \
      --replicate 1 --frozen-manifest '${frozen_manifest}' --split-manifest '${split_manifest}' \
      --workflow-revision '${workflow_revision}' \
      --input-receipt '${input_manifest}' --pmi '${pmi}' --outdir benchmark_results/evaluation
    cp '${input_manifest}' benchmark_results/input_checksums.json
    cp '${split_manifest}' benchmark_results/evaluation/reference_split_manifest.json
    """
}

process EVALUATE_KIDNEY {
    tag "${sample_name}-${run_scope}"
    label 'evaluate'
    publishDir params.outdir, mode: 'copy', overwrite: true
    input:
    path method_results
    path reference_holdout
    path pmi
    path input_receipt
    path input_manifest
    path frozen_manifest
    path tidy_script
    path segbench_src
    path rctd_script
    val sample_name
    val run_scope
    val workflow_revision
    output:
    path 'benchmark_results', emit: results
    script:
    def methodDirs = method_results.collect { "'${it}'" }.join(' ')
    """
    mkdir -p benchmark_results/methods benchmark_results/evaluation
    for d in ${methodDirs}; do cp -r "\$d" benchmark_results/methods/; done
    export SEGBENCH_ENV_CONFIG=/opt/segbench/workflows/cirro/configs/environments.container.yaml
    export PYTHONPATH='${segbench_src}'
    export SEGBENCH_RCTD_SCRIPT='${rctd_script}'
    export RETICULATE_PYTHON=/opt/conda/bin/python
    /opt/conda/bin/Rscript -e 'cfg <- reticulate::py_config(); stopifnot(normalizePath(cfg\$python) == normalizePath(Sys.getenv("RETICULATE_PYTHON"))); cat(sprintf("reticulate_python=%s\\npython_version=%s\\n", cfg\$python, cfg\$version))' \
      > benchmark_results/evaluation/rctd_environment_receipt.txt
    python -m segbench evaluate benchmark_results/methods --dataset kidney_visiumhd \
      --reference-h5ad '${reference_holdout}' --reference-celltype-col lineage \
      --min-reference-cells 50 --rctd-cores '${task.cpus}' --outdir benchmark_results/evaluation
    python -c 'from pathlib import Path; logs=sorted(Path("benchmark_results/methods").glob("*/rctd/rctd.log")); print("\\n".join("===== " + str(p) + " =====\\n" + p.read_text(errors="replace") for p in logs), flush=True)'
    python -c 'import pandas as pd; d=pd.read_csv("benchmark_results/evaluation/comparison_table.csv"); bad=d.loc[~d["rctd_status"].fillna("").astype(str).str.startswith("ok"), ["method", "rctd_status"]]; assert bad.empty, "RCTD gate failed:\\n" + bad.to_string(index=False)'
    python '${tidy_script}' --comparison benchmark_results/evaluation/comparison_table.csv \
      --methods-root benchmark_results/methods --dataset kidney --platform VisiumHD \
      --replicate 1 --frozen-manifest '${frozen_manifest}' --input-receipt '${input_manifest}' \
      --workflow-revision '${workflow_revision}' \
      --pmi '${pmi}' --outdir benchmark_results/evaluation
    cp '${input_manifest}' benchmark_results/input_checksums.json
    cp '${input_receipt}' benchmark_results/effective_seg_input_receipt.json
    """
}

workflow {
    requiredParam('dataset_kind', params.dataset_kind)
    requiredParam('input_manifest', params.input_manifest)
    requiredParam('pmi', params.pmi)
    requiredParam('reference_holdout', params.reference_holdout)
    if (!(params.dataset_kind in ['xenium_lung', 'kidney_visiumhd'])) error "--dataset_kind must be xenium_lung or kidney_visiumhd"
    if (!(params.run_scope in ['smoke', 'full'])) error "--run_scope must be smoke or full"

    def inputDir = params.input_dir
    pmi_ch = Channel.fromPath(resolveDatasetPath(params.pmi, inputDir), checkIfExists: true)
    holdout_ch = Channel.fromPath(resolveDatasetPath(params.reference_holdout, inputDir), checkIfExists: true)
    input_manifest_ch = Channel.fromPath(resolveDatasetPath(params.input_manifest, inputDir), checkIfExists: true)
    frozen_manifest_ch = Channel.fromPath(file("${projectDir}/frozen_manifest.json"), checkIfExists: true)
    tidy_script_ch = Channel.fromPath(file("${projectDir}/bin/make_tidy_outputs.py"), checkIfExists: true)
    resource_script_ch = Channel.fromPath(file("${projectDir}/bin/run_with_resources.py"), checkIfExists: true)
    segbench_src_ch = Channel.fromPath(file("${projectDir}/../../src"), checkIfExists: true)
    rctd_script_ch = Channel.fromPath(file("${projectDir}/../../workflow/scripts/run_rctd.R"), checkIfExists: true)

    if (params.dataset_kind == 'xenium_lung') {
        requiredParam('transcripts', params.transcripts); requiredParam('xenium_dir', params.xenium_dir)
        requiredParam('clusters', params.clusters); requiredParam('reference_train', params.reference_train)
        requiredParam('reference_split_manifest', params.reference_split_manifest)
        transcripts_ch = Channel.fromPath(resolveDatasetPath(params.transcripts, inputDir), checkIfExists: true)
        xenium_ch = Channel.fromPath(resolveDatasetPath(params.xenium_dir, inputDir), checkIfExists: true)
        clusters_ch = Channel.fromPath(resolveDatasetPath(params.clusters, inputDir), checkIfExists: true)
        train_ch = Channel.fromPath(resolveDatasetPath(params.reference_train, inputDir), checkIfExists: true)
        split_manifest_ch = Channel.fromPath(resolveDatasetPath(params.reference_split_manifest, inputDir), checkIfExists: true)
        exact_script_ch = Channel.fromPath(file("${projectDir}/bin/make_exact_xenium_bundle.py"), checkIfExists: true)
        common_script_ch = Channel.fromPath(file("${projectDir}/../../workflow/scripts/_count_correction/prepare_tsu20_common_inputs.py"), checkIfExists: true)
        baysor_prep_ch = Channel.fromPath(file("${projectDir}/bin/prepare_baysor.py"), checkIfExists: true)
        baysor_std_ch = Channel.fromPath(file("${projectDir}/bin/standardize_baysor.py"), checkIfExists: true)
        segger_cli_ch = Channel.fromPath(file("${projectDir}/bin/seeded_segger_cli.py"), checkIfExists: true)
        segger_std_ch = Channel.fromPath(file("${projectDir}/bin/standardize_segger_v2.py"), checkIfExists: true)

        PREP_XENIUM(transcripts_ch, xenium_ch, clusters_ch, train_ch, exact_script_ch,
                    common_script_ch, params.sample_name, params.run_scope, params.smoke_xenium_transcripts)
        PREP_BAYSOR(PREP_XENIUM.out.prepared, baysor_prep_ch, params.sample_name)
        BAYSOR(PREP_BAYSOR.out.input, params.sample_name)
        STANDARDIZE_BAYSOR(PREP_XENIUM.out.prepared, BAYSOR.out.native_output, baysor_std_ch,
                           params.sample_name, params.seed)
        PROSEG(PREP_XENIUM.out.prepared, resource_script_ch, params.sample_name, params.seed)
        SPLIT(PREP_XENIUM.out.prepared, train_ch, resource_script_ch, params.sample_name, params.seed)
        CELLADMIX(PREP_XENIUM.out.prepared, clusters_ch, resource_script_ch, params.sample_name, params.seed)
        TRACER_SEG(PREP_XENIUM.out.prepared.map{ it.resolve('exact_transcripts.parquet') }, pmi_ch,
                   segbench_src_ch, resource_script_ch, params.sample_name, params.seed)
        // Submit the GPU task only after every independent CPU method finishes.
        // This prevents an unavailable GPU instance from head-of-line blocking
        // CPU jobs in a shared Cirro/AWS Batch queue.  Mapping paths to scalar
        // values creates a scheduling barrier without staging method outputs.
        cpu_method_gate = STANDARDIZE_BAYSOR.out.results
            .mix(PROSEG.out.results, SPLIT.out.results, CELLADMIX.out.results, TRACER_SEG.out.results)
            .map { 1 }
            .collect()
        SEGGER(cpu_method_gate, PREP_XENIUM.out.prepared, segger_cli_ch, resource_script_ch,
               params.sample_name, params.seed)
        STANDARDIZE_SEGGER(PREP_XENIUM.out.prepared, SEGGER.out.native_output, segger_std_ch,
                           params.sample_name, params.seed)
        methods = STANDARDIZE_BAYSOR.out.results.mix(PROSEG.out.results, STANDARDIZE_SEGGER.out.results,
                  SPLIT.out.results, CELLADMIX.out.results, TRACER_SEG.out.results).collect()
        EVALUATE_XENIUM(methods, holdout_ch, pmi_ch, split_manifest_ch, input_manifest_ch,
                        frozen_manifest_ch, tidy_script_ch, segbench_src_ch, rctd_script_ch,
                        params.sample_name, params.run_scope, workflow.revision ?: 'unknown')
    } else {
        requiredParam('kidney_seg_input', params.kidney_seg_input)
        requiredParam('visiumhd_matrix', params.visiumhd_matrix); requiredParam('spatial_dir', params.spatial_dir)
        requiredParam('bin2cell_h5', params.bin2cell_h5); requiredParam('spaceranger_dir', params.spaceranger_dir)
        requiredParam('source_image', params.source_image)
        seg_input_ch = Channel.fromPath(resolveDatasetPath(params.kidney_seg_input, inputDir), checkIfExists: true)
        matrix_ch = Channel.fromPath(resolveDatasetPath(params.visiumhd_matrix, inputDir), checkIfExists: true)
        spatial_ch = Channel.fromPath(resolveDatasetPath(params.spatial_dir, inputDir), checkIfExists: true)
        b2c_h5_ch = Channel.fromPath(resolveDatasetPath(params.bin2cell_h5, inputDir), checkIfExists: true)
        spaceranger_ch = Channel.fromPath(resolveDatasetPath(params.spaceranger_dir, inputDir), checkIfExists: true)
        image_ch = Channel.fromPath(resolveDatasetPath(params.source_image, inputDir), checkIfExists: true)
        kidney_prep_ch = Channel.fromPath(file("${projectDir}/bin/prepare_kidney_seg_input.py"), checkIfExists: true)
        bin2cell_runner_ch = Channel.fromPath(file("${projectDir}/adapters/run_bin2cell.py"), checkIfExists: true)
        PREP_KIDNEY_SEG(seg_input_ch, kidney_prep_ch, params.sample_name, params.run_scope, params.smoke_kidney_rows)
        TRACER_SEG(PREP_KIDNEY_SEG.out.prepared.map{ it.resolve('kidney_seg_input.parquet') }, pmi_ch,
                   segbench_src_ch, resource_script_ch, params.sample_name, params.seed)
        TRACER_NOSEG(matrix_ch, spatial_ch, pmi_ch, segbench_src_ch, resource_script_ch,
                     params.sample_name, params.seed, params.run_scope, params.smoke_kidney_transcripts)
        BIN2CELL(b2c_h5_ch, spaceranger_ch, image_ch, resource_script_ch, bin2cell_runner_ch, params.sample_name,
                 params.seed, params.run_scope, params.smoke_kidney_bins)
        methods = TRACER_SEG.out.results.mix(TRACER_NOSEG.out.results, BIN2CELL.out.results).collect()
        EVALUATE_KIDNEY(methods, holdout_ch, pmi_ch,
                        PREP_KIDNEY_SEG.out.prepared.map{ it.resolve('frozen_input_receipt.json') },
                        input_manifest_ch, frozen_manifest_ch, tidy_script_ch, segbench_src_ch,
                        rctd_script_ch,
                        params.sample_name, params.run_scope, workflow.revision ?: 'unknown')
    }
}

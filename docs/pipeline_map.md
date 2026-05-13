# Pipeline map: existing SPLIT Xenium pipeline vs. TRACER benchmark layer

This is an internal map produced during Phase 0 of the TRACER benchmark adaptation. It captures what already exists, what each existing rule expects/produces, and where the new TRACER / cellAdmix / standardization / metrics layers will plug in.

## 1. Existing pipeline shape

`workflow/Snakefile` is the original SPLIT Xenium analysis pipeline. It is driven by **two** YAML files (loaded via `config_split_original/config.yml` + a sibling `experiments.yml`) and built around a multi-level wildcards hierarchy:

```
condition_id / gene_panel_id / donor / sample
  ├── segmentation_id   (10x_<mode>_<um>, baysor, proseg_expected, proseg_mode, segger)
  ├── normalisation_id  (lognorm, sctransform)
  ├── annotation_id     (.../rctd_*, singler, seurat, xgboost)
  └── count_correction_id (split_*, ovrlpy, resolvi_*)
```

Output root: `${output_path}/...`, where `output_path` defaults to `results/`.

### Includes (workflow/Snakefile)
- `rules/reprocess_raw_data.smk` — checkpoints + xeniumranger version sniff (`check10xVersions`).
- `rules/segmentation.smk` — wraps the four direct-segmentation methods.
- `rules/count_correction.smk` — wraps ovrlpy / resolvi / split.
- `rules/standard_seurat_analysis.smk` — Seurat preprocessing + reports.
- `rules/joint_scanpy_analysis.smk`, `coexpression.smk`, `neighborhood_analysis.smk`,
  `cell_type_annotation.smk`, `segmentation_qc.smk`, `doublet_finding.smk`,
  `data_wrapping.smk`, `geo_sub.smk` — heavier downstream analyses.

### Segmentation rules
| Rule | File | Output |
|------|------|--------|
| `run10x` | `rules/_segmentation/10x.smk` | `results/segmentation/{compact_seg_id}/{sample_id}/normalised_results/` — Xenium bundle from `xeniumranger resegment`. Wildcard `compact_segmentation_id` matches `10x_\w+um` (e.g. `10x_5um`). |
| `runBaysor` → `adjustBaysorResults` → `normaliseBaysor` | `rules/_segmentation/baysor.smk` | `.../segmentation/baysor/{sample_id}/raw_results/segmentation.csv`, `.../processed_results/segmentation.csv`, then `xeniumranger import-segmentation` → `.../normalised_results/`. |
| `runProseg` → `runProseg2Baysor` → `convertProsegCountsFormat` → `normaliseProseg` → `mapProsegRawAndNormalisedCells` | `rules/_segmentation/proseg.smk` | `.../proseg/{sample_id}/raw_results/{transcript-metadata.csv.gz, counts.mtx.gz, expected-counts.mtx.gz, cell-metadata.csv.gz, cell-polygons.geojson.gz}`, plus Baysor-format conversion and a normalised xeniumranger bundle. |
| `runSeggerPreprocess` → `runSeggerTrain` → `runSeggerPredict` → `cleanSeggerPredictDir` → `runSegger2Baysor` → `normaliseSegger` | `rules/_segmentation/segger.smk` | `.../segger/{sample_id}/raw_results/segger_transcripts.parquet`, then Baysor-format and a normalised xeniumranger bundle. |

All four converge to the **same normalised output contract**: `…/normalised_results/outs/transcripts.parquet`, `…/cells.parquet` (Xenium-style bundle).

### Count-correction rules
| Rule | File | Notes |
|------|------|-------|
| `runOvrlpy`, `getCorrectedCountsFromOvrlpy` | `rules/_count_correction/ovrlpy.smk` | Operates on transcripts parquet (Xenium-format) or proseg `transcript-metadata.csv.gz`. Outputs `signal_integrity.parquet`, `signal_strength.parquet`, `transcript_info.parquet`, then `corrected_counts.h5`. Script: `workflow/scripts/_count_correction/ovrlpy_sample.py` + `ovrlpy_sample_correction.py`. |
| `runSplit{FullyPurified,SpotClassBalanced,ScoreBalanced}` | `rules/_count_correction/_split/*.smk` | **Hard dependency** on `cell_type_annotation/.../{annotation_id}/post_processed_output.rds` (RCTD). Inputs are Seurat RDS objects from `std_seurat_analysis/.../preprocessed_seurat.rds`. R scripts under `workflow/scripts/_count_correction/`. |
| `runResolvi*` | `rules/_count_correction/_resolvi/*.smk` | GPU/CUDA path; not part of the benchmark plan. |

### Scripts of interest
- `workflow/scripts/utils/{config_utils.py,config_constants.py,raw_data_utils.py,run_time_utils.py}` — config plumbing for the original pipeline (not needed for the benchmark layer).
- `workflow/scripts/_segmentation/*.py` — Baysor/Proseg/Segger glue (CSV adjustments + format conversions). Reusable.
- `workflow/scripts/_count_correction/ovrlpy_sample.py` — reads transcripts parquet (`x_location`,`y_location`,`z_location`,`feature_name`,`qv`,`is_gene`) and runs `ovrlpy.run`. Reusable directly against the benchmark standardized output (which preserves these columns).

## 2. Why we are adding a separate benchmark layer

The existing Snakefile cannot be reused as-is because:
1. It requires a full `experiments.yml` describing conditions/gene panels/donors/samples — the benchmark wants a single-sample one-click run.
2. SPLIT rules require RCTD post-processed objects, which require references and a heavy Seurat path.
3. There is no standardized cross-method output contract — outputs live in method-specific shapes.
4. There is no TRACER hook, no cellAdmix hook, and no benchmark metrics layer.

Per the task brief ("Prefer adding benchmark-level wrapper rules over modifying original rules heavily"), we add a parallel benchmark Snakefile + benchmark-scoped rules and scripts under `workflow/rules/_benchmark/` and `workflow/scripts/_benchmark/`. The existing SPLIT-derived rules are left untouched and may be invoked by the benchmark layer where appropriate via `include:` (e.g., to re-use `runProseg`, `runBaysor`, etc., on a single sample).

## 3. Benchmark layer layout (target)

```
workflow/
  Snakefile_benchmark                # new entry point for the benchmark
  configs/
    benchmark_lung_small.yml         # small/full single-sample benchmark
    benchmark_lung_tiny.yml          # smoke test: xenium_default + TRACER + metrics only
  rules/
    _benchmark/
      benchmark_all.smk              # rule all + target aggregation
      xenium_default.smk             # uses Xenium-provided cells/transcripts directly
      baysor_wrap.smk                # wraps existing runBaysor
      proseg_wrap.smk                # wraps existing runProseg
      segger_wrap.smk                # wraps existing runSegger
      tracer_refinement.smk          # TRACER applied to a standardized base
      celladmix.smk                  # cellAdmix on a standardized base
      split_wrap.smk                 # benchmark targets for SPLIT
      ovrlpy_wrap.smk                # benchmark targets for ovrlpy
      metrics.smk                    # NPMI, marker specificity, summary
  scripts/
    _benchmark/
      standardize_method_output.py   # contract adapter
      standardize_xenium_default.py  # adapter for raw Xenium bundle
      standardize_baysor.py          # adapter for Baysor output
      standardize_proseg.py          # adapter for Proseg output
      standardize_segger.py          # adapter for Segger output
      run_tracer_refine.py           # TRACER wrapper
      run_celladmix.R                # cellAdmix wrapper
      run_split_benchmark.R          # SPLIT thin wrapper around existing script
      run_ovrlpy_benchmark.py        # ovrlpy thin wrapper around existing script
      run_npmi_metrics.py
      run_marker_specificity.py
      collect_metrics.py
      plot_benchmark_summary.py
resources/
  marker_sets/lung_cancer_markers.yml
```

## 4. Standardized output contract

Every method writes to `results/{dataset}/{method}/standardized/`:
- `transcripts.parquet` — one row per molecule. Columns: `transcript_id, feature_name, x_location, y_location, z_location, qv, cell_id_xenium_default, cell_id_method, method, assignment_source`.
- `cells.parquet` — one row per cell. Columns: `cell_id_method, x_centroid, y_centroid, n_transcripts, n_genes, area, method`.
- `cell_by_gene.mtx` + `barcodes.tsv` + `features.tsv` — MatrixMarket triplet.
- `cell_metadata.parquet` — `cell_id_method` + any per-cell metadata (n_transcripts, n_genes, area, optional cell-type label).
- `method_info.json` — provenance record (see Phase 2).

This contract is the single point of contact between method runners and downstream metrics / TRACER refinement / cellAdmix.

## 5. Plug points

| New thing | Plug point | Existing rule it builds on |
|-----------|------------|-----------------------------|
| `xenium_default` standardize | direct from `dataset/.../TSU-20/{transcripts.parquet,cells.parquet}` | no existing rule needed |
| `baysor` standardize | `results/{dataset}/baysor/raw/{segmentation.csv,segmentation_polygons_2d.json}` | wraps `adjustBaysorResults` output |
| `proseg` standardize | `results/{dataset}/proseg/raw/{transcript-metadata.csv.gz, cell-metadata.csv.gz}` | wraps `runProseg` output |
| `segger` standardize | `results/{dataset}/segger/raw/segger_transcripts.parquet` | wraps `cleanSeggerPredictDir` output |
| TRACER refinement | takes standardized `transcripts.parquet` from any base, returns standardized output under `tracer_from_{base}/` | new |
| cellAdmix | takes standardized output + cluster labels → corrected counts | new |
| SPLIT (benchmark) | thin wrapper that takes standardized → calls existing SPLIT R script with the required Seurat/RCTD objects | reuses `split_fully_purified.R` |
| ovrlpy (benchmark) | thin wrapper that takes standardized transcripts (Xenium-format columns are already present) and calls `ovrlpy_sample.py` | reuses script directly |
| Metrics | reads standardized outputs of every method; writes a single `metrics_all_methods.csv/parquet` and figures | new |

## 6. Phase 1 first deliverable (smoke test)

Goal: a single command runs to produce `results/lung_small/summary/metrics_all_methods.csv` with metrics for `xenium_default` and `tracer_from_xenium_default`, against the local TSU-20 dataset, without requiring Baysor/Proseg/Segger/SPLIT/cellAdmix/ovrlpy environments to exist.

This guarantees: dataset path is correct, standardization contract works, TRACER plug-in path works, NPMI + marker metrics work end-to-end, and `collect_metrics.py` produces the summary CSV.

#!/usr/bin/env Rscript
# ============================================================================
# cellAdmix wrapper for the TRACER benchmark layer.
# ============================================================================
# Takes a standardized benchmark output directory and applies cellAdmix's
# admixture-correction routine. Writes:
#   {out_dir}/corrected_counts.mtx
#   {out_dir}/cell_metadata.parquet
#   {out_dir}/method_info.json
#   {out_dir}/admixture_metrics.csv          (best-effort)
#
# Requirements:
#   - kharchenkolab/cellAdmix installed (devtools::install_github)
#   - arrow + Matrix + jsonlite for IO
#
# cellAdmix requires per-cell cluster / cell-type labels. The wrapper accepts
# them via --cluster-labels (a TSV/CSV with `cell_id_method` + `cluster`).
# If labels are not provided, the wrapper exits cleanly with a TODO message
# so the rest of the benchmark can still run.
# ============================================================================

suppressPackageStartupMessages({
  ok <- TRUE
  for (pkg in c("optparse", "Matrix", "jsonlite", "arrow")) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      message(sprintf("[celladmix] missing R package: %s", pkg))
      ok <- FALSE
    } else {
      suppressWarnings(library(pkg, character.only = TRUE))
    }
  }
  if (!ok) {
    quit(status = 2, save = "no")
  }
})

option_list <- list(
  make_option("--standardized-dir", type = "character", help = "Input standardized output directory."),
  make_option("--out-dir", type = "character", help = "Output directory for cellAdmix results."),
  make_option("--cluster-labels", type = "character", default = NA_character_,
              help = "CSV/TSV with columns cell_id_method,cluster (Xenium clusters.csv works after a rename)."),
  make_option("--base-method", type = "character", default = "unknown",
              help = "Name of the base method whose output we are correcting."),
  make_option("--repo-path", type = "character", default = NA_character_,
              help = "Optional path to a local cellAdmix checkout."),
  make_option("--admixture-threshold", type = "double", default = 0.1),
  make_option("--min-cells-per-cluster", type = "integer", default = 25),
  make_option("--log", type = "character", default = NA_character_),
  make_option("--allow-stub", action = "store_true", default = FALSE,
              help = "If cellAdmix is unavailable, write an empty stub output instead of failing.")
)
opt <- parse_args(OptionParser(option_list = option_list))

if (!is.na(opt$log)) {
  dir.create(dirname(opt$log), recursive = TRUE, showWarnings = FALSE)
  log_con <- file(opt$log, open = "wt")
  sink(log_con, type = "output")
  sink(log_con, type = "message")
}

dir.create(opt$`out-dir`, recursive = TRUE, showWarnings = FALSE)

stand_dir <- opt$`standardized-dir`
mtx_path <- file.path(stand_dir, "cell_by_gene.mtx")
barcodes_path <- file.path(stand_dir, "cell_by_gene_barcodes.tsv")
features_path <- file.path(stand_dir, "cell_by_gene_features.tsv")
meta_path <- file.path(stand_dir, "cell_metadata.parquet")

for (p in c(mtx_path, barcodes_path, features_path, meta_path)) {
  if (!file.exists(p)) {
    stop(sprintf("Missing input from standardized contract: %s", p))
  }
}

start_time <- format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
mat <- Matrix::readMM(mtx_path)
barcodes <- readLines(barcodes_path)
features <- readLines(features_path)
rownames(mat) <- barcodes
colnames(mat) <- features

meta <- arrow::read_parquet(meta_path)

cluster_df <- NULL
if (!is.na(opt$`cluster-labels`)) {
  if (file.exists(opt$`cluster-labels`)) {
    sep <- if (grepl("\\.tsv$", opt$`cluster-labels`)) "\t" else ","
    cluster_df <- utils::read.table(
      opt$`cluster-labels`, header = TRUE, sep = sep,
      stringsAsFactors = FALSE, check.names = FALSE
    )
    # Best-effort rename: Xenium clusters.csv uses Barcode,Cluster.
    if (all(c("Barcode", "Cluster") %in% colnames(cluster_df))) {
      cluster_df <- data.frame(
        cell_id_method = as.character(cluster_df$Barcode),
        cluster = as.character(cluster_df$Cluster),
        stringsAsFactors = FALSE
      )
    }
    if (!all(c("cell_id_method", "cluster") %in% colnames(cluster_df))) {
      stop("--cluster-labels file must have columns: cell_id_method,cluster (or Barcode,Cluster).")
    }
  } else {
    message(sprintf("[celladmix] cluster labels not found at %s", opt$`cluster-labels`))
  }
}

if (is.null(cluster_df) || nrow(cluster_df) == 0) {
  if (opt$`allow-stub`) {
    message("[celladmix] No cluster labels provided. Writing stub output (uncorrected counts).")
    Matrix::writeMM(mat, file.path(opt$`out-dir`, "corrected_counts.mtx"))
    arrow::write_parquet(meta, file.path(opt$`out-dir`, "cell_metadata.parquet"))
    info <- list(
      method_name = sprintf("celladmix_from_%s", opt$`base-method`),
      command = paste(commandArgs(trailingOnly = FALSE), collapse = " "),
      input_files = list(mtx_path, barcodes_path, features_path),
      output_files = list(
        file.path(opt$`out-dir`, "corrected_counts.mtx"),
        file.path(opt$`out-dir`, "cell_metadata.parquet")
      ),
      start_time = start_time,
      end_time = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
      threads = NA,
      container_or_env = Sys.getenv("CONDA_DEFAULT_ENV", unset = NA),
      extra = list(stub_no_clusters = TRUE)
    )
    writeLines(
      jsonlite::toJSON(info, pretty = TRUE, auto_unbox = TRUE, null = "null"),
      file.path(opt$`out-dir`, "method_info.json")
    )
    quit(status = 0, save = "no")
  } else {
    stop("cellAdmix requires per-cell cluster labels. Provide --cluster-labels or rerun with --allow-stub.")
  }
}

# Try to load cellAdmix; degrade to stub if unavailable.
if (!is.na(opt$`repo-path`) && nzchar(opt$`repo-path`)) {
  .libPaths(c(opt$`repo-path`, .libPaths()))
}
have_celladmix <- requireNamespace("cellAdmix", quietly = TRUE)
if (!have_celladmix) {
  if (opt$`allow-stub`) {
    message("[celladmix] cellAdmix R package not available; writing stub output.")
    Matrix::writeMM(mat, file.path(opt$`out-dir`, "corrected_counts.mtx"))
    arrow::write_parquet(meta, file.path(opt$`out-dir`, "cell_metadata.parquet"))
    info <- list(
      method_name = sprintf("celladmix_from_%s", opt$`base-method`),
      command = paste(commandArgs(trailingOnly = FALSE), collapse = " "),
      input_files = list(mtx_path, barcodes_path, features_path),
      output_files = list(
        file.path(opt$`out-dir`, "corrected_counts.mtx"),
        file.path(opt$`out-dir`, "cell_metadata.parquet")
      ),
      start_time = start_time,
      end_time = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
      threads = NA,
      container_or_env = Sys.getenv("CONDA_DEFAULT_ENV", unset = NA),
      extra = list(stub_no_celladmix_install = TRUE)
    )
    writeLines(
      jsonlite::toJSON(info, pretty = TRUE, auto_unbox = TRUE, null = "null"),
      file.path(opt$`out-dir`, "method_info.json")
    )
    quit(status = 0, save = "no")
  } else {
    stop("cellAdmix is not installed. Either install kharchenkolab/cellAdmix or rerun with --allow-stub.")
  }
}

# Real cellAdmix path. cellAdmix's exact API depends on version; we probe.
clusters_for_cells <- setNames(cluster_df$cluster, cluster_df$cell_id_method)
clusters_for_cells <- clusters_for_cells[rownames(mat)]
keep <- !is.na(clusters_for_cells)
mat_sub <- mat[keep, , drop = FALSE]
cluster_vec <- clusters_for_cells[keep]
# Filter rare clusters.
tab <- table(cluster_vec)
keep_clusters <- names(tab)[tab >= opt$`min-cells-per-cluster`]
mask <- cluster_vec %in% keep_clusters
mat_sub <- mat_sub[mask, , drop = FALSE]
cluster_vec <- cluster_vec[mask]

corrected <- mat_sub  # default fallback
metrics <- NULL
api_used <- NA_character_

if (exists("correct_admixture", where = asNamespace("cellAdmix"))) {
  api_used <- "cellAdmix::correct_admixture"
  res <- tryCatch(
    cellAdmix::correct_admixture(
      counts = mat_sub,
      clusters = cluster_vec,
      admixture_threshold = opt$`admixture-threshold`
    ),
    error = function(e) {
      message(sprintf("[celladmix] correct_admixture failed: %s", conditionMessage(e)))
      NULL
    }
  )
  if (!is.null(res)) {
    if (inherits(res, "list") && !is.null(res$corrected)) corrected <- res$corrected
    if (inherits(res, "list") && !is.null(res$metrics)) metrics <- res$metrics
  }
} else if (exists("cellAdmix", where = asNamespace("cellAdmix"))) {
  api_used <- "cellAdmix::cellAdmix"
  res <- tryCatch(
    cellAdmix::cellAdmix(mat_sub, cluster_vec),
    error = function(e) {
      message(sprintf("[celladmix] cellAdmix() failed: %s", conditionMessage(e)))
      NULL
    }
  )
  if (!is.null(res)) corrected <- res
} else {
  # No known entry point. Per the project's no-stub policy, fail loudly
  # rather than emit uncorrected counts.
  available <- ls(asNamespace("cellAdmix"))
  stop(sprintf(
    paste0(
      "cellAdmix is installed but no known entry point was found. ",
      "Expected one of: cellAdmix::correct_admixture, cellAdmix::cellAdmix. ",
      "Available exported names: %s. ",
      "Update workflow/scripts/_benchmark/run_celladmix.R to call the correct API for this version."
    ),
    paste(available, collapse = ", ")
  ))
}

# Persist outputs.
Matrix::writeMM(as(corrected, "CsparseMatrix"), file.path(opt$`out-dir`, "corrected_counts.mtx"))
arrow::write_parquet(
  data.frame(cell_id_method = rownames(corrected), cluster = cluster_vec, stringsAsFactors = FALSE),
  file.path(opt$`out-dir`, "cell_metadata.parquet")
)
if (!is.null(metrics)) {
  utils::write.csv(metrics, file.path(opt$`out-dir`, "admixture_metrics.csv"), row.names = FALSE)
}

info <- list(
  method_name = sprintf("celladmix_from_%s", opt$`base-method`),
  command = paste(commandArgs(trailingOnly = FALSE), collapse = " "),
  input_files = list(mtx_path, barcodes_path, features_path),
  output_files = list(
    file.path(opt$`out-dir`, "corrected_counts.mtx"),
    file.path(opt$`out-dir`, "cell_metadata.parquet")
  ),
  start_time = start_time,
  end_time = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
  threads = NA,
  container_or_env = Sys.getenv("CONDA_DEFAULT_ENV", unset = NA),
  extra = list(
    base_method = opt$`base-method`,
    api_used = api_used,
    admixture_threshold = opt$`admixture-threshold`,
    min_cells_per_cluster = opt$`min-cells-per-cluster`
  )
)
writeLines(
  jsonlite::toJSON(info, pretty = TRUE, auto_unbox = TRUE, null = "null"),
  file.path(opt$`out-dir`, "method_info.json")
)

if (!is.na(opt$log)) {
  sink(type = "message")
  sink(type = "output")
  close(log_con)
}

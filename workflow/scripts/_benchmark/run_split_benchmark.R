#!/usr/bin/env Rscript
# ============================================================================
# SPLIT benchmark wrapper.
# ============================================================================
# SPLIT requires (1) a Seurat object with a counts assay and (2) a post-
# processed RCTD doublet-mode object. The original SPLIT pipeline rules
# (workflow/rules/_count_correction/_split/*.smk) wire these dependencies
# in via the multi-condition wildcards system, which the benchmark layer
# does not use.
#
# This wrapper takes:
#   --standardized-dir  a benchmark standardized output for some base method
#   --rctd-rds          path to the RCTD post_processed_output.rds
#   --out-dir           where to write the SPLIT-corrected outputs
#
# If --rctd-rds is missing, the wrapper exits with a clear error UNLESS
# --allow-stub is passed (in which case it writes the uncorrected counts
# back and marks the result as stubbed).
# ============================================================================

suppressPackageStartupMessages({
  for (pkg in c("optparse", "Matrix", "arrow", "jsonlite")) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      stop(sprintf("Missing R package: %s. Install before running.", pkg))
    } else {
      suppressWarnings(library(pkg, character.only = TRUE))
    }
  }
})

option_list <- list(
  make_option("--standardized-dir", type = "character"),
  make_option("--rctd-rds", type = "character", default = NA_character_),
  make_option("--base-method", type = "character", default = "unknown"),
  make_option("--out-dir", type = "character"),
  make_option("--log", type = "character", default = NA_character_),
  make_option("--allow-stub", action = "store_true", default = FALSE)
)
opt <- parse_args(OptionParser(option_list = option_list))

if (!is.na(opt$log)) {
  dir.create(dirname(opt$log), recursive = TRUE, showWarnings = FALSE)
  con <- file(opt$log, open = "wt")
  sink(con, type = "output"); sink(con, type = "message")
}

dir.create(opt$`out-dir`, recursive = TRUE, showWarnings = FALSE)
start_time <- format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")

mtx_path <- file.path(opt$`standardized-dir`, "cell_by_gene.mtx")
barcodes_path <- file.path(opt$`standardized-dir`, "cell_by_gene_barcodes.tsv")
features_path <- file.path(opt$`standardized-dir`, "cell_by_gene_features.tsv")
meta_path <- file.path(opt$`standardized-dir`, "cell_metadata.parquet")

mat <- Matrix::readMM(mtx_path)
rownames(mat) <- readLines(barcodes_path)
colnames(mat) <- readLines(features_path)
meta <- arrow::read_parquet(meta_path)

write_stub <- function(reason) {
  Matrix::writeMM(as(t(mat), "CsparseMatrix"), file.path(opt$`out-dir`, "corrected_counts.mtx"))
  arrow::write_parquet(meta, file.path(opt$`out-dir`, "corrected_counts_metadata.parquet"))
  info <- list(
    method_name = sprintf("split_from_%s", opt$`base-method`),
    command = paste(commandArgs(trailingOnly = FALSE), collapse = " "),
    input_files = list(mtx_path),
    output_files = list(file.path(opt$`out-dir`, "corrected_counts.mtx")),
    start_time = start_time,
    end_time = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
    container_or_env = Sys.getenv("CONDA_DEFAULT_ENV", unset = NA),
    extra = list(stub = TRUE, reason = reason, base_method = opt$`base-method`)
  )
  writeLines(jsonlite::toJSON(info, pretty = TRUE, auto_unbox = TRUE, null = "null"),
             file.path(opt$`out-dir`, "method_info.json"))
}

if (is.na(opt$`rctd-rds`) || !file.exists(opt$`rctd-rds`)) {
  if (opt$`allow-stub`) {
    message("[split] RCTD post-processed RDS not provided/found. Writing stub output.")
    write_stub("missing_rctd")
    quit(status = 0, save = "no")
  } else {
    stop("SPLIT requires --rctd-rds pointing to a valid post-processed RCTD RDS. ",
         "Rerun with --allow-stub if you want to ignore this requirement.")
  }
}

ok_split <- requireNamespace("SPLIT", quietly = TRUE) &&
            requireNamespace("spacexr", quietly = TRUE) &&
            requireNamespace("Seurat", quietly = TRUE)
if (!ok_split) {
  if (opt$`allow-stub`) {
    message("[split] SPLIT / spacexr / Seurat missing. Writing stub output.")
    write_stub("missing_packages")
    quit(status = 0, save = "no")
  } else {
    stop("SPLIT requires SPLIT + spacexr + Seurat. Install them or rerun with --allow-stub.")
  }
}

suppressPackageStartupMessages({
  library(Seurat); library(spacexr); library(SPLIT)
})

# SPLIT expects a counts matrix with cells in columns. Our standardized mtx
# stores cells in rows, so transpose.
counts <- as(t(mat), "CsparseMatrix")

rctd <- readRDS(opt$`rctd-rds`)
res_split <- SPLIT::purify(
  counts = counts,
  rctd = rctd,
  DO_purify_singlets = TRUE
)
Matrix::writeMM(as(res_split$purified_counts, "CsparseMatrix"),
                file.path(opt$`out-dir`, "corrected_counts.mtx"))
arrow::write_parquet(as.data.frame(res_split$cell_meta),
                     file.path(opt$`out-dir`, "corrected_counts_metadata.parquet"))
info <- list(
  method_name = sprintf("split_from_%s", opt$`base-method`),
  command = paste(commandArgs(trailingOnly = FALSE), collapse = " "),
  input_files = list(mtx_path, opt$`rctd-rds`),
  output_files = list(
    file.path(opt$`out-dir`, "corrected_counts.mtx"),
    file.path(opt$`out-dir`, "corrected_counts_metadata.parquet")
  ),
  start_time = start_time,
  end_time = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
  container_or_env = Sys.getenv("CONDA_DEFAULT_ENV", unset = NA),
  extra = list(base_method = opt$`base-method`)
)
writeLines(jsonlite::toJSON(info, pretty = TRUE, auto_unbox = TRUE, null = "null"),
           file.path(opt$`out-dir`, "method_info.json"))

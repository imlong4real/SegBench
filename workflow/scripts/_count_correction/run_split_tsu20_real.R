#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(optparse)
  library(Matrix)
  library(jsonlite)
})

option_list <- list(
  make_option("--xenium-dir", type = "character"),
  make_option("--scrna-h5ad", type = "character"),
  make_option("--celltype-column", type = "character", default = "auto"),
  make_option("--outdir", type = "character"),
  make_option("--common-inputs", type = "character", default = "results/tsu20_tools/common_inputs"),
  make_option("--cores", type = "integer", default = 2),
  make_option("--umi-min", type = "integer", default = 10),
  make_option("--counts-min", type = "integer", default = 10),
  make_option("--umi-min-sigma", type = "numeric", default = 1)
)
opt <- parse_args(OptionParser(option_list = option_list))

dir.create(opt$outdir, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(opt$outdir, "logs"), recursive = TRUE, showWarnings = FALSE)
log_con <- file(file.path(opt$outdir, "logs", "run_split_tsu20_real.log"), open = "wt")
sink(log_con, type = "output")
sink(log_con, type = "message")
start <- Sys.time()

write_info <- function(status, reason = NULL, extra = list(), outputs = list()) {
  info <- c(list(
    method_name = "split_xenium_default",
    status = status,
    stub = FALSE,
    split_version = if (requireNamespace("SPLIT", quietly = TRUE)) as.character(packageVersion("SPLIT")) else NA_character_,
    spacexr_version = if (requireNamespace("spacexr", quietly = TRUE)) as.character(packageVersion("spacexr")) else NA_character_,
    seurat_version = if (requireNamespace("Seurat", quietly = TRUE)) as.character(packageVersion("Seurat")) else NA_character_,
    reference_h5ad = opt$`scrna-h5ad`,
    reference_celltype_column = extra$reference_celltype_column %||% opt$`celltype-column`,
    rctd_mode = "doublet",
    post_processed_RCTD = identical(status, "PASS"),
    DO_purify_singlets = TRUE,
    external_scrna_reference_used = TRUE,
    pseudo_reference_debug = FALSE,
    command = paste(commandArgs(trailingOnly = FALSE), collapse = " "),
    start_time = format(start, "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
    end_time = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
    runtime_seconds = as.numeric(difftime(Sys.time(), start, units = "secs")),
    reason = reason,
    output_files = outputs
  ), extra)
  writeLines(toJSON(info, pretty = TRUE, auto_unbox = TRUE, null = "null"),
             file.path(opt$outdir, "method_info.json"))
}

`%||%` <- function(a, b) if (is.null(a) || length(a) == 0 || is.na(a)) b else a

tryCatch({
  missing_pkgs <- c("Seurat", "spacexr", "SPLIT", "arrow", "dplyr")[
    !vapply(c("Seurat", "spacexr", "SPLIT", "arrow", "dplyr"), requireNamespace, logical(1), quietly = TRUE)
  ]
  if (length(missing_pkgs) > 0) {
    stop(sprintf("Missing R packages: %s", paste(missing_pkgs, collapse = ", ")))
  }
  suppressPackageStartupMessages({
    library(Seurat)
    library(spacexr)
    library(SPLIT)
    library(arrow)
    library(dplyr)
  })

  common <- opt$`common-inputs`
  info <- fromJSON(file.path(common, "common_inputs_info.json"))
  ref_counts <- readMM(file.path(common, "scrna_reference_counts.mtx"))
  spatial_counts <- readMM(file.path(common, "xenium_counts.mtx"))
  genes <- readLines(file.path(common, "scrna_reference_genes.tsv"))
  ref_cells <- readLines(file.path(common, "scrna_reference_cells.tsv"))
  spatial_cells <- readLines(file.path(common, "xenium_barcodes.tsv"))
  ref_meta <- read.csv(file.path(common, "scrna_reference_cell_metadata.csv"), stringsAsFactors = FALSE)
  spatial_meta <- arrow::read_parquet(file.path(common, "xenium_cell_metadata_with_clusters.parquet")) %>%
    as.data.frame()

  rownames(ref_counts) <- genes
  colnames(ref_counts) <- ref_cells
  rownames(spatial_counts) <- genes
  colnames(spatial_counts) <- spatial_cells
  ref_counts <- as(ref_counts, "CsparseMatrix")
  spatial_counts <- as(spatial_counts, "CsparseMatrix")
  ref_counts@x <- round(ref_counts@x)
  spatial_counts@x <- round(spatial_counts@x)

  ref_labels <- factor(ref_meta$celltype)
  names(ref_labels) <- ref_meta$cell_id
  ref_labels <- droplevels(ref_labels[colnames(ref_counts)])
  keep_ref <- !is.na(ref_labels)
  ref_counts <- ref_counts[, keep_ref, drop = FALSE]
  ref_labels <- droplevels(ref_labels[keep_ref])

  spatial_meta <- spatial_meta[match(colnames(spatial_counts), spatial_meta$cell_id), ]
  coords <- data.frame(
    x = spatial_meta$x_centroid,
    y = spatial_meta$y_centroid,
    row.names = colnames(spatial_counts)
  )
  keep_spatial <- complete.cases(coords)
  spatial_counts <- spatial_counts[, keep_spatial, drop = FALSE]
  coords <- coords[keep_spatial, , drop = FALSE]
  spatial_meta <- spatial_meta[keep_spatial, , drop = FALSE]

  xe <- CreateSeuratObject(
    counts = spatial_counts,
    assay = "Xenium",
    meta.data = spatial_meta
  )

  ref.obj <- Reference(ref_counts, cell_types = ref_labels, min_UMI = 10, require_int = TRUE)
  test.obj <- SpatialRNA(coords, spatial_counts, require_int = TRUE)
  rctd <- create.RCTD(
    test.obj,
    ref.obj,
    UMI_min = opt$`umi-min`,
    counts_MIN = opt$`counts-min`,
    UMI_min_sigma = opt$`umi-min-sigma`,
    max_cores = opt$cores,
    CELL_MIN_INSTANCE = 25
  )
  rctd <- run.RCTD(rctd, doublet_mode = "doublet")
  saveRDS(rctd, file.path(opt$outdir, "RCTD_raw.rds"))

  rctd <- SPLIT::run_post_process_RCTD(rctd = rctd)
  saveRDS(rctd, file.path(opt$outdir, "post_processed_RCTD.rds"))

  counts_for_split <- tryCatch(
    GetAssayData(xe, assay = "Xenium", layer = "counts"),
    error = function(e) GetAssayData(xe, assay = "Xenium", slot = "counts")
  )
  res_split <- SPLIT::purify(
    counts = counts_for_split,
    rctd = rctd,
    DO_purify_singlets = TRUE
  )
  saveRDS(res_split, file.path(opt$outdir, "split_result.rds"))
  writeMM(as(res_split$purified_counts, "CsparseMatrix"), file.path(opt$outdir, "purified_counts.mtx"))
  write.csv(as.data.frame(res_split$cell_meta), file.path(opt$outdir, "cell_meta.csv"), row.names = TRUE)
  arrow::write_parquet(as.data.frame(res_split$cell_meta), file.path(opt$outdir, "cell_meta.parquet"))

  xe_purified <- CreateSeuratObject(
    counts = res_split$purified_counts,
    meta.data = as.data.frame(res_split$cell_meta),
    assay = "Xenium"
  )
  saveRDS(xe_purified, file.path(opt$outdir, "xe_purified.rds"))

  outputs <- list(
    file.path(opt$outdir, "RCTD_raw.rds"),
    file.path(opt$outdir, "post_processed_RCTD.rds"),
    file.path(opt$outdir, "split_result.rds"),
    file.path(opt$outdir, "purified_counts.mtx"),
    file.path(opt$outdir, "cell_meta.csv"),
    file.path(opt$outdir, "cell_meta.parquet"),
    file.path(opt$outdir, "xe_purified.rds")
  )
  write_info("PASS", extra = list(
    reference_celltype_column = info$reference_celltype_column,
    n_reference_cells = ncol(ref_counts),
    n_spatial_cells = ncol(spatial_counts),
    n_shared_genes = nrow(spatial_counts)
  ), outputs = outputs)
}, error = function(e) {
  msg <- conditionMessage(e)
  message(msg)
  common_info <- tryCatch(fromJSON(file.path(opt$`common-inputs`, "common_inputs_info.json")), error = function(e) list())
  write_info("BLOCKED", reason = msg, extra = list(
    reference_celltype_column = common_info$reference_celltype_column %||% opt$`celltype-column`,
    n_reference_cells = common_info$n_reference_cells %||% NA_integer_,
    n_spatial_cells = common_info$n_spatial_cells %||% NA_integer_,
    n_shared_genes = common_info$n_shared_genes %||% NA_integer_
  ))
  quit(status = 1, save = "no")
}, finally = {
  sink(type = "message")
  sink(type = "output")
  close(log_con)
})

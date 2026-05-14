#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(optparse)
  library(jsonlite)
  library(Matrix)
})

celladmix_fun <- function(name) {
  get(name, envir = asNamespace("cellAdmix"), inherits = FALSE)
}

option_list <- list(
  make_option("--xenium-dir", type = "character"),
  make_option("--clusters", type = "character"),
  make_option("--outdir", type = "character"),
  make_option("--common-inputs", type = "character", default = "results/tsu20_tools/common_inputs"),
  make_option("--num-factors", type = "integer", default = 10),
  make_option("--nmol-dsamp", type = "integer", default = 10000),
  make_option("--n-cells-nmf", type = "integer", default = 2000),
  make_option("--cores", type = "integer", default = 2),
  make_option("--bridge-cells", type = "integer", default = 200)
)
opt <- parse_args(OptionParser(option_list = option_list))

dir.create(opt$outdir, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(opt$outdir, "logs"), recursive = TRUE, showWarnings = FALSE)
log_con <- file(file.path(opt$outdir, "logs", "run_celladmix_tsu20_real.log"), open = "wt")
sink(log_con, type = "output")
sink(log_con, type = "message")
start <- Sys.time()

write_info <- function(status, reason = NULL, extra = list(), outputs = list()) {
  info <- c(list(
    method_name = "celladmix_xenium_default",
    status = status,
    stub = FALSE,
    celladmix_version = if (requireNamespace("cellAdmix", quietly = TRUE)) as.character(packageVersion("cellAdmix")) else NA_character_,
    temporary_celltype_source = opt$clusters,
    external_scrna_reference_used = FALSE,
    debug_cluster_labels = TRUE,
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

tryCatch({
  missing_pkgs <- c("cellAdmix", "arrow", "dplyr", "Matrix")[
    !vapply(c("cellAdmix", "arrow", "dplyr", "Matrix"), requireNamespace, logical(1), quietly = TRUE)
  ]
  if (length(missing_pkgs) > 0) {
    stop(sprintf("Missing R packages: %s", paste(missing_pkgs, collapse = ", ")))
  }
  suppressPackageStartupMessages({
    library(cellAdmix)
    library(arrow)
    library(dplyr)
  })

  df <- arrow::read_parquet(file.path(opt$`common-inputs`, "xenium_transcripts_for_celladmix.parquet")) %>%
    as.data.frame()
  cell_meta <- arrow::read_parquet(file.path(opt$`common-inputs`, "xenium_cell_metadata_with_clusters.parquet")) %>%
    as.data.frame()
  cell_meta <- cell_meta[!is.na(cell_meta$celltype), ]
  cell_meta <- data.frame(
    x = cell_meta$x_centroid,
    y = cell_meta$y_centroid,
    z = 0,
    cell = as.character(cell_meta$cell_id),
    celltype = as.character(cell_meta$celltype),
    cluster = as.character(cell_meta$cluster),
    stringsAsFactors = FALSE
  )
  rownames(cell_meta) <- cell_meta$cell
  df <- df[df$cell %in% cell_meta$cell, c("x", "y", "z", "gene", "cell", "celltype", "mol_id")]
  mol_counts <- table(df$cell)
  keep_cells <- names(mol_counts)[mol_counts > 2]
  if (length(keep_cells) == 0) {
    stop("cellAdmix requires at least one cell with more than two molecules")
  }
  dropped_low_molecule_cells <- setdiff(cell_meta$cell, keep_cells)
  message(sprintf("Dropping %d cells with two or fewer molecules before cellAdmix", length(dropped_low_molecule_cells)))
  df <- df[df$cell %in% keep_cells, , drop = FALSE]
  cell_meta <- cell_meta[cell_meta$cell %in% keep_cells, , drop = FALSE]
  df$mol_id <- as.character(df$mol_id)
  rownames(df) <- df$mol_id

  set.seed(1)
  nmf_rds <- file.path(opt$outdir, "nmf_result.rds")
  crf_rds <- file.path(opt$outdir, "crf_result.rds")
  knn_rds <- file.path(opt$outdir, "knn_result.rds")
  if (file.exists(nmf_rds)) {
    message("Reusing cached cellAdmix NMF result")
    res <- readRDS(nmf_rds)
  } else {
    df_sub <- cellAdmix::samp_ct_equal(df, cell_meta, num.cells.samp = opt$`n-cells-nmf`)
    res <- cellAdmix::run_knn_nmf(
      df_sub,
      k = opt$`num-factors`,
      h = 20,
      nmol.dsamp = opt$`nmol-dsamp`,
      n.cores = opt$cores
    )
    saveRDS(res, nmf_rds)
  }
  if (file.exists(crf_rds)) {
    message("Reusing cached cellAdmix CRF result")
    crf_res <- readRDS(crf_rds)
  } else {
    crf_res <- cellAdmix::run_crf_all(
      df,
      res,
      num.nn = 10,
      same.label.ratio = 5,
      normalize.by = "gene",
      n.cores = opt$cores
    )
    saveRDS(crf_res, crf_rds)
  }
  if (file.exists(knn_rds)) {
    message("Reusing cached cellAdmix molecule KNN result")
    knn_res <- readRDS(knn_rds)
  } else {
    knn_res <- celladmix_fun("get_mol_knn")(df, knn_k = 10)
    saveRDS(knn_res, knn_rds)
  }
  bridge_raw <- cellAdmix::run_bridge_test(
    df,
    crf_res,
    cell_meta,
    knn_res[[1]],
    knn_res[[2]],
    ncells.samp = opt$`bridge-cells`,
    knn.k = 20,
    n.cores = opt$cores
  )
  all_ctypes <- unique(cell_meta$celltype)
  bridge_res <- cellAdmix::extract_bridge_res(
    bridge_raw,
    all.ctypes = all_ctypes,
    n.factors = opt$`num-factors`,
    nmf.type = "joint",
    p.thresh = 0.1,
    adj.pvals = FALSE
  )
  fp_checks <- celladmix_fun("check_fp")(
    df,
    cell_meta,
    crf_res,
    bridge_res,
    do_clean = TRUE,
    knn_k = 100,
    median_thresh = 0.1
  )

  df$factor <- crf_res[, 1]
  remove_key <- paste(df$factor, df$celltype, sep = "_")
  removed <- df[remove_key %in% fp_checks, ]
  cleaned <- df[!(remove_key %in% fp_checks), ]

  arrow::write_parquet(cleaned, file.path(opt$outdir, "cleaned_transcripts.parquet"))
  arrow::write_parquet(removed, file.path(opt$outdir, "removed_transcripts.parquet"))
  factor_assignments <- data.frame(mol_id = df$mol_id, factor = df$factor, cell = df$cell, celltype = df$celltype)
  write.csv(factor_assignments, file.path(opt$outdir, "factor_assignments.csv"), row.names = FALSE)

  counts <- sparseMatrix(
    i = match(cleaned$gene, sort(unique(cleaned$gene))),
    j = match(cleaned$cell, sort(unique(cleaned$cell))),
    x = 1L,
    dims = c(length(sort(unique(cleaned$gene))), length(sort(unique(cleaned$cell)))),
    dimnames = list(sort(unique(cleaned$gene)), sort(unique(cleaned$cell)))
  )
  writeMM(as(counts, "CsparseMatrix"), file.path(opt$outdir, "corrected_counts.mtx"))
  arrow::write_parquet(cell_meta, file.path(opt$outdir, "cell_metadata.parquet"))
  saveRDS(list(
    nmf = res,
    crf = crf_res,
    bridge_raw = bridge_raw,
    bridge_res = bridge_res,
    fp_checks = fp_checks
  ), file.path(opt$outdir, "celladmix_result.rds"))

  outputs <- list(
    file.path(opt$outdir, "cleaned_transcripts.parquet"),
    file.path(opt$outdir, "removed_transcripts.parquet"),
    file.path(opt$outdir, "factor_assignments.csv"),
    file.path(opt$outdir, "corrected_counts.mtx"),
    file.path(opt$outdir, "cell_metadata.parquet"),
    file.path(opt$outdir, "celladmix_result.rds"),
    nmf_rds,
    crf_rds,
    knn_rds
  )
  write_info("DEBUG_PASS", outputs = outputs, extra = list(
    n_input_transcripts = nrow(df),
    n_removed_transcripts = nrow(removed),
    n_retained_transcripts = nrow(cleaned),
    n_cells = length(unique(df$cell)),
    n_clusters = length(unique(df$celltype)),
    n_dropped_low_molecule_cells = length(dropped_low_molecule_cells),
    num_factors = opt$`num-factors`,
    nmol_dsamp = opt$`nmol-dsamp`
  ))
}, error = function(e) {
  msg <- conditionMessage(e)
  message(msg)
  write_info("BLOCKED", reason = msg)
  quit(status = 1, save = "no")
}, finally = {
  sink(type = "message")
  sink(type = "output")
  close(log_con)
})

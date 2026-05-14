#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  if (!requireNamespace("remotes", quietly = TRUE)) {
    install.packages("remotes", repos = "https://cloud.r-project.org")
  }
})

install_cran <- function(pkgs) {
  missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) > 0) {
    install.packages(missing, repos = "https://cloud.r-project.org")
  }
}

install_bioc <- function(pkgs) {
  if (!requireNamespace("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager", repos = "https://cloud.r-project.org")
  }
  missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) > 0) {
    BiocManager::install(missing, ask = FALSE, update = FALSE)
  }
}

install_cran(c(
  "Matrix", "data.table", "dplyr", "ggplot2", "jsonlite", "optparse",
  "arrow", "remotes", "RANN", "sparseMatrixStats", "hdf5r"
))
install_bioc(c("Biobase", "BiocParallel"))

if (!requireNamespace("CRF", quietly = TRUE)) {
  install.packages(
    "https://cran.r-project.org/src/contrib/Archive/CRF/CRF_0.4-3.tar.gz",
    repos = NULL,
    type = "source"
  )
}
if (!requireNamespace("NMF", quietly = TRUE)) {
  install.packages("NMF", repos = "https://cloud.r-project.org")
}
if (!requireNamespace("cellAdmix", quietly = TRUE)) {
  remotes::install_github("kharchenkolab/cellAdmix", upgrade = "never")
}
if (!requireNamespace("spacexr", quietly = TRUE)) {
  remotes::install_github("dmcable/spacexr", upgrade = "never")
}
if (!requireNamespace("SPLIT", quietly = TRUE)) {
  remotes::install_github("bdsc-tds/SPLIT", upgrade = "never")
}

pkgs <- c(
  "SPLIT", "spacexr", "Seurat", "Matrix", "data.table", "dplyr",
  "ggplot2", "remotes", "cellAdmix", "CRF", "arrow", "RANN",
  "sparseMatrixStats", "hdf5r"
)
status <- data.frame(
  package = pkgs,
  installed = vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)
)
print(status)
if (!all(status$installed)) {
  quit(status = 1, save = "no")
}
sessionInfo()

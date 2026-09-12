# NSCLC Xenium: what each method actually read

Reconstructed from run receipts and the files themselves, not from configuration
alone. Two discrepancies were found; both are recorded here with their fixes.

## The shared input

`filtered_df_standardized.parquet` is the raw Xenium export restricted to real
genes at QV >= 20. Verified by recomputing the filter from the raw table:

| stage | rows |
|---|---|
| raw `transcripts.parquet` (532 features, 230 control/blank) | 1,643,068 |
| real genes only, **all** QV | 1,639,198 |
| **real genes AND QV >= 20** | **1,552,421** |

1,552,421 matches the standardized file exactly, and that file's QV minimum is
exactly 20.00 with zero control probes. There is no `lungcancer_df.parquet` on
this system; the standardized parquet is the intended input.

| | |
|---|---|
| path | `TRACER/datasets/dataset/lung_cancer_xenium_10x/TSU-20/filtered_df_standardized.parquet` |
| SHA-256 | `c95693aebabad66d58d7140c19d17af144b8a1a327026c45c9a92e9fbfc62e7e` |
| rows | 1,552,421 |
| genes | 302 (0 control/blank) |
| QV | min 20.00, max 40.00 |

A stricter `df_finetuned.parquet` (QV >= 30.02, 1,436,900 rows, SHA
`3b09576863c4...`) exists in the same directory and is **not** used by any
method.

## Per-method provenance

| method | spatial source | SHA-256 | QV / filter | input tx | scRNA reference | PMI source |
|---|---|---|---|---|---|---|
| Baysor | `filtered_df_standardized.parquet` | `c95693ae…62e7e` | QV>=20, 302 genes | 1,552,421 | `lung_cancer_50k.h5ad` | -- |
| ProSeg | same | `c95693ae…62e7e` | same | 1,551,332 | same | -- |
| Segger | -- | -- | -- | **not run** (see `segger_migration.md`) | -- | -- |
| SPLIT | same | `c95693ae…62e7e` | same | no transcript-level output | same | -- |
| CellAdmix | `common_inputs/xenium_transcripts_for_celladmix.parquet` | -- | **none (fixed)** | **1,637,205** | same | -- |
| TRACER | `filtered_df_standardized.parquet` | `c95693ae…62e7e` | QV>=20 | 1,552,421 | -- | `lung_cancer_npmi.csv.gz` |

The scRNA reference exists at two paths -- one in this repository, one under
TRACER -- which are **byte-identical**, SHA-256
`8e234190f502243ca31ea8f6b446b566f037f7d88199103072aa3ea8b5fd4ea3`. Either path
names the same data.

## Discrepancy 1: cellAdmix read an unfiltered population

cellAdmix's wrapper passes `--xenium-dir` to the R driver; `--transcripts` is
read only to log a provenance line and never reaches the algorithm. The driver
consumes `xenium_transcripts_for_celladmix.parquet`, which
`prepare_tsu20_common_inputs.py` built from the raw export by dropping
UNASSIGNED molecules and control probes -- **with no QV floor**.

The result: cellAdmix processed **1,637,205** molecules against the
**1,552,421** every other method saw, including roughly 85,000 at QV < 20. Its
molecule identifiers are re-indexed as `mol_id`, so they share nothing with the
standardized table and the mismatch could not be caught by an id comparison.

Its transcript counts were therefore not comparable: `frac_assigned` 0.9270 and
mean transcripts per profile 27.70 are computed over a larger denominator than
the rest of the suite.

**Fixed** in `prepare_tsu20_common_inputs.py`: a `--min-qv` option, default
20.0, applied before the UNASSIGNED and control-probe filters, with the raw /
post-QV / post-filter counts logged. **cellAdmix must be re-run on the
regenerated input before its transcript-level numbers are comparable.** Its
cell-level and reference-based metrics are unaffected in kind but will shift.

(An earlier note in this repository described this as transcript duplication.
It is not: the table holds 1,635,024 rows with 1,635,024 distinct ids. It is a
different molecule population, not a repeated one.)

## Discrepancy 2: the cPMI panel's source cohort

TRACER's panel, `lung_cancer_npmi.csv.gz`, is built from **GSE127465**
(`n_cells_i / p_i` = 54,773 on all 44,850 rows), not from `lung_cancer_50k.h5ad`.
Where a specification calls for the 50k object as the cPMI source, that
condition is currently not met. See `comparison_audit.md` sections 3 and 8.

## Mean transcripts per profile

`n_assigned / n_entities`:

| method | profiles | assigned | fraction | mean tx/profile |
|---|---|---|---|---|
| baseline_10x | 58,449 | 1,550,517 | 1.000 | 26.53 |
| celladmix | 54,721 | 1,515,722 | 0.927 | 27.70 (inflated denominator) |
| tracer | 55,967 | 1,424,202 | 0.917 | 25.45 |
| proseg | 56,930 | 1,351,617 | 0.871 | 23.74 |
| baysor | 17,691 | 1,539,147 | 0.991 | 87.00 |

The ~26 figure is 1.55M molecules over ~58k cells on a **302-plex** panel, which
is the expected order for a panel that size. Baysor's 87 follows from emitting
17,691 cells where the others emit ~56,000: fewer, larger cells.

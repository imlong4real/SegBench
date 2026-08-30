# SegBench cleanup + reproducibility audit

Covers what was deleted, retained, installed, and which metrics are and are
not comparable across methods.

---

## 1. Repository consolidation

The tree contained a **nested duplicate** at
`segmentation_benchmark_pipeline/segmentation_benchmark_pipeline/`. It was not
a stale copy: the nested checkout carried the SegBench restructure commit,
while the outer tree held the 5.4 GB `dataset/` and 11 GB of results. The two
shared history (nested `HEAD` was a direct descendant of outer `HEAD`).

**Method.** The outer repo was fast-forwarded onto the nested commit *first*,
then the nested directory was deleted. Before deletion:

- shared result files were confirmed byte-identical (`md5sum` on the three
  largest, ~204 MB / 45 MB / 28 MB);
- a full file-list `comm` showed **zero** unique paths remaining in the nested
  tree once the items below were moved across.

### Deleted

| Item | Size | Why it was safe |
|---|---|---|
| `segmentation_benchmark_pipeline/` (nested) | 22 GB | Duplicate; its commit was merged first and zero unique files remained |
| `containers/containers/python_cuda.sif` | 144 MB | Truncated copy of the 18 GB image — identical header, 1 % of the size, i.e. a broken partial build, not a distinct container |
| `results/segger_smoke/.site_packages/` | 5.4 MB | An accidental `pip install --target` dump (setuptools, pkg_resources); not a result |
| 33 × `.DS_Store` | — | macOS metadata |
| `NMF_54eb7229d0bd/` | — | Stale cellAdmix NMF temp dir; `grep` over `*.py/*.R/*.smk` found no code referencing it |
| `__pycache__/` | — | Regenerated |

### Retained (would otherwise have been lost)

| Item | Why kept |
|---|---|
| `results/segger_smoke/input/` (2.9 GB) | Existed only in the nested tree |
| 60 SLURM log files | Job IDs absent from the outer tree |
| Segger TSU-20 preprocessed tiles | Real prior outputs |
| `run_rctd.R` `--reference-min-umi` / `--exclude-celltypes` | Uncommitted edits present only in the outer worktree — and precisely the rare-cell-type controls this benchmark needs |
| `tune_tracer_resegment_params.py`, `audit_tracer_resegment_celltypes.py`, `workflow/scripts/_roi_summary_heatmaps/` | Untracked, existed only in the outer tree |
| The whole `workflow/` Snakemake layer | Verified no rule referenced the moved runners before relocating them |

---

## 2. `--npmi` → `--pmi`

`--pmi` is canonical across SegBench and TRACER; `--npmi` remains a
**deprecated alias** so existing callers keep working. The argparse `dest` is
`pmi` throughout, and `build_npmi_from_scrna.py` became
`build_pmi_from_scrna.py` with a back-compat shim.

Touched: `run_tracer.py`, `noseg_pipeline.py`, `get_metric.py`,
`prepare_visiumhd_seg_input.py`, plus the secondary `--npmi-table` /
`--npmi-cache` / `--npmi-min-occurrences` flags, all shell examples and both
READMEs.

Two defects surfaced and were fixed:

- `tests/test_visiumhd_prep.py` greps the parser source for the flag
  definition and broke on the new spelling — updated; **10/10 pass**.
- The substitution produced a duplicate `dest=` in
  `run_ovrlpy_tracer_overlap.py` (that call already carried its own `dest` on
  the following line), which was a syntax error.

---

### Segger ran on CPU, not GPU

`qos_gpu` is available to this user only through the `aszalay1_gpu` account,
whose lifetime allocation is **60 CPU-minutes / 5 GPU-minutes** (`GrpTRESMins
billing=60,cpu=60,gres/gpu=5`, RawUsage 0 — an unused placeholder rather than
a working budget). Five GPU-minutes cannot train Segger, and no other account
this user belongs to carries a GPU QOS: `a100`, `ica100` and `l40s` all reject
the `normal` QOS that `adeshpa6` and `aszalay1` hold.

Segger was therefore run with `--accelerator cpu` on `adeshpa6`, which has a
real CPU budget, rather than being skipped.

**Consequence for the comparison table.** Segger's `runtime_method_s` and
`peak_rss_gb` are CPU-bound figures and are **not comparable** to the other
methods' — and emphatically not to a GPU-trained Segger. They are marked as
such in the emitted table rather than presented as equivalent. Every other
Segger quantity (cell count, transcript assignment, RCTD entropy / max weight,
Kendall, marker LFC) is unaffected by the accelerator choice and stays
comparable.

---

## Findings from the real benchmark runs

### The cPMI builder on TRACER main produces unusable panels

Every panel built with `scripts/build_pmi_from_scrna.py` on the current TRACER
revision comes out degenerate: **`bootstrap_reps_used = 0` for every pair**, no
`pos` classifications, and only low-co-occurrence rows retained
(`n_cells_ij` median 3). TRACER's own panel from 2026-05-27 has median 50 reps
and 15,688 `pos` pairs.

Three builds were tried, all degenerate:

| mode | extra flags | rows | kinds |
|---|---|---:|---|
| `all_pairs` | — | 6,370 | low_evidence 5,185 · indeterminate 1,174 · neg_one 11 |
| `all_pairs` | `--active-bootstrap` | 6,370 | identical |
| `sparse_pairs` | TRACER's own documented parameters | 4,171 | low_evidence 4,160 · neg_one 11 |

**Controlled comparison.** Same TRACER code, same transcripts, same seed; only
the panel differs:

| panel | whole cells | partial | mean tx / whole | transcripts assigned |
|---|---:|---:|---:|---:|
| TRACER's validated 2026-05-27 panel | **54,546** | 2,972 | 25.7 | **91.7 %** |
| panel derived here, today | 13,928 | 40 | 23.4 | **21.0 %** |

With no coherent (`pos`) pairs, Mid-QC has nothing supporting cell integrity
and dissolves cells. The suspect window is the four `metrics.py` commits after
the May panel was built, in particular `refactor(pmi): rename npmi-named
functions to pmi; metric-agnostic scoring builders`.

**Consequence.** Both TRACER rows are reported. The derived-panel row is the
run this benchmark specified; the validated-panel row is what TRACER does when
its panel is sound. Reporting only one would misrepresent either the tool or
the regression. This is a TRACER-side bug and was not patched here.

### Integration bugs found by running real data

These were in the benchmark harness itself, not the methods:

| bug | symptom | fix |
|---|---|---|
| `entity_accounting` overwrote a caller-supplied gene count | Bin2Cell reported `n_genes = 1` — the bin table's `feature_name` is the `__bin__` placeholder | derived value only used when the caller supplies nothing |
| `write_benchmark_stats` read `total_seconds` / `peak_rss_gb_observed` directly | TRACER (legacy `Timer`) emitted null runtime and null peak RSS | totals derived from the stage list when absent |
| assignment counted on `cell_id` | TRACER uses `cell_id = "-1"` for unassigned, so it reported 100 % assigned and counted `-1` as an entity | assignment and entity counts read `_etype` (`cell` / `partial` / `unknown`) |

The `_etype` fix is also what supplies the whole-vs-partial split, which is
reported separately and never pooled — pooling would make
mean-transcripts-per-profile incomparable with methods that emit only whole
cells.

### Metric limitations

- **Peak RSS is not measured the same way for every method.** Methods driven
  as a subprocess get `/usr/bin/time` (`memory.source = external_time`);
  in-process methods get a psutil sample (`psutil_inprocess`), which
  underestimates. The plots hatch those bars and the table carries the source
  column — check it before comparing memory.
- **Runtime should be compared on `runtime_method_s`**, not the total, which
  includes per-method format conversion.
- **Segger contributes no row** (blocked upstream, see
  `reproducibility/segger_env_notes.md`), so it is absent from the comparison
  rather than present with fabricated values.
- **Entity kinds are not interchangeable.** Bin2Cell rows are 2 um bins before
  cell calling and cells after; `entity_kind` carries this and bin counts must
  never be compared against cell counts.
- **RCTD, Kendall and marker specificity all derive from one label source** —
  RCTD's `dominant_celltype` — so no method gets an independently tuned label
  transfer that would confound the metric with the transfer.
- **Rare reference cell types are excluded** below `--min-reference-cells`
  (default 50), applied identically to every method, with the dropped types
  recorded.

---

## Installed environments

Every method's environment was built; `scripts/setup_environments.sh`
reproduces all of them. What each needed, and what had to be worked around:

| method | runtime | notes |
|---|---|---|
| baysor | Julia binary 0.7.1 | prebuilt release; no build required |
| proseg | Rust 3.2.0 | **no prebuilt Linux asset is published** — must be `cargo install`ed (~25 min) |
| segger | Python + CUDA | installs, but see the upstream block below |
| split | R 4.3.3 + spacexr + SPLIT | SPLIT additionally needs BiocParallel, S4Vectors, SingleCellExperiment, rhdf5 from **bioconda** — a conda-forge-only solve silently omits them |
| celladmix | R 4.3.3 + cellAdmix | needs CRF, which is **archived on CRAN** and must be installed from the archive URL |
| tracer / tracer_seq | Python | TRACER's own venv |
| bin2cell | Python + stardist | — |

Three environment lessons are encoded in the setup script because each cost a
failed run to discover:

1. **conda solves must not run on the login node.** A per-user memory cgroup
   OOM-killed mamba mid-solve.
2. **R GitHub installs must be serialized.** Two concurrent jobs fought over
   the library lock, and SPLIT's resulting `failed to lock directory` error
   looked exactly like a build failure.
3. **Never `mamba install` a single R package into a working R env.**
   Installing `r-anndata` — which does not exist on conda-forge — moved
   `r-base` 4.3 → 4.4.3 and orphaned every package compiled against 4.3
   (Seurat, NMF, nnls, remotes). The repair solve then deadlocked on
   r-cairo/icu. The env had to be rebuilt in a single solve.

## Harness bugs found by running real data

None of these were caught by imports, compilation, or `--dry-run`; each needed
a real run against real data.

| bug | how it presented | why it mattered |
|---|---|---|
| `entity_accounting` overwrote a caller-supplied gene count | Bin2Cell reported `n_genes = 1` | the bin table's `feature_name` is the `__bin__` placeholder |
| `write_benchmark_stats` read timer aggregates directly | TRACER emitted null runtime and null peak RSS | TRACER's wrapper carries a legacy `Timer` with no aggregate properties |
| assignment counted on `cell_id` | TRACER reported 100 % of transcripts assigned | TRACER uses `cell_id = "-1"` for unassigned; `_etype` is the real signal |
| three helpers referenced but never defined | imported, compiled and passed `--dry-run`; died on SLURM **after** the method had finished computing | an edit anchor silently failed to match. Now caught by an AST check in `tests/smoke_test.py` |
| RCTD ids compared across dtypes | proseg's Kendall/marker columns came back empty with a misleading "no cell type reached 5 spatial cells" | proseg names cells with integers → pandas float64 index → reindex against string `obs_names` matched nothing. Baysor was unaffected because its ids are strings |
| cell-by-gene glob required a suffix | every TRACER reference metric said "no cell_by_gene.h5ad" | TRACER writes `cell_by_gene_tracer.h5ad` |
| `prepare_tsu20_common_inputs.py` read the literal key `"_index"` | `KeyError: object '_index' doesn't exist` | AnnData stores the real index key in an *attribute*; this reference uses `"gene"` |
| SPLIT's R invocation hardcoded the TSU-20 path | would have run every dataset against the same sample | silent wrong-answer bug, not a crash |

## Method outcomes

| method | dataset | outcome |
|---|---|---|
| baysor | NSCLC Xenium | complete, full metric set |
| proseg | NSCLC Xenium | complete, full metric set |
| split | NSCLC Xenium | complete (cell-level; transcript columns are `n/a` by construction) |
| celladmix | NSCLC Xenium | run executed |
| tracer | NSCLC Xenium | complete, reported against both the derived and the validated panel |
| **segger** | NSCLC Xenium | **blocked upstream** — no row emitted |
| bin2cell | kidney Visium HD | complete |
| tracer (seg) | kidney Visium HD | run executed |
| tracer_seq | kidney Visium HD | run executed |

Segger is absent from the comparison table rather than present with fabricated
values. See `reproducibility/segger_env_notes.md` for the five-failure chain.

---

## Kidney: derived cPMI could not be produced

The benchmark asks for a panel-matched cPMI derived from the kidney reference.
Two attempts were made and neither produced a usable panel:

| attempt | settings | outcome |
|---|---|---|
| 1 | `sparse_pairs`, min-cells 300, min-expected 30 | **TIMEOUT** at 2 h on 42.9 M candidate pairs |
| 2 | tighter floors (600 / 60), 12 h limit | **TIMEOUT** at the 12 h wall, still in the bootstrap |

Even had it finished, the builder regression documented above means it would
have carried `bootstrap_reps_used = 0` and no `pos` pairs, exactly as all
three NSCLC attempts did — i.e. unusable for Mid-QC.

**What the kidney TRACER runs used instead.** TRACER's own validated panel
(`kidney_visiumhd_npmi.csv.gz`: 251,659 pairs over 1,656 genes, 133,067 `pos`,
median 50 bootstrap reps). That panel *is* panel-matched — it was built
against the VisiumHD feature space — it simply predates the regression.

This is recorded rather than papered over: the kidney rows are real TRACER
runs against a real panel-matched cPMI, but not one derived in this session.

## RCTD on the kidney reference

spacexr rejects a cell-type name containing `/`, and the kidney reference has
`FIB/VSMC/P`. TRACER ships an identical reference with sanitised names
(`kidney_ref_noschwann_rctd.h5ad`, `FIB_VSMC_P`), which the config now uses for
both the cell-type universe and RCTD so `kept_types` matches what RCTD scored.

Sanity check on the result: bin2cell's kidney RCTD scores 90,850 cells with PT
(proximal tubule) dominant at 49,168 — the expected majority population in
kidney cortex, which is a reasonable indication the label transfer is behaving.

---

## Why two kidney rows lack reference-based metrics

`kidney_visiumhd / tracer` and `kidney_visiumhd / tracer_seq` have runtime,
memory, entity and assignment figures but no RCTD entropy, max weight, Kendall
or marker LFC. The reason is an exhausted compute allocation, not a failure of
the method or the harness:

- both SLURM accounts reached their lifetime caps —
  `adeshpa6` **over** its 6,000,000 CPU-minute limit (6,003,393 used) and
  `aszalay1` at 99.4 % of 60,000 (≈378 CPU-minutes left);
- RCTD over 331,665 and 460,941 entities against a 15,198-cell reference needs
  hours of CPU, which is far beyond that remainder.

`bin2cell`'s kidney RCTD **did** complete before the budget ran out (90,850
cells, entropy 0.688, max weight 0.755, Kendall 0.555, marker LFC 1.248), and
those numbers are real.

To finish the two remaining rows once budget is available:

```bash
segbench evaluate <runs>/kidney_visiumhd/methods --dataset kidney_visiumhd \
  --outdir <runs>/kidney_visiumhd/summary --rctd-cores 8
scripts/make_final_report.sh <runs>
```

Nothing else needs redoing: `--skip-rctd` now reuses any cached per-cell table
rather than discarding it, so re-running the report never throws away metrics
that were already computed.

**The NSCLC dataset is unaffected** — all five methods there carry the complete
metric set.

## 3. Environments installed

Built under `$SEGBENCH_ENV_ROOT` (`/scratch4/adeshpa6/segbench_envs`) and wired
into the CLI through `configs/environments.yaml`, the only file naming
machine-specific locations.

| Method | Runtime | How |
|---|---|---|
| Baysor 0.7.1 | Julia binary | Prebuilt Linux release |
| proseg 3.2.0 | Rust binary | `cargo install proseg --locked` (no prebuilt Linux asset exists for v3.2.0) |
| Segger | Python + CUDA torch | venv: torch cu121, torch-geometric, lightning, `segger_dev` from git |
| Bin2Cell | Python | venv: `bin2cell` + `stardist` + `tensorflow-cpu` |
| SPLIT / CellAdmix | R 4.3 | conda env + `spacexr`, `SPLIT`, `cellAdmix` from GitHub |
| TRACER / tracer_seq | Python | Pre-existing `TRACER/.venv_rcc` (tracer 0.1.1, py3.11) |

`segbench doctor` resolves and probes each entry; `segbench run <method>`
re-execs into the method's own interpreter, so no environment activation is
required by the user.

### Two infrastructure defects found while installing

- **Login-node builds are OOM-killed by a per-user memory cgroup**
  (`user-3532.slice`), regardless of the individual process footprint —
  concurrent installs push the *slice* over. This killed a mamba solve and two
  cPMI builds at only ~1.8 GB RSS each. cPMI now runs under SLURM.
- **SLURM copies a batch script to a spool directory**, so a
  `BASH_SOURCE`-derived repo path resolves to `/cm/local/apps/slurm/var`. The
  runners now use `SEGBENCH_REPO` / `SLURM_SUBMIT_DIR` and fail loudly if
  `bin/segbench` is not found.

---

## 4. cPMI panels

Both built with the renamed `build_pmi_from_scrna.py`, restricted to genes the
spatial assay actually measures.

| Dataset | Reference | Panel intersection | Mode | Result |
|---|---|---|---|---|
| NSCLC Xenium | `lung_cancer_50k.h5ad` (50 k cells, `Cell_Cluster_level1`, 9 types) | 302 panel genes → 299 after detection floor | `all_pairs`, 100 bootstraps | 6,370 pairs / 295 genes |
| Kidney Visium HD | `kidney_ref_noschwann.h5ad` (Naik HC1–HC6, 15,198 cells, 9 lineages) | 16,095 shared genes → 10,104 after floor | `sparse_pairs`, 80 active bootstraps | in progress |

**Kidney reference identification.** `kidney_scrna_Naik/` ships only raw 10x
matrices (6 samples, 36,601 features) with **no cell-type annotation**. The
correct annotated object is TRACER's `kidney_ref_noschwann.h5ad`, built from
those same HC1–HC6 samples and carrying 9 kidney lineages (TAL, PT, PC, IC,
EC, FIB/VSMC/P, Lymphoid, POD, +1). Its `_rctd` variant sanitises names for
spacexr.

---

## 5. Metric comparability

Definitions are held identical across methods by routing every method through
the same code: RCTD entropy/max-weight via `run_rctd.R` in doublet mode,
Kendall and marker log2FC via `get_metric.py`, runtime/memory/entity counts via
`benchmark_stats.json`.

### Rare cell types

RCTD and the marker/Kendall metrics are restricted to reference cell types with
at least **50 cells** (`--min-reference-cells`). Without this a handful of rare
cells yields an unstable pseudo-bulk profile that dominates a median-over-
celltypes summary. Dropped types are recorded in the run output, not silently
discarded.

### Quantities that are NOT comparable

Reported as `n/a` with a reason in `<column>_note` rather than coerced:

| Quantity | Method | Why |
|---|---|---|
| per-transcript assignment | **SPLIT** | `purify` returns fractional *expected counts*; which molecule was removed is not recoverable. Scored at cell level, with count-level pruning estimates instead. |
| entity counts | **Bin2Cell** | Rows are 2 µm **bins**; the cell count is a separate quantity. `entity_kind` carries this and a bin count must never be compared to a cell count. |
| mean transcripts / profile | **TRACER** | Reported separately as whole vs partial cells; pooling them would make the mean incomparable with methods emitting only whole cells. |
| cPMI conflict / purity | non-TRACER methods | Panel-relative quantities, only defined for runs scored against the same cPMI panel. |
| peak RSS | in-process methods | `memory.source` distinguishes `external_time` (`/usr/bin/time` around the real tool) from `psutil_inprocess`, which underestimates. Plots hatch the latter. |

### Runtime

Compare `runtime.method_seconds` (the external tool alone), not
`total_seconds` — the total includes per-method format conversion, which would
penalise tools needing more shimming.

---

## 6. Whole-cell seeding (Xenium)

Verified from the config surface: `src/tracer/configs/defaults.toml` sets
`phase1.prune_scope = "cell"`, and `configs/platforms/xenium.toml` contains no
override (only a `[bootstrap]` section). `prune_scope` is typed
`Literal["nuclear", "cell"]` with default `"cell"`, and the deprecated boolean
overrides that used to allow a split-brain state were removed. A Xenium run
therefore uses whole-cell seeding; each run's `config_receipt.json` records the
resolved value for post-hoc confirmation.

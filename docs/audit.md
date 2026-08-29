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

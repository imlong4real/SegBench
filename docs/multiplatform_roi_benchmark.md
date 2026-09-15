# Multiplatform frozen-ROI SegBench benchmark

Six imaging segmentation / refinement methods on four platforms, each run on a
preregistered fixed-physical-window ROI, scored by one packaged evaluation
suite against a held-out scRNA reference.

Campaign root: `results/segbench_multiplatform_v1/`
Cirro project: BTC-DST-Development `d57d7407-4c32-4256-bd2e-aa1e037569aa`
Cirro process: `segbench-roi-0-2-0-cirro`

---

## 1. ROI design (frozen before any method ran)

`results/segbench_multiplatform_v1/roi_design/roi_manifest_frozen.json`

The design is a **fixed physical window**, not a fixed transcript count,
because transcript density differs across these platforms by more than an
order of magnitude:

| Platform | Transcripts | Occupied tissue | Density |
|---|---:|---:|---:|
| Xenium5K cervical | 87,050,061 | 100.93 mm² | 0.86M tx/mm² |
| Atera cervical | 728,918,209 | 72.68 mm² | **10.03M tx/mm²** |
| CosMx lung (Lung5_Rep1) | 37,138,747 | 19.26 mm² | 1.93M tx/mm² |
| MERFISH mouse ileum | 819,665 | 0.53 mm² | 1.54M tx/mm² |

Atera is **11.6× denser than Xenium5K on the same tissue type**. A fixed
transcript budget would therefore have compared a postage stamp of Atera
against a large field of Xenium5K.

### Protocol

1. Stream each platform's effective input population and accumulate a 25 µm
   fine histogram of transcript coordinates. **Only x/y are read.**
2. Choose one common square window side `L`, identical for every applicable
   platform, on input-only feasibility grounds.
3. Tile each tissue into non-overlapping `L × L` windows anchored at the fine
   grid origin; only windows entirely inside the grid are considered, so every
   window has exactly the same physical area.
4. Eligibility (input-only): `occ(w)` = fraction of the window's 25 µm
   sub-bins holding ≥1 transcript; eligible iff `occ(w) ≥ 0.90`.
5. Rank eligible windows by transcripts per mm².
6. Select the windows nearest the 25th / 50th / 75th percentile of that
   distribution (linear-interpolated percentile, nearest by absolute
   difference, row-major tie-break).

`L = 500 µm` (0.25 mm²), chosen from the ladder {250, 500, 750, 1000}:

* **Compute** — the heaviest ROI that actually executes is Atera q75 at 3.34M
  transcripts, ≈2.2× the largest completed SegBench run (NSCLC Xenium TSU-20,
  1.55M). At 750 µm and 1000 µm, Atera reaches 7.5M and 13.2M per ROI per
  method, with cellAdmix the known-slow method.
* **Statistics** — at 250 µm an ROI holds only ~460–560 cells on the sparser
  platforms, too few for stable RCTD / marker log2FC / pseudobulk correlation.
* **Percentile stability** — 500 µm leaves 374 / 264 / 59 eligible windows
  (Xenium5K / Atera / CosMx). At 1000 µm CosMx retains only **10**, which makes
  a 25th/75th-percentile selection coarse and unstable.
* 500 µm is an exact multiple of the 25 µm design grid.

**Occupancy threshold robustness.** The 500 µm occupancy distribution is
strongly bimodal on Xenium5K (p10 = 0.02, p25 = 0.95) — off-tissue windows sit
near 0, tissue windows near 1 — so any threshold inside that gap selects
essentially the same set. Sweeping `occ_min` over {0.75, 0.85, 0.90, 0.95,
0.99} moves the eligible-window count by ≤12% and the median-ROI transcript
count by ≤4% on every platform
(`roi_design/logs/explore_occ_sensitivity.log`).

### Selected ROIs

| Dataset | Platform | q25 | q50 (primary) | q75 |
|---|---|---:|---:|---:|
| xenium5k_cervical | Xenium5K | 123,180 | **173,995** | 264,425 |
| atera_cervical | Atera | 1,930,637 | **2,679,679** | 3,335,246 |
| cosmx_nsclc | CosMx | 341,980 | **483,319** | 599,248 |
| merfish_mouse_ileum | MERFISH | — | **819,665** (whole tissue) | — |

q50 is the primary comparison; q25/q75 are predefined density sensitivities.
All transcripts inside a window are kept — no subsampling, no transcript
target. MERFISH is run whole-tissue and is not ROI-selected: its occupancy
never reaches 0.90 (max 0.84) because the ileum lumen is genuine void space.

### What ROI selection did *not* use

Cell types, biological markers, segmentation outputs, benchmark metrics. The
selection consumes integer count grids derived from x/y only.

> **Supersedes the earlier ROI set.** The previous cross-platform ROIs in
> `dataset/*/roi_summary.json` were selected by *highest mean RCTD problem
> score* — a benchmark metric — at a different window size per dataset and
> against a fixed ~1.5M transcript target. They are not used here.

---

## 2. Effective input populations

| Platform | Population | Note |
|---|---|---|
| Xenium5K, Atera | `filtered_df.parquet` — `qv > 30` **and** `is_gene` | frozen upstream by the dataset's own two-pass stream filter |
| CosMx | `NegPrb*` removed; px → µm at **0.18 µm/px** | verified: FOV = 5472 × 3648 px = 985 × 657 µm, matching the `fov_positions` grid pitch |
| MERFISH | all molecules in `input_transcripts_um.parquet` | µm coordinates; `roi_transcripts.parquet` is in **pixels** and is not used |

No method-specific QV or control-probe filtering is applied anywhere, and the
frozen ROI parquet is the single input every method in a run consumes.

### MERFISH original segmentation mask

The "original mask" is **`membrane_prior_3d`** — the Cellpose membrane prior
the Petukhov Baysor run consumed — joined from the upstream
`baysor_segmentation.csv` on `mol_id → transcript_id` (1:1 and total over all
819,665 molecules).

Verified (`inputs/merfish_original_mask_report.json`):

| Field | Labels | Assigned |
|---|---:|---:|
| `membrane_prior_3d` (used) | 7,722 | 48.5% |
| `prior_segmentation` | — | 99.1% agreement with the above |
| `cell` (Baysor output) | 8,211 | 98.0% |
| `membrane_prior_z85` | — | 6.9% |
| `dapi_prior_z85` | — | 5.4% |

The `cell_id` delivered in `input_transcripts_um.parquet` equals Baysor's
output for 98.0% of molecules but the membrane prior for only 0.005%, so the
delivered column is Baysor's **result**, not the prior. Using it as the
"original" baseline would have made a method's output the input of the
refinement methods. Baysor's output is retained only as
`baysor_reference_cell_id` (provenance; never read by a method).

---

## 3. Reference splits (SPLIT leakage protection)

One deterministic stratified split per dataset, frozen before production:

| Reference | Train | Held-out | Genes |
|---|---:|---:|---:|
| cervical (Xenium5K view) | 17,552 | 4,388 | 5,051 / 5,100 |
| cervical (Atera view) | 17,552 | 4,388 | 17,786 / 18,022 |
| lung (CosMx view) | 40,001 | 9,999 | 958 / 960 |
| mouse ileum | 3,141 | 786 | 236 / 241 |

Stratified by cell type, per-stratum lexicographic order then PCG64
permutation, seed 20260909, 20% held out. The two cervical views share an
**identical cell partition** (verified byte-identical) and differ only in gene
restriction, so Xenium5K and Atera are scored against the same held-out cells.

SPLIT and the cPMI builders see only the training subset. The held-out subset
is delivered only to the evaluator, and scores every method identically.

---

## 4. cPMI panels

| Panel | Source | Edges | Genes | SHA-256 (first 16) |
|---|---|---:|---:|---|
| CosMx lung | built, `9a6b078` | 397,192 | 911 | `7a82f9bacd8661e9` |
| MERFISH ileum | built, `9a6b078` | 1,413 | 55 | `cb05d48b1fe223b5` |
| Cervical → Xenium5K | filtered | 6,393,629 | 3,754 | `7297aaa8b79cf58b` |
| Cervical → Atera | filtered | 70,518,829 | 12,466 | `67191e2c4cfe025f` |

Built panels use the pinned builder `imlong4real/TRACER@9a6b078` with the
frozen parameters (`--strategy rep --arms xgt1`, cPMI promoted into the `PMI`
column, seed 20260909, `min_det_cells` 25 on the unique draw, 25 depth bins,
full-library depth), from the **training** split only.

The cervical panel is used **as delivered** — never rebuilt or retuned per ROI.
The only operation is dropping edges whose genes the platform does not measure.
Source: `cervical_xgt1_cpmi_balanced.csv.gz`,
sha256 `7e79cb411bf5e61fc7a09ba4f804d48164715d27c49f660e23b5efd3af0833b9`,
70,518,829 edges over 12,466 genes.

---

## 5. Methods

Baysor, ProSeg, Segger (GPU), SPLIT, cellAdmix, TRACER Seg — all six on every
platform, from the identical frozen ROI parquet. A method that genuinely
cannot consume a platform's input representation is recorded as
`not_applicable` with the technical reason in `not_applicable.tsv`; the
benchmark is not modified to force it.

Per-ROI method inputs are derived once, in `PREP_ROI`, from the frozen
parquet: Xenium-named transcripts, nucleus geometry, an entity table, a
cell-by-gene matrix and cluster labels.

---

### Measured method feasibility at high plex

The Atera panel (17,868 genes) is 3.8× Xenium5K, 18.6× CosMx and 74× MERFISH,
and two methods hit hard resource limits there that they do not hit anywhere
else. Measured on the *smallest* Atera ROI (q25, 1.93M transcripts):

| Method | Request | Peak RSS | Outcome |
|---|---:|---:|---|
| cellAdmix attempt 1 | 160 GiB | 159.6 GiB | SIGKILL (OOM) after 3m53s |
| cellAdmix attempt 2 | 320 GiB | 295.9 GiB | exit 1, R allocation failure, after 32m17s |
| Baysor attempt 1 | 64 GiB | — | AWS Batch OOM, **no exit status reported** |

Consequences, all of which are recorded rather than worked around:

* Baysor's base allocation was raised 64 → 128 GiB, and a *null* exit status
  now escalates memory like exit 1. AWS Batch reports a container OOM either
  way, so matching on exit code alone silently mis-classified Baysor's failure
  as non-retryable.
* Atera runs are given a 512 GiB ceiling, so SPLIT/cellAdmix escalate
  160 → 320 → 480 GiB. The ROI is **not** shrunk to make a method fit.
* A method that still exhausts its retries is marked `not_applicable` for that
  platform with its measured peak RSS and failure mode, and the remaining
  methods and the held-out evaluation complete normally. Before this, one
  method's terminal failure aborted the whole ROI and discarded the other five.

## 6. Evaluation

The packaged suite already used for NSCLC Xenium and kidney VisiumHD:
runtime, CPU time, peak host RSS, GPU runtime/peak VRAM (reported separately),
total entities, whole/partial entities, assigned/unassigned/recovered
transcripts, median transcripts per entity, RCTD entropy, RCTD max weight,
marker specificity log2FC, Kendall/Pearson/Spearman vs the held-out reference,
and TRACER coherence/purity/conflict where applicable.

Density-normalised additions, so platform density cannot make the
computational comparison misleading:

* `runtime_per_1m_transcripts` — seconds per 1M transcripts
* `peak_rss_gb_per_1m_transcripts` — GiB per 1M transcripts
* `entities_per_mm2` — entities per mm² of frozen ROI area

---

## 7. Caveats

These are properties of the data and the design, not defects to be tuned away.

1. **The MERFISH cPMI panel is thin** — 1,413 edges over 55 of 241 panel
   genes. The ileum reference has 3,141 training cells, of which the balanced
   draw leaves 1,863 unique, so the frozen 25-cell detection floor removes most
   genes. TRACER's coherence/purity/conflict on MERFISH therefore rests on 55
   genes. The frozen builder parameters were kept rather than lowered for this
   one dataset.

2. **cPMI edge counts are wildly unequal between the two cervical platforms** —
   Atera retains all 70.5M reference edges (its WTA panel covers the whole
   reference) while Xenium5K retains 6.4M (9.1%). Any Atera-vs-Xenium5K
   comparison of a cPMI-derived metric is partly a comparison of panel breadth.
   Compare methods *within* a platform.

3. **The cervical population is `qv > 30`**, while the frozen NSCLC Xenium
   benchmark used `qv > 20`. Both cervical platforms share the same threshold,
   so within-cervical comparisons are unaffected; cross-campaign comparisons
   against NSCLC are not on an identical QV footing.

4. **The MERFISH original mask assigns only 48.5% of molecules.** The
   refinement methods (SPLIT, cellAdmix, TRACER Seg) therefore start from a
   genuinely sparse prior on that platform, unlike the ~80–90% vendor
   assignment elsewhere.

5. **Nucleus geometry in the ROI bundles is derived, not vendor-supplied** —
   the convex hull of a cell's `overlaps_nucleus` molecules, with a circle
   fallback. Vendor nucleus polygons are not carried through the frozen ROI
   parquet. Methods that consume nucleus geometry see this approximation.

6. **Cluster labels for cellAdmix are k-means (k = 12) on the ROI's own
   cell-by-gene matrix** (sparse truncated SVD, seed 20260909), not the
   vendor's graph clustering, which does not exist for a 500 µm sub-window.
   Unsupervised and input-only, but not identical to the lung graph's
   `graphclust` labels.

7. **MERFISH whole tissue is 0.53 mm²** — roughly two ROIs' worth of area,
   and the entire delivered Petukhov molecule set. It is comparable in scale
   to the other platforms' ROIs but is not a density-matched selection.

8. **MERFISH reference metrics rest on 3 of 6 cell types.** The evaluator's
   `--min-reference-cells 50` floor — identical to the NSCLC and kidney runs,
   and deliberately not relaxed for one dataset — drops Enteroendocrine (19
   held-out cells), Paneth (41) and Tuft (16), leaving Enterocyte, Goblet and
   Stem_TA. Those three sit on one differentiation axis and are
   transcriptionally similar, so RCTD entropy, RCTD max weight, marker log2FC
   and the pseudobulk correlations have less to discriminate on MERFISH than
   on the other platforms, where every cell type clears the floor
   (cervical 10/10, CosMx 9/9). Read MERFISH reference-concordance numbers as
   a three-way problem, not a six-way one. This was determined from the frozen
   split before any run, not from results.

---

## 8. Reproducing

```bash
# 1. density scan  (input coordinates only)
workflow/scripts/_roi_design/scan_density.py --key <key> --parquet <src> ...

# 2. inspect the window-size ladder, then freeze
workflow/scripts/_roi_design/select_rois.py --explore ...
workflow/scripts/_roi_design/select_rois.py --freeze --window-um 500 --occ-min 0.90 ...

# 3. cut the frozen ROIs
workflow/scripts/_roi_design/extract_rois.py --manifest roi_manifest_frozen.json ...

# 4. references and panels
workflows/cirro/bin/freeze_reference_split.py ...
workflows/cirro/bin/build_lung_panel_from_split.py ...
workflow/scripts/_roi_design/filter_cpmi_by_genes.py ...

# 5. launch and collect
workflow/scripts/_roi_design/launch_cirro_runs.py --run full --tier primary ...
workflow/scripts/_roi_design/collect_results.py --ledger ... --outdir ...
```

### Segger: why `--cells-min-counts 1`

Segger v0.2.0 builds its reference AnnData with `cells_min_counts=10`, which
drops cells below that threshold from `adata.obs`, but its segmentation graph
then left-joins *every* transcript passing the segmentation mask against that
table. Transcripts of a dropped cell therefore receive a **null**
`cell_encoding`. The very next line fills nulls for `cell_cluster` and not for
`cell_encoding`, so the null survives into
`setup_segmentation_graph`, where Polars materialises the nullable column as
`float64` with NaN and PyTorch refuses it as an index:

```
IndexError: tensors used as indices must be long, int, byte or bool tensors
```

Every platform here trips it, because the original mask always contains some
small cells:

| Platform | cells | with <10 transcripts | transcripts affected |
|---|---:|---:|---:|
| Atera q25 | 3,496 | 19 (0.5%) | 88 |
| CosMx q25 | 1,022 | 15 (1.5%) | 61 |
| Xenium5K q25 | 2,024 | 150 (7.4%) | 839 |
| MERFISH whole | 7,722 | **2,405 (31.1%)** | 10,446 |

MERFISH is worst because the Cellpose membrane prior produces many small
partial cells.

`--cells-min-counts 1` keeps every original-mask cell in the AnnData, so no
transcript can join to a missing row. It is a compatibility setting, not a
quality knob: it is applied identically on every platform, it was chosen before
any Segger metric was seen, and it makes Segger's seed set exactly the original
mask that SPLIT, cellAdmix and TRACER Seg also refine — the alternative
(dropping small cells from the boundary tables) would have given Segger a
different input population from every other method.

### Reading `peak_rss_gb`

`run_with_resources.py` records **the sum of live process-tree RSS**, sampled
each second. Shared pages are therefore counted once per forked worker, so the
figure overstates real memory for any multi-process method — and for a method
that forks heavily it can exceed the container's own allocation, which is proof
the number is an over-count rather than a requirement.

Baysor on Atera q50 reports **685 GiB** against a container capped well below
that. Read it as "Baysor forks many workers over a 17,868-gene panel", not as a
685 GiB requirement. The tidy tables carry `peak_rss_exceeds_request` (1/0) so
these rows are visible rather than silently trusted, and
`peak_rss_gb_per_1m_transcripts` inherits the same caveat.

The metric is deliberately **not** changed mid-campaign: the frozen NSCLC Xenium
and kidney VisiumHD runs used the same summed definition, and switching to PSS
or max-single-process now would make this campaign incomparable with them.
Sound cross-method comparison here should lean on `cpu_time_seconds` and
wall-clock runtime, which are not affected.

### TRACER: whole cells and partials are reported separately

TRACER emits two kinds of entity — a whole cell, and a partial fragment it
could not stitch — and they are not one population. Partials carry far fewer
transcripts (CosMx q50: median 35 against a whole cell's 139) and score very
differently, so a single pooled number averages two distributions, weighted by
whichever is more numerous. Partials are 60% of TRACER entities on Atera, 62%
on CosMx and 31% on MERFISH.

Measured at q50:

| ROI | metric | whole | partial | pooled (previously reported) |
|---|---|---:|---:|---:|
| Atera | RCTD entropy | **0.0167** | 0.6860 | 0.2609 |
| Atera | RCTD max weight | 0.9982 | 0.7259 | 0.9420 |
| CosMx | RCTD entropy | 0.2089 | 0.6844 | 0.4883 |
| CosMx | RCTD max weight | 0.9532 | 0.7216 | 0.8407 |
| MERFISH | RCTD entropy | 0.1613 | 0.2677 | 0.2009 |
| MERFISH | RCTD max weight | 0.9644 | 0.9298 | 0.9529 |

cPMI purity and conflict move the same way on every platform (whole: higher
purity, lower conflict).

**Pooling can change the ranking, not just the value.** On MERFISH the pooled
TRACER entropy (0.201) is worse than SPLIT (0.182), while TRACER's whole cells
(0.161) are better than SPLIT. On Atera the pooled figure is roughly 16× the
whole-cell value.

**This is not a like-for-like cross-method comparison.** Baysor, ProSeg and
cellAdmix have no whole/partial distinction — every entity they emit is a whole
cell by construction. TRACER-whole is a subset *TRACER itself selected*, so part
of its advantage on that subset is selection, not segmentation quality. Read the
split as "TRACER's whole cells against its own partials". The pooled row is
retained alongside, because it is the only TRACER figure directly comparable to
the other methods.

Two mechanisms deliver this:

* `src/segbench/methods/tracer.py` writes `whole_partial_status` into
  `cell_by_gene_tracer.h5ad`, so runs from that revision onward can be
  stratified natively by the packaged evaluator.
* `workflow/scripts/_roi_design/stratify_tracer_entities.py` recomputes the
  exactly-recoverable metrics (RCTD entropy/max weight from the published
  per-cell weights, cPMI purity/conflict from `cell_scores.tsv.gz`) for runs
  that predate it, without re-running anything.

Both join on **`tracer_id`**, never `cell_id`: a partial shares its parent's
cell_id, so cell_id resolves only 38% of entities while tracer_id resolves 100%.
Pseudobulk correlation and marker log2FC are deliberately not recomputed
post-hoc — they aggregate across cells by predicted type, and reimplementing
that would risk diverging from the packaged evaluator.

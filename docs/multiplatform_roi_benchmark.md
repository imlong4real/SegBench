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

# Audit: why the headline table overstates SPLIT

The first comparison table ranked SPLIT ahead of every other method on all four
reference-based metrics. This document tests that result and finds it is
substantially an artefact of two things the table did not control for:
**which cells each method is scored on**, and **the fact that SPLIT optimises
against the same reference the benchmark evaluates with**.

Nothing here says SPLIT is a bad method. It says the original table did not
support the comparison that was being read off it.

---

## 1. SPLIT is scored on 62% of the cells

SPLIT is a purification method: it consumes the vendor segmentation and
returns cleaned profiles. It does not return all of them.

| Stage | Cells | Lost |
|---|---|---|
| Vendor 10x Xenium segmentation | 58,449 | — |
| SPLIT purified output | 40,811 | **17,638** (30.2%) |
| Scored by the benchmark's RCTD | 36,261 | 4,550 more |

**SPLIT's published metrics describe 62.0% of the original cells.** TRACER's
describe 70.3%, and the two populations are not the same cells.

The wrapper had already recorded the first drop --
`n_cells_original_dropped_by_split: 17638` in `split_pruning_summary.json` --
but nothing propagated it into the comparison table, so it never reached the
reader.

### Where the cells go

Two mechanisms, both selective, neither a bug:

1. **Depth threshold.** `workflow/scripts/_count_correction/run_split_tsu20_real.R`
   builds RCTD with `UMI_min = 10` and `counts_MIN = 10`. Retained cells have a
   hard floor at exactly 10 total counts; 12,820 vendor cells fall below it.
2. **RCTD spot-class rejection.** The remaining ~4,818 dropped cells clear the
   depth floor but are not confidently typed, so purification has nothing to
   act on.

A third filter then applies: the benchmark's own RCTD re-imposes `UMI_min = 10`
on the *purified* counts, which are lower than the originals because
purification removes counts. That is the 40,811 -> 36,261 step.

| | n | mean counts | median | min | max |
|---|---|---|---|---|---|
| dropped | 17,638 | 8.7 | 7 | 1 | 100 |
| retained | 40,811 | 34.2 | 28 | **10** | 271 |

### The dropped cells are the hard ones

This is the decisive measurement. TRACER scores every cell, so it can be
evaluated separately on the two populations SPLIT splits the data into:

| TRACER restricted to | n | RCTD entropy | max weight |
|---|---|---|---|
| cells SPLIT **kept** | 35,242 | 0.527 | 0.830 |
| cells SPLIT **dropped** | 5,828 | **0.948** | **0.561** |

Entropy 0.948 is close to the maximum for this cell-type universe: these are
cells no method can confidently type. SPLIT is never asked to.

---

## 2. Matched-cell comparison

Restricting every comparable method to the 35,242 cells scored for all of
them:

| | native | matched | change |
|---|---|---|---|
| split entropy | 0.4031 | 0.3985 | -0.005 |
| tracer entropy | 0.5857 | 0.5270 | **-0.059** |
| gap | 0.183 | **0.129** | -30% |

SPLIT barely moves, because the matched set is almost exactly its own native
set. TRACER improves substantially, because matching removes the hard cells it
was being penalised for scoring. **About 30% of SPLIT's apparent entropy
advantage is selection.** The remainder is real under this metric -- but see
the next section for what that metric actually measures.

### Against a do-nothing baseline

The vendor segmentation, scored on the same 35,242 cells, is the reference
point the original table never had:

| method (matched cells) | entropy | max weight | entropy reduction vs baseline |
|---|---|---|---|
| baseline_10x (do nothing) | 0.7505 | 0.7441 | -- |
| tracer | 0.5270 | 0.8300 | 0.2235 (29.8%) |
| split | 0.3985 | 0.8875 | **0.3521 (46.9%)** |

Both methods genuinely improve on doing nothing, and on this metric SPLIT
improves more even after matching. That is the fair statement of SPLIT's
result: its lead over TRACER is smaller than the headline table implied, not
absent. What the number cannot tell you is how much of that lead is
segmentation quality and how much is agreement with the reference SPLIT was
optimised against -- which is what section 3 is about.

### Which methods can be matched at all

Only methods that preserve the vendor cell-id space:

| method | matched? | why |
|---|---|---|
| baseline_10x | yes | vendor segmentation itself |
| split | yes | purifies vendor cells in place |
| tracer | yes | refines assignments within vendor cells |
| proseg, baysor | **no** | de-novo segmentation: their cells are new objects with no cell-for-cell correspondence |
| celladmix | see run | refinement; correspondence checked empirically |

For the de-novo methods there is no matched comparison to make. That is a
property of the methods, and is reported rather than papered over.

---

## 3. Reference circularity

The four reference-based metrics are **not independent of SPLIT's objective**.

SPLIT purifies by running RCTD against the scRNA reference and removing the
count mass that does not fit the dominant singlet profile. The benchmark then
scores it by running RCTD against the same reference and asking how
concentrated the weights are.

A method that removes exactly the counts which make RCTD weights diffuse will
score well on a metric defined as "RCTD weights are not diffuse". That is not
evidence about segmentation quality; it is the optimiser reporting its own
objective.

Confirmed from `rctd.log`: the benchmark RCTD used **all 50,000** reference
cells (`[ref] 50000 cells across 9 celltypes`) -- the same object SPLIT was
given.

These four are therefore relabelled **reference-concordance** metrics, not
accuracy metrics:

- `rctd_entropy_median`
- `rctd_max_weight_median`
- `kendall_tau_median`
- `marker_logfc_median`

The relabelling applies to *every* method. It is not a penalty aimed at SPLIT;
it states what the number measures.

### Held-out evaluation, and why the obvious split does not work

The reference carries a study-disjoint split, which at first looks like a ready
made independent cohort:

| `id` | cells | studies |
|---|---|---|
| Reference | 43,606 | GSE131907 (14,539), KU_loom (13,603), GSE148071 (8,715), GSE136246 (5,960), GSE153935 (789) |
| Validation | 6,394 | **GSE127465 (5,219)**, GSE119911 (1,175) |

The two halves share no study, so Validation is genuinely external *to the RCTD
reference*. But it is not external to TRACER.

**TRACER's cPMI panel is built from GSE127465.** From the panel's own build
receipt (`results/reference_npmi/npmi_build_summary.json`):

    --reference-h5ad datasets/lung_cancer_scrna_GSE127465/processed/h5ad/
                     lung_scrna_GSE127465_harmonized.h5ad

and GSE127465 supplies **81.6%** of the Validation cells. Scoring TRACER
against Validation would be scoring it largely against the cohort its own gene
pair statistics came from -- the same circularity SPLIT has with the full
reference, just via a different route.

So the two methods are entangled with *different* slices of the reference:

| method | optimises against | circular with respect to |
|---|---|---|
| SPLIT | the full 50k reference, via RCTD | the entire evaluation reference |
| TRACER | a cPMI panel built from GSE127465 | GSE127465 |
| baseline_10x | nothing | -- |

The only cohort disjoint from **both** is **GSE119911**, at 1,175 cells -- and
it is too thin to carry the full comparison. Its composition, against the
50-cell minimum the pipeline already applies:

| cell type | GSE119911 | >= 50? |
|---|---|---|
| T | 574 | yes |
| Myeloid | 319 | yes |
| Cancer | 206 | yes |
| Mast | 33 | no |
| B | 33 | no |
| Fibroblasts | 6 | no |
| Plasma | 4 | no |
| Endothelial | 0 | absent |
| Ciliated | 0 | absent |

**Three of the nine cell types survive.** So the conclusion to state plainly:

> A fully independent held-out evaluation is **not possible** with this
> reference. Every cohort large enough to score against is one that either
> SPLIT (via RCTD) or TRACER (via its cPMI panel) was already fitted to.

What *is* possible is a restricted check on GSE119911 over T, Myeloid and
Cancer -- the three most abundant types, 1,099 of its 1,175 cells. That is
reported as a partial, low-power check, not as the independent evaluation the
benchmark would need. Fixing this properly means sourcing an external lung
scRNA cohort that contributed to neither the RCTD reference nor the cPMI
panel, which is a data-acquisition task rather than an analysis one.

---

## 4. What the corrected table reports

Three views per method, so a lead earned by dropping cells is visible as such:

- `native_*` -- the method's own output, on whatever cells it emitted. This is
  what the original table showed.
- `matched_*` -- the same metric on the fixed cell set common to all
  comparable methods.
- `heldout_*` -- pseudo-bulk metrics against the study-disjoint Validation
  cohort.

Reproduce with:

```bash
scripts/audit_selection_bias.py --dataset nsclc_xenium
```

Outputs `corrected_comparison.csv`, `cell_funnel.csv` and
`audit_provenance.json` (which records the matched cell count, the excluded
methods and their reasons, and the held-out cohort membership).

---

## 5. Scoring the baseline: what it cost, and what was scored instead

The vendor segmentation had never been scored by the benchmark's own RCTD, so
there was no "do nothing" row to compare the purification methods against.
Adding one turned out to be the most expensive item in this audit.

Two problems had to be fixed first:

1. **The reference was loaded whole before genes were restricted.**
   `run_rctd.R` does subset the reference to the spatial panel, but only after
   `read_h5ad` has pulled the full 50,000 x 72,131 scRNA matrix into R, so peak
   memory is set by a matrix ~180x larger than the one actually used. Under a
   5 GB cgroup that is killed before RCTD starts.
   `segbench.evaluate.panel_restricted_reference` now writes a panel-only copy
   of the reference once (360 MB -> 47 MB), keyed by a hash of the panel, and
   hands R a file that is already narrow.
2. **The parallel worker cluster is fragile on a contended node.** A 58,449-cell
   doublet-mode run held 4 cores for over nine hours at ~37% CPU each (login
   node load average 18-25, 45 users) and then died with
   `unserialize(node$con): error reading from connection` -- a SOCK worker lost.
   The master hung and two workers were orphaned onto init, still spinning.

Rather than pay that cost again for a number the comparison does not need, the
baseline was scored on **the matched cell set only** (35,242 cells): that is
precisely the population the corrected table compares on, and it is 40% smaller.
All 35,242 matched cells are present in the vendor object with >= 10 counts, so
nothing is lost to the depth filter.

**Consequence to read honestly:** there is a `matched_*` row for `baseline_10x`
but no `native_*` row. The vendor segmentation's metrics over all 58,449 of its
own cells were not computed. That does not affect any claim made here -- every
claim about baseline is a matched-set claim -- but the table must not be read as
if the missing cell were merely omitted for space.

---

## 6. Is TRACER's result about TRACER, or about its panel?

TRACER and SPLIT differ in two ways at once: the algorithm, and the cohort
whose statistics it consumes. Section 3 showed the cohorts are different
(GSE127465 for TRACER's cPMI panel, the full 50k reference for SPLIT). This
section holds the segmentation and the algorithm fixed and swaps only the
panel's source cohort.

A second cPMI panel was built from the **same 50k reference SPLIT purifies
against**, with every build parameter identical to TRACER's own panel
(`--mode sparse_pairs --min-cells-expressed 10
--min-expected-cooccurrence 10 --pmi-abs-threshold 0.2 --bootstrap-n 100
--seed 1`, same active-bootstrap settings, same 302-gene spatial panel).

The first attempt at this got it wrong, in a way worth recording because the
failure mode is silent.

TRACER's run consumes `lung_cancer_npmi.csv.gz`, which is an **`all_pairs`**
build: 44,850 rows, 42,770 carrying an NPMI value. The replacement panel was
built with `--mode sparse_pairs` and produced 4,171 pairs. That is not a
provenance swap -- it is a different build mode, 10.3x fewer usable pairs.

Run with it, TRACER scored **2,120 cells of 58,449** (3.6%). The method did not
produce a worse answer; it produced almost no answer. Reporting an entropy from
those 2,120 cells beside 35,242 matched cells would have been a textbook
instance of the very selection effect this document is about -- a number
computed on 3.6% of the data, presented as if comparable.

Rebuilding in `all_pairs` mode did not rescue it either, and the first
explanation offered here for that -- that the 50k reference is too shallow to
support cPMI -- **was wrong, and is retracted.** Measuring the two references
directly refutes it:

| over the 300 panel genes | 50k RCTD reference | GSE127465 |
|---|---|---|
| cells | 50,000 | 54,773 |
| counts per cell | 907 mean / 92 median | 60 mean / 52 median |
| per-gene detection rate | 0.096 | 0.054 |
| **fraction of gene pairs with expected co-occurrence >= 10** | **80.7%** | 64.8% |

The 50k reference is the *deeper* of the two and should clear the evidence
threshold on **more** pairs, not fewer. (An earlier version of this measurement
sampled the first 5,000 cells of each object rather than a random 8,000; with
files ordered by study that is a biased slice, and it gave the opposite answer
for the mean. Random sampling is what the table above uses.)

What the panel files actually record settles the provenance question that the
missing build receipt left open. Every row carries `n_cells_i` and `p_i`, whose
ratio is the source cohort's cell count:

- `lung_cancer_npmi.csv.gz` -> **54,773** cells on all 44,850 rows = GSE127465
- the rebuilt panel -> **50,000** cells on all 6,370 rows = the 50k reference

So TRACER's panel is confirmed to come from GSE127465 (section 3 stands), and
the rebuild did read the reference it was given.

### Where this actually failed

The builder computed what it should: `n_candidate_pairs_pre = 44,551`, of which
8,063 were dropped for expected co-occurrence below 10, leaving 36,477
candidates. The file it then wrote holds 6,370 rows whose mean expected
co-occurrence is **3.31**, with only 11 at or above 10 -- that is, the written
rows are drawn from the pairs that were *rejected*, not the 36,477 that
survived. The 29,394 well-evidenced pairs GSE127465's panel carries have no
counterpart in the output.

**This is a tooling problem, not a property of the reference.** Until it is
understood, the honest position is that the swap has not been performed --
not that it cannot be.

**Therefore the question "is TRACER's result about TRACER or about its panel?"
is still open** -- blocked on a panel-writing step that is not yet understood,
with the reference itself measured and adequate. The degenerate run that scored
2,120 cells must not be presented as the answer.

---

## 7. The reference-free metric, and what it says

ovrlpy's vertical signal integrity (VSI) asks whether the transcripts a method
assigned to a cell come from a vertically coherent signal. It is computed from
transcript coordinates and gene identities alone -- **no cell-type reference is
involved**, so it is the one metric here that neither SPLIT nor TRACER can be
optimising against.

The integrity map depends only on the coordinates, which every method on this
sample shares, so it is fitted once (1,552,421 transcripts, 108 s) and each
method's transcript-to-cell assignment is scored against the identical map.

| method | cells | VSI median | VSI mean | frac cells VSI<0.5 | tx assigned |
|---|---|---|---|---|---|
| **tracer** | 55,967 | **0.6470** | 0.5875 | 0.3188 | 0.917 |
| celladmix | 54,664 | 0.6457 | 0.5818 | 0.3272 | 0.925 |
| baseline_10x | 58,449 | 0.6268 | 0.5649 | 0.3537 | 0.999 |
| proseg | 56,930 | 0.6229 | 0.5622 | 0.3535 | 0.871 |
| baysor | 17,691 | 0.5474 | 0.5121 | 0.4390 | 0.991 |
| **split** | -- | **not measurable** | | | |

Read against the do-nothing baseline (0.6268):

- **tracer** +0.0202 and **celladmix** +0.0189: both leave cells sitting in
  more vertically coherent signal than the vendor segmentation did.
- **proseg** -0.0039: indistinguishable from doing nothing on this metric.
- **baysor** -0.0794, with 43.9% of its cells below 0.5 against 35.4% for the
  baseline. Baysor emits 17,691 cells where the others emit ~56,000, so its
  cells are far larger, and larger cells span more vertically incoherent
  tissue. The metric is measuring a real consequence of that choice.

### The finding that matters

**SPLIT cannot be scored on this metric at all.** It emits purified per-cell
count profiles, not transcript-level assignments (`not_transcript_level_
reason.txt` in its output directory records this), so there is nothing to lay
against the integrity map.

That produces the sharpest statement this audit can make:

> On the four metrics that are scored against the same reference SPLIT
> optimises against, SPLIT ranks first. On the one metric that is independent
> of that reference, SPLIT cannot be evaluated, and TRACER ranks first.

This is not evidence that SPLIT is worse. It is evidence that **the original
ranking rested entirely on metrics that could not distinguish segmentation
quality from agreement with SPLIT's own objective**, and that the only
available independent check does not cover SPLIT. A benchmark that reports
SPLIT as the winner without both of those facts alongside is overstating what
it measured.

### The panel builder drops every bootstrapped pair

The blocker is now pinned to a specific defect, and it is worth reporting
upstream because it silently produces a file that looks like a panel.

The `all_pairs` rebuild's own diagnostics say what it computed:

| classification | pairs |
|---|---|
| `pos` (confident positive association) | 11,585 |
| `neg` (confident negative) | 11,204 |
| `unsettled` | 10,418 |
| `dead_zone` | 4,974 |
| **subtotal: successfully bootstrapped** | **38,181** |
| `low_evidence` | 5,185 |
| `indeterminate` | 1,174 |
| `neg_one` | 11 |
| **subtotal: not bootstrappable** | **6,370** |

and `n_output_pairs: 6,370`.

The written file is *exactly* the pairs the bootstrap could not evaluate. All
38,181 that it did evaluate -- including 22,789 with a confident sign -- are
discarded at write time. The reference was fine and the statistics were
computed; only the last step is broken.

For contrast, TRACER's own panel carries every class (`pos` 15,688, `neg`
4,004, `unsettled` 11,110, `dead_zone` 2,873, `low_evidence` 9,079,
`indeterminate` 2,080). It was built on **2026-05-27**; the builder script was
last modified **2026-06-01**. The panel in use predates the regression, which
is why nothing downstream has noticed it.

Practical consequences:

1. **Any cPMI panel built with the current script is unusable**, and fails
   quietly -- it emits a well-formed CSV with the right schema and a plausible
   row count, containing only the pairs that carry no information.
2. The 2x2 stays blocked on this, not on the data.
3. Anything else in this project that rebuilds a panel is affected the same
   way. That includes the `kidney_visiumhd` reference panel if it is ever
   regenerated.

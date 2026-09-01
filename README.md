<p align="center">
  <img src="asset/segbench_logo.png" alt="SegBench" width="360">
</p>

<h1 align="center">SegBench</h1>
<p align="center"><b>A reproducible benchmark for cell segmentation and profile refinement in spatial transcriptomics</b></p>

<!-- badges: start -->
<p align="center">

[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![Methods](https://img.shields.io/badge/methods-8-green)](docs/methods.md)
[![Platforms](https://img.shields.io/badge/platforms-Xenium%20%7C%20VisiumHD-orange)](docs/methods.md)
[![Self-test](https://img.shields.io/badge/self--test-segbench%20selftest-informational)](tests/)

</p>
<!-- badges: end -->

SegBench runs cell segmentation and profile-refinement methods on the same
spatial transcriptomics data, under one input contract and one metric suite, so
their outputs can actually be compared.

---

## Supported methods

**Imaging / molecule-resolved (Xenium, MERFISH, CosMx)**

| Method | Kind | Entity |
|---|---|---|
| [Baysor](https://github.com/kharchenkolab/Baysor) | segmentation | cell |
| [ProSeg](https://github.com/dcjones/proseg) | segmentation | cell |
| [Segger](https://github.com/dpeerlab/segger) ¹ | segmentation | cell |
| [SPLIT](https://github.com/kharchenkolab/SPLIT) | refinement | cell |
| [CellAdmix](https://github.com/kharchenkolab/celladmix) | refinement | cell |
| [TRACER (Seg)](https://github.com/imlong4real/TRACER) | refinement | cell |

¹ Registered and wired, but not yet completed on this cluster — see
[docs/comparison_audit.md](docs/comparison_audit.md).

**Sequencing / array-based (Visium HD)**

| Method | Kind | Entity |
|---|---|---|
| [Bin2Cell](https://github.com/Teichlab/bin2cell) | cell-calling | cell |
| [TRACER (Seg)](https://github.com/imlong4real/TRACER) | refinement | cell |
| [TRACER (No-seg)](https://github.com/imlong4real/TRACER) | refinement | bin |

---

## Install

SegBench runs from a clone; no `pip install` required.

```bash
git clone https://github.com/imlong4real/segmentation_benchmark_pipeline.git
cd segmentation_benchmark_pipeline
cp configs/environments.local.example.sh configs/environments.local.sh
# edit configs/environments.local.sh to point at your interpreters and data
source configs/environments.local.sh
```

Each method lives in its own environment, declared in
[`configs/environments.yaml`](configs/environments.yaml). Build them with:

```bash
scripts/setup_environments.sh
```

Check what resolves on your machine, and what the suite would do:

```bash
segbench doctor
segbench selftest
```

---

## Run

Xenium — one method, then the whole imaging suite:

```bash
segbench run tracer --dataset nsclc_xenium --outdir runs/nsclc/tracer
```

```bash
segbench suite imaging --dataset nsclc_xenium --outdir runs/nsclc
```

Visium HD:

```bash
segbench suite sequencing --dataset kidney_visiumhd --outdir runs/kidney
```

Score the runs and build the comparison table:

```bash
segbench evaluate --outdir runs/nsclc && segbench report --outdir runs/nsclc
```

---

## Outputs and metrics

Every method writes the same layout, whatever it produced internally:

```
<outdir>/<method>/
├── outputs/                  standardized cell-by-gene h5ad + transcripts parquet
├── benchmark_stats.json      runtime, peak RSS, entity counts
├── config_receipt.json       resolved args, input digests, versions, git commit
└── run.log                   one file per run
```

Scored per method:

| Group | Metrics |
|---|---|
| Cost | runtime (method only), peak RSS |
| Output | entities produced, mean transcripts per profile, fraction of transcripts assigned |
| Reference concordance | RCTD entropy, RCTD max weight, Kendall tau vs scRNA, marker specificity log2FC |
| Reference-free | ovrlpy vertical signal integrity |
| Coherence | cPMI conflict |

Reference-concordance metrics are scored against an scRNA reference; where a
method also *optimises* against that reference, the metric measures agreement
with it rather than segmentation quality. See the audit before ranking anything
on them.

TRACER's cPMI panels are consumed, not built, by SegBench. The upstream panel
builder currently has a defect that produces an unusable panel without erroring
— regenerate a panel only after checking
[docs/comparison_audit.md](docs/comparison_audit.md).

---

## Documentation

| | |
|---|---|
| [docs/methods.md](docs/methods.md) | per-method inputs, invocation, and output contract |
| [docs/adding_a_method.md](docs/adding_a_method.md) | adding a method to the registry |
| [docs/comparison_audit.md](docs/comparison_audit.md) | **benchmark caveats, selection effects, reference circularity, known-bad tooling** |
| [docs/audit.md](docs/audit.md) | repository and pipeline hygiene audit |
| [reproducibility/](reproducibility/) | container recipes, environment locks, checksums |

**Read [docs/comparison_audit.md](docs/comparison_audit.md) before drawing
conclusions from any comparison table.** It documents which metrics are
independent of the evaluation reference and which are not, how many cells each
method is actually scored on, and the current state of upstream tooling.

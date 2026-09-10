#!/usr/bin/env python3
"""Build the frozen lung cPMI panel from the training split only.

This is a thin memory-safe adapter around TRACER's exact
``feat/lung-panel-builders`` implementation.  The split h5ad contains only the
Xenium panel but carries true whole-transcriptome depth in
``obs['_full_library_total_counts']``; all balancing, gene-admission, presence,
depth-binning and cPMI calculations are delegated to the pinned builder code.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp


BUILDER_COMMIT = "9a6b078b9127c1f87e591ace5f935e00cde4d7da"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_builder(path: Path):
    spec = importlib.util.spec_from_file_location("tracer_lung_panel_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load builder: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-h5ad", type=Path, required=True)
    parser.add_argument("--builder-script", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--celltype-col", default="Cell_Cluster_level1")
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--min-det-cells", type=int, default=25)
    parser.add_argument("--n-depth-bins", type=int, default=25)
    parser.add_argument("--prefix", default="lung_")
    args = parser.parse_args()

    builder = load_builder(args.builder_script)
    conflict_module = load_builder(
        args.builder_script.parent.parent / "src" / "tracer" / "conflict_reference.py")
    build_depth_corrected_reference = conflict_module.build_depth_corrected_reference

    a = ad.read_h5ad(args.train_h5ad)
    if "_full_library_total_counts" not in a.obs:
        raise SystemExit("training split lacks _full_library_total_counts")
    if args.celltype_col not in a.obs:
        raise SystemExit(f"training split lacks {args.celltype_col}")
    counts = a.layers["counts"] if "counts" in a.layers else a.X
    counts = counts.tocsr() if sp.issparse(counts) else sp.csr_matrix(counts)
    genes = a.var_names.astype(str).to_numpy()
    depth = a.obs["_full_library_total_counts"].to_numpy(dtype=float)
    celltypes = a.obs[args.celltype_col].astype(str).to_numpy()

    draw = builder.balanced_draw(celltypes, strategy="rep", seed=args.seed)
    unique_draw = np.unique(draw)
    transformed = builder.cp10k_log1p(counts, depth)
    detected = np.asarray((transformed[unique_draw] >= 1).sum(0)).ravel()
    keep = detected >= args.min_det_cells
    kept_genes = genes[keep]
    result = build_depth_corrected_reference(
        counts=transformed[draw][:, keep],
        genes=np.asarray(kept_genes, dtype=object),
        depth=depth[draw],
        min_count=1,
        min_det_cells=1,
        n_depth_bins=args.n_depth_bins,
        depth_metric="total_counts",
    )
    args.outdir.mkdir(parents=True, exist_ok=True)
    builder.write_panel(
        result.edges, args.outdir, f"{args.prefix}xgt1_cpmi_balanced_rep",
        promote="cPMI")
    panel = args.outdir / f"{args.prefix}xgt1_cpmi_balanced_rep.csv.gz"
    edges = result.edges
    stats = {
        "schema_version": "1.0",
        "builder_commit": BUILDER_COMMIT,
        "builder_script": args.builder_script.name,
        "adapter": "build_lung_panel_from_split.py",
        "parameters": {
            "strategy": "rep", "presence_arm": "xgt1",
            "estimator_promoted_to_PMI": "cPMI", "seed": args.seed,
            "min_det_cells_on_unique_draw": args.min_det_cells,
            "n_depth_bins": args.n_depth_bins,
            "depth_metric": "full-library total_counts",
        },
        "training_reference": {
            "name": args.train_h5ad.name, "sha256": sha256(args.train_h5ad),
            "n_cells": int(a.n_obs), "n_panel_genes": int(a.n_vars),
        },
        "balanced_draw": {
            "n_rows": int(len(draw)), "n_unique_cells": int(len(unique_draw)),
        },
        "panel": {
            "name": panel.name, "sha256": sha256(panel),
            "n_edges": int(len(edges)), "n_genes": int(len(set(edges.gene_i) | set(edges.gene_j))),
            "pmi_median": float(np.nanmedian(edges.cPMI)),
            "pmi_positive_fraction": float(np.mean(edges.cPMI > 0.2)),
            "pmi_negative_fraction": float(np.mean(edges.cPMI < -0.2)),
            "observed_codetection_median": float(np.nanmedian(edges.O)),
        },
    }
    (args.outdir / "lung_cpmi_builder_receipt.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n")
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

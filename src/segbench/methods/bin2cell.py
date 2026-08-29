#!/usr/bin/env python3
"""Bin2Cell runner — Visium HD 2um bins -> cells.

Bin2Cell (https://github.com/Teichlab/bin2cell) turns Visium HD's regular 2um
bin grid into biologically meaningful cells. It segments the paired H&E (and
optionally a gene-expression density image) with StarDist, expands the labels,
and groups bins that fall inside one label into a single cell.

This is the **sequencing-modality** counterpart to the imaging wrappers: the
input is a binned AnnData + image rather than a molecule table, and the natural
"entity" is a bin before cell calling and a cell after it.

PIPELINE (stages are timed individually)
========================================
    load_inputs     read the 2um-bin AnnData (spaceranger `binned_outputs`)
    convert_inputs  destripe + write the scaled H&E / GEX images
    run_method      StarDist segmentation (H&E, and optionally GEX) + label
                    expansion + bin-to-cell grouping
    convert_outputs cell-by-gene h5ad + a bin->cell assignment table
    validate_schema check the emitted contract
    write_outputs   provenance + benchmark_stats.json

OUTPUT
======
Because Visium HD has no individual transcripts, the transcript contract is
expressed at **bin** level: one row per 2um bin, with ``cell_id`` naming the
cell the bin was assigned to (or ``UNASSIGNED``). ``x``/``y`` are the bin's
spatial coordinates and ``feature_name`` is left as ``__bin__`` because a bin
carries a whole expression vector, not one gene. The assigned/unassigned
statistics therefore read as *bins* assigned to cells, which is the directly
comparable quantity to "transcripts assigned" on the imaging side.

EXAMPLE
=======
    segbench run bin2cell \\
      --input-h5ad  dataset/visium_hd/square_002um/filtered_feature_bc_matrix.h5 \\
      --source-image dataset/visium_hd/Visium_HD_He.tif \\
      --spaceranger-dir dataset/visium_hd/square_002um \\
      --outdir benchmark_output/visium_hd/bin2cell \\
      --sample-name VisiumHD_demo --mpp 0.5 --prob-thresh 0.01
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .. import REPO_ROOT as _REPO_ROOT
from .. import common as rc
from .. import stats as stx
from . import _base

METHOD = "bin2cell"
#: Placeholder gene label: a Visium HD bin holds a full expression vector, so
#: there is no single feature_name to report per row.
BIN_FEATURE = "__bin__"


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _base.add_common_args(p, method=METHOD)
    p.add_argument("--input-h5ad", type=Path, default=None,
                   help="2um-bin matrix: filtered_feature_bc_matrix.h5 or a .h5ad.")
    p.add_argument("--spaceranger-dir", type=Path, default=None,
                   help="spaceranger square_002um dir (for spatial/ positions).")
    p.add_argument("--source-image", type=Path, default=None,
                   help="Full-resolution H&E image (tif).")
    p.add_argument("--labels-npz", type=Path, default=None,
                   help="Precomputed label image (.npz). Skips StarDist entirely — "
                        "use this when StarDist is unavailable.")
    # --- bin2cell parameters (names mirror the bin2cell API) ---
    p.add_argument("--mpp", type=float, default=0.5,
                   help="Microns per pixel for the scaled image.")
    p.add_argument("--prob-thresh", type=float, default=0.01,
                   help="StarDist probability threshold (H&E).")
    p.add_argument("--nms-thresh", type=float, default=0.5)
    p.add_argument("--stardist-model", default="2D_versatile_he",
                   help="StarDist pretrained model name for H&E.")
    p.add_argument("--prob-thresh-gex", type=float, default=0.05,
                   help="StarDist probability threshold for the GEX image.")
    p.add_argument("--stardist-model-gex", default="2D_versatile_fluo")
    p.add_argument("--use-gex", action="store_true",
                   help="Also segment a gene-expression density image and "
                        "combine it with the H&E labels (bin2cell 'salvage').")
    p.add_argument("--expand-microns", type=float, default=2.0,
                   help="Label expansion distance in microns.")
    p.add_argument("--min-counts", type=int, default=1,
                   help="Drop bins below this total count before segmentation.")
    p.add_argument("--no-destripe", action="store_true",
                   help="Skip bin2cell.destripe() (row/column bias correction).")
    return p


def _load_bins(args, *, log):
    """Read the 2um-bin AnnData via bin2cell's reader when possible."""
    import anndata as ad
    path = _base.require_input(args, "input_h5ad", "--input-h5ad")
    if path.suffix == ".h5ad":
        adata = ad.read_h5ad(path)
    else:
        try:
            import bin2cell as b2c
            adata = b2c.read_visium(
                str(args.spaceranger_dir or path.parent),
                source_image_path=str(args.source_image) if args.source_image else None,
            )
        except Exception as exc:
            log.warning("bin2cell.read_visium failed (%s); falling back to "
                        "scanpy.read_10x_h5.", exc)
            import scanpy as sc
            adata = sc.read_10x_h5(str(path))
    adata.var_names_make_unique()
    log.info("Loaded %d bins x %d genes", adata.n_obs, adata.n_vars)
    return adata


def _bin_assignment_table(adata, *, cell_col: str, log) -> pd.DataFrame:
    """One row per bin: coordinates + the cell it was assigned to."""
    obs = adata.obs
    if "spatial" in adata.obsm:
        xy = np.asarray(adata.obsm["spatial"], dtype="float32")
        x, y = xy[:, 0], xy[:, 1]
    else:  # fall back to array coordinates when the image was never attached
        x = obs.get("array_col", pd.Series(np.arange(adata.n_obs))).to_numpy("float32")
        y = obs.get("array_row", pd.Series(np.arange(adata.n_obs))).to_numpy("float32")
        log.warning("No obsm['spatial']; using array_row/array_col as coordinates.")

    cid = obs[cell_col] if cell_col in obs.columns else pd.Series(
        ["UNASSIGNED"] * adata.n_obs, index=obs.index)
    df = pd.DataFrame({
        "x": x, "y": y,
        "feature_name": BIN_FEATURE,
        "cell_id": cid.astype(str).to_numpy(),
        "transcript_id": obs.index.astype(str).to_numpy(),   # bin barcode
        "n_counts_bin": np.asarray(
            adata.X.sum(axis=1)).ravel().astype("float32"),
    })
    # bin2cell writes 0 / NaN for bins that landed outside every label.
    return rc.standardize_transcripts(
        df, method="Bin2Cell", unassigned_extra=("0", "0.0", "nan", "-1"), log=log)


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    _base.resolve_config(args, method=METHOD)

    outputs_dir = args.outdir / "outputs"
    bins_path = outputs_dir / f"{METHOD}_bin_assignments.parquet"
    rc.prepare_outdir(args.outdir, bins_path, args.overwrite)
    log = rc.setup_logging(args.outdir, "segbench.bin2cell")
    log.info("=== bin2cell === sample=%s seed=%d", args.sample_name, args.seed)
    np.random.seed(args.seed)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    stage_dir = args.outdir / "work"
    stage_dir.mkdir(parents=True, exist_ok=True)
    timer = rc.StageTimer(log)

    if args.dry_run:
        log.info("[dry-run] input_h5ad=%s source_image=%s labels_npz=%s",
                 args.input_h5ad, args.source_image, args.labels_npz)
        log.info("[dry-run] would write %s", bins_path)
        return 0

    try:
        import bin2cell as b2c
    except ImportError as exc:
        raise SystemExit(
            "bin2cell is not importable "
            f"({exc}). Install it with `pip install bin2cell` (it also needs "
            "stardist + tensorflow unless you pass --labels-npz).") from exc
    b2c_version = getattr(b2c, "__version__", "unknown")
    log.info("bin2cell version: %s", b2c_version)

    # --- load_inputs --------------------------------------------------------
    with timer.time("load_inputs"):
        adata = _load_bins(args, log=log)
        n_bins_input = int(adata.n_obs)
        if args.min_counts and args.min_counts > 0:
            import scanpy as sc
            sc.pp.filter_cells(adata, min_counts=args.min_counts)
            log.info("Kept %d/%d bins with >= %d counts",
                     adata.n_obs, n_bins_input, args.min_counts)

    # --- convert_inputs -----------------------------------------------------
    he_png = stage_dir / "he.tiff"
    with timer.time("convert_inputs"):
        if not args.no_destripe:
            b2c.destripe(adata)
            log.info("Applied bin2cell.destripe().")
        if args.source_image and args.labels_npz is None:
            b2c.scaled_he_image(adata, mpp=args.mpp, save_path=str(he_png))
            log.info("Wrote scaled H&E image: %s", he_png)

    # --- run_method ---------------------------------------------------------
    labels_key = "labels_he"
    with timer.time("run_method"):
        if args.labels_npz is not None:
            log.info("Using precomputed labels: %s (StarDist skipped)", args.labels_npz)
            b2c.insert_labels(adata, labels_npz_path=str(args.labels_npz),
                              basis="spatial", spatial_key="spatial",
                              mpp=args.mpp, labels_key=labels_key)
        else:
            if not args.source_image:
                raise SystemExit(
                    "bin2cell needs either --source-image (to segment with "
                    "StarDist) or --labels-npz (a precomputed label image).")
            labels_npz = stage_dir / "he.npz"
            b2c.stardist(image_path=str(he_png), labels_npz_path=str(labels_npz),
                         stardist_model=args.stardist_model,
                         prob_thresh=args.prob_thresh, nms_thresh=args.nms_thresh)
            b2c.insert_labels(adata, labels_npz_path=str(labels_npz),
                              basis="spatial", spatial_key="spatial",
                              mpp=args.mpp, labels_key=labels_key)
            log.info("StarDist H&E segmentation complete.")

        b2c.expand_labels(adata, labels_key=labels_key,
                          expanded_labels_key="labels_he_expanded",
                          max_bin_distance=None,
                          algorithm="volume_ratio",
                          expand_microns=args.expand_microns)
        final_labels = "labels_he_expanded"

        if args.use_gex:
            gex_png, gex_npz = stage_dir / "gex.tiff", stage_dir / "gex.npz"
            b2c.grid_image(adata, "n_counts_adjusted", mpp=args.mpp,
                           sigma=5, save_path=str(gex_png))
            b2c.stardist(image_path=str(gex_png), labels_npz_path=str(gex_npz),
                         stardist_model=args.stardist_model_gex,
                         prob_thresh=args.prob_thresh_gex, nms_thresh=args.nms_thresh)
            b2c.insert_labels(adata, labels_npz_path=str(gex_npz), basis="array",
                              mpp=args.mpp, labels_key="labels_gex")
            b2c.salvage_secondary_labels(adata, primary_label=final_labels,
                                         secondary_label="labels_gex",
                                         labels_key="labels_joint")
            final_labels = "labels_joint"
            log.info("Combined H&E + GEX labels into %s", final_labels)

        cdata = b2c.bin_to_cell(adata, labels_key=final_labels,
                                spatial_keys=["spatial"])
        log.info("bin_to_cell: %d cells from %d bins", cdata.n_obs, adata.n_obs)

    # --- convert_outputs ----------------------------------------------------
    cells_h5ad = outputs_dir / f"{METHOD}_cell_by_gene.h5ad"
    bins_h5ad = outputs_dir / f"{METHOD}_bins_annotated.h5ad"
    with timer.time("convert_outputs"):
        std = _bin_assignment_table(adata, cell_col=final_labels, log=log)
        std.to_parquet(bins_path, index=False, compression="snappy")
        log.info("Wrote bin assignments: %s", bins_path)
        cdata.write_h5ad(cells_h5ad)
        log.info("Wrote cell-by-gene: %s (%d cells x %d genes)",
                 cells_h5ad, cdata.n_obs, cdata.n_vars)
        try:
            adata.write_h5ad(bins_h5ad)
        except Exception as e:  # secondary artifact only
            log.warning("Could not write annotated bins h5ad (%s).", e)
            bins_h5ad = None

    # --- validate_schema ----------------------------------------------------
    with timer.time("validate_schema"):
        rc.validate_schema(
            std, method=METHOD, out_path=bins_path,
            in_path=str(args.input_h5ad),
            report_path=args.outdir / "schema_validation_report.json", log=log,
            extra={"entity_kind": "bin", "n_cells": int(cdata.n_obs),
                   "bin2cell_version": b2c_version})

    # --- write_outputs ------------------------------------------------------
    outs = [str(bins_path), str(cells_h5ad)] + ([str(bins_h5ad)] if bins_h5ad else [])
    with timer.time("write_outputs"):
        rc.write_provenance(
            outdir=args.outdir, method=METHOD, sample_name=args.sample_name,
            args=args, timer=timer, repo_root=_REPO_ROOT,
            inputs={"input_h5ad": str(args.input_h5ad or ""),
                    "source_image": str(args.source_image or ""),
                    "labels_npz": str(args.labels_npz or "")},
            outputs=outs, method_version=b2c_version, runner_kind="python", log=log,
            extra_config={"mpp": args.mpp, "prob_thresh": args.prob_thresh,
                          "expand_microns": args.expand_microns,
                          "use_gex": args.use_gex,
                          "stardist_model": args.stardist_model},
            summary_extra_lines=[
                "Bin2Cell is a sequencing-modality cell-calling method: rows in "
                "the standardized table are 2um BINS, not transcripts.",
                f"{cdata.n_obs} cells called from {adata.n_obs} bins."])

    acct = stx.transcript_accounting(std, n_input=n_bins_input)
    stx.write_benchmark_stats(
        outdir=args.outdir, method=METHOD, modality="sequencing",
        sample_name=args.sample_name, timer=timer, dataset=args.dataset,
        transcripts=acct,
        entities=stx.entity_accounting(
            std, entity_kind="cell", n_entities=int(cdata.n_obs),
            n_genes=int(cdata.n_vars)),
        qc={"n_bins_input": n_bins_input,
            "n_bins_after_filter": int(adata.n_obs),
            "n_bins_assigned_to_cell": acct["n_assigned"],
            "median_bins_per_cell": float(np.median(
                cdata.obs["bin_count"])) if "bin_count" in cdata.obs else None,
            "mpp": float(args.mpp),
            "expand_microns": float(args.expand_microns),
            "used_stardist": args.labels_npz is None,
            "used_gex_salvage": bool(args.use_gex)},
        method_version=b2c_version, outputs=outs,
        notes="Rows are 2um bins; 'assigned' means the bin fell inside a called cell.")

    log.info("DONE. Total wall: %.1fs | %d cells", timer.total_seconds, cdata.n_obs)
    return 0


if __name__ == "__main__":
    sys.exit(main())

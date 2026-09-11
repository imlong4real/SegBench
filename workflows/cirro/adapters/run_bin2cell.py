#!/usr/bin/env python3
"""Cirro adapter for explicit Bin2Cell StarDist prediction orchestration.

Bin2Cell 0.3.4 defaults to 4096-pixel tiles.  Its tiled StarDist call requires
each input dimension to be at least that large, which is not true for the
deterministic smoke ROI.  The smoke path therefore uses StarDist's ordinary
small-image predictor; full runs retain Bin2Cell's tiled predictor.  The model,
normalization, thresholds, and downstream Bin2Cell operations are unchanged.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _install_sparse_safe_insert_labels(b2c, *, chunk_size: int = 1_000_000):
    """Install Bin2Cell 0.3.4 label insertion compatible with recent SciPy.

    SciPy 1.17 returns a sparse object for paired CSR point indexing, whereas
    Bin2Cell 0.3.4 assumes ``np.asarray(...)`` produces a numeric vector.  The
    latter instead becomes an object array and fails when assigned to
    ``adata.obs``.  Keep the upstream coordinate/mpp semantics, but explicitly
    densify only the selected labels and do so in bounded chunks.
    """
    def sparse_safe_insert_labels(
        adata,
        labels_npz_path,
        basis="spatial",
        spatial_key="spatial",
        mpp=None,
        labels_key="labels",
    ):
        import numpy as np
        import scipy.sparse

        labels_path = Path(labels_npz_path).resolve()
        labels_sparse = scipy.sparse.load_npz(labels_path)
        adata.uns.setdefault("bin2cell", {}).setdefault(
            "labels_npz_paths", {})[labels_key] = str(labels_path)

        # ``scaled_he_image`` crops the morphology image by default and stores
        # coordinates relative to that crop.  Bin2Cell's tutorial expects that
        # generated key to be supplied to ``insert_labels``.  The SegBench core
        # wrapper historically passed the full-image ``spatial`` key instead,
        # which produces all-zero labels for a cropped smoke ROI.  Prefer the
        # deterministic default crop key, or the sole recorded crop key.
        effective_spatial_key = spatial_key
        if basis == "spatial" and spatial_key == "spatial":
            default_crop_key = "spatial_cropped_150_buffer"
            cropped_keys = sorted(
                str(key) for key in adata.obsm
                if str(key).startswith("spatial_cropped_"))
            if default_crop_key in cropped_keys:
                effective_spatial_key = default_crop_key
            elif len(cropped_keys) == 1:
                effective_spatial_key = cropped_keys[0]
            elif len(cropped_keys) > 1:
                raise RuntimeError(
                    "Multiple cropped spatial coordinate keys are present; "
                    "cannot choose one safely: " + ", ".join(cropped_keys))
        adata.uns["bin2cell"].setdefault(
            "label_coordinate_keys", {})[labels_key] = effective_spatial_key

        coords = np.asarray(
            b2c.get_mpp_coords(
                adata, basis=basis, spatial_key=effective_spatial_key, mpp=mpp),
            dtype=np.int64,
        )
        mask = (
            (coords[:, 0] >= 0)
            & (coords[:, 0] < labels_sparse.shape[0])
            & (coords[:, 1] >= 0)
            & (coords[:, 1] < labels_sparse.shape[1])
        )
        valid = np.flatnonzero(mask)
        labels = np.zeros(adata.n_obs, dtype=labels_sparse.dtype)
        for start in range(0, valid.size, chunk_size):
            take = valid[start:start + chunk_size]
            selected = labels_sparse[coords[take, 0], coords[take, 1]]
            if scipy.sparse.issparse(selected):
                selected = selected.toarray()
            labels[take] = np.asarray(selected).reshape(-1)
        adata.obs[labels_key] = labels

    b2c.insert_labels = sparse_safe_insert_labels


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction-mode", choices=("direct", "tiled"), required=True)
    parser.add_argument("--block-size", type=int, required=True)
    parser.add_argument("method_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.prediction_mode == "tiled" and args.block_size <= 384:
        parser.error("--block-size must exceed min_overlap + 2*context (384)")

    method_args = list(args.method_args)
    if method_args and method_args[0] == "--":
        method_args.pop(0)

    import bin2cell as b2c
    _install_sparse_safe_insert_labels(b2c)

    if args.prediction_mode == "direct":
        def direct_stardist(
            image_path,
            labels_npz_path,
            stardist_model="2D_versatile_he",
            block_size=4096,
            min_overlap=128,
            context=128,
            model=None,
            model_axes=None,
            **kwargs,
        ):
            import numpy as np
            import scipy.sparse
            from stardist.models import StarDist2D

            if model is not None and model_axes is None:
                raise ValueError("model_axes is required with a custom StarDist model")
            img = b2c.load_image(
                image_path,
                gray=(stardist_model == "2D_versatile_fluo"),
                dtype=np.float16,
            )
            img = b2c.normalize(img)
            if model is None:
                model = StarDist2D.from_pretrained(stardist_model)
                model_axes = "YXC" if stardist_model == "2D_versatile_he" else "YX"
            labels, _ = model.predict_instances(img, axes=model_axes, **kwargs)
            labels_sparse = scipy.sparse.csr_matrix(labels)
            scipy.sparse.save_npz(labels_npz_path, labels_sparse)
            print(f"Found {len(np.unique(labels_sparse.data))} objects")

        b2c.stardist = direct_stardist
    else:
        upstream_stardist = b2c.stardist

        def stardist_with_explicit_block_size(*positional, **kwargs):
            kwargs.setdefault("block_size", args.block_size)
            return upstream_stardist(*positional, **kwargs)

        b2c.stardist = stardist_with_explicit_block_size

    from segbench.methods.bin2cell import main as bin2cell_main

    return int(bin2cell_main(method_args) or 0)


if __name__ == "__main__":
    sys.exit(main())

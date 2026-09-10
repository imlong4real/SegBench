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

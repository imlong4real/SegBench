#!/usr/bin/env python3
"""Regression test for recent-SciPy sparse labels in the Cirro adapter."""
from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import scipy.sparse

REPO = Path(__file__).resolve().parents[1]
ADAPTER = REPO / "workflows" / "cirro" / "adapters" / "run_bin2cell.py"


def _load_adapter():
    spec = importlib.util.spec_from_file_location("run_bin2cell", ADAPTER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_sparse_point_labels_are_numeric_and_chunked() -> None:
    module = _load_adapter()
    coords = np.asarray([[0, 1], [1, 2], [2, 0], [9, 9]], dtype=np.int64)
    observed = {}

    def get_mpp_coords(*args, **kwargs):
        observed["spatial_key"] = kwargs["spatial_key"]
        return coords

    fake_b2c = SimpleNamespace(get_mpp_coords=get_mpp_coords)
    module._install_sparse_safe_insert_labels(fake_b2c, chunk_size=2)

    adata = SimpleNamespace(
        n_obs=len(coords), obs=pd.DataFrame(index=["a", "b", "c", "outside"]),
        uns={}, obsm={"spatial": coords, "spatial_cropped_150_buffer": coords},
    )
    labels = scipy.sparse.csr_matrix(
        np.asarray([[0, 11, 0], [0, 0, 22], [33, 0, 0]], dtype=np.int32))
    with tempfile.TemporaryDirectory() as td:
        labels_path = Path(td) / "labels.npz"
        scipy.sparse.save_npz(labels_path, labels)
        fake_b2c.insert_labels(
            adata, str(labels_path), basis="spatial", spatial_key="spatial", mpp=0.5)

        assert adata.obs["labels"].tolist() == [11, 22, 33, 0]
        assert adata.obs["labels"].dtype == np.dtype("int32")
        assert adata.uns["bin2cell"]["labels_npz_paths"]["labels"] == str(
            labels_path.resolve())
        assert observed["spatial_key"] == "spatial_cropped_150_buffer"
        assert adata.uns["bin2cell"]["label_coordinate_keys"]["labels"] == (
            "spatial_cropped_150_buffer")

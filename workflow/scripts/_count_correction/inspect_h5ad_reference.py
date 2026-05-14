#!/usr/bin/env python3
"""Inspect an h5ad scRNA reference without importing anndata.

The base Python environment used for this benchmark can have anndata/scanpy
import conflicts, so this script reads AnnData's HDF5 layout directly with
h5py. It prints and writes a concise summary for choosing RCTD/cellAdmix
reference inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


CELLTYPE_HINTS = (
    "cell_type",
    "celltype",
    "cell_type_major",
    "annotation",
    "annotations",
    "predicted_cell_type",
    "lineage",
    "broad_cell_type",
    "subtype",
    "level1",
    "level2",
    "level3",
    "harmonised_level4",
)
SAMPLE_HINTS = ("sample", "patient", "donor", "orig.ident", "batch")


def _decode(x: Any) -> Any:
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    if isinstance(x, np.bytes_):
        return x.astype(str)
    return x


def read_h5ad_column(group: h5py.Group, name: str) -> np.ndarray:
    obj = group[name]
    if isinstance(obj, h5py.Dataset):
        return np.array([_decode(x) for x in obj[:]])
    if isinstance(obj, h5py.Group) and "categories" in obj and "codes" in obj:
        categories = np.array([_decode(x) for x in obj["categories"][:]], dtype=object)
        codes = obj["codes"][:]
        out = np.empty(len(codes), dtype=object)
        out[:] = None
        mask = codes >= 0
        out[mask] = categories[codes[mask]]
        return out
    raise TypeError(f"Unsupported h5ad column encoding for {name}")


def read_h5ad_names(f: h5py.File, axis: str) -> list[str]:
    g = f[axis]
    key = "_index" if "_index" in g else next(iter(g.keys()))
    return [str(_decode(x)) for x in read_h5ad_column(g, key)]


def extract_gene_panel_names(path: Path) -> set[str]:
    data = json.loads(path.read_text())
    targets = data.get("payload", {}).get("targets", [])
    names = set()
    for target in targets:
        tdata = target.get("type", {}).get("data", {})
        if "name" in tdata:
            names.add(str(tdata["name"]))
    return names


def matrix_integerish(f: h5py.File, key: str) -> tuple[bool, str]:
    data = f[f"{key}/data"]
    sample = data[: min(len(data), 200_000)]
    if sample.size == 0:
        return True, "empty matrix"
    frac = float(np.mean(np.abs(sample - np.round(sample)) > 1e-6))
    return frac < 1e-4, f"sample_non_integer_fraction={frac:.6g}, sample_min={sample.min()}, sample_max={sample.max()}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--xenium-gene-panel", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    h5ad = Path(args.h5ad)
    panel = Path(args.xenium_gene_panel)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    with h5py.File(h5ad, "r") as f:
        shape = tuple(int(x) for x in f["X"].attrs["shape"])
        obs_cols = list(f["obs"].keys())
        var_cols = list(f["var"].keys())
        obs_names = read_h5ad_names(f, "obs")
        gene_names = read_h5ad_column(f["var"], "feature_name") if "feature_name" in f["var"] else read_h5ad_names(f, "var")
        gene_names = [str(x) for x in gene_names]
        panel_genes = extract_gene_panel_names(panel)
        overlap = sorted(set(gene_names) & panel_genes)

        candidate_celltypes = [
            c for c in obs_cols if any(h in c.lower() for h in CELLTYPE_HINTS)
        ]
        candidate_samples = [
            c for c in obs_cols if any(h in c.lower() for h in SAMPLE_HINTS)
        ]

        lines.append(f"h5ad: {h5ad}")
        lines.append(f"shape_obs_cells_by_var_genes: {shape}")
        lines.append(f"n_obs_names: {len(obs_names)}")
        lines.append(f"n_var_genes: {len(gene_names)}")
        lines.append(f"var_columns: {', '.join(var_cols)}")
        lines.append(f"obs_columns: {', '.join(obs_cols)}")
        lines.append(f"candidate_cell_type_columns: {', '.join(candidate_celltypes) or 'NONE'}")
        lines.append(f"candidate_sample_patient_columns: {', '.join(candidate_samples) or 'NONE'}")
        lines.append(f"gene_id_format_first_10: {', '.join(gene_names[:10])}")
        lines.append(f"xenium_panel_gene_count: {len(panel_genes)}")
        lines.append(f"overlap_with_xenium_gene_panel: {len(overlap)}")
        lines.append(f"overlap_genes: {', '.join(overlap)}")

        integerish, detail = matrix_integerish(f, "X")
        lines.append(f"X_counts_assessment: {'counts_like' if integerish else 'not_integer_counts'} ({detail})")
        if "raw" in f:
            lines.append("raw_counts_exist: true")
        else:
            lines.append("raw_counts_exist: false")
        layer_names = list(f.get("layers", {}).keys()) if "layers" in f else []
        lines.append(f"layers: {', '.join(layer_names) or 'NONE'}")
        for layer in layer_names:
            integerish, detail = matrix_integerish(f, f"layers/{layer}")
            lines.append(f"layer_{layer}_counts_assessment: {'counts_like' if integerish else 'not_integer_counts'} ({detail})")

        lines.append("top_20_obs_columns_unique_counts:")
        for col in obs_cols[:20]:
            vals = read_h5ad_column(f["obs"], col)
            lines.append(f"  {col}: {len(set(map(str, vals)))} unique")

        lines.append("candidate_cell_type_unique_counts:")
        for col in candidate_celltypes:
            vals = read_h5ad_column(f["obs"], col)
            levels = sorted(set(map(str, vals)))
            preview = ", ".join(levels[:20])
            lines.append(f"  {col}: {len(levels)} unique [{preview}]")

    text = "\n".join(lines) + "\n"
    out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

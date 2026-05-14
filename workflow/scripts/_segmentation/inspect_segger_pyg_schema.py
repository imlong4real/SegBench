"""
Inspect PyG dataset .pt files saved by Segger's create_dataset_fast.py.

Reports which files have / lack edge_label (and edge_label_index) per split,
and writes a CSV + JSON summary.

Usage:
    python3 inspect_segger_pyg_schema.py \
        --dataset-dir results/segger_TSU20/preprocessed_data \
        --outdir results/segger_TSU20/debug
"""

import argparse
import csv
import glob
import json
import os
import sys

import torch


def inspect_file(fpath):
    """Return a dict describing edge_label presence for every edge store."""
    try:
        data = torch.load(fpath, map_location="cpu", weights_only=False)
    except Exception as exc:
        return {"path": fpath, "load_error": str(exc)}

    record = {"path": fpath, "type": type(data).__name__}

    if hasattr(data, "edge_types"):
        # HeteroData
        record["graph_type"] = "HeteroData"
        record["node_types"] = [str(t) for t in data.node_types]
        record["edge_types"] = [str(t) for t in data.edge_types]
        record["edge_stores"] = []
        for etype in data.edge_types:
            estore = data[etype]
            keys = list(estore.keys())
            has_el = "edge_label" in keys
            has_eli = "edge_label_index" in keys
            eli_shape = list(estore.edge_label_index.shape) if has_eli else None
            el_shape = list(estore.edge_label.shape) if has_el else None
            record["edge_stores"].append(
                {
                    "edge_type": str(etype),
                    "keys": keys,
                    "has_edge_label": has_el,
                    "has_edge_label_index": has_eli,
                    "edge_label_shape": el_shape,
                    "edge_label_index_shape": eli_shape,
                }
            )
    else:
        # Homogeneous Data
        record["graph_type"] = "HomoData"
        keys = list(data.keys())
        has_el = "edge_label" in keys
        has_eli = "edge_label_index" in keys
        record["keys"] = keys
        record["has_edge_label"] = has_el
        record["has_edge_label_index"] = has_eli
        record["edge_label_shape"] = (
            list(data.edge_label.shape) if has_el else None
        )
        record["edge_label_index_shape"] = (
            list(data.edge_label_index.shape) if has_eli else None
        )

    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    splits = ["train_tiles", "val_tiles", "test_tiles"]
    all_records = []
    summary = {s: {"total": 0, "has_edge_label": 0, "missing_edge_label": 0} for s in splits}

    for split in splits:
        pt_dir = os.path.join(args.dataset_dir, split, "processed")
        pt_files = sorted(glob.glob(os.path.join(pt_dir, "*.pt")))
        print(f"\n{split}: {len(pt_files)} files in {pt_dir}")
        summary[split]["total"] = len(pt_files)

        for fpath in pt_files:
            rec = inspect_file(fpath)
            rec["split"] = split

            if "load_error" in rec:
                print(f"  ERROR loading {os.path.basename(fpath)}: {rec['load_error']}")
                all_records.append(rec)
                continue

            has_el = False
            if rec["graph_type"] == "HeteroData":
                has_el = any(
                    es["has_edge_label"] for es in rec["edge_stores"]
                )
            else:
                has_el = rec.get("has_edge_label", False)

            rec["any_edge_label"] = has_el
            if has_el:
                summary[split]["has_edge_label"] += 1
            else:
                summary[split]["missing_edge_label"] += 1

            all_records.append(rec)

    # Print summary table
    print("\n=== Summary ===")
    print(f"{'Split':<15} {'Total':>6} {'HasEdgeLabel':>14} {'Missing':>10}")
    for split in splits:
        s = summary[split]
        print(
            f"{split:<15} {s['total']:>6} {s['has_edge_label']:>14} {s['missing_edge_label']:>10}"
        )

    # Write JSON
    json_path = os.path.join(args.outdir, "pyg_schema_summary.json")
    with open(json_path, "w") as fh:
        json.dump({"summary": summary, "records": all_records}, fh, indent=2)
    print(f"\nWrote: {json_path}")

    # Write flat CSV (one row per file)
    csv_path = os.path.join(args.outdir, "pyg_schema_summary.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["split", "filename", "graph_type", "any_edge_label", "load_error"])
        for rec in all_records:
            writer.writerow(
                [
                    rec.get("split", ""),
                    os.path.basename(rec["path"]),
                    rec.get("graph_type", ""),
                    rec.get("any_edge_label", ""),
                    rec.get("load_error", ""),
                ]
            )
    print(f"Wrote: {csv_path}")


if __name__ == "__main__":
    sys.exit(main())

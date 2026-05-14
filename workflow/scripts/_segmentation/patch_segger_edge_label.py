"""
Patch PyG .pt files that are missing edge_label (and/or edge_label_index).

Segger's predict_fast.py collates all three dataset splits (train/val/test)
via PyG DataLoaders. If any tile in a batch lacks edge_label while others in
the same batch have it, PyG's collate raises KeyError: 'edge_label'.

This script adds zero-valued edge_label (and edge_label_index if also missing)
to tiles that lack them, making the split schema consistent.

Writes patched tiles back to the same location. Backs up originals first to
<processed_dir>_backup/ (skipped if backup already exists).

Usage:
    python3 patch_segger_edge_label.py \
        --dataset-dir results/segger_TSU20/preprocessed_data \
        --splits test_tiles [train_tiles val_tiles] \
        --report-csv results/segger_TSU20/debug/edge_label_patch_report.csv
"""

import argparse
import csv
import glob
import os
import shutil
import sys

import torch


def _infer_edge_label_size(estore, key_eli="edge_label_index", key_ei="edge_index"):
    """Return the number of supervision edges from edge_label_index, or 0."""
    if key_eli in estore:
        eli = estore[key_eli]
        if eli.dim() == 2:
            return eli.shape[1]
        elif eli.dim() == 1:
            return eli.shape[0]
    return 0


def patch_data(data):
    """
    Mutate data in-place: add edge_label / edge_label_index to stores that
    lack them. Returns True if any patch was applied.
    """
    patched = False

    if hasattr(data, "edge_types"):
        # HeteroData
        for etype in data.edge_types:
            estore = data[etype]
            has_el = "edge_label" in estore
            has_eli = "edge_label_index" in estore

            if has_el and has_eli:
                continue  # already complete

            n = _infer_edge_label_size(estore)

            if not has_eli:
                estore.edge_label_index = torch.zeros((2, 0), dtype=torch.long)

            if not has_el:
                estore.edge_label = torch.zeros(n, dtype=torch.float)

            patched = True
    else:
        # Homogeneous Data — check as attribute dict
        has_el = "edge_label" in data
        has_eli = "edge_label_index" in data

        if not (has_el and has_eli):
            n = _infer_edge_label_size(data)
            if not has_eli:
                data.edge_label_index = torch.zeros((2, 0), dtype=torch.long)
            if not has_el:
                data.edge_label = torch.zeros(n, dtype=torch.float)
            patched = True

    return patched


def process_split(split_processed_dir, dry_run=False):
    """Back up and patch all .pt files in split_processed_dir."""
    pt_files = sorted(glob.glob(os.path.join(split_processed_dir, "*.pt")))
    if not pt_files:
        print(f"  No .pt files in {split_processed_dir}")
        return []

    # Backup directory: sibling of processed, named processed_backup
    backup_dir = split_processed_dir + "_backup"
    if not os.path.isdir(backup_dir):
        print(f"  Backing up {len(pt_files)} files → {backup_dir}")
        if not dry_run:
            shutil.copytree(split_processed_dir, backup_dir)
    else:
        print(f"  Backup already exists at {backup_dir} — skipping copy")

    rows = []
    n_patched = 0
    for fpath in pt_files:
        fname = os.path.basename(fpath)
        try:
            data = torch.load(fpath, map_location="cpu", weights_only=False)
        except Exception as exc:
            print(f"  ERROR loading {fname}: {exc}")
            rows.append({"file": fpath, "status": "load_error", "detail": str(exc)})
            continue

        if dry_run:
            rows.append({"file": fpath, "status": "dry_run", "detail": ""})
            continue

        patched = patch_data(data)
        if patched:
            torch.save(data, fpath)
            n_patched += 1
            rows.append({"file": fpath, "status": "patched", "detail": "edge_label added"})
        else:
            rows.append({"file": fpath, "status": "ok", "detail": ""})

    print(f"  Patched {n_patched}/{len(pt_files)} files")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True,
                    help="Path to Segger preprocessed_data directory")
    ap.add_argument("--splits", nargs="+",
                    default=["test_tiles"],
                    help="Which split subdirs to patch (default: test_tiles)")
    ap.add_argument("--report-csv", default=None,
                    help="Where to write a CSV patch report")
    ap.add_argument("--dry-run", action="store_true",
                    help="Inspect only; do not write any files")
    args = ap.parse_args()

    all_rows = []
    for split in args.splits:
        processed_dir = os.path.join(args.dataset_dir, split, "processed")
        print(f"\n=== {split} ({processed_dir}) ===")
        if not os.path.isdir(processed_dir):
            print(f"  Directory not found — skipping")
            continue
        rows = process_split(processed_dir, dry_run=args.dry_run)
        for r in rows:
            r["split"] = split
        all_rows.extend(rows)

    if args.report_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.report_csv)), exist_ok=True)
        with open(args.report_csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["split", "file", "status", "detail"])
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nWrote patch report: {args.report_csv}")

    patched = sum(1 for r in all_rows if r.get("status") == "patched")
    errors = sum(1 for r in all_rows if r.get("status") == "load_error")
    print(f"\nTotal patched: {patched}  Errors: {errors}")


if __name__ == "__main__":
    sys.exit(main())

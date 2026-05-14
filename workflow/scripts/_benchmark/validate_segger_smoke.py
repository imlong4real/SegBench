"""
Validate the Segger smoke-test outputs and write a JSON report.

Usage:
    python workflow/scripts/_benchmark/validate_segger_smoke.py

Output:
    results/segger_smoke/segger_smoke_validation.json
"""

import json
import os
import sys
from datetime import datetime

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))

CHECKS = {
    "input_subset_parquet": "results/segger_smoke/input/transcripts_100k.parquet",
    "input_summary_json": "results/segger_smoke/input/input_summary.json",
    "xenium_bundle_transcripts": "results/segger_smoke/input/xenium_bundle/transcripts.parquet",
    "xenium_bundle_nuclei": "results/segger_smoke/input/xenium_bundle/nucleus_boundaries.parquet",
    "preprocessed_data_dir": "results/segger_smoke/preprocessed_data",
    "trained_model_dir": "results/segger_smoke/trained_model",
    "segger_output_dir": "results/segger_smoke/segger_output",
    "container_sif": "containers/python_cuda.sif",
}

EXPECTED_MAX_ROWS = 100_000


def check_file_nonempty(path):
    return os.path.isfile(path) and os.path.getsize(path) > 0


def check_dir_nonempty(path):
    return os.path.isdir(path) and bool(os.listdir(path))


def check_parquet_rows(path):
    try:
        import pyarrow.parquet as pq
        t = pq.read_table(path)
        return len(t)
    except Exception:
        try:
            import pandas as pd
            df = pd.read_parquet(path)
            return len(df)
        except Exception:
            return -1


def main():
    results = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "repo_dir": REPO_DIR,
        "checks": {},
        "status": "UNKNOWN",
        "blockers": [],
    }

    for name, rel_path in CHECKS.items():
        abs_path = os.path.join(REPO_DIR, rel_path)
        if rel_path.endswith("/") or not os.path.splitext(rel_path)[1]:
            ok = check_dir_nonempty(abs_path)
        else:
            ok = check_file_nonempty(abs_path)
        results["checks"][name] = {"path": rel_path, "ok": ok}
        if not ok:
            results["blockers"].append(f"MISSING or EMPTY: {rel_path}")

    # Row count check on input subset
    subset_path = os.path.join(REPO_DIR, CHECKS["input_subset_parquet"])
    if os.path.exists(subset_path):
        n_rows = check_parquet_rows(subset_path)
        results["checks"]["input_subset_parquet"]["rows"] = n_rows
        if n_rows > EXPECTED_MAX_ROWS:
            results["blockers"].append(
                f"input subset has {n_rows} rows > {EXPECTED_MAX_ROWS}"
            )
        elif n_rows <= 0:
            results["blockers"].append("input subset row count unreadable or 0")

    # Check segger output for non-stub files
    out_dir = os.path.join(REPO_DIR, CHECKS["segger_output_dir"])
    if os.path.isdir(out_dir):
        output_files = []
        for root, _, files in os.walk(out_dir):
            for f in files:
                fp = os.path.join(root, f)
                size = os.path.getsize(fp)
                output_files.append({"file": os.path.relpath(fp, REPO_DIR), "size": size})
        results["segger_output_files"] = output_files
        if not output_files:
            results["blockers"].append("segger_output_dir is empty — no output files")
        elif all(f["size"] == 0 for f in output_files):
            results["blockers"].append("all segger output files are empty (stubs)")

    if results["blockers"]:
        results["status"] = "BLOCKED"
    else:
        results["status"] = "PASS"

    out_path = os.path.join(REPO_DIR, "results/segger_smoke/segger_smoke_validation.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)

    print(f"Status: {results['status']}")
    if results["blockers"]:
        for b in results["blockers"]:
            print(f"  BLOCKER: {b}")
    print(f"Report: {out_path}")

    return 0 if results["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

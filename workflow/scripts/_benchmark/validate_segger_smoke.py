"""
Validate Segger TSU-20 full-run outputs and write a JSON report.

Usage:
    python workflow/scripts/_benchmark/validate_segger_smoke.py

Output:
    results/segger_TSU20/segger_validation.json
"""

import json
import os
import sys
from datetime import datetime

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))

CHECKS = {
    "xenium_bundle_transcripts": "results/segger_smoke/input/xenium_bundle/TSU-20/transcripts.parquet",
    "xenium_bundle_nuclei": "results/segger_smoke/input/xenium_bundle/TSU-20/nucleus_boundaries.parquet",
    "preprocessed_data_dir": "results/segger_TSU20/preprocessed_data",
    "trained_model_dir": "results/segger_TSU20/trained_model",
    "segger_output_dir": "results/segger_TSU20/segger_output",
    "container_sif": "containers/python_cuda.sif",
}


def check_file_nonempty(path):
    return os.path.isfile(path) and os.path.getsize(path) > 0


def check_dir_nonempty(path):
    return os.path.isdir(path) and bool(os.listdir(path))


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

    out_path = os.path.join(REPO_DIR, "results/segger_TSU20/segger_validation.json")
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

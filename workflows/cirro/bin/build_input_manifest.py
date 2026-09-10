#!/usr/bin/env python3
"""Write a deterministic SHA-256 inventory for a Cirro upload directory."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        processed = 0
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
            processed += len(block)
            if processed % (512 << 20) < len(block):
                print(f"hashing {path.name}: {processed / (1 << 30):.1f} GiB", file=sys.stderr,
                      flush=True)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve(); output = args.output.resolve()
    files = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.resolve() != output):
        print(f"hashing {path.relative_to(root).as_posix()}", file=sys.stderr, flush=True)
        files.append({
            "relative_path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    manifest = {
        "schema_version": "1.0",
        "dataset": args.dataset,
        "source_root": str(args.source_root.resolve()) if args.source_root else None,
        "source_files_preserved": True,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_files": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "files": files,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

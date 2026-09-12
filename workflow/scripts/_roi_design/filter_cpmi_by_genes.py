#!/usr/bin/env python3
"""Restrict a prebuilt cPMI panel to a platform's gene set.

The cervical panel is *used as delivered*, never rebuilt or retuned per ROI —
the only operation applied is dropping edges whose genes the platform does not
measure.  Both endpoints must be present for an edge to survive.

The filter runs in awk: the panel is a multi-GB gzip and this process lives
inside a hard 5 GB user cgroup, so the edge table is never materialised.

Writes the filtered panel plus a receipt recording source path/SHA-256, the
gene overlap, and the effective edge count.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import sys
from pathlib import Path

AWK = r'''
BEGIN { FS=","; OFS="," }
NR==FNR { if ($0 != "") gene[$0]=1; ngene++; next }
FNR==1  { print; header=1; next }
{
  total++
  if (($1 in gene) && ($2 in gene)) { kept++; seen[$1]=1; seen[$2]=1; print }
}
END {
  n=0; for (g in seen) n++
  printf("STATS total=%d kept=%d genes_in_panel=%d genes_in_filter=%d\n",
         total, kept, n, ngene) > "/dev/stderr"
}
'''


def sha256(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        fd = fh.fileno()
        n = 0
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
            n += 1
            if n % 64 == 0:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", required=True, help="Source cPMI csv.gz")
    ap.add_argument("--genes", required=True, help="One gene per line")
    ap.add_argument("--out", required=True, help="Filtered cPMI csv.gz")
    ap.add_argument("--receipt", required=True)
    ap.add_argument("--platform", default="")
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    panel, genes, out = Path(a.panel), Path(a.genes), Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    gene_list = [g.strip() for g in genes.read_text().splitlines() if g.strip()]

    # The awk program goes in a real file: `-f /dev/stdin` would collide with
    # the `-` data argument, and awk would try to parse the panel as program
    # text.
    with tempfile.NamedTemporaryFile("w", suffix=".awk", delete=False) as fh:
        fh.write(AWK)
        prog = fh.name
    try:
        cmd = (f"set -o pipefail; zcat {panel!s} | "
               f"awk -f {prog} {genes!s} - | gzip -c > {out!s}")
        proc = subprocess.run(["bash", "-c", cmd], text=True, capture_output=True)
    finally:
        os.unlink(prog)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"filter failed with exit {proc.returncode}")

    stats = {}
    for line in proc.stderr.splitlines():
        if line.startswith("STATS "):
            for kv in line[6:].split():
                k, v = kv.split("=")
                stats[k] = int(v)
    if not stats:
        raise SystemExit(f"awk produced no stats; stderr was:\n{proc.stderr}")

    # Header is written but counted separately, so `total` is the edge count.
    receipt = {
        "schema_version": "roi-design-1.0",
        "operation": "gene-overlap filter only; the panel is NOT rebuilt, "
                     "retuned, or re-estimated per ROI",
        "label": a.label,
        "platform": a.platform,
        "source_panel": str(panel.resolve()),
        "source_panel_sha256": sha256(panel),
        "source_panel_bytes": panel.stat().st_size,
        "source_edges": stats["total"],
        "platform_gene_list": str(genes.resolve()),
        "platform_gene_list_sha256": sha256(genes),
        "platform_genes": len(gene_list),
        "filtered_panel": str(out.resolve()),
        "filtered_panel_sha256": sha256(out),
        "filtered_panel_bytes": out.stat().st_size,
        "effective_edges": stats["kept"],
        "genes_with_at_least_one_edge": stats["genes_in_panel"],
        "edge_retention": stats["kept"] / max(stats["total"], 1),
        "gene_overlap_fraction_of_platform":
            stats["genes_in_panel"] / max(len(gene_list), 1),
    }
    Path(a.receipt).write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

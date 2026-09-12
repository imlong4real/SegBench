#!/usr/bin/env python3
"""Build the MERFISH benchmark input carrying the ORIGINAL segmentation mask.

The delivered `input_transcripts_um.parquet` carries Baysor's *output* in its
`cell_id` column.  Using that as the original mask would hand every refinement
method (SPLIT, cellAdmix, TRACER Seg) a method result as its input, and would
make Baysor its own baseline.

The original mask is the 3-D membrane (Cellpose) prior segmentation that the
Petukhov Baysor run consumed.  It lives upstream in `baysor_segmentation.csv`
as `membrane_prior_3d`, joined to the transcripts through `mol_id` ->
`transcript_id`.

Verified before use (enforced below, not assumed):
  * the join is 1:1 and total;
  * `membrane_prior_3d` is a label field, not a mask flag;
  * it agrees with the `prior_segmentation` column Baysor actually consumed;
  * Baysor's own `cell` output is a different, denser field, and is what the
    delivered `cell_id` column actually holds.

Emits the SegBench transcript contract with `cell_id` = membrane prior label
(0 -> UNASSIGNED).  Baysor's output is preserved as `baysor_reference_cell_id`
for provenance only; no method reads it.

Everything streams row group by row group: this runs inside a hard 5 GB user
cgroup shared with unrelated processes, where a whole-table read of even this
small file peaks at 546 MB and is OOM-killed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

UNASSIGNED = "UNASSIGNED"
PRIOR_COL = "membrane_prior_3d"
CONST = {"platform": "MERFISH", "sample": "merfish_mouse_ileum",
         "roi_id": "merfish_mouse_ileum__whole", "original_segmentation": PRIOR_COL}


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


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


def read_segmentation(path: Path, columns: list[str]) -> dict[str, np.ndarray]:
    parts: list[dict[str, np.ndarray]] = []
    for chunk in pd.read_csv(path, usecols=columns,
                             dtype={c: np.int64 for c in columns},
                             chunksize=100_000):
        parts.append({c: chunk[c].to_numpy(np.int64) for c in columns})
        del chunk
    out = {c: np.concatenate([p[c] for p in parts]) for c in columns}
    del parts
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transcripts", required=True)
    ap.add_argument("--baysor-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    a = ap.parse_args()

    tx_path, seg_path, out_path = Path(a.transcripts), Path(a.baysor_csv), Path(a.out)

    seg_cols = ["mol_id", PRIOR_COL, "prior_segmentation", "cell",
                "membrane_prior_z85", "dapi_prior_z85"]
    seg = read_segmentation(seg_path, seg_cols)
    seg_ids = seg["mol_id"]
    prior, prior_seg, baysor = seg[PRIOR_COL], seg["prior_segmentation"], seg["cell"]

    pf = pq.ParquetFile(str(tx_path))
    n_rows = pf.metadata.num_rows

    # ---- verification of the prior's identity ------------------------------
    checks: dict[str, object] = {}
    if n_rows != seg_ids.size:
        raise SystemExit(f"row-count mismatch: transcripts {n_rows} vs "
                         f"segmentation {seg_ids.size}")
    if np.unique(seg_ids).size != seg_ids.size:
        raise SystemExit("mol_id is not unique in the segmentation table")
    n_prior_labels = int(np.unique(prior[prior != 0]).size)
    if n_prior_labels < 100:
        raise SystemExit(f"{PRIOR_COL} has only {n_prior_labels} labels; it is not "
                         f"a segmentation label field")
    agree = float((prior == prior_seg).mean())
    if agree < 0.95:
        raise SystemExit(f"{PRIOR_COL} agrees with prior_segmentation only "
                         f"{agree:.3f} of the time; the prior identity is unclear")
    checks["join"] = "1:1 and total on transcript_id == mol_id"
    checks["n_molecules"] = int(n_rows)
    checks[f"{PRIOR_COL}_labels"] = n_prior_labels
    checks[f"{PRIOR_COL}_assigned_fraction"] = float((prior != 0).mean())
    checks["agreement_with_prior_segmentation"] = agree
    checks["baysor_output_labels"] = int(np.unique(baysor[baysor != 0]).size)
    checks["baysor_output_assigned_fraction"] = float((baysor != 0).mean())
    checks["membrane_prior_z85_assigned_fraction"] = float((seg["membrane_prior_z85"] != 0).mean())
    checks["dapi_prior_z85_assigned_fraction"] = float((seg["dapi_prior_z85"] != 0).mean())

    order = np.argsort(seg_ids, kind="stable")
    sorted_ids = seg_ids[order]

    # ---- stream ------------------------------------------------------------
    writer: pq.ParquetWriter | None = None
    seen = 0
    n_assigned = 0
    genes: set[str] = set()
    labels_used: set[int] = set()
    match_baysor = 0
    match_prior = 0
    cache_fd = os.open(str(tx_path), os.O_RDONLY)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for rg in range(pf.metadata.num_row_groups):
        tbl = pf.read_row_group(rg, use_threads=False)
        ids = tbl.column("transcript_id").to_numpy().astype(np.int64)
        loc = np.searchsorted(sorted_ids, ids)
        if loc.max(initial=0) >= sorted_ids.size or not np.array_equal(sorted_ids[loc], ids):
            raise SystemExit("unmapped molecules after the join")
        pos = order[loc]
        prior_tx = prior[pos]
        baysor_tx = baysor[pos]

        delivered = pd.to_numeric(pd.Series(tbl.column("cell_id").to_pylist()),
                                  errors="coerce").fillna(0).to_numpy(np.int64)
        match_baysor += int((delivered == baysor_tx).sum())
        match_prior += int((delivered == prior_tx).sum())

        new_cell = np.where(
            prior_tx == 0, UNASSIGNED,
            np.char.add("mp_", prior_tx.astype(np.int64).astype(str)))
        n_assigned += int((prior_tx != 0).sum())
        labels_used.update(np.unique(prior_tx[prior_tx != 0]).tolist())
        genes.update(set(tbl.column("feature_name").to_pylist()))

        out = tbl.append_column("baysor_reference_cell_id",
                                tbl.column("cell_id").cast(pa.string()))
        out = out.set_column(out.schema.get_field_index("cell_id"), "cell_id",
                             pa.array(new_cell, type=pa.string()))
        for name, value in CONST.items():
            col = pa.array([value] * out.num_rows, type=pa.string())
            idx = out.schema.get_field_index(name)
            out = (out.set_column(idx, name, col) if idx >= 0
                   else out.append_column(name, col))
        if writer is None:
            writer = pq.ParquetWriter(str(out_path), out.schema, compression="zstd")
        writer.write_table(out)
        seen += out.num_rows
        del tbl, out, ids, pos, prior_tx, baysor_tx, delivered, new_cell
        os.posix_fadvise(cache_fd, 0, 0, os.POSIX_FADV_DONTNEED)
        pa.default_memory_pool().release_unused()
    if writer is not None:
        writer.close()
    os.close(cache_fd)
    if seen != n_rows:
        raise SystemExit(f"wrote {seen} rows but the source has {n_rows}")

    checks["delivered_cell_id_equals_baysor_output"] = match_baysor / n_rows
    checks["delivered_cell_id_equals_membrane_prior_3d"] = match_prior / n_rows
    # The delivered cell_id is Baysor's output.  It is not a perfect match
    # because Baysor's ~2% noise molecules (is_noise=True, cell=0) carry a
    # different sentinel downstream, so the floor sits below that 2% gap.
    if checks["delivered_cell_id_equals_baysor_output"] < 0.95:
        raise SystemExit(
            f"delivered cell_id matches Baysor's output only "
            f"{checks['delivered_cell_id_equals_baysor_output']:.4f} of the time; "
            f"the provenance assumed by this script does not hold")
    # ...and it must NOT already be the prior, or there would be nothing to fix.
    if checks["delivered_cell_id_equals_membrane_prior_3d"] > 0.05:
        raise SystemExit(
            f"delivered cell_id already matches {PRIOR_COL} "
            f"{checks['delivered_cell_id_equals_membrane_prior_3d']:.4f} of the "
            f"time; re-check which field is the original mask")

    report = {
        "schema_version": "roi-design-1.0",
        "purpose": "MERFISH original segmentation mask = membrane_prior_3d, the "
                   "Cellpose membrane prior consumed by the Petukhov Baysor run",
        "transcripts_source": str(tx_path.resolve()),
        "transcripts_sha256": sha256(tx_path),
        "segmentation_source": str(seg_path.resolve()),
        "segmentation_sha256": sha256(seg_path),
        "output": str(out_path.resolve()),
        "output_sha256": sha256(out_path),
        "output_rows": int(seen),
        "n_genes": len(genes),
        "n_original_entities": len(labels_used),
        "frac_assigned": n_assigned / n_rows,
        "unassigned_sentinel": UNASSIGNED,
        "baysor_output_retained_as": "baysor_reference_cell_id (provenance only; "
                                     "never read by a method)",
        "peak_rss_mb": round(rss_mb(), 1),
        "verification": checks,
    }
    Path(a.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

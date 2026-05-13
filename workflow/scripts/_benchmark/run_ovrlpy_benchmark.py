#!/usr/bin/env python3
"""Run ovrlpy on a standardized transcripts table.

This is a thin wrapper that reuses ovrlpy directly: the standardized
transcripts.parquet already carries the columns ovrlpy needs (after
renaming x_location/y_location/z_location/feature_name → x/y/z/gene).

Outputs:
    {out_dir}/signal_integrity.parquet
    {out_dir}/signal_strength.parquet
    {out_dir}/transcript_info.parquet
    {out_dir}/pseudocell_summary.parquet  (best-effort)
    {out_dir}/method_info.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from _io_contract import MethodInfo, load_standardized_transcripts  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ovrlpy on a standardized transcripts table.")
    p.add_argument("--standardized-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--cell-diameter", type=float, default=10.0)
    p.add_argument("--n-expected-celltypes", type=int, default=30)
    p.add_argument("--allow-stub", action="store_true")
    p.add_argument("--log", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.log:
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(args.log, "w", encoding="utf-8")
        sys.stdout = log_fh
        sys.stderr = log_fh

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start = dt.datetime.utcnow().isoformat() + "Z"

    stand_dir = Path(args.standardized_dir)
    transcripts_path = stand_dir / "transcripts.parquet"

    try:
        import ovrlpy  # type: ignore
    except Exception as e:
        if not args.allow_stub:
            raise SystemExit(
                f"ovrlpy could not be imported ({e}). Install ovrlpy or rerun with --allow-stub."
            )
        print(f"[ovrlpy-benchmark] ovrlpy missing ({e}); writing stub outputs.")
        for fname in ("signal_integrity.parquet", "signal_strength.parquet", "transcript_info.parquet"):
            pd.DataFrame().to_parquet(out_dir / fname)
        info = MethodInfo(
            method_name="ovrlpy",
            command=" ".join(sys.argv),
            input_files=[str(transcripts_path)],
            output_files=[str(out_dir / "signal_integrity.parquet")],
            start_time=start,
            container_or_env=os.environ.get("CONDA_DEFAULT_ENV"),
            extra={"stub": True},
        )
        info.end_time = dt.datetime.utcnow().isoformat() + "Z"
        (out_dir / "method_info.json").write_text(json.dumps(info.to_dict(), indent=2))
        return

    df = load_standardized_transcripts(transcripts_path).rename(
        columns={
            "x_location": "x",
            "y_location": "y",
            "z_location": "z",
            "feature_name": "gene",
            "cell_id_method": "cell_id",
        }
    )
    df["gene"] = df["gene"].astype("category")

    # ovrlpy.cell_integrity_from_transcripts expects an integer cell_id with
    # an int "unassigned" sentinel. Encode the standardized string cell_id
    # to int codes here and persist the lookup for downstream use.
    import numpy as np  # local to keep top-level minimal

    cid_str = df["cell_id"].astype("string")
    empty_mask = cid_str.isna() | cid_str.isin(["UNASSIGNED", "", "0"])
    codes, uniques = pd.factorize(cid_str)
    codes = codes.astype("int32")
    codes[empty_mask.to_numpy()] = -1
    df["cell_id"] = codes
    # Persist mapping so users can match VSI rows back to string cell ids.
    pd.DataFrame({"code": list(range(len(uniques))), "cell_id_method": list(uniques)}).to_parquet(
        out_dir / "cell_id_code_map.parquet"
    )

    ovrlpy_version = getattr(ovrlpy, "__version__", "unknown")
    print(f"[ovrlpy-benchmark] ovrlpy version: {ovrlpy_version}")

    # ovrlpy >= 1.0 exposes an Ovrlp class. ovrlpy < 1.0 exposed an
    # ovrlpy.run(...) helper. We support both.
    if hasattr(ovrlpy, "Ovrlp"):
        # New API.
        ovrlp = ovrlpy.Ovrlp(
            df,
            n_components=int(args.n_expected_celltypes),
            coordinate_keys=("x", "y", "z"),
            gene_key="gene",
        )
        # The Ovrlp class exposes a high-level analyse() pipeline; if it
        # is unavailable, we fall back to step-by-step methods.
        if hasattr(ovrlp, "analyse"):
            ovrlp.analyse()
        else:
            for step in ("fit_signatures", "fit_pseudocells", "fit_transcripts"):
                if hasattr(ovrlp, step):
                    getattr(ovrlp, step)()
        # ovrlpy 1.x stores artifacts on the Ovrlp instance as:
        #   integrity_map: 2D ndarray (signal-integrity grid)
        #   signal_map:    2D ndarray (signal-strength grid)
        #   pseudocells:   AnnData
        import numpy as np

        integrity_map = getattr(ovrlp, "integrity_map", None)
        signal_map = getattr(ovrlp, "signal_map", None)
        pseudocells = getattr(ovrlp, "pseudocells", None)

        if integrity_map is not None:
            pd.DataFrame(np.asarray(integrity_map)).to_parquet(
                out_dir / "signal_integrity.parquet"
            )
        else:
            pd.DataFrame().to_parquet(out_dir / "signal_integrity.parquet")

        if signal_map is not None:
            pd.DataFrame(np.asarray(signal_map)).to_parquet(
                out_dir / "signal_strength.parquet"
            )
        else:
            pd.DataFrame().to_parquet(out_dir / "signal_strength.parquet")

        if pseudocells is not None:
            try:
                ps_df = pseudocells.obs.copy()
                ps_df["x"] = pseudocells.obsm["spatial"][:, 0] if "spatial" in pseudocells.obsm else np.nan
                ps_df["y"] = pseudocells.obsm["spatial"][:, 1] if "spatial" in pseudocells.obsm else np.nan
                ps_df.to_parquet(out_dir / "pseudocell_summary.parquet")
            except Exception as e:
                print(f"[ovrlpy-benchmark] pseudocell extraction failed: {e}")
                pd.DataFrame().to_parquet(out_dir / "pseudocell_summary.parquet")

        # Per-cell VSI from the same analyse'd Ovrlp object.
        try:
            vsi = ovrlpy.cell_integrity_from_transcripts(
                ovrlp, cell_id="cell_id", unassigned=-1
            )
            vsi_df = vsi.to_pandas() if hasattr(vsi, "to_pandas") else pd.DataFrame(vsi)
            vsi_df.to_parquet(out_dir / "cell_signal_integrity.parquet")
        except Exception as e:
            print(f"[ovrlpy-benchmark] cell_integrity_from_transcripts skipped: {e}")
            pd.DataFrame().to_parquet(out_dir / "cell_signal_integrity.parquet")

        pd.DataFrame().to_parquet(out_dir / "transcript_info.parquet")
    else:
        # Old API.
        signal_integrity, signal_strength, _viz = ovrlpy.run(
            df=df,
            cell_diameter=args.cell_diameter,
            n_expected_celltypes=args.n_expected_celltypes,
        )
        pd.DataFrame(signal_integrity).to_parquet(out_dir / "signal_integrity.parquet")
        pd.DataFrame(signal_strength).to_parquet(out_dir / "signal_strength.parquet")
        cols = [c for c in ("x_pixel", "y_pixel", "n_pixel", "z_delim") if c in df.columns]
        if cols:
            df[cols].to_parquet(out_dir / "transcript_info.parquet")
        else:
            pd.DataFrame().to_parquet(out_dir / "transcript_info.parquet")

    info = MethodInfo(
        method_name="ovrlpy",
        method_version=getattr(ovrlpy, "__version__", None),
        command=" ".join(sys.argv),
        input_files=[str(transcripts_path)],
        output_files=[
            str(out_dir / "signal_integrity.parquet"),
            str(out_dir / "signal_strength.parquet"),
            str(out_dir / "transcript_info.parquet"),
        ],
        start_time=start,
        container_or_env=os.environ.get("CONDA_DEFAULT_ENV"),
        extra={
            "cell_diameter": args.cell_diameter,
            "n_expected_celltypes": args.n_expected_celltypes,
        },
    )
    info.end_time = dt.datetime.utcnow().isoformat() + "Z"
    (out_dir / "method_info.json").write_text(json.dumps(info.to_dict(), indent=2))


if __name__ == "__main__":
    main()

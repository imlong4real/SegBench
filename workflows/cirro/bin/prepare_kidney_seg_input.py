#!/usr/bin/env python3
"""Freeze the effective kidney segmented-mode transcript/bin table."""
from __future__ import annotations
import argparse, hashlib, json, shutil
from pathlib import Path
import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""): h.update(b)
    return h.hexdigest()


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--input",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True); p.add_argument("--max-rows",type=int,default=0)
    args=p.parse_args(); args.output.parent.mkdir(parents=True,exist_ok=True)
    if not args.max_rows:
        shutil.copy2(args.input,args.output); n_rows=None; selection="full"
    else:
        df=pd.read_parquet(args.input)
        n_rows=int(len(df)); center=np.array([df.x.median(),df.y.median()])
        distance=np.square(df[["x","y"]].to_numpy(dtype=float)-center).sum(axis=1)
        idx=np.argpartition(distance,args.max_rows-1)[:args.max_rows]
        df=df.iloc[np.sort(idx)].copy(); df.to_parquet(args.output,index=False,compression="snappy")
        selection=f"{len(df)} rows nearest global coordinate median"
    if n_rows is None:
        import pyarrow.parquet as pq
        n_rows=pq.ParquetFile(args.input).metadata.num_rows
    import pyarrow.parquet as pq
    effective_rows=pq.ParquetFile(args.output).metadata.num_rows
    receipt={"source_name":args.input.name,"source_sha256":sha256(args.input),
             "source_rows":int(n_rows),"selection":selection,"effective_rows":int(effective_rows),
             "effective_sha256":sha256(args.output)}
    (args.output.parent/"frozen_input_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")


if __name__=="__main__": main()

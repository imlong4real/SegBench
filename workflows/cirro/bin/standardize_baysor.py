#!/usr/bin/env python3
"""Standardize native Baysor output and attach immutable/resource provenance."""
from __future__ import annotations
import argparse, hashlib, json, logging, shutil
from pathlib import Path
import pandas as pd


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""): h.update(b)
    return h.hexdigest()


def parse_time(path: Path) -> dict:
    values={}
    if path.exists():
        for line in path.read_text(errors="replace").splitlines():
            if ":" in line:
                key,value=line.strip().split(":",1); values[key.strip()]=value.strip()
    rss=values.get("Maximum resident set size (kbytes)")
    return {"source":"GNU time -v", "peak_rss_gb":float(rss)/(1024**2) if rss else None,
            "user_cpu_seconds":float(values.get("User time (seconds)","nan")),
            "system_cpu_seconds":float(values.get("System time (seconds)","nan"))}


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--transcripts",type=Path,required=True)
    p.add_argument("--segmentation",type=Path,required=True); p.add_argument("--time-file",type=Path,required=True)
    p.add_argument("--native-dir",type=Path,required=True); p.add_argument("--outdir",type=Path,required=True)
    p.add_argument("--sample-name",required=True); p.add_argument("--seed",type=int,required=True)
    p.add_argument("--requested-cpus",type=int,required=True)
    p.add_argument("--requested-memory-gb",type=float,required=True)
    p.add_argument("--wall-seconds",type=float,default=None); args=p.parse_args()
    from segbench.methods.baysor import standardize_baysor_segmentation
    from segbench.common import build_cell_by_gene_h5ad
    source=pd.read_parquet(args.transcripts); out=args.outdir/"outputs"; native=args.outdir/"native"
    out.mkdir(parents=True,exist_ok=True); native.mkdir(parents=True,exist_ok=True)
    std=standardize_baysor_segmentation(args.segmentation,source,log=logging.getLogger("baysor"))
    stdp=out/"baysor_transcripts_standardized.parquet"; std.to_parquet(stdp,index=False,compression="snappy")
    cbg=out/"baysor_cell_by_gene.h5ad"; build_cell_by_gene_h5ad(std,out_path=cbg,log=logging.getLogger("baysor"))
    shutil.copy2(args.segmentation,native/args.segmentation.name)
    for file in args.native_dir.iterdir():
        if file.is_file() and file != args.segmentation: shutil.copy2(file,native/file.name)
    resources=parse_time(args.time_file); assigned=~std.cell_id.astype(str).eq("UNASSIGNED")
    resource={"schema_version":"1.0","method":"baysor","wall_clock_seconds":args.wall_seconds,
              "host":{"cpu_requested":args.requested_cpus,
                      "memory_requested_gb":args.requested_memory_gb,
                      "cpu_time_seconds":resources["user_cpu_seconds"]+resources["system_cpu_seconds"],
                      "average_cpu_cores_used":((resources["user_cpu_seconds"]+resources["system_cpu_seconds"])/args.wall_seconds
                                                if args.wall_seconds else None),
                      "peak_rss_gb":resources["peak_rss_gb"]},
              "gpu":{"gpu_requested":0,"model":None,"peak_vram_mb":None},
              "notes":{"peak_rss":"GNU time peak host RSS; GPU memory not applicable"}}
    (args.outdir/"resource_usage.json").write_text(json.dumps(resource,indent=2)+"\n")
    stats={"schema_version":"1.0","status":"ok","method":"baysor","method_version":"0.7.1",
           "method_commit":"109850599ea026b7d70c7cf96bc6de14740f827d","sample_name":args.sample_name,
           "entity_kind":"cell","seed":args.seed,
           "transcripts":{"n_total":int(len(std)),"n_assigned":int(assigned.sum()),
                          "n_unassigned":int((~assigned).sum()),"frac_assigned":float(assigned.mean())},
           "entities":{"n_entities":int(std.loc[assigned,"cell_id"].nunique()),
                       "n_whole_cells":int(std.loc[assigned,"cell_id"].nunique()),"n_partial_cells":0,
                       "median_transcripts_per_entity":float(std.loc[assigned,"cell_id"].value_counts().median())
                       if assigned.any() else None,
                       "median_transcripts_per_whole_cell":float(std.loc[assigned,"cell_id"].value_counts().median())
                       if assigned.any() else None},
           "runtime":{"total_seconds":args.wall_seconds,"method_seconds":args.wall_seconds},
           "memory":{"peak_rss_gb":resources["peak_rss_gb"],"source":"GNU time -v"},
           "inputs":{"transcripts":{"name":args.transcripts.name,"sha256":sha256(args.transcripts)}},
           "outputs":[str(stdp),str(cbg)]}
    (args.outdir/"benchmark_stats.json").write_text(json.dumps(stats,indent=2)+"\n")
    receipt={"method":"Baysor","version":"0.7.1","commit":"109850599ea026b7d70c7cf96bc6de14740f827d",
             "release_artifact_sha256":"f37f02695a068fd360ade7abd0f91b07c5199a3cce6e91ded550689459fbe5c0",
             "parameters":{"scale":8.0,"min_molecules_per_cell":50,"polygon_format":"none","seed":args.seed},
             "input_sha256":sha256(args.transcripts),"output_sha256":sha256(stdp)}
    (args.outdir/"config_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")


if __name__=="__main__": main()

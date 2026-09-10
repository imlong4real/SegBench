#!/usr/bin/env python3
"""Turn SegBench's wide comparison into manuscript-ready tidy tables."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import pandas as pd


ENTITY = {"n_entities","n_whole_cells","n_partial_cells","n_partial_only_cells",
          "n_whole_and_partial_cells","n_transcripts_total","n_transcripts_assigned",
          "n_transcripts_unassigned","frac_assigned","median_transcripts_per_entity",
          "median_transcripts_per_whole_cell","median_transcripts_per_partial_cell",
          "mean_transcripts_per_profile","mean_transcripts_per_whole_cell",
          "mean_transcripts_per_partial_cell"}
RESOURCE = {"runtime_total_s","runtime_method_s","peak_rss_gb","peak_rss_source"}
RCTD = {"rctd_entropy_median","rctd_max_weight_median","rctd_n_cells_scored","rctd_status"}
REFERENCE = {"kendall_tau_median","pearson_r_median","spearman_rho_median","n_celltypes_scored"}
MARKER = {"marker_logfc_median","n_marker_celltypes"}
TRACER = {"cpmi_purity","cpmi_conflict","cpmi_relative_purity","cpmi_relative_conflict"}


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""): h.update(b)
    return h.hexdigest()


def unit(metric: str) -> str:
    if metric.endswith("_s") or metric.endswith("seconds"): return "seconds"
    if metric.endswith("_gb"): return "GiB"
    if metric.startswith("n_"): return "count"
    if "frac" in metric or metric.endswith("percent"): return "fraction"
    return "unitless"


def subset(long: pd.DataFrame, metrics: set[str], path: Path) -> None:
    long[long.metric.isin(metrics)].to_csv(path, sep="\t", index=False)


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--comparison",type=Path,required=True)
    p.add_argument("--methods-root",type=Path,required=True); p.add_argument("--dataset",required=True)
    p.add_argument("--platform",required=True); p.add_argument("--replicate",default="1")
    p.add_argument("--frozen-manifest",type=Path,required=True)
    p.add_argument("--split-manifest",type=Path); p.add_argument("--input-receipt",type=Path,required=True)
    p.add_argument("--pmi",type=Path,required=True); p.add_argument("--outdir",type=Path,required=True)
    args=p.parse_args(); args.outdir.mkdir(parents=True,exist_ok=True)
    wide=pd.read_csv(args.comparison)
    idcols={"dataset","method","entity_kind","status","run_dir"}
    rows=[]
    for _,row in wide.iterrows():
        for metric,value in row.items():
            if metric in idcols or metric.endswith("_note"): continue
            note=row.get(f"{metric}_note")
            applicable=not pd.isna(value)
            rows.append({"dataset":args.dataset,"platform":args.platform,"method":row.method,
                         "replicate":args.replicate,"metric":metric,
                         "value":value if applicable else "NA","unit":unit(metric),
                         "applicable":str(bool(applicable)).lower(),
                         "provenance":("held-out evaluation reference" if metric in RCTD|REFERENCE|MARKER
                                       else (str(note) if isinstance(note,str) else "SegBench standardized contract"))})
    long=pd.DataFrame(rows)
    long.to_csv(args.outdir/"benchmark_summary.tsv",sep="\t",index=False)
    subset(long,ENTITY,args.outdir/"entity_summary.tsv")
    subset(long,RCTD,args.outdir/"rctd_metrics.tsv")
    subset(long,REFERENCE,args.outdir/"reference_correlations.tsv")
    subset(long,MARKER,args.outdir/"marker_specificity.tsv")
    subset(long,TRACER,args.outdir/"tracer_qc.tsv")

    resources=[]; resource_rows=[]
    receipts=[]
    for method_dir in sorted(p.parent for p in args.methods_root.rglob("benchmark_stats.json")):
        rp=method_dir/"resource_usage.json"; cp=method_dir/"config_receipt.json"
        if rp.exists():
            resource=json.loads(rp.read_text()); resources.append(resource)
            host=resource.get("host",{}); gpu=resource.get("gpu",{})
            resource_rows.append({
                "dataset":args.dataset,"platform":args.platform,
                "method":resource.get("method",method_dir.name),"replicate":args.replicate,
                "wall_clock_seconds":resource.get("wall_clock_seconds"),
                "cpu_requested":host.get("cpu_requested"),
                "cpu_time_seconds":host.get("cpu_time_seconds"),
                "average_cpu_cores_used":host.get("average_cpu_cores_used"),
                "host_memory_requested_gb":host.get("memory_requested_gb"),
                "peak_host_rss_gb":host.get("peak_rss_gb"),
                "gpu_requested":gpu.get("gpu_requested"),"gpu_model":gpu.get("model"),
                "peak_gpu_vram_mb":gpu.get("peak_vram_mb"),
                "gpu_utilization_mean_percent":gpu.get("utilization_mean_percent"),
                "gpu_utilization_max_percent":gpu.get("utilization_max_percent"),
                "gpu_active_runtime_seconds_sampled":gpu.get("active_runtime_seconds_sampled"),
                "host_gpu_memory_separate":True,
            })
        if cp.exists(): receipts.append({"method_directory":method_dir.name,"receipt":json.loads(cp.read_text())})
    (args.outdir/"resource_usage.json").write_text(json.dumps(resources,indent=2)+"\n")
    pd.DataFrame(resource_rows).to_csv(args.outdir/"resource_usage.tsv",sep="\t",index=False)
    frozen=json.loads(args.frozen_manifest.read_text())
    manifest={"schema_version":"1.0","dataset":args.dataset,"platform":args.platform,
              "replicate":args.replicate,"frozen_configuration":frozen,
              "input_receipt":json.loads(args.input_receipt.read_text()),
              "effective_pmi":{"name":args.pmi.name,"sha256":sha256(args.pmi)},
              "reference_split":(json.loads(args.split_manifest.read_text()) if args.split_manifest else None),
              "method_receipts":receipts,"resource_usage":resources}
    manifest_path=args.outdir/"benchmark_manifest.json"
    manifest_path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    manifest["plot_ready_outputs"]={p.name:sha256(p) for p in sorted(args.outdir.glob("*.tsv"))}
    manifest_path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")


if __name__=="__main__": main()

# HPC Environment Check

**Date:** 2026-05-14  
**Host:** dsailogin (login node — Rocky Linux 8)

## System

| Item | Value |
|------|-------|
| hostname | dsailogin |
| whoami | lyuan13 |
| pwd | /weka/home/lyuan13/segmentation_benchmark_pipeline |
| OS | Linux 4.18.0-477.27.1.el8_8.x86_64 |

## Tools (login node PATH)

| Tool | Available | Path / Version |
|------|-----------|----------------|
| git | YES | /usr/bin/git — git version 2.39.3 |
| python | YES | /usr/bin/python — Python 3.8.16 |
| conda | NO | not in default PATH; available via `module load anaconda3/2024.02-1` |
| mamba | NO | not in default PATH |
| snakemake | NO | not in default PATH |
| apptainer | **LOGIN: NO** | **COMPUTE: YES** — /usr/bin/apptainer version 1.4.4-1.el9 |
| singularity | **LOGIN: NO** | **COMPUTE: YES** — symlink to apptainer 1.4.4 |
| sbatch | YES | /opt/mprov/slurm/current/bin/sbatch — slurm 24.11.6 |
| srun | YES | /opt/mprov/slurm/current/bin/srun |
| salloc | YES | /opt/mprov/slurm/current/bin/salloc |
| nvidia-smi | NO (login node) | Available on GPU nodes |

## Module System

Available via `module load`:
- `anaconda3/2024.02-1` — Conda + Python 3.11
- No apptainer/singularity module; binary is in `/usr/bin/` on compute nodes only

## SLURM Cluster Partitions

| Partition | GPUs | Available Nodes | Notes |
|-----------|------|-----------------|-------|
| `l40s` (default) | L40S x8/node | l01-l08 (some draining) | Mixed state |
| `a100` | A100 x8/node | c001-c015 | **Mostly drained/draining** |
| `nvl` | H100 x4/node | n01-n16 | **Healthy — recommended** |
| `h100` | H100 x4/node | h01-h16 | Healthy |
| `cpu` | none | cpu001-cpu080 | CPU only |

**Recommended GPU partition for Segger:** `nvl` or `h100`

## User Account

Slurm account: `adeshpa6`

## Key Paths

| Item | Path |
|------|------|
| Repo | /weka/home/lyuan13/segmentation_benchmark_pipeline |
| Input parquet | /weka/home/lyuan13/TRACER/tutorials/lung_cancer/data/lung_cancer_df.parquet |
| Container def | reproducibility/python_cuda/python_cuda.def |
| Container target | containers/python_cuda.sif |

## Parquet Input Schema

File: `lung_cancer_df.parquet` (1,436,900 rows — Xenium format)

Columns: `x`, `y`, `z`, `feature_name`, `cell_id`, `nucleus_distance`, `transcript_id`, `fov_name`, `qv`, `overlaps_nucleus`

**Note:** Instructions referenced `/home/lyuan13/TRACER/tutorials/lung_cancer/data/filtered_df.parquet`  
Actual file path: `/weka/home/lyuan13/TRACER/tutorials/lung_cancer/data/lung_cancer_df.parquet`  
(`/home/lyuan13` and `/weka/home/lyuan13` resolve to the same location on this cluster.)

## Critical Finding

Apptainer is **not available on the login node** — all container build and run operations must be submitted as Slurm jobs (`sbatch`) or run inside interactive allocations.

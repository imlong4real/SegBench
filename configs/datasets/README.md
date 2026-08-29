# Dataset configs

One file per sample. A dataset config supplies the *inputs* (paths, sample
name, platform); a method config supplies the *parameters*. `segbench` merges
them, so the same dataset can be run through any method:

```bash
segbench run proseg --dataset tsu20_xenium --outdir benchmark_output/tsu20/proseg
segbench run baysor --dataset tsu20_xenium --outdir benchmark_output/tsu20/baysor
```

Paths may be absolute, repo-relative, or use `${SEGBENCH_DATA}` — which
expands to `$SEGBENCH_DATA` if set and `<repo>/dataset` otherwise. Keeping
machine-specific roots in that one environment variable is what lets these
files be committed without hard-coding anyone's home directory:

```bash
export SEGBENCH_DATA=/scratch/$USER/spatial_data
```

The files here are **templates** describing the expected schema. Copy one and
point it at your data; they are not expected to resolve on a fresh clone.

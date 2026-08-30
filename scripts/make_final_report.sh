#!/usr/bin/env bash
# Score every method on both datasets and emit ONE unified summary + plots.
#
#   scripts/make_final_report.sh [RUNS_ROOT] [OUT_DIR]
#
# Runs `segbench evaluate` per dataset (each dataset needs its own scRNA
# reference, so they cannot share one call), then concatenates the per-dataset
# tables into the single deliverable CSV.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
[ -f configs/environments.local.sh ] && source configs/environments.local.sh \
                                     || source configs/environments.local.example.sh

# Runs root: explicit arg > $SEGBENCH_RUNS > scratch under $USER. The last
# is only a guess -- on shared-allocation machines the scratch directory is
# named for the group, not the login, so set SEGBENCH_RUNS there.
RUNS="${1:-${SEGBENCH_RUNS:-/scratch4/$USER/segbench_runs}}"
OUT="${2:-$RUNS/final_report}"
mkdir -p "$OUT"

for DS in nsclc_xenium kidney_visiumhd; do
  ROOT="$RUNS/$DS/methods"
  [ -d "$ROOT" ] || { echo "[skip] $DS — no runs at $ROOT"; continue; }
  echo; echo "=== scoring $DS ==="
  ./bin/segbench evaluate "$ROOT" --dataset "$DS" \
      --outdir "$RUNS/$DS/summary" --rctd-cores "${RCTD_CORES:-8}" ${EVAL_EXTRA:-}
done

echo; echo "=== unified summary ==="
"${SEGBENCH_PYTHON:-python3}" - "$RUNS" "$OUT" <<'PY'
import sys
from pathlib import Path
import pandas as pd

runs, out = Path(sys.argv[1]), Path(sys.argv[2])
frames = []
for ds in ("nsclc_xenium", "kidney_visiumhd"):
    f = runs / ds / "summary" / "comparison_table.csv"
    if f.exists():
        frames.append(pd.read_csv(f))
if not frames:
    sys.exit("no per-dataset tables found")

df = pd.concat(frames, ignore_index=True)
lead = ["dataset", "method", "entity_kind", "status",
        "runtime_method_s", "runtime_total_s", "peak_rss_gb", "peak_rss_source",
        "n_entities", "n_whole_cells", "n_partial_cells",
        "mean_transcripts_per_profile",
        "mean_transcripts_per_whole_cell", "mean_transcripts_per_partial_cell",
        "n_transcripts_total", "n_transcripts_assigned", "frac_assigned",
        "rctd_entropy_median", "rctd_max_weight_median",
        "kendall_tau_median", "marker_logfc_median",
        "cpmi_purity", "cpmi_conflict"]
cols = [c for c in lead if c in df.columns] \
     + [c for c in df.columns if c not in lead and not c.endswith("_note")] \
     + [c for c in df.columns if c.endswith("_note")]
df = df[cols]
out.mkdir(parents=True, exist_ok=True)
csv = out / "segbench_summary.csv"
df.to_csv(csv, index=False)
print(f"Wrote {csv}  ({len(df)} rows across {df.dataset.nunique()} datasets)")

sys.path.insert(0, str(Path.cwd() / "src"))
try:
    from segbench import report as rp
    rp.comparison_figure(df, out / "segbench_comparison",
                         title="SegBench — all methods, both datasets")
    rp.runtime_memory_scatter(df, out / "segbench_cost")
    rp.write_markdown_summary(df, out / "segbench_summary.md",
                              dataset="all methods, both datasets")
    print(f"Wrote plots + markdown into {out}")
except Exception as exc:
    print(f"(plots skipped: {exc})")
PY
echo; echo "Done — deliverable is $OUT/segbench_summary.csv"

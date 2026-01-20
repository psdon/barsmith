#!/usr/bin/env bash
set -euo pipefail

# Lightweight harness to exercise Barsmith on the ES 30m desktop slice for both
# directions and a couple of depths. This is intended for manual CPU/memory
# profiling rather than automated tests.
#
# Usage (from barsmith/):
#   ./tmp/profile_es_pre2025.sh

DATA="${DATA:-$HOME/Desktop/es_30m_pre2025.csv}"
OUT_BASE="${OUT_BASE:-tmp/profile_runs}"

mkdir -p "$OUT_BASE"

run_case () {
  local direction="$1"
  local depth="$2"
  local label="${direction}_d${depth}"
  local out_dir="$OUT_BASE/$label"

  echo
  echo "== Running Barsmith: direction=${direction}, max-depth=${depth}, out=${out_dir} =="
  time cargo run -p barsmith_cli -- \
    comb \
    --csv "$DATA" \
    --direction "$direction" \
    --target next_bar_color_and_wicks \
    --output-dir "$out_dir" \
    --max-depth "$depth" \
    --min-sample-size 500 \
    --logic-mode and \
    --include-date-start 2024-01-01 \
    --end-year 2024 \
    --batch-size 5000 \
    --max-prefetch-combos 500 \
    --n-jobs -1 \
    --report-metrics top10
}

run_case long 1
run_case short 1
run_case long 2
run_case short 2

echo
echo "Profile runs complete. Inspect $OUT_BASE and your system profiler for CPU/memory behaviour."

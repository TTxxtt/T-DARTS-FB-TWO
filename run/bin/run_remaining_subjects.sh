#!/bin/bash
# Autonomous driver: run the frozen hierarchical method and the RF-only control
# for the remaining subjects (002/004/006/007/008/009), wait for the searches,
# then retrain both arms under the matched protocol (1500/200, val_nll
# selection, screening-only) and print the nine-subject comparison.
#
# Safe to run under nohup: it only orchestrates sbatch and polls the output
# tree, and it writes everything to sh_log/remaining_subjects.log.
set -u

RUNS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RUNS"
mkdir -p sh_log

LOG="$RUNS/sh_log/remaining_subjects.log"
exec >> "$LOG" 2>&1
echo "=== remaining-subjects driver start $(date) ==="

SUBS=(002 004 006 007 008 009)
SEEDS=(20250901 20250902 20250903)
# 0-based indices for the submit scripts: 002,004,006,007,008,009
REMAINING_0BASED="1 3 5 6 7 8"

count_hier_search() {
  local n=0
  for s in "${SUBS[@]}"; do for d in "${SEEDS[@]}"; do
    [ -f "outputs/hier/bci42a/hier_search_s${s}_seed${d}/genotype.json" ] && n=$((n + 1))
  done; done
  echo "$n"
}
count_rf_search() {
  local n=0
  for s in "${SUBS[@]}"; do for d in "${SEEDS[@]}"; do
    [ -f "outputs/rfsearch/bci42a/search_s${s}_seed${d}/genotype.json" ] && n=$((n + 1))
  done; done
  echo "$n"
}
count_hier_retrain() {
  local n=0
  for s in "${SUBS[@]}"; do for d in "${SEEDS[@]}"; do
    [ -f "outputs/hier_retrain/bci42a/train_s${s}_seed${d}_hier/final_summary.json" ] && n=$((n + 1))
  done; done
  echo "$n"
}
count_rf_retrain() {
  local n=0
  for s in "${SUBS[@]}"; do for d in "${SEEDS[@]}"; do
    [ -f "outputs/rf_retrain_nll/bci42a/train_s${s}_seed${d}_rf/final_summary.json" ] && n=$((n + 1))
  done; done
  echo "$n"
}

wait_until() {  # label, counter-function, target
  local label=$1 counter=$2 target=$3
  while [ "$("$counter")" -lt "$target" ]; do sleep 60; done
  echo "[$(date '+%m-%d %H:%M')] $label complete ($("$counter")/$target)"
}

echo "--- submitting searches for subjects ${SUBS[*]} ---"
SUBJECTS="$REMAINING_0BASED" SEEDS="${SEEDS[*]}" bash bin/submit_hier_search.sh
SUBJECTS="$REMAINING_0BASED" SEEDS="${SEEDS[*]}" bash bin/submit_rf_search.sh

echo "--- waiting for 18 hier + 18 rf search genotypes ---"
wait_until "hier searches" count_hier_search 18
wait_until "rf searches" count_rf_search 18

echo "--- submitting retrains (hier + matched RF-only, val_nll) ---"
SUBJECTS="$REMAINING_0BASED" SEEDS="${SEEDS[*]}" bash bin/submit_hier_retrain.sh
BEST_METRIC=val_nll OUTARM=rf_retrain_nll \
  SUBJECTS="$REMAINING_0BASED" SEEDS="${SEEDS[*]}" bash bin/submit_rf_retrain.sh

echo "--- waiting for 18 + 18 retrains ---"
wait_until "hier retrains" count_hier_retrain 18
wait_until "rf retrains" count_rf_retrain 18

echo "--- final paired comparison over all nine subjects ---"
python3 tools/compare_hierarchical_vs_rf_only.py \
  --anchor-search-arm hier --anchor-search-phase hier_search \
  --anchor-retrain-arm hier_retrain --anchor-leaf-arm hier \
  --rf-search-arm rfsearch --rf-search-phase search \
  --rf-retrain-arm rf_retrain_nll --rf-leaf-arm rf \
  --subjects 001 002 003 004 005 006 007 008 009 \
  --seeds "${SEEDS[*]}" \
  --json outputs/nine_subject_hier_vs_rf.json

echo "=== remaining-subjects driver DONE $(date) ==="

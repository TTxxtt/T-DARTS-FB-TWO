#!/bin/bash
# Submit Arm B: four-mechanism hierarchical search + its Stage 1/2 retrain.
#
# Grid: 9 subjects x 1 seed = 9 searches + 9 retrains = 18 runs.
#
#   seed 20190821, the same seed Arm A used.  Arm A is the vendored upstream
#   FBNAS chain, whose randSeed is hardcoded inside the frozen `ho.py`; rather
#   than monkeypatch the baseline's RNG, both arms run at that one seed.  That
#   is a deliberate trade: a single seed and an exactly paired comparison,
#   instead of three seeds and a patched baseline.
#
# Arm A is NOT submitted here and is NOT rerun.  Its nine chains already exist
# under run/outputs/fbnas/bci42a/ses2Test/, with opt_choice.csv (the searched
# architecture), cali_bn_acc.npy (the calibrated traversal) and results.csv
# (the Session-2 evaluation).  tools/extract_arm_a_session2.py reads them.
#
# Two ledgers, because the retrain depends on the search: a retrain is only
# submitted once its subject's genotype.json exists and the search's own
# final_summary.json says it finished.  `drain` therefore keeps looping while
# either stage is incomplete.
#
# Quota-aware and resumable, like the other drivers: each job goes to a
# partition with a free card, and the ledgers mean re-running submits only what
# is missing.
set -u

RUNS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$RUNS/.." && pwd)"
cd "$RUNS"

MODE="${1:-run}"

# 0-based subject index; SUB+1 is the dataset's id.
SUBJECTS="${SUBJECTS:-0 1 2 3 4 5 6 7 8}"
SEEDS="${SEEDS:-20190821}"
SEARCH_OUTARM="${SEARCH_OUTARM:-operator_armB}"
RETRAIN_OUTARM="${RETRAIN_OUTARM:-operator_armB_retrain}"
# GPUFEE06/GPUFEE08 are excluded by request.  Widest partition first.
PARTITIONS="${PARTITIONS:-GPUFEE04 GPUFEE05 GPUFEE02}"
DRAIN_SLEEP="${DRAIN_SLEEP:-120}"
QUEUE_AHEAD="${QUEUE_AHEAD:-0}"

SEARCH_LEDGER="$RUNS/sh_log/operator_armB_search_submitted.txt"
RETRAIN_LEDGER="$RUNS/sh_log/operator_armB_retrain_submitted.txt"

# The stage segment is written into the format string on purpose: it is the one
# place that guarantees an Arm B leaf can never be spelled like a frozen arm's.
search_leaf_of()  { printf "operator_armB_search_s%03d_seed%s" "$(( $1 + 1 ))" "$2"; }
retrain_leaf_of() { printf "train_s%03d_seed%s_armB" "$(( $1 + 1 ))" "$2"; }
search_dir_of()   { echo "$RUNS/outputs/$SEARCH_OUTARM/bci42a/$(search_leaf_of "$@")"; }
retrain_dir_of()  { echo "$RUNS/outputs/$RETRAIN_OUTARM/bci42a/$(retrain_leaf_of "$@")"; }

count_grid() { echo $(( $(echo "$SUBJECTS" | wc -w) * $(echo "$SEEDS" | wc -w) )); }
echo "=== grid: $(count_grid) searches + $(count_grid) retrains (subjects=$SUBJECTS, seeds='$SEEDS') ==="

if [ "$MODE" = "dry" ]; then
  # A preview that writes a file or fires a job is not a preview.
  for sub in $SUBJECTS; do for seed in $SEEDS; do
    echo "search  $(search_leaf_of "$sub" "$seed")"
    echo "retrain $(retrain_leaf_of "$sub" "$seed")  (after the search above finishes)"
  done; done
  exit 0
fi

mkdir -p sh_log/armBsearch sh_log/armBtrain "outputs/$SEARCH_OUTARM" "outputs/$RETRAIN_OUTARM" \
         "logs/$SEARCH_OUTARM" "logs/$RETRAIN_OUTARM"
[ -f "$SEARCH_LEDGER" ] || : > "$SEARCH_LEDGER"
[ -f "$RETRAIN_LEDGER" ] || : > "$RETRAIN_LEDGER"

# --- capacity, checked now and never assumed ------------------------------
free_cards() {
  local P=$1 nodes gpus total used
  read -r nodes gpus < <(sinfo -p "$P" -h -o "%D %G" 2>/dev/null | head -1)
  total=$(echo "$gpus" | sed 's/.*gpu:\([0-9]*\).*/\1/')
  if [ -z "$total" ] || ! [ "$total" -gt 0 ] 2>/dev/null; then echo 0; return; fi
  used=$(squeue -p "$P" -h -t RUNNING -o "%b" 2>/dev/null | sed 's/gres:gpu://' | paste -sd+ | bc)
  echo $(( total * nodes - ${used:-0} ))
}

build_pool() {
  POOL=()
  local P n
  for P in $PARTITIONS; do
    n=$(free_cards "$P")
    echo "  $P: $n free card(s)" >&2
    if [ "$n" -eq 0 ] && [ "$QUEUE_AHEAD" != 0 ]; then
      echo "    (queue-ahead: contributing $QUEUE_AHEAD slot(s) with no free card)" >&2
      n="$QUEUE_AHEAD"
    fi
    for _ in $(seq 1 "$n" 2>/dev/null); do POOL+=("$P"); done
  done
}

submit_one() {
  local P=$1 kind=$2 sub=$3 seed=$4 leaf out
  if [ "$kind" = "search" ]; then
    leaf=$(search_leaf_of "$sub" "$seed")
    out=$(SUB=$sub SEED=$seed OUTARM=$SEARCH_OUTARM sbatch -p "$P" bin/run_arm_b_search_job.sh 2>&1)
  else
    leaf=$(retrain_leaf_of "$sub" "$seed")
    out=$(SUB=$sub SEED=$seed OUTARM=$RETRAIN_OUTARM SEARCH_OUTARM=$SEARCH_OUTARM \
      sbatch -p "$P" bin/run_arm_b_retrain_job.sh 2>&1)
  fi
  if echo "$out" | grep -q "Submitted batch job"; then
    # Record before reporting: the ledger is what makes a re-run safe, and a
    # job accepted but not recorded would be submitted twice.
    if [ "$kind" = "search" ]; then echo "$leaf" >> "$SEARCH_LEDGER"
    else echo "$leaf" >> "$RETRAIN_LEDGER"; fi
    echo "  -> $P $kind $leaf | $out"
    return 0
  fi
  if echo "$out" | grep -q "QOSMaxSubmitJobPerUserLimit"; then
    echo "  !! per-user submit cap reached (normal QOS MaxSubmitPU)"; return 3
  fi
  echo "  !! sbatch failed on $P for $leaf: $out" >&2
  return 1
}

# A retrain may only go out once the architecture it trains is frozen on disk.
search_ready() {
  local dir; dir=$(search_dir_of "$1" "$2")
  [ -f "$dir/genotype.json" ] && [ -f "$dir/final_summary.json" ]
}

one_pass() {
  build_pool
  if [ ${#POOL[@]} -eq 0 ]; then
    echo "!! no free GPU cards on: $PARTITIONS" >&2
    return 5
  fi
  echo "  pool: ${#POOL[@]} card(s) on $(printf '%s\n' "${POOL[@]}" | sort -u | paste -sd, -)"
  local idx=0 submitted=0 skipped=0 blocked=0
  local kind sub seed leaf dir ledger

  for kind in search retrain; do
    for sub in $SUBJECTS; do for seed in $SEEDS; do
      if [ "$kind" = "search" ]; then
        leaf=$(search_leaf_of "$sub" "$seed"); dir=$(search_dir_of "$sub" "$seed")
        ledger="$SEARCH_LEDGER"
      else
        if ! search_ready "$sub" "$seed"; then
          blocked=$((blocked + 1)); continue
        fi
        leaf=$(retrain_leaf_of "$sub" "$seed"); dir=$(retrain_dir_of "$sub" "$seed")
        ledger="$RETRAIN_LEDGER"
      fi
      if grep -qxF "$leaf" "$ledger" 2>/dev/null; then
        skipped=$((skipped + 1)); continue
      fi
      # allocate() creates the run dir with exist_ok=False, so a leftover
      # directory -- a smoke test, an interrupted attempt -- would make the job
      # die with FileExistsError long after it started.  Treat it as accounted
      # for and say so.
      if [ -e "$dir" ]; then
        echo "  -- leftover run dir, not resubmitting: $leaf"
        echo "$leaf" >> "$ledger"; skipped=$((skipped + 1)); continue
      fi
      # The rotation index is incremented here in the main shell, never inside a
      # command substitution: a substitution runs in a subshell, so a counter
      # bumped in there would never advance.
      P=${POOL[$(( idx % ${#POOL[@]} ))]}
      idx=$(( idx + 1 ))
      submit_one "$P" "$kind" "$sub" "$seed"
      case $? in
        0) submitted=$((submitted + 1)) ;;
        3) return 4 ;;
        *) echo "    (will retry on the next pass)" ;;
      esac
    done; done
  done
  echo "  pass: $submitted submitted, $skipped accounted for, $blocked retrain(s) waiting on their search"
  return 0
}

missing() {
  local kind=$1 n=0 sub seed leaf ledger
  for sub in $SUBJECTS; do for seed in $SEEDS; do
    if [ "$kind" = "search" ]; then
      leaf=$(search_leaf_of "$sub" "$seed"); ledger="$SEARCH_LEDGER"
    else
      leaf=$(retrain_leaf_of "$sub" "$seed"); ledger="$RETRAIN_LEDGER"
    fi
    grep -qxF "$leaf" "$ledger" 2>/dev/null || n=$((n + 1))
  done; done
  echo "$n"
}

one_pass
rc=$?
left_s=$(missing search); left_r=$(missing retrain)
echo "=== remaining: $left_s search, $left_r retrain ==="

if [ "$MODE" = "drain" ]; then
  while [ "$left_s" -gt 0 ] || [ "$left_r" -gt 0 ]; do
    sleep "$DRAIN_SLEEP"
    one_pass; rc=$?
    left_s=$(missing search); left_r=$(missing retrain)
    echo "[$(date '+%F %T')] remaining: $left_s search, $left_r retrain"
  done
  echo "=== DRAIN COMPLETE: all $(count_grid) searches and $(count_grid) retrains submitted ==="
  echo "=== note: 'submitted' is not 'finished' -- count final_summary.json to know that ==="
  exit 0
fi
[ "$rc" = 4 ] && echo "=== submit cap reached; re-run this script later to submit the rest ==="
[ "$rc" = 5 ] && exit 1
exit 0

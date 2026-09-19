#!/bin/bash
# Submit the Tier-1 global-family operator-separability pilot.
#
# Grid: 3 subjects x 3 seeds x 5 families = 45 runs.  Each run fixes one V2
# temporal family for all three bands and trains it from scratch on the shared
# Session-0 231/57 split, screening only (Session 1 never opened).
#
# This is Tier-1: it asks whether a family preference survives seed noise when
# one family serves all three bands at once.  It does NOT answer whether
# subjects prefer different families per frequency band -- a band-specific
# preference averages out across the three bands and is invisible here by
# construction.  A WEAK_GLOBAL verdict from tools/analyze_operator_v2.py
# therefore bounds global separability only and must not be read as "abandon
# the per-band search"; the follow-up is the band-specific probe.  See
# docs/operator_separability_v2.md sections 9-10.
#
# The cluster caps submissions per user (normal QOS: MaxSubmitPU=30), so 45
# runs cannot all be in flight at once.  This script is therefore *resumable*:
# every run it submits is recorded in a ledger, and re-running it submits only
# what is still missing.  Run it again after the first wave drains:
#
#     ./bin/submit_operator_v2.sh            # one pass
#     ./bin/submit_operator_v2.sh drain      # keep topping up until done
#     ./bin/submit_operator_v2.sh dry        # print the grid, submit nothing
set -u

RUNS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$RUNS/.." && pwd)"
cd "$RUNS"

MODE="${1:-run}"

# 0-based subject index; SUB+1 is the dataset's id (2,4,5 -> 003,005,006).
SUBJECTS="${SUBJECTS:-2 4 5}"
OPS="${OPS:-dilated gated local_attention dynamic band_gated}"
SEEDS="${SEEDS:-20250901 20250902 20250903}"
RF="${RF:-57}"
OUTARM="${OUTARM:-operator_v2}"
# GPUFEE06/GPUFEE08 are excluded by request.  Widest partition first.
PARTITIONS="${PARTITIONS:-GPUFEE04 GPUFEE05 GPUFEE02}"
DRAIN_SLEEP="${DRAIN_SLEEP:-120}"

LEDGER="$RUNS/sh_log/opv2_submitted.txt"

count_grid() { echo $(( $(echo "$SUBJECTS" | wc -w) * $(echo "$OPS" | wc -w) * $(echo "$SEEDS" | wc -w) )); }
echo "=== grid: $(count_grid) runs (subjects=$SUBJECTS seeds='$SEEDS') ==="

if [ "$MODE" = "dry" ]; then
  # A preview that writes a file or fires a job is not a preview.
  for sub in $SUBJECTS; do for op in $OPS; do for seed in $SEEDS; do
    echo "$sub $op $seed"
  done; done; done
  exit 0
fi

mkdir -p sh_log/opv2 "outputs/$OUTARM" "logs/$OUTARM"
[ -f "$LEDGER" ] || : > "$LEDGER"

leaf_of() { printf "train_s%03d_seed%s_operator_v2_%s" "$(( $1 + 1 ))" "$3" "$2"; }
run_dir_of() { echo "$RUNS/outputs/$OUTARM/bci42a/$(leaf_of "$@")"; }

# --- capacity, checked now and never assumed ------------------------------
# A static per-partition cap is not a substitute for looking.  On 2026-09-18 a
# job went to GPUFEE02 (2 cards, both already taken) and sat PENDING behind
# another user while GPUFEE04 had idle cards; the cap said "2, go ahead".
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
    for _ in $(seq 1 "$n" 2>/dev/null); do POOL+=("$P"); done
  done
}

submit_one() {
  local P=$1 sub=$2 op=$3 seed=$4 leaf
  leaf=$(leaf_of "$sub" "$op" "$seed")
  local out
  out=$(SUB=$sub OP=$op TRAIN_SEED=$seed RF=$RF OUTARM=$OUTARM \
    sbatch -p "$P" bin/run_operator_v2_job.sh 2>&1)
  if echo "$out" | grep -q "Submitted batch job"; then
    # Record before reporting: the ledger is what makes a re-run safe, and a
    # job that was accepted but not recorded would be submitted twice.
    echo "$leaf" >> "$LEDGER"
    echo "  -> $P $leaf | $out"
    return 0
  fi
  if echo "$out" | grep -q "QOSMaxSubmitJobPerUserLimit"; then
    echo "  !! per-user submit cap reached (normal QOS MaxSubmitPU)"; return 3
  fi
  echo "  !! sbatch failed on $P for $leaf: $out" >&2
  return 1
}

# Each pass walks the grid once, skipping anything already recorded, completed
# on disk, or in flight.  allocate() creates the run dir with exist_ok=False,
# so a leftover directory (a smoke test, a previous attempt) would make the job
# die with FileExistsError long after it started -- treat any existing dir as
# "do not resubmit" and say so rather than letting it fail an hour in.
one_pass() {
  build_pool
  if [ ${#POOL[@]} -eq 0 ]; then
    echo "!! no free GPU cards on: $PARTITIONS" >&2
    return 5
  fi
  echo "  pool: ${#POOL[@]} card(s) on $(printf '%s\n' "${POOL[@]}" | sort -u | paste -sd, -)"
  local idx=0 submitted=0 skipped=0 blocked=0
  local sub op seed leaf dir
  for sub in $SUBJECTS; do for op in $OPS; do for seed in $SEEDS; do
    leaf=$(leaf_of "$sub" "$op" "$seed")
    dir=$(run_dir_of "$sub" "$op" "$seed")
    if grep -qxF "$leaf" "$LEDGER" 2>/dev/null; then
      skipped=$((skipped + 1)); continue
    fi
    if [ -e "$dir" ]; then
      echo "  -- leftover run dir, not resubmitting: $leaf"
      echo "$leaf" >> "$LEDGER"; skipped=$((skipped + 1)); continue
    fi
    # The rotation index is incremented here in the main shell, never inside a
    # command substitution: a substitution runs in a subshell, so a counter
    # bumped in there would never advance.  That exact bug put 13 runs onto the
    # 4-card partition on 2026-09-18.
    P=${POOL[$(( idx % ${#POOL[@]} ))]}
    idx=$(( idx + 1 ))
    submit_one "$P" "$sub" "$op" "$seed"
    case $? in
      0) submitted=$((submitted + 1)) ;;
      3) blocked=1; return 4 ;;
      *) echo "    (will retry on the next pass)" ;;
    esac
  done; done; done
  echo "  pass: $submitted submitted, $skipped already accounted for"
  return 0
}

remaining() {
  local n=0 leaf
  for sub in $SUBJECTS; do for op in $OPS; do for seed in $SEEDS; do
    leaf=$(leaf_of "$sub" "$op" "$seed")
    grep -qxF "$leaf" "$LEDGER" 2>/dev/null || n=$((n + 1))
  done; done; done
  echo "$n"
}

one_pass
rc=$?
left=$(remaining)
echo "=== remaining: $left of $(count_grid) ==="

if [ "$MODE" = "drain" ]; then
  while [ "$left" -gt 0 ]; do
    sleep "$DRAIN_SLEEP"
    one_pass; rc=$?
    left=$(remaining)
    echo "[$(date '+%F %T')] remaining: $left of $(count_grid)"
  done
  echo "=== DRAIN COMPLETE: all $(count_grid) runs submitted ==="
  exit 0
fi
[ "$rc" = 4 ] && echo "=== submit cap reached; re-run this script later to submit the rest ==="
[ "$rc" = 5 ] && exit 1
exit 0

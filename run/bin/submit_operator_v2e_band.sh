#!/bin/bash
# Submit the band-specific mechanism probe.
#
# Grid: 3 subjects x 3 seeds x 9 configurations = 81 runs.  A configuration is
# one band swapping to one family; the other two bands stay on dilated_e.
#
#   Low  -> dynamic_e | gated_e | band_gated_e
#   Mid  -> dynamic_e | gated_e | band_gated_e
#   High -> dynamic_e | gated_e | band_gated_e
#
# The all-dilated baseline is NOT part of this grid and is NOT retrained.  Its
# nine runs already exist as the Expressive dilated_e arm under
# run/outputs/operator_v2e/, and tools/analyze_operator_v2e_band.py pairs
# against them.  Submitting them again would put a second, differently-seeded
# copy of the same configuration into a paired comparison.
#
# Screening only: Session 1 is never opened by this stage, and the job script
# has no flag that could open it.
#
# Quota-aware and resumable, like the other drivers: each job goes to a
# partition with a free card, and the ledger means re-running submits only what
# is missing.
set -u

RUNS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$RUNS/.." && pwd)"
cd "$RUNS"

MODE="${1:-run}"

# 0-based subject index; SUB+1 is the dataset's id (2,4,5 -> 003,005,006).
SUBJECTS="${SUBJECTS:-2 4 5}"
SEEDS="${SEEDS:-20250901 20250902 20250903}"
# The 9 configurations, spelled as "band:family".
CONFIGS="${CONFIGS:-low:dynamic_e low:gated_e low:band_gated_e mid:dynamic_e mid:gated_e mid:band_gated_e high:dynamic_e high:gated_e high:band_gated_e}"
RF="${RF:-57}"
OUTARM="${OUTARM:-operator_v2e_band}"
# GPUFEE06/GPUFEE08 are excluded by request.  Widest partition first.
PARTITIONS="${PARTITIONS:-GPUFEE04 GPUFEE05 GPUFEE02}"
DRAIN_SLEEP="${DRAIN_SLEEP:-120}"

LEDGER="$RUNS/sh_log/operator_v2e_band_submitted.txt"

count_grid() { echo $(( $(echo "$SUBJECTS" | wc -w) * $(echo "$CONFIGS" | wc -w) * $(echo "$SEEDS" | wc -w) )); }
echo "=== grid: $(count_grid) runs (subjects=$SUBJECTS, 9 configurations, seeds='$SEEDS') ==="

if [ "$MODE" = "dry" ]; then
  # A preview that writes a file or fires a job is not a preview.
  for sub in $SUBJECTS; do for cfg in $CONFIGS; do for seed in $SEEDS; do
    echo "$sub $cfg $seed"
  done; done; done
  exit 0
fi

mkdir -p sh_log/opv2eband "outputs/$OUTARM" "logs/$OUTARM"
[ -f "$LEDGER" ] || : > "$LEDGER"

# The stage segment is written into the format string on purpose.  It is the one
# place that guarantees a band-probe leaf can never be spelled like a global
# grid's leaf, whatever OUTARM is set to.
leaf_of() {
  local band="${2%%:*}" family="${2##*:}"
  printf "train_s%03d_seed%s_operator_v2e_band_%s_%s" "$(( $1 + 1 ))" "$3" "$band" "$family"
}
run_dir_of() { echo "$RUNS/outputs/$OUTARM/bci42a/$(leaf_of "$@")"; }

# --- capacity, checked now and never assumed ------------------------------
# A static per-partition cap is not a substitute for looking: on 2026-09-18 a
# job went to GPUFEE02 (2 cards, both taken) and sat PENDING behind another user
# while GPUFEE04 had idle cards.
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
  local P=$1 sub=$2 cfg=$3 seed=$4 leaf band family
  leaf=$(leaf_of "$sub" "$cfg" "$seed")
  band="${cfg%%:*}"; family="${cfg##*:}"
  local out
  out=$(SUB=$sub BAND=$band FAMILY=$family TRAIN_SEED=$seed RF=$RF OUTARM=$OUTARM \
    sbatch -p "$P" bin/run_operator_v2e_band_job.sh 2>&1)
  if echo "$out" | grep -q "Submitted batch job"; then
    # Record before reporting: the ledger is what makes a re-run safe, and a
    # job accepted but not recorded would be submitted twice.
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

one_pass() {
  build_pool
  if [ ${#POOL[@]} -eq 0 ]; then
    echo "!! no free GPU cards on: $PARTITIONS" >&2
    return 5
  fi
  echo "  pool: ${#POOL[@]} card(s) on $(printf '%s\n' "${POOL[@]}" | sort -u | paste -sd, -)"
  local idx=0 submitted=0 skipped=0
  local sub cfg seed leaf dir
  for sub in $SUBJECTS; do for cfg in $CONFIGS; do for seed in $SEEDS; do
    leaf=$(leaf_of "$sub" "$cfg" "$seed")
    dir=$(run_dir_of "$sub" "$cfg" "$seed")
    if grep -qxF "$leaf" "$LEDGER" 2>/dev/null; then
      skipped=$((skipped + 1)); continue
    fi
    # allocate() creates the run dir with exist_ok=False, so a leftover
    # directory -- a smoke test, an interrupted attempt -- would make the job
    # die with FileExistsError long after it started.  Treat it as accounted
    # for and say so.
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
    submit_one "$P" "$sub" "$cfg" "$seed"
    case $? in
      0) submitted=$((submitted + 1)) ;;
      3) return 4 ;;
      *) echo "    (will retry on the next pass)" ;;
    esac
  done; done; done
  echo "  pass: $submitted submitted, $skipped already accounted for"
  return 0
}

remaining() {
  local n=0 leaf
  for sub in $SUBJECTS; do for cfg in $CONFIGS; do for seed in $SEEDS; do
    leaf=$(leaf_of "$sub" "$cfg" "$seed")
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

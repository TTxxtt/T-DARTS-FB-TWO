#!/bin/bash
# Idempotent, quota-aware Stage-2 driver.  Submits a --stage2-only job for every
# finished Stage-1 retrain whose Stage-2 is not complete, never exceeding the
# cluster QOS per-user submit cap (CAP_STAGE2, default 28 < the observed 30),
# and tops up as running jobs clear.  Safe under nohup: it exits only once every
# run carries a Stage-2 test readout.  A per-run sentinel tracks dispatched jobs
# across restarts; a submission rejected by QOS drops its sentinel so the next
# pass retries it.
set -u

RUNS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RUNS"

CAP_STAGE2="${CAP_STAGE2:-28}"
SLEEP_STAGE2="${SLEEP_STAGE2:-60}"
SUBJECTS="${SUBJECTS:-0 1 2 3 4 5 6 7 8}"
SEEDS="${SEEDS:-20250901 20250902 20250903}"
ARMS="${ARMS:-hier rf}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-600}"
SENTINEL_DIR="$RUNS/sh_log/stage2_sentinels"
mkdir -p sh_log/stage2 "$SENTINEL_DIR"

# Job records: "arm:outarm:sub(0-based):subj:seed"
jobs=()
for arm in $ARMS; do
  case "$arm" in
    hier) outarm=hier_retrain ;;
    rf)   outarm=rf_retrain_nll ;;
    *)    echo "!! unknown arm '$arm'"; continue ;;
  esac
  for sub in $SUBJECTS; do
    subj=$(printf "%03d" $((sub + 1)))
    for seed in $SEEDS; do
      jobs+=( "$arm:$outarm:$sub:$subj:$seed" )
    done
  done
done
total=${#jobs[@]}

done_job() { # echo 0 if the run already has a Stage-2 test readout
  IFS=: read -r arm outarm _ _ subj seed <<< "$1"
  grep -q '"test_threshold"' \
    "$RUNS/outputs/$outarm/bci42a/train_s${subj}_seed${seed}_${arm}/final_summary.json" 2>/dev/null
}

# dispatch: claim+submit one job; return 0 if a brand-new submission went out.
dispatch() {
  IFS=: read -r arm outarm sub subj seed <<< "$1"
  local sentinel="$SENTINEL_DIR/${outarm}_s${subj}_seed${seed}_${arm}"
  if done_job "$1"; then rm -f "$sentinel"; return 1; fi
  if [ -f "$sentinel" ]; then return 1; fi
  : > "$sentinel"
  if sbatch --export=ALL,SUB=$sub,SEED=$seed,ARM=$arm,OUTARM=$outarm,STAGE2_EPOCHS=$STAGE2_EPOCHS \
        --job-name="stage2_${arm}_${subj}_${seed}" "$RUNS/bin/run_stage2_job.sh" >/dev/null 2>&1; then
    return 0
  else
    rm -f "$sentinel"; return 1
  fi
}

echo "=== stage2 driver: $total candidate jobs, cap=$CAP_STAGE2, sleep=${SLEEP_STAGE2}s $(date) ==="
while :; do
  submitted=$(squeue -h -u "$USER" 2>/dev/null | grep -cE 'stage2_(hier|rf)_' || true)
  [ -n "$submitted" ] && [ "$submitted" -ge 0 ] || submitted=0
  available=$(( CAP_STAGE2 - submitted ))
  [ "$available" -lt 0 ] && available=0

  remaining=0
  new=0
  for entry in "${jobs[@]}"; do
    done_job "$entry" && continue
    remaining=$((remaining + 1))
    if [ "$available" -gt 0 ] && dispatch "$entry"; then
      new=$((new + 1))
      available=$((available - 1))
    fi
  done

  if [ "$remaining" -eq 0 ]; then
    echo "=== stage2 driver DONE: all $total runs carry a Stage-2 test readout $(date) ==="
    break
  fi
  if [ "$new" -eq 0 ]; then
    echo "[$(date '+%H:%M:%S')] no new submissions (remaining=$remaining in_scheduler=$submitted) -- waiting $SLEEP_STAGE2 s"
  else
    echo "[$(date '+%H:%M:%S')] submitted $new this pass (remaining=$remaining in_scheduler=$submitted free=$available)"
  fi
  sleep "$SLEEP_STAGE2"
done
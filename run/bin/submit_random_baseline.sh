#!/bin/bash
# Submit the random-architecture baseline: 9 subjects x 5 frozen random
# genotypes, plus the standard-DARTS searches needed to give every subject a
# searched comparator in the same 14^6 space.
#
# The genotypes are NOT generated here. They are sampled once by
# tools/sample_random_genotypes.py into run/genotypes/random_k5/s<NNN>/ and
# frozen before any result existed -- that pre-registration is the whole point,
# so this driver only consumes them and refuses to run if they are missing.
set -u

RUNS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RUNS"
mkdir -p sh_log/randomret sh_log/darts outputs/random_k5 logs/random_k5

TRAIN_SEED="${TRAIN_SEED:-20250901}"
GENO_ROOT="$RUNS/genotypes/random_k5"
COUNT="${COUNT:-5}"
# Subjects whose standard-DARTS@1e-3 search still has to run.  003/005/006
# already have one (run/outputs/darts_lr1e3), so they are excluded.
SEARCH_SUBS="${SEARCH_SUBS:-0 1 3 6 7 8}"

for sub in $(seq 0 8); do
  s=$(printf "%03d" $((sub + 1)))
  [ -f "$GENO_ROOT/s${s}/manifest.json" ] || { echo "!! missing frozen sample for s${s} in $GENO_ROOT" >&2; exit 1; }
done

AGENDA="$RUNS/sh_log/random_baseline_agenda.txt"
: > "$AGENDA"
for sub in $(seq 0 8); do
  s=$(printf "%03d" $((sub + 1)))
  for k in $(seq 0 $((COUNT - 1))); do
    # The genotype directory carries the "s" prefix (s001) while the subject
    # index and the file name do not; the file already begins with "g", so the
    # gene token below is the whole file stem and must not be re-prefixed.
    printf 'random %s s%s g%03d\n' "$sub" "$s" "$k" >> "$AGENDA"
  done
done
for sub in $SEARCH_SUBS; do printf 'search %s\n' "$sub" >> "$AGENDA"; done

echo "=== to run: $(wc -l < "$AGENDA") jobs (TRAIN_SEED=$TRAIN_SEED) ==="
[ "${1:-run}" = "dry" ] && { cat "$AGENDA"; exit 0; }

# Account-wide cap is QOS MaxSubmitPU=30 (running+pending), not per-partition.
want=30
submit_one() {
  local r02 r04 r05 entry P cap used out
  r02=$(squeue -u "$USER" -p GPUFEE02 -h -t RUNNING,PENDING 2>/dev/null | wc -l)
  r04=$(squeue -u "$USER" -p GPUFEE04 -h -t RUNNING,PENDING 2>/dev/null | wc -l)
  r05=$(squeue -u "$USER" -p GPUFEE05 -h -t RUNNING,PENDING 2>/dev/null | wc -l)
  # GPUFEE05 first: it is the smallest node (4 GPUs) and sits idle most of the
  # time, so filling it costs the least queue wait.  GPUFEE06/08 are excluded by
  # request even though they carry spare GPUs.  Caps are the partition's GPU
  # count, not the account-wide QOS limit, which caps the total at 30 anyway.
  for entry in "GPUFEE05 4 $r05" "GPUFEE02 2 $r02" "GPUFEE04 20 $r04"; do
    P=$(echo "$entry" | awk '{print $1}')
    cap=$(echo "$entry" | awk '{print $2}')
    used=$(echo "$entry" | awk '{print $3}')
    [ "$used" -lt "$cap" ] || continue
    if [ "$1" = "search" ]; then
      out=$(SUB="$2" SEED="$TRAIN_SEED" OUTARM=darts_lr1e3 EPOCHS=200 ALPHA_LR=1e-3 \
        ALPHA_UPDATE_MODE=minibatch DECODE_MODE=last \
        sbatch -p "$P" bin/run_darts_job.sh 2>&1)
    else
      out=$(SUB="$2" TRAIN_SEED="$TRAIN_SEED" OUTARM=random_k5 \
        GENOTYPE="$GENO_ROOT/$3/$4.json" \
        sbatch -p "$P" bin/run_random_retrain_job.sh 2>&1)
    fi
    if echo "$out" | grep -q "Submitted batch job"; then
      echo "  -> $P $1 $2 ${3:-}${4:-} | $out"; return 0
    fi
  done
  return 1
}

while [ -s "$AGENDA" ]; do
  run=$(squeue -u "$USER" -o %.8T 2>/dev/null | grep -c RUNNING)
  pend=$(squeue -u "$USER" -o %.8T 2>/dev/null | grep -c PENDING)
  if [ $((run + pend)) -ge $want ]; then sleep 30; continue; fi
  line=$(head -1 "$AGENDA"); sed -i '1d' "$AGENDA"
  # Skip a job whose output leaf already exists -- either it was submitted by an
  # earlier run of this driver, or the genotype was retrained by hand.  Without
  # this the job re-queues forever: train_retrain.py refuses an existing leaf
  # with mkdir(exist_ok=False), sbatch still succeeds, and the agenda line
  # below is restored on every failure.
  set -- $line
  if [ "$1" = "random" ]; then
    subj=$(printf "%03d" $(($2 + 1)))
    leaf="$RUNS/outputs/random_k5/bci42a/train_s${subj}_seed${TRAIN_SEED}_$4"
    [ -e "$leaf" ] && { echo "  -- skip (exists): $line"; continue; }
  fi
  # shellcheck disable=SC2086
  submit_one $line || { echo "$line" >> "$AGENDA"; sleep 30; }
done
echo "=== ALL RANDOM-BASELINE JOBS SUBMITTED ==="

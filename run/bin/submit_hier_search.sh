#!/bin/bash
# Submit the frozen hierarchical method (hard operator search at RF57, then RF
# search) for the pre-registered subjects and seeds.  Each seed runs its own
# complete chain; no cross-seed majority vote.  Session 1 stays closed.
set -u

RUNS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RUNS"

# Pre-registered 0-based subject indices; SUB+1 gives 001/003/005.
SUBJECTS="${SUBJECTS:-0 2 4}"
SEEDS="${SEEDS:-20250901 20250902 20250903}"
PHASE_A_EPOCHS="${PHASE_A_EPOCHS:-300}"
PHASE_B_EPOCHS="${PHASE_B_EPOCHS:-200}"
ALPHA_LR="${ALPHA_LR:-3e-4}"
OUTARM="${OUTARM:-hier}"

mkdir -p sh_log/hier "outputs/$OUTARM" "logs/$OUTARM"

AGENDA="$RUNS/sh_log/hier_agenda.txt"
: > "$AGENDA"
for sub in $SUBJECTS; do
  SUBJ=$(printf "%03d" $((sub + 1)))
  for seed in $SEEDS; do
    # Resume-safe: never resubmit a chain whose run directory already exists.
    if [ -e "$RUNS/outputs/$OUTARM/bci42a/hier_search_s${SUBJ}_seed${seed}" ]; then
      echo "  skip existing sub=$SUBJ seed=$seed"
      continue
    fi
    echo "$sub $seed" >> "$AGENDA"
  done
done
echo "=== to run: $(wc -l < "$AGENDA") chains (subjects=$SUBJECTS seeds=$SEEDS phaseA=$PHASE_A_EPOCHS phaseB=$PHASE_B_EPOCHS) ==="
[ "${1:-run}" = "dry" ] && { cat "$AGENDA"; exit 0; }

submit_one() {
  local sub=$1 seed=$2
  local r04 r05 entry P cap used out
  r04=$(squeue -u "$USER" -p GPUFEE04 -h -t RUNNING,PENDING 2>/dev/null | wc -l)
  r05=$(squeue -u "$USER" -p GPUFEE05 -h -t RUNNING,PENDING 2>/dev/null | wc -l)
  for entry in "GPUFEE04 20 $r04" "GPUFEE05 2 $r05"; do
    P=$(echo "$entry" | awk '{print $1}')
    cap=$(echo "$entry" | awk '{print $2}')
    used=$(echo "$entry" | awk '{print $3}')
    [ "$used" -lt "$cap" ] || continue
    out=$(SUB=$sub SEED=$seed PHASE_A_EPOCHS=$PHASE_A_EPOCHS PHASE_B_EPOCHS=$PHASE_B_EPOCHS \
      ALPHA_LR=$ALPHA_LR OUTARM=$OUTARM \
      sbatch -p "$P" bin/run_hier_search_job.sh 2>&1)
    if echo "$out" | grep -q "Submitted batch job"; then
      echo "  -> $P sub=$sub seed=$seed | $out"; return 0
    fi
  done
  return 1
}

while [ -s "$AGENDA" ]; do
  read -r sub seed < "$AGENDA"
  sed -i '1d' "$AGENDA"
  submit_one "$sub" "$seed" || { echo "$sub $seed" >> "$AGENDA"; sleep 20; }
done
echo "=== ALL HIERARCHICAL SEARCHES SUBMITTED ==="

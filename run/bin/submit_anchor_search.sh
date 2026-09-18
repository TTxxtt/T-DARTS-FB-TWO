#!/bin/bash
# Submit the full two-phase anchored search (operator then RF) on Session 0.
# Default is the pre-registered Subject 003, three seeds; each seed runs its
# own complete chain, and no cross-seed majority vote is taken.  Session 1 is
# never opened.  Same quota-aware loop as the other drivers.
set -u

RUNS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RUNS"

# Pre-registered 0-based subject index; SUB+1 gives the dataset's 003.
SUBJECTS="${SUBJECTS:-2}"
SEEDS="${SEEDS:-20250901 20250902 20250903}"
OP_EPOCHS="${OP_EPOCHS:-200}"
RF_EPOCHS="${RF_EPOCHS:-200}"
ALPHA_LR="${ALPHA_LR:-3e-4}"
NO_DUP="${NO_DUP:-1}"
OUTARM="${OUTARM:-anchored}"

mkdir -p sh_log/anchor "outputs/$OUTARM" "logs/$OUTARM"

AGENDA="$RUNS/sh_log/anchor_agenda.txt"
: > "$AGENDA"
for sub in $SUBJECTS; do
  for seed in $SEEDS; do echo "$sub $seed" >> "$AGENDA"; done
done
echo "=== to run: $(wc -l < "$AGENDA") jobs (subjects=$SUBJECTS seeds=$SEEDS op_epochs=$OP_EPOCHS rf_epochs=$RF_EPOCHS no_dup=$NO_DUP) ==="
[ "${1:-run}" = "dry" ] && { cat "$AGENDA"; exit 0; }

submit_one() {
  local sub=$1 seed=$2
  local r04 r05 r02 entry P cap used out
  r04=$(squeue -u "$USER" -p GPUFEE04 -h -t RUNNING,PENDING 2>/dev/null | wc -l)
  r05=$(squeue -u "$USER" -p GPUFEE05 -h -t RUNNING,PENDING 2>/dev/null | wc -l)
  r02=$(squeue -u "$USER" -p GPUFEE02 -h -t RUNNING,PENDING 2>/dev/null | wc -l)
  for entry in "GPUFEE04 20 $r04" "GPUFEE05 2 $r05" "GPUFEE02 2 $r02"; do
    P=$(echo "$entry" | awk '{print $1}')
    cap=$(echo "$entry" | awk '{print $2}')
    used=$(echo "$entry" | awk '{print $3}')
    [ "$used" -lt "$cap" ] || continue
    out=$(SUB=$sub SEED=$seed OP_EPOCHS=$OP_EPOCHS RF_EPOCHS=$RF_EPOCHS \
      ALPHA_LR=$ALPHA_LR NO_DUP=$NO_DUP OUTARM=$OUTARM \
      sbatch -p "$P" bin/run_anchor_search_job.sh 2>&1)
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
echo "=== ALL ANCHORED SEARCHES SUBMITTED ==="

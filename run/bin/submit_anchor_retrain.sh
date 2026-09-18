#!/bin/bash
# Retrain each seed's own complete anchored genotype from scratch.  Submit only
# after submit_anchor_search.sh has finished: every job preflights its seed's
# genotype.json and exits before taking a GPU if it is missing.  No cross-seed
# majority vote is formed -- seed k's genotype is retrained as seed k.
set -u

RUNS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RUNS"

SUBJECTS="${SUBJECTS:-2}"
SEEDS="${SEEDS:-20250901 20250902 20250903}"
SEARCH_ARM="${SEARCH_ARM:-anchored}"
OUTARM="${OUTARM:-anchor_retrain}"

mkdir -p sh_log/anchor_retrain "outputs/$OUTARM" "logs/$OUTARM"

AGENDA="$RUNS/sh_log/anchor_retrain_agenda.txt"
: > "$AGENDA"
for sub in $SUBJECTS; do
  for seed in $SEEDS; do echo "$sub $seed" >> "$AGENDA"; done
done
echo "=== to run: $(wc -l < "$AGENDA") retrains (subjects=$SUBJECTS seeds=$SEEDS) ==="
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
    out=$(SUB=$sub SEED=$seed SEARCH_ARM=$SEARCH_ARM OUTARM=$OUTARM \
      sbatch -p "$P" bin/run_anchor_retrain_job.sh 2>&1)
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
echo "=== ALL ANCHOR RETRAINS SUBMITTED ==="

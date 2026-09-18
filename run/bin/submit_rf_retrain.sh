#!/bin/bash
# Retrain the RF-only search genotypes from scratch.  Submit only after
# submit_rf_search.sh has finished: each job preflights its seed's
# genotype.json and exits before taking a GPU if it is missing.
set -u

RUNS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RUNS"

SUBJECTS="${SUBJECTS:-2}"
SEEDS="${SEEDS:-20250901 20250902 20250903}"
SEARCH_ARM="${SEARCH_ARM:-rfsearch}"
OUTARM="${OUTARM:-rfretrain}"
BEST_METRIC="${BEST_METRIC:-val_inacc}"

mkdir -p sh_log/rfretrain "outputs/$OUTARM" "logs/$OUTARM"

AGENDA="$RUNS/sh_log/rfretrain_agenda.txt"
: > "$AGENDA"
for sub in $SUBJECTS; do
  SUBJ=$(printf "%03d" $((sub + 1)))
  for seed in $SEEDS; do
    # Resume-safe: a finished retrain is never submitted again.
    if [ -f "$RUNS/outputs/$OUTARM/bci42a/train_s${SUBJ}_seed${seed}_rf/final_summary.json" ]; then
      echo "  skip existing sub=$SUBJ seed=$seed"
      continue
    fi
    echo "$sub $seed" >> "$AGENDA"
  done
done
echo "=== to run: $(wc -l < "$AGENDA") retrains (subjects=$SUBJECTS seeds=$SEEDS) ==="
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
    out=$(SUB=$sub SEED=$seed SEARCH_ARM=$SEARCH_ARM OUTARM=$OUTARM BEST_METRIC=$BEST_METRIC \
      sbatch -p "$P" bin/run_rf_retrain_job.sh 2>&1)
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
echo "=== ALL RF RETRAINS SUBMITTED ==="

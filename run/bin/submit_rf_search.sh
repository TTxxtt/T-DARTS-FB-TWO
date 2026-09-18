#!/bin/bash
# Submit Phase B RF-only searches: both paths fixed to dilated, RF searched
# over 15/29/57/113 with --no-duplicate-paths.  Same quota-aware loop as the
# other drivers.  Default is the pre-registered Subject 003, three seeds.
set -u

RUNS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RUNS"

# Pre-registered 0-based subject index; SUB+1 gives the dataset's 003.
SUBJECTS="${SUBJECTS:-2}"
SEEDS="${SEEDS:-20250901 20250902 20250903}"
EPOCHS="${EPOCHS:-200}"
ALPHA_LR="${ALPHA_LR:-3e-4}"
OUTARM="${OUTARM:-rfsearch}"

mkdir -p sh_log/rfsearch "outputs/$OUTARM" "logs/$OUTARM"

AGENDA="$RUNS/sh_log/rfsearch_agenda.txt"
: > "$AGENDA"
for sub in $SUBJECTS; do
  SUBJ=$(printf "%03d" $((sub + 1)))
  for seed in $SEEDS; do
    # Resume-safe: a run directory already on disk means the job exists (or
    # finished), so it is not submitted again.
    if [ -e "$RUNS/outputs/$OUTARM/bci42a/search_s${SUBJ}_seed${seed}" ]; then
      echo "  skip existing sub=$SUBJ seed=$seed"
      continue
    fi
    echo "$sub $seed" >> "$AGENDA"
  done
done
echo "=== to run: $(wc -l < "$AGENDA") jobs (subjects=$SUBJECTS seeds=$SEEDS epochs=$EPOCHS) ==="
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
    out=$(SUB=$sub SEED=$seed EPOCHS=$EPOCHS ALPHA_LR=$ALPHA_LR \
      sbatch -p "$P" bin/run_rf_search_job.sh 2>&1)
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
echo "=== ALL RF SEARCHES SUBMITTED ==="

#!/bin/bash
# Submit the fixed-genotype operator-separability pilot: Path0 = dilated_rf57
# everywhere, Path1 = {dilated, normal, dwsep, lkdw}_rf57, identical split,
# seed, budget and stopping rule.  Same quota-aware loop as the other drivers.
#
# Start with the default single subject and single seed.  Only after seeing a
# difference that exceeds run-to-run noise should OPS be narrowed to the
# interesting arms and SEEDS extended -- running all four arms at three seeds
# before the pilot says anything is wasted GPU time.
set -u

RUNS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$RUNS/.." && pwd)"
cd "$RUNS"

# Pre-registered 0-based subject index; SUB+1 gives the dataset's 003.
SUBJECTS="${SUBJECTS:-2}"
OPS="${OPS:-dilated normal dwsep lkdw}"
SEEDS="${SEEDS:-20250901}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
PATIENCE="${PATIENCE:-50}"
GENOTYPE_DIR="${GENOTYPE_DIR:-$RUNS/genotypes/opsep_rf57}"
OUTARM="${OUTARM:-opsep_rf57}"

mkdir -p sh_log/opsep "outputs/$OUTARM" "logs/$OUTARM"

AGENDA="$RUNS/sh_log/opsep_agenda.txt"
: > "$AGENDA"
for sub in $SUBJECTS; do
  for op in $OPS; do
    for seed in $SEEDS; do echo "$sub $op $seed" >> "$AGENDA"; done
  done
done
echo "=== to run: $(wc -l < "$AGENDA") jobs (subjects=$SUBJECTS ops='$OPS' seeds=$SEEDS) ==="
# Dry mode stops before the generator runs: a preview that writes files is not
# a preview.
[ "${1:-run}" = "dry" ] && { cat "$AGENDA"; exit 0; }

if [ ! -f "$GENOTYPE_DIR/manifest.json" ]; then
  echo "=== generating the four fixed genotypes in $GENOTYPE_DIR ==="
  source /gpfs/home/W125221190/anaconda3/bin/activate
  conda activate eeg
  python "$REPO/tools/make_operator_separability_genotypes.py" \
    --output-dir "$GENOTYPE_DIR" || exit 1
fi

submit_one() {
  local sub=$1 op=$2 seed=$3
  local r04 r05 r02 entry P cap used out
  r04=$(squeue -u "$USER" -p GPUFEE04 -h -t RUNNING,PENDING 2>/dev/null | wc -l)
  r05=$(squeue -u "$USER" -p GPUFEE05 -h -t RUNNING,PENDING 2>/dev/null | wc -l)
  r02=$(squeue -u "$USER" -p GPUFEE02 -h -t RUNNING,PENDING 2>/dev/null | wc -l)
  for entry in "GPUFEE04 20 $r04" "GPUFEE05 2 $r05" "GPUFEE02 2 $r02"; do
    P=$(echo "$entry" | awk '{print $1}')
    cap=$(echo "$entry" | awk '{print $2}')
    used=$(echo "$entry" | awk '{print $3}')
    [ "$used" -lt "$cap" ] || continue
    out=$(SUB=$sub OP=$op TRAIN_SEED=$seed MAX_EPOCHS=$MAX_EPOCHS PATIENCE=$PATIENCE \
      sbatch -p "$P" bin/run_operator_separability_job.sh 2>&1)
    if echo "$out" | grep -q "Submitted batch job"; then
      echo "  -> $P sub=$sub op=$op seed=$seed | $out"; return 0
    fi
  done
  return 1
}

while [ -s "$AGENDA" ]; do
  line=$(head -1 "$AGENDA"); sed -i '1d' "$AGENDA"
  # shellcheck disable=SC2086
  submit_one $line || { echo "$line" >> "$AGENDA"; sleep 20; }
done
echo "=== ALL OPERATOR-SEPARABILITY JOBS SUBMITTED ==="

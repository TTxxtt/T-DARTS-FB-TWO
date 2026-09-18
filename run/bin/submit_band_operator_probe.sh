#!/bin/bash
# Fixed-RF57 single-band operator specificity probe: 3 bands x 3 non-anchor
# operators + the all-dilated baseline = 10 configurations, one training seed,
# 50 epochs, screening-only.  The three "<band>_dilated" cells collapse to the
# baseline (same six genes), so they are not submitted again.
set -u

RUNS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$RUNS/.." && pwd)"
cd "$RUNS"

# Pre-registered 0-based subject index; SUB+1 gives the dataset's 003.
SUBJECTS="${SUBJECTS:-2}"
TRAIN_SEED="${TRAIN_SEED:-20250901}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
PATIENCE="${PATIENCE:-50}"
GENOTYPE_DIR="${GENOTYPE_DIR:-$RUNS/genotypes/band_operator_rf57}"
OUTARM="${OUTARM:-band_op}"

mkdir -p sh_log/bandop "outputs/$OUTARM" "logs/$OUTARM"

if [ ! -f "$GENOTYPE_DIR/manifest.json" ]; then
  echo "=== generating the fixed-RF57 probe genotypes in $GENOTYPE_DIR ==="
  source /gpfs/home/W125221190/anaconda3/bin/activate
  conda activate eeg
  python "$REPO/tools/make_band_operator_genotypes.py" --output-dir "$GENOTYPE_DIR" || exit 1
fi

AGENDA="$RUNS/sh_log/band_operator_agenda.txt"
: > "$AGENDA"
for sub in $SUBJECTS; do
  for arm in baseline_dd low_normal low_dwsep low_lkdw mid_normal mid_dwsep mid_lkdw high_normal high_dwsep high_lkdw; do
    echo "$sub $arm" >> "$AGENDA"
  done
done
echo "=== to run: $(wc -l < "$AGENDA") jobs (subjects=$SUBJECTS seed=$TRAIN_SEED epochs=$MAX_EPOCHS) ==="
[ "${1:-run}" = "dry" ] && { cat "$AGENDA"; exit 0; }

submit_one() {
  local sub=$1 arm=$2
  local r04 r05 r02 entry P cap used out
  r04=$(squeue -u "$USER" -p GPUFEE04 -h -t RUNNING,PENDING 2>/dev/null | wc -l)
  r05=$(squeue -u "$USER" -p GPUFEE05 -h -t RUNNING,PENDING 2>/dev/null | wc -l)
  r02=$(squeue -u "$USER" -p GPUFEE02 -h -t RUNNING,PENDING 2>/dev/null | wc -l)
  for entry in "GPUFEE04 20 $r04" "GPUFEE05 2 $r05" "GPUFEE02 2 $r02"; do
    P=$(echo "$entry" | awk '{print $1}')
    cap=$(echo "$entry" | awk '{print $2}')
    used=$(echo "$entry" | awk '{print $3}')
    [ "$used" -lt "$cap" ] || continue
    out=$(SUB=$sub ARM_NAME=$arm GENOTYPE_JSON="$GENOTYPE_DIR/$arm.json" \
      TRAIN_SEED=$TRAIN_SEED MAX_EPOCHS=$MAX_EPOCHS PATIENCE=$PATIENCE OUTARM=$OUTARM \
      sbatch -p "$P" bin/run_band_operator_job.sh 2>&1)
    if echo "$out" | grep -q "Submitted batch job"; then
      echo "  -> $P sub=$sub $arm | $out"; return 0
    fi
  done
  return 1
}

while [ -s "$AGENDA" ]; do
  read -r sub arm < "$AGENDA"
  sed -i '1d' "$AGENDA"
  submit_one "$sub" "$arm" || { echo "$sub $arm" >> "$AGENDA"; sleep 20; }
done
echo "=== ALL BAND-OPERATOR PROBE JOBS SUBMITTED ==="

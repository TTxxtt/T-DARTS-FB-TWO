#!/bin/bash
# Pre-registered random RF baseline.  Round 1: 12 sampled no-duplicate RF-only
# genotypes plus the majority-vote genotype, all under one training seed,
# 50 epochs, screening-only (Session 1 never opened).  Round 2 (only if the
# majority lands in the front of the distribution): re-run a chosen set of arms
# under the remaining seeds, e.g.
#
#   ARMS="majority g010 g007 g005" SEEDS="20250902 20250903" \
#     bash bin/submit_rf_random_baseline.sh
#
# The sampling seed is fixed and the sample is written once, before any score
# exists -- that ordering is the pre-registration, so do not re-run the
# generator against another seed after looking at results.
set -u

RUNS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$RUNS/.." && pwd)"
cd "$RUNS"

# Pre-registered 0-based subject index; SUB+1 gives the dataset's 003.
SUBJECTS="${SUBJECTS:-2}"
TRAIN_SEED="${TRAIN_SEED:-20250901}"
SEEDS="${SEEDS:-$TRAIN_SEED}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
PATIENCE="${PATIENCE:-50}"
COUNT="${COUNT:-12}"
GENO_SEED="${GENO_SEED:-20250917}"
GENOTYPE_DIR="${GENOTYPE_DIR:-$RUNS/genotypes/rf_random_${COUNT}_seed${GENO_SEED}}"
OUTARM="${OUTARM:-rf_random}"
# Empty means round 1: all sampled genotypes plus the majority vote.
ARMS="${ARMS:-}"

mkdir -p sh_log/rfrandom "outputs/$OUTARM" "logs/$OUTARM"

AGENDA="$RUNS/sh_log/rf_random_agenda.txt"
: > "$AGENDA"
for sub in $SUBJECTS; do
  for seed in $SEEDS; do
    if [ -n "$ARMS" ]; then
      for arm in $ARMS; do echo "$sub arm $arm $seed" >> "$AGENDA"; done
    else
      for gid in $(seq 0 $((COUNT - 1))); do echo "$sub random $gid $seed" >> "$AGENDA"; done
      echo "$sub majority 0 $seed" >> "$AGENDA"
    fi
  done
done
echo "=== to run: $(wc -l < "$AGENDA") jobs (subjects=$SUBJECTS seeds='$SEEDS' arms='${ARMS:-all}' epochs=$MAX_EPOCHS) ==="
[ "${1:-run}" = "dry" ] && { cat "$AGENDA"; exit 0; }

if [ ! -f "$GENOTYPE_DIR/manifest.json" ]; then
  echo "=== generating the pre-registered baseline in $GENOTYPE_DIR ==="
  source /gpfs/home/W125221190/anaconda3/bin/activate
  conda activate eeg
  python "$REPO/tools/sample_rf_only_genotypes.py" \
    --output-dir "$GENOTYPE_DIR" --seed "$GENO_SEED" --count "$COUNT" || exit 1
fi

submit_one() {
  local sub=$1 mode=$2 arg=$3 seed=$4
  local r04 r05 r02 entry P cap used out
  r04=$(squeue -u "$USER" -p GPUFEE04 -h -t RUNNING,PENDING 2>/dev/null | wc -l)
  r05=$(squeue -u "$USER" -p GPUFEE05 -h -t RUNNING,PENDING 2>/dev/null | wc -l)
  r02=$(squeue -u "$USER" -p GPUFEE02 -h -t RUNNING,PENDING 2>/dev/null | wc -l)
  case "$mode" in
    random)   ARCH_ENV="GID=$arg GENOTYPE_DIR=$GENOTYPE_DIR" ;;
    majority) ARCH_ENV="GENOTYPE_JSON=$GENOTYPE_DIR/majority.json ARM_NAME=majority" ;;
    arm)      ARCH_ENV="GENOTYPE_JSON=$GENOTYPE_DIR/$arg.json ARM_NAME=$arg" ;;
    *)        echo "!! unknown mode $mode" >&2; return 1 ;;
  esac
  for entry in "GPUFEE04 20 $r04" "GPUFEE05 2 $r05" "GPUFEE02 2 $r02"; do
    P=$(echo "$entry" | awk '{print $1}')
    cap=$(echo "$entry" | awk '{print $2}')
    used=$(echo "$entry" | awk '{print $3}')
    [ "$used" -lt "$cap" ] || continue
    out=$(env SUB=$sub $ARCH_ENV TRAIN_SEED=$seed MAX_EPOCHS=$MAX_EPOCHS PATIENCE=$PATIENCE OUTARM=$OUTARM \
      sbatch -p "$P" bin/run_random_genotype_screen_job.sh 2>&1)
    if echo "$out" | grep -q "Submitted batch job"; then
      echo "  -> $P sub=$sub $mode $arg seed=$seed | $out"; return 0
    fi
  done
  return 1
}

while [ -s "$AGENDA" ]; do
  read -r sub mode arg seed < "$AGENDA"
  sed -i '1d' "$AGENDA"
  submit_one "$sub" "$mode" "$arg" "$seed" || { echo "$sub $mode $arg $seed" >> "$AGENDA"; sleep 20; }
done
echo "=== ALL RANDOM-RF BASELINE JOBS SUBMITTED ==="

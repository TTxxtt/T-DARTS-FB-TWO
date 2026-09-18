#!/bin/bash
# Architecture-landscape screening sweep: COUNT sampled genotypes plus the
# Top-1 genotype of each existing DARTS search arm, run on the pre-registered
# subjects and training seeds.  Same quota-aware loop as submit_darts.sh.
#
# The genotype sample is written once, here, before any score exists -- that
# ordering is the pre-registration, so do not re-run the generator against a
# different --geno-seed after looking at results.
set -u

RUNS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$RUNS/.." && pwd)"
cd "$RUNS"

# Pre-registered 0-based subject indices; SUB+1 gives the dataset's 001..009.
# {0,4,8} maps to 001/005/009.
SUBJECTS="${SUBJECTS:-0 4 8}"
SEEDS="${SEEDS:-20250901 20250902 20250903}"
COUNT="${COUNT:-20}"
GENO_SEED="${GENO_SEED:-20250901}"
DARTS_ARMS="${DARTS_ARMS:-darts_200ep darts_lr1e3}"
GENOTYPE_DIR="${GENOTYPE_DIR:-$RUNS/genotypes/random_${COUNT}_seed${GENO_SEED}}"

mkdir -p sh_log/landscape outputs/landscape logs/landscape

AGENDA="$RUNS/sh_log/landscape_agenda.txt"
: > "$AGENDA"
# Subject outermost, so the first 60 jobs complete one subject's whole
# distribution (20 genotypes x 3 seeds) and can be read while the rest run.
for sub in $SUBJECTS; do
  for gid in $(seq 0 $((COUNT - 1))); do
    for seed in $SEEDS; do echo "random $sub $gid $seed" >> "$AGENDA"; done
  done
done
# The searched genotypes last: they are only interpretable as a percentile of
# the sampled distribution, so there is nothing to learn from them earlier.
for sub in $SUBJECTS; do
  for arm in $DARTS_ARMS; do
    for seed in $SEEDS; do echo "search $sub $arm $seed" >> "$AGENDA"; done
  done
done

echo "=== to run: $(wc -l < "$AGENDA") jobs (subjects=$SUBJECTS seeds=$SEEDS) ==="
# Dry mode stops here, before the generator runs: a preview that writes files is
# not a preview, and the genotype sample is the one artifact that must not be
# produced twice.
[ "${1:-run}" = "dry" ] && { cat "$AGENDA"; exit 0; }

# Generate the sample if it is not already on disk.  Re-running this against an
# existing directory is left as an error by the generator itself, which refuses
# to overwrite -- the manifest is the record of what was pre-registered.
if [ ! -f "$GENOTYPE_DIR/manifest.json" ]; then
  echo "=== generating $COUNT genotypes in $GENOTYPE_DIR ==="
  source /gpfs/home/W125221190/anaconda3/bin/activate
  conda activate eeg
  python "$REPO/tools/sample_random_genotypes.py" \
    --count "$COUNT" --seed "$GENO_SEED" --output-dir "$GENOTYPE_DIR" || exit 1
fi

submit_one() {
  local mode=$1 sub=$2 arg=$3 seed=$4
  local r04 r05 r02 entry P cap used out
  r04=$(squeue -u "$USER" -p GPUFEE04 -h -t RUNNING,PENDING 2>/dev/null | wc -l)
  r05=$(squeue -u "$USER" -p GPUFEE05 -h -t RUNNING,PENDING 2>/dev/null | wc -l)
  r02=$(squeue -u "$USER" -p GPUFEE02 -h -t RUNNING,PENDING 2>/dev/null | wc -l)
  for entry in "GPUFEE04 20 $r04" "GPUFEE05 2 $r05" "GPUFEE02 2 $r02"; do
    P=$(echo "$entry" | awk '{print $1}')
    cap=$(echo "$entry" | awk '{print $2}')
    used=$(echo "$entry" | awk '{print $3}')
    [ "$used" -lt "$cap" ] || continue
    if [ "$mode" = "search" ]; then
      out=$(SUB=$sub SEARCH_ARM=$arg TRAIN_SEED=$seed sbatch -p "$P" bin/run_random_genotype_screen_job.sh 2>&1)
    else
      out=$(SUB=$sub GID=$arg TRAIN_SEED=$seed sbatch -p "$P" bin/run_random_genotype_screen_job.sh 2>&1)
    fi
    if echo "$out" | grep -q "Submitted batch job"; then
      echo "  -> $P $mode sub=$sub arg=$arg seed=$seed | $out"; return 0
    fi
  done
  return 1
}

want=30
while [ -s "$AGENDA" ]; do
  run=$(squeue -u "$USER" -o %.8T 2>/dev/null | grep -c RUNNING)
  pend=$(squeue -u "$USER" -o %.8T 2>/dev/null | grep -c PENDING)
  if [ $((run + pend)) -ge $want ]; then sleep 20; continue; fi
  line=$(head -1 "$AGENDA"); sed -i '1d' "$AGENDA"
  # shellcheck disable=SC2086
  submit_one $line || { echo "$line" >> "$AGENDA"; sleep 20; }
done
echo "=== ALL LANDSCAPE SCREENING SUBMITTED ==="

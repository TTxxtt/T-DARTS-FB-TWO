#!/bin/bash
#SBATCH --job-name=randret
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH -o sh_log/randomret/%j.log
#SBATCH -e sh_log/randomret/%j.err
#SBATCH -p GPUFEE04
#SBATCH --constraint="Python"

# Full retrain of ONE pre-registered random genotype: Stage 1 (Session-0
# screening) then Stage 2 (train+val merged, threshold-B stop) and a Session-1
# test read-out -- the same protocol the searched architectures get, so the two
# are directly comparable.
#
# This is run_random_genotype_screen_job.sh minus --screening-only.  That script
# is a Stage-1 *screen* (validation NLL only, no Session-1 read-out), so its
# output cannot be compared against a searched architecture's test accuracy.
# Kept separate rather than parameterised so the existing screen stays
# reproducible from unchanged inputs.

# Slurm copies this script into /var/spool/slurm/<jobid>/ before running it, so
# BASH_SOURCE points at the spool copy and deriving RUNS from it would cd into
# /var/spool.  SLURM_SUBMIT_DIR is the directory sbatch was invoked from, which
# the submit_*.sh drivers set to run/ by cd-ing there first.
RUNS="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO="${TDARTS_REPO:-$(cd "$RUNS/.." && pwd)}"
DATA="${DATA_ROOT:-$REPO/../FBNAS-master-main/data/bci42a/multiviewPython}"
SUB=${SUB:?need SUB (0..8)}
# The SAME training seed the searched arm uses.  Only the architecture differs;
# changing the training seed here would fold seed noise into the comparison the
# random baseline exists to make.
TRAIN_SEED=${TRAIN_SEED:-20250901}
OUTARM=${OUTARM:-random_k5}
# The frozen genotype file, sampled and written before any result existed.
GENOTYPE=${GENOTYPE:?need GENOTYPE (path to a sampled genotype json)}
GENO=$(basename "$GENOTYPE" .json)
MAX_EPOCHS=${MAX_EPOCHS:-1500}
PATIENCE=${PATIENCE:-200}
STAGE2=${STAGE2:-600}
PRELOAD=${PRELOAD:-1}
NUM_WORKERS=${NUM_WORKERS:-0}
SUBJ=$(printf "%03d" $((SUB + 1)))

if [ ! -f "$GENOTYPE" ]; then
  echo "!! genotype not found: $GENOTYPE" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1

# --- cluster-specific environment (edit per site) -----------------------
source /gpfs/apps/gcc/9.1.0/env.sh
source /gpfs/home/W125221190/anaconda3/bin/activate
conda activate eeg
# ------------------------------------------------------------------------

cd "$RUNS"

EXTRA=""
[ "$PRELOAD" = 1 ] && EXTRA="$EXTRA --preload-data"
# STAGE2_MIN=N floors Stage 2's length: the threshold break is not allowed
# before epoch N.  Unset keeps the historical rule.  Only runs that would have
# stopped below N are affected, so a floor is applied by re-running exactly
# those few rather than the whole arm.
STAGE2_MIN=${STAGE2_MIN:-}
[ -n "$STAGE2_MIN" ] && EXTRA="$EXTRA --stage2-min-epochs $STAGE2_MIN"

echo "=== random retrain sub=$SUBJ geno=$GENO seed=$TRAIN_SEED host=$(hostname) $(date) ==="
# --arm "$GENO" keeps the five random architectures of one subject in five
# distinct leaves (train_s003_seed20250901_g000, ...); with --arm "" they would
# all collide on one directory and the second job would be refused by
# mkdir(exist_ok=False).
srun python "$REPO/train_retrain.py" \
  --genotype-json "$GENOTYPE" \
  --data-root "$DATA" \
  --output-root "$RUNS/outputs/$OUTARM" \
  --log-root "$RUNS/logs/$OUTARM" \
  --dataset bci42a \
  --arm "$GENO" \
  --subject "$SUBJ" \
  --seed "$TRAIN_SEED" \
  --initialization random \
  --max-epochs "$MAX_EPOCHS" \
  --patience "$PATIENCE" \
  --stage2-epochs "$STAGE2" \
  --num-workers "$NUM_WORKERS" \
  $EXTRA
echo "=== random retrain sub=$SUBJ geno=$GENO done rc=$? $(date) ==="

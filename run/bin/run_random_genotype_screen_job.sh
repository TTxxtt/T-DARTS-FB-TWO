#!/bin/bash
#SBATCH --job-name=randgeno
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH -o sh_log/rfrandom/%j.log
#SBATCH -e sh_log/rfrandom/%j.err
#SBATCH -p GPUFEE04
#SBATCH --constraint="Python"

# One architecture, one subject, one seed: Stage 1 only, on the FBNAS
# 231/57 Session-0 split.  Deliberately no `set -e`, so the trailing
# `rc=$?` marker is still printed when srun fails -- that line is the only
# record in the log of why a job produced nothing.

# Slurm copies this script into /var/spool/slurm/<jobid>/ before running it, so
# BASH_SOURCE points at that spool copy and deriving RUNS from it would cd into
# /var/spool.  SLURM_SUBMIT_DIR is the directory sbatch was invoked from, which
# the submit_*.sh drivers set to run/ by cd-ing there first.
RUNS="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO="${TDARTS_REPO:-$(cd "$RUNS/.." && pwd)}"
DATA="${DATA_ROOT:-$REPO/../FBNAS-master-main/data/bci42a/multiviewPython}"
SUB=${SUB:?need SUB (0..8)}
TRAIN_SEED=${TRAIN_SEED:-20250901}
MAX_EPOCHS=${MAX_EPOCHS:-1500}
PATIENCE=${PATIENCE:-200}
OUTARM=${OUTARM:-landscape}
PRELOAD=${PRELOAD:-1}
# The splits are 231/57, so forked workers are pure overhead once preloading
# removes the I/O they would have overlapped.
NUM_WORKERS=${NUM_WORKERS:-0}

# SUB is a 0-based index into the nine BCI-IV-2a subjects and the dataset spells
# them 001..009, so the lookup name needs the +1 offset.  Without it SUB=0 asks
# for a nonexistent "Subject000" and subject 009 is never submitted at all.
SUBJ=$(printf "%03d" $((SUB + 1)))

# The architecture under test is named one of three ways, and all go through
# the identical screening protocol below so that their scores are directly
# comparable:
#   GENOTYPE_JSON=<path> an explicit genotype file (e.g. the majority-vote
#                        genotype); ARM_NAME names the run leaf
#   SEARCH_ARM=<arm>     decode the Top-1 genotype of an existing search run
#   (default)            take a sampled genotype from GENOTYPE_DIR/g<GID>.json
if [ -n "${GENOTYPE_JSON:-}" ]; then
  if [ ! -f "$GENOTYPE_JSON" ]; then
    echo "!! genotype not found: $GENOTYPE_JSON" >&2
    exit 1
  fi
  ARM=${ARM_NAME:?need ARM_NAME when GENOTYPE_JSON is set}
  ARCH_ARGS="--genotype-json $GENOTYPE_JSON"
elif [ -n "${SEARCH_ARM:-}" ]; then
  SEARCH_DIR=${SEARCH_DIR:-$RUNS/outputs/$SEARCH_ARM/bci42a/search_s${SUBJ}_seed${TRAIN_SEED}}
  if [ ! -f "$SEARCH_DIR/config.json" ]; then
    echo "!! search run not found: $SEARCH_DIR" >&2
    exit 1
  fi
  ARCH_ARGS="--search-dir $SEARCH_DIR"
  ARM="$SEARCH_ARM"
else
  GID=${GID:?need GID (0..19 for the default set)}
  GENOTYPE_DIR=${GENOTYPE_DIR:-$RUNS/genotypes/random_20_seed20250901}
  GNAME=$(printf "g%03d" "$GID")
  GENOTYPE_JSON="$GENOTYPE_DIR/$GNAME.json"
  if [ ! -f "$GENOTYPE_JSON" ]; then
    echo "!! genotype not found: $GENOTYPE_JSON" >&2
    exit 1
  fi
  ARCH_ARGS="--genotype-json $GENOTYPE_JSON"
  ARM="$GNAME"
fi

EXTRA=""
[ "$PRELOAD" = 1 ] && EXTRA="$EXTRA --preload-data"

# --- cluster-specific environment (kept consistent with the existing jobs) --
source /gpfs/apps/gcc/9.1.0/env.sh
source /gpfs/home/W125221190/anaconda3/bin/activate
conda activate eeg
# ---------------------------------------------------------------------------

cd "$RUNS"
echo "=== screen sub=$SUBJ arm=$ARM seed=$TRAIN_SEED host=$(hostname) $(date) ==="
echo "=== preload=$PRELOAD num_workers=$NUM_WORKERS ==="
# --screening-only is what keeps Session 1 closed: the file is never opened, so
# the test set cannot leak into architecture selection no matter what the
# aggregation downstream chooses to do.  It also skips Stage 2, leaving Stage
# 1's validation-best epoch as the screening score.
srun python "$REPO/train_retrain.py" \
  $ARCH_ARGS \
  --data-root "$DATA" \
  --output-root "$RUNS/outputs/$OUTARM" \
  --log-root "$RUNS/logs/$OUTARM" \
  --dataset bci42a \
  --arm "$ARM" \
  --subject "$SUBJ" \
  --seed "$TRAIN_SEED" \
  --initialization random \
  --max-epochs "$MAX_EPOCHS" \
  --patience "$PATIENCE" \
  --num-workers "$NUM_WORKERS" \
  --screening-only \
  $EXTRA
echo "=== screen sub=$SUBJ arm=$ARM done rc=$? $(date) ==="

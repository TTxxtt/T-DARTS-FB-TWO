#!/bin/bash
#SBATCH --job-name=dartsret
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH -o sh_log/darts/%j.log
#SBATCH -e sh_log/darts/%j.err
#SBATCH -p GPUFEE04
#SBATCH --constraint="Python"

# Slurm runs the spool copy under /var/spool/slurm/<jobid>/, not this file, so
# BASH_SOURCE cannot be used to find the repository.  SLURM_SUBMIT_DIR is where
# sbatch was invoked (the submit_*.sh drivers cd to run/ first).
RUNS="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO="${TDARTS_REPO:-$(cd "$RUNS/.." && pwd)}"
DATA="${DATA_ROOT:-$REPO/../FBNAS-master-main/data/bci42a/multiviewPython}"
SUB=${SUB:?need SUB (0..8)}
SEED=${SEED:-20250901}
# See run_darts_job.sh: SUB is a 0-based subject index and the dataset names
# subjects 001..009, so the directory/lookup name needs the +1 offset.
SUBJ=$(printf "%03d" $((SUB + 1)))

# OUTARM is where this retrain writes; SEARCH_ARM is where it reads the search
# from.  They default to the same tree, but separating them lets a retrain be
# run against an archived search (e.g. the 200-epoch one kept as evidence that
# alpha does not converge at that budget).
OUTARM=${OUTARM:-darts}
SEARCH_ARM=${SEARCH_ARM:-$OUTARM}
SEARCH_DIR="${SEARCH_DIR:-$RUNS/outputs/$SEARCH_ARM/bci42a/search_s${SUBJ}_seed${SEED}}"

export PYTHONUNBUFFERED=1

# --- cluster-specific environment (edit per site) -----------------------
source /gpfs/apps/gcc/9.1.0/env.sh
source /gpfs/home/W125221190/anaconda3/bin/activate
conda activate eeg
# ------------------------------------------------------------------------

cd "$RUNS"

# Fail fast rather than after a GPU has been allocated: this job reads artefacts
# the search job must have written first, and they do not exist until it reaches
# its final epoch.
if [ ! -f "$SEARCH_DIR/config.json" ] || [ ! -f "$SEARCH_DIR/metrics.jsonl" ]; then
  echo "!! $SEARCH_DIR has no config.json/metrics.jsonl -- has the search finished?" >&2
  exit 1
fi

# --- C-version defaults (set any of these to 0 in the environment to opt out) --
# OBSERVE_TEST=1  log Session-1 accuracy every epoch.  Observation only: no
#                 stop, checkpoint or selection decision reads it, and the
#                 reported test number is still the single post-training
#                 evaluation in final_summary.json -- never a peak off this curve.
# PRELOAD=1       hold the trials in memory instead of re-unpickling them on
#                 every pass.  Numerically identical, but a training epoch makes
#                 three passes over the split, so without it the run re-reads
#                 ~650 MB of GPFS per epoch (measured ~0.7 s of the 1.62 s epoch).
OBSERVE_TEST=${OBSERVE_TEST:-1}
PRELOAD=${PRELOAD:-1}
# With PRELOAD there is no I/O left to overlap, and the splits are tiny
# (231/57/288), so forked workers are pure overhead -- the upstream baseline
# uses num_workers=0 for the same reason.
NUM_WORKERS=${NUM_WORKERS:-0}

EXTRA=""
[ "$OBSERVE_TEST" = 1 ] && EXTRA="$EXTRA --observe-test"
[ "$PRELOAD" = 1 ] && EXTRA="$EXTRA --preload-data"
# STAGE2_FIXED=N runs Stage 2 for exactly N epochs and disables the
# val_nll < stage-1-terminal-train_nll early stop, which otherwise ends a
# subject's Stage 2 as soon as its validation NLL dips below the Stage-1
# threshold (s009 stopped at epoch 9 of a 600-epoch budget).  Unset keeps the
# historical rule.  A fixed-epoch run is OFF-PROTOCOL: its Session-1 reading is
# comparable only to another fixed-epoch run, never to the threshold-stopped
# archive.
STAGE2_FIXED=${STAGE2_FIXED:-}
[ -n "$STAGE2_FIXED" ] && EXTRA="$EXTRA --stage2-fixed-epochs $STAGE2_FIXED"
# STAGE2_MIN=N floors Stage 2's length: the threshold break is not allowed
# before epoch N.  Unlike STAGE2_FIXED the run still stops on the threshold, so
# it stays on the same rule as the archive -- only the handful of runs that
# stopped after a few epochs are extended.  Give one of the two, not both.
STAGE2_MIN=${STAGE2_MIN:-}
[ -n "$STAGE2_MIN" ] && EXTRA="$EXTRA --stage2-min-epochs $STAGE2_MIN"

echo "=== DARTS retrain sub=$SUBJ seed=$SEED host=$(hostname) $(date) ==="
echo "=== observe_test=$OBSERVE_TEST preload=$PRELOAD num_workers=$NUM_WORKERS stage2_fixed=${STAGE2_FIXED:-none} ==="
# --arm "" keeps the leaf as the plain train_s<subject>_seed<seed>, matching the
# search leaf beside it.
srun python "$REPO/train_retrain.py" \
  --search-dir "$SEARCH_DIR" \
  --data-root "$DATA" \
  --output-root "$RUNS/outputs/$OUTARM" \
  --log-root "$RUNS/logs/$OUTARM" \
  --dataset bci42a \
  --arm "" \
  --subject "$SUBJ" \
  --seed "$SEED" \
  --initialization random \
  --num-workers "$NUM_WORKERS" \
  $EXTRA
echo "=== DARTS retrain sub=$SUBJ done rc=$? $(date) ==="

#!/bin/bash
#SBATCH --job-name=dartsfvret
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH -o sh_log/fullval/%j.log
#SBATCH -e sh_log/fullval/%j.err
#SBATCH -p GPUFEE04
#SBATCH --constraint="Python"

# Retrain the genotype a fullval search exported.  This is run_darts_retrain_job.sh
# with one deliberate difference: it reads --genotype-json, never --search-dir.
#
# With --search-dir, train_retrain.py re-derives the architecture from the last
# logged epoch of metrics.jsonl (train_retrain.py:255 calls extract_genotype at
# epoch=search_epochs).  That would silently discard the EMA genotype and
# retrain the last-epoch argmax instead -- the two differ exactly when the
# experiment is interesting, so the failure would be invisible.  Reading the
# exported file makes which genotype gets retrained explicit in the job.
#
# Kept as a separate script rather than an option on the original so the
# already-published DARTS retrains stay reproducible from unchanged inputs.

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

OUTARM=${OUTARM:-darts_fullval}
SEARCH_ARM=${SEARCH_ARM:-$OUTARM}
SEARCH_DIR="${SEARCH_DIR:-$RUNS/outputs/$SEARCH_ARM/bci42a/search_s${SUBJ}_seed${SEED}}"
# genotype_ema.json is the point of the arm, but genotype_last.json is written
# by the same search and is retrainable with GENOTYPE=genotype_last.json, which
# is what makes "same search, different decoding" a clean comparison.
GENOTYPE=${GENOTYPE:-$SEARCH_DIR/genotype_ema.json}

export PYTHONUNBUFFERED=1

# --- cluster-specific environment (edit per site) -----------------------
source /gpfs/apps/gcc/9.1.0/env.sh
source /gpfs/home/W125221190/anaconda3/bin/activate
conda activate eeg
# ------------------------------------------------------------------------

cd "$RUNS"

# Fail fast rather than after a GPU has been allocated: this job reads an
# artefact the search job must have written first, and it does not exist until
# that job reaches its final epoch.
if [ ! -f "$SEARCH_DIR/config.json" ] || [ ! -f "$SEARCH_DIR/metrics.jsonl" ]; then
  echo "!! $SEARCH_DIR has no config.json/metrics.jsonl -- has the search finished?" >&2
  exit 1
fi
if [ ! -f "$GENOTYPE" ]; then
  echo "!! genotype not found: $GENOTYPE" >&2
  echo "   (a search that ran before --decode-mode was added writes genotype.json only)" >&2
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

echo "=== fullval retrain sub=$SUBJ seed=$SEED host=$(hostname) $(date) ==="
echo "=== genotype=$(basename "$GENOTYPE") observe_test=$OBSERVE_TEST preload=$PRELOAD num_workers=$NUM_WORKERS ==="
# --arm "" keeps the leaf as the plain train_s<subject>_seed<seed>, matching the
# search leaf beside it.
srun python "$REPO/train_retrain.py" \
  --genotype-json "$GENOTYPE" \
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
echo "=== fullval retrain sub=$SUBJ done rc=$? $(date) ==="

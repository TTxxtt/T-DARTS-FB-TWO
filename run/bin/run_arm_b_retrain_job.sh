#!/bin/bash
#SBATCH --job-name=armBtrain
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH -o sh_log/armBtrain/%j.log
#SBATCH -e sh_log/armBtrain/%j.err
#SBATCH -p GPUFEE04
#SBATCH --constraint="Python"

# Arm B's Stage 1 + Stage 2, on the architecture the search froze.
#
# Same two-stage FBNAS protocol the official Arm A chain runs, through the same
# implementation every other arm here uses (`train_retrain.py`):
#
#   Stage 1  Session-0 231/57, early stop on validation accuracy
#            (max 1500 epochs, patience 200) -- upstream's
#            `bestVarToCheck: valInacc` / `NoDecrease: numEpochs 200`.
#            The default --best-metric is used deliberately: the frozen repo
#            arms pass val_nll, but Arm A is the *upstream* chain and it stops
#            on accuracy, so matching it means using the default.
#   Stage 2  restore the best Stage-1 state, continue on all 288 Session-0
#            trials, stop when validation NLL falls below Stage 1's terminal
#            train NLL (upstream `continueAfterEarlystop=True`), cap 600.
#   Test     one Session-2 evaluation at the end.
#
# --observe-test is ON: Session 2's metrics are logged every epoch for
# observation only.  No stop, checkpoint or selection decision reads them --
# `evaluate` is no_grad and the Stage-2 threshold reads validation NLL.
#
# This is the ONLY Arm B job that reads the dataset's second session.

RUNS="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO="${TDARTS_REPO:-$(cd "$RUNS/.." && pwd)}"
DATA="${DATA_ROOT:-$REPO/../FBNAS-master-main/data/bci42a/multiviewPython}"
SUB=${SUB:?need SUB (0..8)}
SEED=${SEED:-20190821}
OUTARM=${OUTARM:-operator_armB_retrain}
SEARCH_OUTARM=${SEARCH_OUTARM:-operator_armB}
MAX_EPOCHS=${MAX_EPOCHS:-1500}
PATIENCE=${PATIENCE:-200}
STAGE2_EPOCHS=${STAGE2_EPOCHS:-600}
# Defaults to the value the protocol froze; it exists so a rerun can be pinned
# to the same setting rather than silently drifting with a new default.
BEST_METRIC=${BEST_METRIC:-val_inacc}
OBSERVE_TEST=${OBSERVE_TEST:-1}
PRELOAD=${PRELOAD:-1}
NUM_WORKERS=${NUM_WORKERS:-0}

SUBJ=$(printf "%03d" $((SUB + 1)))
SEARCH_DIR="${SEARCH_DIR:-$RUNS/outputs/$SEARCH_OUTARM/bci42a/operator_armB_search_s${SUBJ}_seed${SEED}}"
GENOTYPE_JSON="$SEARCH_DIR/genotype.json"

if [ ! -f "$GENOTYPE_JSON" ]; then
  echo "!! $GENOTYPE_JSON not found -- has the Arm B search for this subject finished?" >&2
  exit 1
fi
# The architecture must be frozen before the second session is opened, so check
# that what we are about to retrain is a finished search and not a half-written
# file from a job still running.
if [ ! -f "$SEARCH_DIR/final_summary.json" ]; then
  echo "!! $SEARCH_DIR/final_summary.json missing -- the search has not finished" >&2
  exit 1
fi
if ! grep -q '"band_family_rf"' "$GENOTYPE_JSON"; then
  echo "!! $GENOTYPE_JSON is not a band-dialect genotype (scheme band_family_rf)" >&2
  exit 1
fi

EXTRA=""
[ "$PRELOAD" = 1 ] && EXTRA="$EXTRA --preload-data"
[ "$OBSERVE_TEST" = 1 ] && EXTRA="$EXTRA --observe-test"

# --- cluster-specific environment (kept consistent with the existing jobs) --
source /gpfs/apps/gcc/9.1.0/env.sh
source /gpfs/home/W125221190/anaconda3/bin/activate
conda activate eeg
# ---------------------------------------------------------------------------

cd "$RUNS"
echo "=== armB retrain sub=$SUBJ seed=$SEED host=$(hostname) $(date) ==="
echo "=== genotype=$GENOTYPE_JSON observe_test=$OBSERVE_TEST best_metric=$BEST_METRIC ==="
echo "=== session2 WILL be read once at the end of this run ==="
srun python "$REPO/train_retrain.py" \
  --genotype-json "$GENOTYPE_JSON" \
  --data-root "$DATA" \
  --output-root "$RUNS/outputs/$OUTARM" \
  --log-root "$RUNS/logs/$OUTARM" \
  --dataset bci42a \
  --arm "armB" \
  --subject "$SUBJ" \
  --seed "$SEED" \
  --initialization random \
  --max-epochs "$MAX_EPOCHS" \
  --patience "$PATIENCE" \
  --best-metric "$BEST_METRIC" \
  --stage2-epochs "$STAGE2_EPOCHS" \
  --num-workers "$NUM_WORKERS" \
  $EXTRA
rc=$?
echo "=== armB retrain sub=$SUBJ rc=$rc $(date) ==="
exit $rc

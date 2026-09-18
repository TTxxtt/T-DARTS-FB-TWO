#!/bin/bash
#SBATCH --job-name=rfretrain
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH -o sh_log/rfretrain/%j.log
#SBATCH -e sh_log/rfretrain/%j.err
#SBATCH -p GPUFEE04
#SBATCH --constraint="Python"

# Retrain one searched RF-only genotype from scratch, with the same screening
# protocol as the random-genotype landscape runs: Stage 1 on the 231/57
# Session-0 split, Session 1 never opened, validation-best state reported with
# accuracy/NLL/kappa/macro-F1.  The genotype comes from the RF search job's
# genotype.json, and --no-duplicate-paths is re-checked here so a search that
# somehow exported a duplicate pair fails before taking a GPU.

RUNS="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO="${TDARTS_REPO:-$(cd "$RUNS/.." && pwd)}"
DATA="${DATA_ROOT:-$REPO/../FBNAS-master-main/data/bci42a/multiviewPython}"
SUB=${SUB:?need SUB (0..8)}
SEED=${SEED:-20250901}
MAX_EPOCHS=${MAX_EPOCHS:-1500}
PATIENCE=${PATIENCE:-200}
# Selection metric for the best checkpoint.  The frozen comparison uses
# BEST_METRIC=val_nll for every arm; the default preserves old runs.
BEST_METRIC=${BEST_METRIC:-val_inacc}
SEARCH_ARM=${SEARCH_ARM:-rfsearch}
OUTARM=${OUTARM:-rfretrain}
PRELOAD=${PRELOAD:-1}
NUM_WORKERS=${NUM_WORKERS:-0}

SUBJ=$(printf "%03d" $((SUB + 1)))
SEARCH_DIR="${SEARCH_DIR:-$RUNS/outputs/$SEARCH_ARM/bci42a/search_s${SUBJ}_seed${SEED}}"
GENOTYPE_JSON="$SEARCH_DIR/genotype.json"

if [ ! -f "$GENOTYPE_JSON" ]; then
  echo "!! $GENOTYPE_JSON not found -- has the RF search for this subject/seed finished?" >&2
  exit 1
fi

EXTRA=""
[ "$PRELOAD" = 1 ] && EXTRA="$EXTRA --preload-data"

# --- cluster-specific environment (kept consistent with the existing jobs) --
source /gpfs/apps/gcc/9.1.0/env.sh
source /gpfs/home/W125221190/anaconda3/bin/activate
conda activate eeg
# ---------------------------------------------------------------------------

cd "$RUNS"
echo "=== RF retrain sub=$SUBJ seed=$SEED host=$(hostname) $(date) ==="
echo "=== genotype=$GENOTYPE_JSON max_epochs=$MAX_EPOCHS patience=$PATIENCE ==="
srun python "$REPO/train_retrain.py" \
  --genotype-json "$GENOTYPE_JSON" \
  --data-root "$DATA" \
  --output-root "$RUNS/outputs/$OUTARM" \
  --log-root "$RUNS/logs/$OUTARM" \
  --dataset bci42a \
  --arm "rf" \
  --subject "$SUBJ" \
  --seed "$SEED" \
  --initialization random \
  --max-epochs "$MAX_EPOCHS" \
  --patience "$PATIENCE" \
  --best-metric "$BEST_METRIC" \
  --num-workers "$NUM_WORKERS" \
  --screening-only \
  --no-duplicate-paths \
  $EXTRA
echo "=== RF retrain sub=$SUBJ done rc=$? $(date) ==="

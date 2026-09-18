#!/bin/bash
#SBATCH --job-name=hier_retrain
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH -o sh_log/hier_retrain/%j.log
#SBATCH -e sh_log/hier_retrain/%j.err
#SBATCH -p GPUFEE04
#SBATCH --constraint="Python"

# From-scratch retrain of one hierarchical genotype under the frozen protocol:
# max_epochs=1500, patience=200, best checkpoint by validation NLL (the rule
# every arm of the final comparison uses), screening-only so Session 1 stays
# closed, validation-best read-out with accuracy/NLL/kappa/macro-F1.

RUNS="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO="${TDARTS_REPO:-$(cd "$RUNS/.." && pwd)}"
DATA="${DATA_ROOT:-$REPO/../FBNAS-master-main/data/bci42a/multiviewPython}"
SUB=${SUB:?need SUB (0..8)}
SEED=${SEED:-20250901}
MAX_EPOCHS=${MAX_EPOCHS:-1500}
PATIENCE=${PATIENCE:-200}
BEST_METRIC=${BEST_METRIC:-val_nll}
SEARCH_ARM=${SEARCH_ARM:-hier}
OUTARM=${OUTARM:-hier_retrain}
PRELOAD=${PRELOAD:-1}
NUM_WORKERS=${NUM_WORKERS:-0}

SUBJ=$(printf "%03d" $((SUB + 1)))
SEARCH_DIR="${SEARCH_DIR:-$RUNS/outputs/$SEARCH_ARM/bci42a/hier_search_s${SUBJ}_seed${SEED}}"
GENOTYPE_JSON="$SEARCH_DIR/genotype.json"

if [ ! -f "$GENOTYPE_JSON" ]; then
  echo "!! $GENOTYPE_JSON not found -- has the hierarchical search for this seed finished?" >&2
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
echo "=== hier retrain sub=$SUBJ seed=$SEED host=$(hostname) $(date) ==="
echo "=== genotype=$GENOTYPE_JSON max_epochs=$MAX_EPOCHS patience=$PATIENCE best_metric=$BEST_METRIC ==="
srun python "$REPO/train_retrain.py" \
  --genotype-json "$GENOTYPE_JSON" \
  --data-root "$DATA" \
  --output-root "$RUNS/outputs/$OUTARM" \
  --log-root "$RUNS/logs/$OUTARM" \
  --dataset bci42a \
  --arm "hier" \
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
echo "=== hier retrain sub=$SUBJ done rc=$? $(date) ==="

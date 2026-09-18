#!/bin/bash
#SBATCH --job-name=bandop
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH -o sh_log/bandop/%j.log
#SBATCH -e sh_log/bandop/%j.err
#SBATCH -p GPUFEE04
#SBATCH --constraint="Python"

# One fixed-RF57 single-band operator probe configuration: path 0 dilated_rf57
# everywhere, path 1 varied in one band only.  Screening-only, Session 1 never
# opened, 50 epochs for every configuration.  --no-duplicate-paths is
# deliberately absent: baseline_dd has two identical structures by design.

RUNS="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO="${TDARTS_REPO:-$(cd "$RUNS/.." && pwd)}"
DATA="${DATA_ROOT:-$REPO/../FBNAS-master-main/data/bci42a/multiviewPython}"
SUB=${SUB:?need SUB (0..8)}
ARM_NAME=${ARM_NAME:?need ARM_NAME}
GENOTYPE_JSON=${GENOTYPE_JSON:?need GENOTYPE_JSON}
TRAIN_SEED=${TRAIN_SEED:-20250901}
MAX_EPOCHS=${MAX_EPOCHS:-50}
PATIENCE=${PATIENCE:-50}
OUTARM=${OUTARM:-band_op}
PRELOAD=${PRELOAD:-1}
NUM_WORKERS=${NUM_WORKERS:-0}

SUBJ=$(printf "%03d" $((SUB + 1)))
if [ ! -f "$GENOTYPE_JSON" ]; then
  echo "!! genotype not found: $GENOTYPE_JSON" >&2
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
echo "=== band-op probe sub=$SUBJ arm=$ARM_NAME seed=$TRAIN_SEED $(date) ==="
srun python "$REPO/train_retrain.py" \
  --genotype-json "$GENOTYPE_JSON" \
  --data-root "$DATA" \
  --output-root "$RUNS/outputs/$OUTARM" \
  --log-root "$RUNS/logs/$OUTARM" \
  --dataset bci42a \
  --arm "$ARM_NAME" \
  --subject "$SUBJ" \
  --seed "$TRAIN_SEED" \
  --initialization random \
  --max-epochs "$MAX_EPOCHS" \
  --patience "$PATIENCE" \
  --num-workers "$NUM_WORKERS" \
  --screening-only \
  $EXTRA
echo "=== band-op probe sub=$SUBJ arm=$ARM_NAME done rc=$? $(date) ==="

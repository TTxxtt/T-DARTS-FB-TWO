#!/bin/bash
#SBATCH --job-name=anchor
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH -o sh_log/anchor/%j.log
#SBATCH -e sh_log/anchor/%j.err
#SBATCH -p GPUFEE04
#SBATCH --constraint="Python"

# Full two-phase anchored search on Session 0:
#   phase A  path 0 fixed to dilated, path 1 searches the four operator
#            families with both paths on one synchronised RF (29/57/113,
#            RF 15 excluded because the families collide there);
#   phase B  phase A's operators are frozen and each path searches the full
#            FBNAS RF ladder 15/29/57/113.  With NO_DUP=1 the two RFs of a
#            band are decoded jointly so a band never exports two identical
#            structures.
# Session 1 is never read; the job writes a genotype.json for the retrain.

RUNS="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO="${TDARTS_REPO:-$(cd "$RUNS/.." && pwd)}"
DATA="${DATA_ROOT:-$REPO/../FBNAS-master-main/data/bci42a/multiviewPython}"
SUB=${SUB:?need SUB (0..8)}
SEED=${SEED:-20250901}
OP_EPOCHS=${OP_EPOCHS:-200}
RF_EPOCHS=${RF_EPOCHS:-200}
ALPHA_LR=${ALPHA_LR:-3e-4}
OUTARM=${OUTARM:-anchored}
NO_DUP=${NO_DUP:-1}
INHERIT=${INHERIT:-0}
PRELOAD=${PRELOAD:-1}
NUM_WORKERS=${NUM_WORKERS:-0}

SUBJ=$(printf "%03d" $((SUB + 1)))

export PYTHONUNBUFFERED=1

# --- cluster-specific environment (kept consistent with the other jobs) ----
source /gpfs/apps/gcc/9.1.0/env.sh
source /gpfs/home/W125221190/anaconda3/bin/activate
conda activate eeg
# ---------------------------------------------------------------------------

cd "$RUNS"

EXTRA=""
[ "$NO_DUP" = 1 ] && EXTRA="$EXTRA --no-duplicate-paths"
[ "$INHERIT" = 1 ] && EXTRA="$EXTRA --inherit-weights"
[ "$PRELOAD" = 1 ] && EXTRA="$EXTRA --preload-data"

echo "=== anchored search sub=$SUBJ seed=$SEED host=$(hostname) $(date) ==="
echo "=== op_epochs=$OP_EPOCHS rf_epochs=$RF_EPOCHS alpha_lr=$ALPHA_LR no_dup=$NO_DUP inherit=$INHERIT ==="
srun python "$REPO/train_anchor_search.py" \
  --data-root "$DATA" \
  --output-root "$RUNS/outputs/$OUTARM" \
  --log-root "$RUNS/logs/$OUTARM" \
  --dataset bci42a \
  --arm "" \
  --subject "$SUBJ" \
  --seed "$SEED" \
  --operator-epochs "$OP_EPOCHS" \
  --rf-epochs "$RF_EPOCHS" \
  --warmup-epochs 20 \
  --alpha-lr "$ALPHA_LR" \
  --num-workers "$NUM_WORKERS" \
  $EXTRA
echo "=== anchored search sub=$SUBJ done rc=$? $(date) ==="

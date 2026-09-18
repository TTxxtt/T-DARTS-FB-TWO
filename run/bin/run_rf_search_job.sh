#!/bin/bash
#SBATCH --job-name=rfsearch
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH -o sh_log/rfsearch/%j.log
#SBATCH -e sh_log/rfsearch/%j.err
#SBATCH -p GPUFEE04
#SBATCH --constraint="Python"

# Phase B: RF-only search on Session 0.  Both paths are fixed to `dilated`;
# only the receptive field (15/29/57/113) of each path is searched.  Session 1
# is never read -- the job writes a genotype, and the retrain stage consumes it.

RUNS="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO="${TDARTS_REPO:-$(cd "$RUNS/.." && pwd)}"
DATA="${DATA_ROOT:-$REPO/../FBNAS-master-main/data/bci42a/multiviewPython}"
SUB=${SUB:?need SUB (0..8)}
SEED=${SEED:-20250901}
EPOCHS=${EPOCHS:-200}
ALPHA_LR=${ALPHA_LR:-3e-4}
OUTARM=${OUTARM:-rfsearch}
# The main RF experiment forbids a band's two paths selecting the same RF:
# with one fixed operator, equal RF means two identical filters and no
# multi-scale structure.
NO_DUP=${NO_DUP:-1}
# One train pass plus three validation passes per epoch, and the splits are
# 231/57, so preloading removes the GPFS re-reads and forked workers are pure
# overhead.
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
[ "$PRELOAD" = 1 ] && EXTRA="$EXTRA --preload-data"

echo "=== RF search sub=$SUBJ seed=$SEED host=$(hostname) $(date) ==="
echo "=== epochs=$EPOCHS alpha_lr=$ALPHA_LR no_dup=$NO_DUP preload=$PRELOAD ==="
srun python "$REPO/train_rf_search.py" \
  --data-root "$DATA" \
  --output-root "$RUNS/outputs/$OUTARM" \
  --log-root "$RUNS/logs/$OUTARM" \
  --dataset bci42a \
  --arm "" \
  --subject "$SUBJ" \
  --seed "$SEED" \
  --epochs "$EPOCHS" \
  --warmup-epochs 20 \
  --alpha-lr "$ALPHA_LR" \
  --num-workers "$NUM_WORKERS" \
  $EXTRA
echo "=== RF search sub=$SUBJ done rc=$? $(date) ==="

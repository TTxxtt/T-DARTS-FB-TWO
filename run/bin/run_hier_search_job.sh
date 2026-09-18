#!/bin/bash
#SBATCH --job-name=hier
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH -o sh_log/hier/%j.log
#SBATCH -e sh_log/hier/%j.err
#SBATCH -p GPUFEE04
#SBATCH --constraint="Python"

# Frozen hierarchical method: hard operator search at fixed RF57 (Gumbel
# one-hot, annealed temperature), then RF search over the frozen operators,
# then export this seed's own genotype.  Session 1 is never read and no
# cross-seed majority vote is formed.

RUNS="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO="${TDARTS_REPO:-$(cd "$RUNS/.." && pwd)}"
DATA="${DATA_ROOT:-$REPO/../FBNAS-master-main/data/bci42a/multiviewPython}"
SUB=${SUB:?need SUB (0..8)}
SEED=${SEED:-20250901}
PHASE_A_EPOCHS=${PHASE_A_EPOCHS:-300}
PHASE_B_EPOCHS=${PHASE_B_EPOCHS:-200}
WARMUP_A=${WARMUP_A:-30}
WARMUP_B=${WARMUP_B:-20}
ALPHA_LR=${ALPHA_LR:-3e-4}
TAU_START=${TAU_START:-1.0}
TAU_END=${TAU_END:-0.3}
NO_DUP=${NO_DUP:-1}
OUTARM=${OUTARM:-hier}
PRELOAD=${PRELOAD:-1}
NUM_WORKERS=${NUM_WORKERS:-0}

SUBJ=$(printf "%03d" $((SUB + 1)))

export PYTHONUNBUFFERED=1

# --- cluster-specific environment (kept consistent with the existing jobs) --
source /gpfs/apps/gcc/9.1.0/env.sh
source /gpfs/home/W125221190/anaconda3/bin/activate
conda activate eeg
# ---------------------------------------------------------------------------

cd "$RUNS"

EXTRA=""
[ "$NO_DUP" = 1 ] && EXTRA="$EXTRA --no-duplicate-paths"
[ "$PRELOAD" = 1 ] && EXTRA="$EXTRA --preload-data"

echo "=== hierarchical search sub=$SUBJ seed=$SEED host=$(hostname) $(date) ==="
echo "=== phaseA=$PHASE_A_EPOCHS(warmup $WARMUP_A, tau $TAU_START->$TAU_END) phaseB=$PHASE_B_EPOCHS(warmup $WARMUP_B) no_dup=$NO_DUP ==="
srun python "$REPO/train_hierarchical_search.py" \
  --data-root "$DATA" \
  --output-root "$RUNS/outputs/$OUTARM" \
  --log-root "$RUNS/logs/$OUTARM" \
  --dataset bci42a \
  --arm "" \
  --subject "$SUBJ" \
  --seed "$SEED" \
  --phase-a-epochs "$PHASE_A_EPOCHS" \
  --phase-b-epochs "$PHASE_B_EPOCHS" \
  --warmup-a "$WARMUP_A" \
  --warmup-b "$WARMUP_B" \
  --alpha-lr "$ALPHA_LR" \
  --tau-start "$TAU_START" \
  --tau-end "$TAU_END" \
  --num-workers "$NUM_WORKERS" \
  $EXTRA
echo "=== hierarchical search sub=$SUBJ done rc=$? $(date) ==="

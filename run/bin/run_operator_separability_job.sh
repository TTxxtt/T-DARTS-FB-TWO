#!/bin/bash
#SBATCH --job-name=opsep
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH -o sh_log/opsep/%j.log
#SBATCH -e sh_log/opsep/%j.err
#SBATCH -p GPUFEE04
#SBATCH --constraint="Python"

# One path-1 operator, one subject, one seed.  Path 0 is dilated_rf57 in every
# band; OP selects path 1 (dilated|normal|dwsep|lkdw), all at RF 57.  The four
# arms share every other knob, so a score difference is attributable to the
# operator and nothing else.  Deliberately no `--no-duplicate-paths`: the
# dilated+dilated arm is the control with two identical paths.

RUNS="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO="${TDARTS_REPO:-$(cd "$RUNS/.." && pwd)}"
DATA="${DATA_ROOT:-$REPO/../FBNAS-master-main/data/bci42a/multiviewPython}"
SUB=${SUB:?need SUB (0..8)}
OP=${OP:?need OP (dilated|normal|dwsep|lkdw)}
TRAIN_SEED=${TRAIN_SEED:-20250901}
# Short, identical budget for all four arms; the point is a ranking, not a
# converged accuracy.  PATIENCE=MAX_EPOCHS disables early stopping so every arm
# receives exactly the same number of updates.
MAX_EPOCHS=${MAX_EPOCHS:-50}
PATIENCE=${PATIENCE:-50}
GENOTYPE_DIR=${GENOTYPE_DIR:-$RUNS/genotypes/opsep_rf57}
OUTARM=${OUTARM:-opsep_rf57}
PRELOAD=${PRELOAD:-1}
NUM_WORKERS=${NUM_WORKERS:-0}

SUBJ=$(printf "%03d" $((SUB + 1)))
GENOTYPE_JSON="$GENOTYPE_DIR/path1_${OP}_rf57.json"
if [ ! -f "$GENOTYPE_JSON" ]; then
  echo "!! genotype not found: $GENOTYPE_JSON" >&2
  echo "!! run tools/make_operator_separability_genotypes.py first" >&2
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
echo "=== opsep sub=$SUBJ op=$OP seed=$TRAIN_SEED host=$(hostname) $(date) ==="
echo "=== max_epochs=$MAX_EPOCHS patience=$PATIENCE preload=$PRELOAD ==="
# --screening-only keeps Session 1 closed and skips Stage 2; the read-out is
# Stage 1's validation-best state (accuracy/NLL/kappa) on the shared 231/57
# split.  --arm keeps each operator's leaf distinct without a separate root.
srun python "$REPO/train_retrain.py" \
  --genotype-json "$GENOTYPE_JSON" \
  --data-root "$DATA" \
  --output-root "$RUNS/outputs/$OUTARM" \
  --log-root "$RUNS/logs/$OUTARM" \
  --dataset bci42a \
  --arm "$OP" \
  --subject "$SUBJ" \
  --seed "$TRAIN_SEED" \
  --initialization random \
  --max-epochs "$MAX_EPOCHS" \
  --patience "$PATIENCE" \
  --num-workers "$NUM_WORKERS" \
  --screening-only \
  $EXTRA
echo "=== opsep sub=$SUBJ op=$OP done rc=$? $(date) ==="

#!/bin/bash
#SBATCH --job-name=fixed_retrain
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH -o sh_log/fixeddil/%j.log
#SBATCH -e sh_log/fixeddil/%j.err
#SBATCH -p GPUFEE04
#SBATCH --constraint="Python"

# Stage-1 screening retrain (from scratch) of ONE fixed dilated-only genotype
# from the per-subject architecture table (run/genotypes/fixeddil/).  Same
# frozen protocol as hier_retrain: max_epochs=1500, patience=200, best by
# validation NLL, screening-only so Session 1 stays closed.  Stage 2 runs
# separately via --stage2-only (run_stage2_job.sh).

RUNS="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO="${TDARTS_REPO:-$(cd "$RUNS/.." && pwd)}"
DATA="${DATA_ROOT:-$REPO/../FBNAS-master-main/data/bci42a/multiviewPython}"
SUB=${SUB:?need SUB (0..8)}
SEED=${SEED:-20190821}
OUTARM="${OUTARM:-fixeddil}"
GENOTYPE_JSON="${GENOTYPE_JSON:-$RUNS/genotypes/fixeddil/s$(printf "%03d" $((SUB+1))).json}"
MAX_EPOCHS=${MAX_EPOCHS:-1500}
PATIENCE=${PATIENCE:-200}
PRELOAD=${PRELOAD:-1}
NUM_WORKERS=${NUM_WORKERS:-0}

SUBJ=$(printf "%03d" $((SUB + 1)))
[ -f "$GENOTYPE_JSON" ] || { echo "!! $GENOTYPE_JSON not found" >&2; exit 1; }
EXTRA=""
[ "$PRELOAD" = 1 ] && EXTRA="$EXTRA --preload-data"

source /gpfs/apps/gcc/9.1.0/env.sh
source /gpfs/home/W125221190/anaconda3/bin/activate
conda activate eeg

cd "$RUNS"
echo "=== fixeddil stage1 sub=$SUBJ seed=$SEED host=$(hostname) $(date) ==="
echo "=== genotype=$GENOTYPE_JSON max_epochs=$MAX_EPOCHS patience=$PATIENCE best_metric=val_nll ==="
srun python "$REPO/train_retrain.py" \
  --genotype-json "$GENOTYPE_JSON" \
  --data-root "$DATA" \
  --output-root "$RUNS/outputs/$OUTARM" \
  --log-root "$RUNS/logs/$OUTARM" \
  --dataset bci42a \
  --arm "fixeddil" \
  --subject "$SUBJ" \
  --seed "$SEED" \
  --initialization random \
  --max-epochs "$MAX_EPOCHS" \
  --patience "$PATIENCE" \
  --best-metric val_nll \
  --num-workers "$NUM_WORKERS" \
  --screening-only \
  --no-duplicate-paths \
  $EXTRA
echo "=== fixeddil stage1 sub=$SUBJ rc=$? $(date) ==="
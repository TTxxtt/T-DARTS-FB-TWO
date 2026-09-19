#!/bin/bash
#SBATCH --job-name=armBsearch
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH -o sh_log/armBsearch/%j.log
#SBATCH -e sh_log/armBsearch/%j.err
#SBATCH -p GPUFEE04
#SBATCH --constraint="Python"

# Arm B's search: four-mechanism hierarchical, FBNAS random-subnet sampling.
#
#   Phase A  four temporal families at RF57, single path  -> 4^3  = 64 candidates
#   Phase B  the frozen family over the RF ladder, 1 or 2  -> 10^3 = 1000 candidates
#
# Architecture is chosen by calibrated validation accuracy, matching upstream
# NAS.nas_phase, so the only thing that differs from the dilated-only arm is the
# search space itself.
#
# Session 1 (the dataset's SECOND recording session, code session=1) is never
# read here.  This job exports an architecture; the retrain job is what opens
# the second session, and only after the architecture is frozen.

RUNS="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO="${TDARTS_REPO:-$(cd "$RUNS/.." && pwd)}"
DATA="${DATA_ROOT:-$REPO/../FBNAS-master-main/data/bci42a/multiviewPython}"
SUB=${SUB:?need SUB (0..8)}
SEED=${SEED:-20190821}
OUTARM=${OUTARM:-operator_armB}
PHASE_A_EPOCHS=${PHASE_A_EPOCHS:-200}
PHASE_B_EPOCHS=${PHASE_B_EPOCHS:-200}
PRELOAD=${PRELOAD:-1}
NUM_WORKERS=${NUM_WORKERS:-0}

SUBJ=$(printf "%03d" $((SUB + 1)))

EXTRA=""
[ "$PRELOAD" = 1 ] && EXTRA="$EXTRA --preload-data"

# --- cluster-specific environment (kept consistent with the existing jobs) --
source /gpfs/apps/gcc/9.1.0/env.sh
source /gpfs/home/W125221190/anaconda3/bin/activate
conda activate eeg
# ---------------------------------------------------------------------------

cd "$RUNS"
echo "=== armB search sub=$SUBJ seed=$SEED host=$(hostname) $(date) ==="
echo "=== phase_a_epochs=$PHASE_A_EPOCHS phase_b_epochs=$PHASE_B_EPOCHS outarm=$OUTARM ==="
srun python "$REPO/train_arm_b_search.py" \
  --data-root "$DATA" \
  --output-root "$RUNS/outputs/$OUTARM" \
  --log-root "$RUNS/logs/$OUTARM" \
  --dataset bci42a \
  --arm "operator_armB" \
  --subject "$SUBJ" \
  --seed "$SEED" \
  --phase-a-epochs "$PHASE_A_EPOCHS" \
  --phase-b-epochs "$PHASE_B_EPOCHS" \
  --num-workers "$NUM_WORKERS" \
  $EXTRA
rc=$?
echo "=== armB search sub=$SUBJ rc=$rc $(date) ==="
exit $rc

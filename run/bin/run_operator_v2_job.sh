#!/bin/bash
#SBATCH --job-name=opv2
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH -o sh_log/opv2/%j.log
#SBATCH -e sh_log/opv2/%j.err
#SBATCH -p GPUFEE04
#SBATCH --constraint="Python"

# One Tier-1 global-family run: a single operator family is used for ALL THREE
# bands, and only the family varies across arms.  Every other knob -- split,
# seed, RF, budget, stopping rule -- is identical, so a score difference is
# attributable to the family and nothing else.
#
# Session 1 stays closed: --read-session1 is deliberately absent.  The script
# would have to be edited to open it, and the entry guard in
# train_operator_v2.py is what the test suite watches.
#
# This is *not* the old opsep probe.  run_operator_separability_job.sh drives
# train_retrain.py with fixed path-1 genotypes (dilated|normal|dwsep|lkdw);
# this one drives train_operator_v2.py with the five V2 families and a single
# 12-channel path per band.  The two answer different questions.

RUNS="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO="${TDARTS_REPO:-$(cd "$RUNS/.." && pwd)}"
DATA="${DATA_ROOT:-$REPO/../FBNAS-master-main/data/bci42a/multiviewPython}"
SUB=${SUB:?need SUB 0-based subject index, SUB+1 gives the dataset id}
OP=${OP:?need OP one of dilated gated local_attention dynamic band_gated}
TRAIN_SEED=${TRAIN_SEED:-20250901}
RF=${RF:-57}
# Defaults mirror train_operator_v2.py; override only to change the protocol.
MAX_EPOCHS=${MAX_EPOCHS:-1500}
PATIENCE=${PATIENCE:-200}
OUTARM=${OUTARM:-operator_v2}
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
echo "=== opv2 sub=$SUBJ op=$OP seed=$TRAIN_SEED rf=$RF host=$(hostname) $(date) ==="
echo "=== max_epochs=$MAX_EPOCHS patience=$PATIENCE preload=$PRELOAD session1=closed ==="
# --device cuda is passed explicitly: train_operator_v2.py raises if CUDA was
# asked for and is unavailable, so a broken allocation fails loudly instead of
# silently running the whole budget on CPU.
#
# --output-root is absolute on purpose.  The script's own default is the
# relative "outputs", which would land under whichever directory the job
# happened to start in rather than in the run/ tree the rest of the tooling
# reads.  --arm is left at its default: it resolves to <arm>_<operator>, which
# is what keeps the five families in five distinct leaves.
srun python "$REPO/train_operator_v2.py" \
  --operator "$OP" \
  --subject "$SUBJ" \
  --seed "$TRAIN_SEED" \
  --rf "$RF" \
  --data-root "$DATA" \
  --output-root "$RUNS/outputs/$OUTARM" \
  --log-root "$RUNS/logs/$OUTARM" \
  --dataset bci42a \
  --epochs "$MAX_EPOCHS" \
  --patience "$PATIENCE" \
  --device cuda \
  --num-workers "$NUM_WORKERS" \
  $EXTRA
echo "=== opv2 sub=$SUBJ op=$OP done rc=$? $(date) ==="

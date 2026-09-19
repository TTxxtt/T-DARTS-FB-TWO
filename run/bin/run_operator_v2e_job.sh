#!/bin/bash
#SBATCH --job-name=opv2e
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH -o sh_log/opv2e/%j.log
#SBATCH -e sh_log/opv2e/%j.err
#SBATCH -p GPUFEE04
#SBATCH --constraint="Python"

# One Expressive-V2 global-family run: a single operator family is used for ALL
# THREE bands and only the family varies across arms.  Every other knob -- split,
# seed, RF, budget, stopping rule -- is identical to the Matched generation's, so
# a difference between generations is attributable to the families and not to
# the protocol.
#
# Session 1 stays closed: --read-session1 is deliberately absent, and the entry
# point's guard is what the test suite watches.
#
# This is a *copy* of run_operator_v2_job.sh rather than a parameterisation of
# it.  The Matched job script is the provenance record for the 45 runs already
# on disk under run/outputs/operator_v2/, and its ledger has to stay idempotent;
# sharing one script would also mean sharing the OUTARM default, which is
# exactly the mistake that would let an Expressive run land in the frozen tree.

RUNS="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO="${TDARTS_REPO:-$(cd "$RUNS/.." && pwd)}"
DATA="${DATA_ROOT:-$REPO/../FBNAS-master-main/data/bci42a/multiviewPython}"
SUB=${SUB:?need SUB 0-based subject index, SUB+1 gives the dataset id}
TRAIN_SEED=${TRAIN_SEED:-20250901}
RF=${RF:-57}
# Defaults mirror train_operator_v2e.py; override only to change the protocol.
MAX_EPOCHS=${MAX_EPOCHS:-1500}
PATIENCE=${PATIENCE:-200}
OUTARM=${OUTARM:-operator_v2e}
PRELOAD=${PRELOAD:-1}
NUM_WORKERS=${NUM_WORKERS:-0}

# Exactly one target: a pilot candidate, or a capacity control.  The controls
# are ablation arms and are not reachable through --operator, so launching one
# takes a deliberate CAPACITY_CONTROL= here.  Requiring one-or-the-other keeps
# the two from being confused by an inherited environment variable.
OP=${OP:-}
CAPACITY_CONTROL=${CAPACITY_CONTROL:-}
if [ -n "$OP" ] && [ -n "$CAPACITY_CONTROL" ]; then
  echo "!! set OP (a pilot candidate) or CAPACITY_CONTROL (an ablation), not both" >&2
  exit 1
fi
if [ -z "$OP" ] && [ -z "$CAPACITY_CONTROL" ]; then
  echo "!! need OP one of dilated_e gated_e local_attention_e dynamic_e band_gated_e," >&2
  echo "!! or CAPACITY_CONTROL one of wide_dilated_2p5_e wide_dilated_5_e" >&2
  exit 1
fi
if [ -n "$CAPACITY_CONTROL" ]; then
  TARGET_FLAG="--capacity-control $CAPACITY_CONTROL"
  TARGET_NAME="$CAPACITY_CONTROL"
  ROLE="capacity_control"
else
  TARGET_FLAG="--operator $OP"
  TARGET_NAME="$OP"
  ROLE="candidate"
fi

SUBJ=$(printf "%03d" $((SUB + 1)))

EXTRA=""
[ "$PRELOAD" = 1 ] && EXTRA="$EXTRA --preload-data"

# --- cluster-specific environment (kept consistent with the existing jobs) --
source /gpfs/apps/gcc/9.1.0/env.sh
source /gpfs/home/W125221190/anaconda3/bin/activate
conda activate eeg
# ---------------------------------------------------------------------------

cd "$RUNS"
echo "=== opv2e sub=$SUBJ target=$TARGET_NAME role=$ROLE seed=$TRAIN_SEED rf=$RF host=$(hostname) $(date) ==="
echo "=== max_epochs=$MAX_EPOCHS patience=$PATIENCE preload=$PRELOAD session1=closed ==="
# --device cuda is passed explicitly so a broken allocation fails loudly instead
# of silently running the whole budget on CPU.
#
# --output-root is absolute on purpose: the entry point's own default is the
# relative "outputs", which would land under whatever directory the job started
# in rather than in the run/ tree the tooling reads.
# shellcheck disable=SC2086
srun python "$REPO/train_operator_v2e.py" \
  $TARGET_FLAG \
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
echo "=== opv2e sub=$SUBJ target=$TARGET_NAME done rc=$? $(date) ==="

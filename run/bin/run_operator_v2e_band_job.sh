#!/bin/bash
#SBATCH --job-name=opv2eband
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH -o sh_log/opv2eband/%j.log
#SBATCH -e sh_log/opv2eband/%j.err
#SBATCH -p GPUFEE04
#SBATCH --constraint="Python"

# One band-specific mechanism probe run: exactly ONE band replaces the anchor
# and the other two stay on dilated_e.  Everything else -- split, seed, RF,
# budget, stopping rule -- is identical to both global grids, so a difference is
# attributable to the one band that moved.
#
# Session 1 stays closed: the flag that opens it is deliberately absent from the
# srun call below, and the entry point's own default is what the test suite
# watches -- this script has no variable that could expand into it.
#
# A copy rather than a parameterisation of run_operator_v2e_job.sh, for the same
# reason that script is a copy of the Matched one: the earlier job scripts are
# the provenance record for their archived runs, and sharing one would mean
# sharing an OUTARM default -- which is exactly how a run lands in the wrong
# frozen tree.

RUNS="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO="${TDARTS_REPO:-$(cd "$RUNS/.." && pwd)}"
DATA="${DATA_ROOT:-$REPO/../FBNAS-master-main/data/bci42a/multiviewPython}"
SUB=${SUB:?need SUB 0-based subject index, SUB+1 gives the dataset id}
TRAIN_SEED=${TRAIN_SEED:-20250901}
RF=${RF:-57}
# Defaults mirror train_operator_v2e_band.py; override only to change the protocol.
MAX_EPOCHS=${MAX_EPOCHS:-1500}
PATIENCE=${PATIENCE:-200}
OUTARM=${OUTARM:-operator_v2e_band}
PRELOAD=${PRELOAD:-1}
NUM_WORKERS=${NUM_WORKERS:-0}

# The varying band and the family placed in it.  The other two bands are the
# anchor by construction -- the entry point refuses two varying bands, and this
# script cannot express one in the first place.
BAND=${BAND:-}
FAMILY=${FAMILY:-}
case "$BAND" in
  low)   LOW="$FAMILY"; MID=dilated_e; HIGH=dilated_e ;;
  mid)   LOW=dilated_e; MID="$FAMILY"; HIGH=dilated_e ;;
  high)  LOW=dilated_e; MID=dilated_e; HIGH="$FAMILY" ;;
  *)
    echo "!! need BAND one of low mid high, got '${BAND}'" >&2
    exit 1
    ;;
esac
case "$FAMILY" in
  dynamic_e|gated_e|band_gated_e) ;;
  *)
    echo "!! need FAMILY one of dynamic_e gated_e band_gated_e, got '${FAMILY}'" >&2
    echo "!! local_attention_e is out of scope for this stage, and dilated_e is" >&2
    echo "!! the anchor -- replacing it with itself is the baseline, not a probe" >&2
    exit 1
    ;;
esac

SUBJ=$(printf "%03d" $((SUB + 1)))

EXTRA=""
[ "$PRELOAD" = 1 ] && EXTRA="$EXTRA --preload-data"

# --- cluster-specific environment (kept consistent with the existing jobs) --
source /gpfs/apps/gcc/9.1.0/env.sh
source /gpfs/home/W125221190/anaconda3/bin/activate
conda activate eeg
# ---------------------------------------------------------------------------

cd "$RUNS"
echo "=== opv2eband sub=$SUBJ band=$BAND family=$FAMILY seed=$TRAIN_SEED rf=$RF host=$(hostname) $(date) ==="
echo "=== low=$LOW mid=$MID high=$HIGH max_epochs=$MAX_EPOCHS patience=$PATIENCE session1=closed ==="
# --device cuda is passed explicitly so a broken allocation fails loudly instead
# of silently running the whole budget on CPU.
#
# --output-root is absolute on purpose: the entry point's own default is the
# relative "outputs", which would land under whatever directory the job started
# in rather than in the run/ tree the tooling reads.
srun python "$REPO/train_operator_v2e_band.py" \
  --low "$LOW" \
  --mid "$MID" \
  --high "$HIGH" \
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
echo "=== opv2eband sub=$SUBJ band=$BAND family=$FAMILY done rc=$? $(date) ==="

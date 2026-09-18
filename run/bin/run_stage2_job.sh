#!/bin/bash
#SBATCH --job-name=stage2
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH -o sh_log/stage2/%j.log
#SBATCH -e sh_log/stage2/%j.err
#SBATCH -p GPUFEE04
#SBATCH --constraint="Python"

# Stage 2 of the frozen protocol for ONE run, resumed from its finished Stage-1
# summary.  Session 0 (train+val) is re-trained from the Stage-1 best.pt while
# recording the first epoch each of TWO stopping thresholds is crossed, and the
# Session-1 test is read out under both:
#   A = Stage-1 validation-best NLL (historic threshold here)
#   B = Stage-1 terminal training loss (baseModel.py:417-421, faithful FBNAS)
# --stage2-only therefore reuses existing probe runs without re-running Stage 1.
#
# ARM selects which existing arm's probe to resume:
#   hier -> OUTARM=hier_retrain, run_dir ..._hier, genotype in hier search
#   rf   -> OUTARM=rf_retrain_nll, run_dir ..._rf,  genotype in rf search

# RUNS (= run/) must come from SLURM_SUBMIT_DIR, not from BASH_SOURCE: sbatch
# executes a spool copy of this script, so "${BASH_SOURCE[0]}" points at
# /var/spool/slurm/... and dirname would resolve RUNS to the spool dir.  The
# submitter therefore must `cd run/` before sbatch so SLURM_SUBMIT_DIR is run/.
RUNS="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO="${TDARTS_REPO:-$(cd "$RUNS/.." && pwd)}"
DATA="${DATA_ROOT:-$REPO/../FBNAS-master-main/data/bci42a/multiviewPython}"
SUB=${SUB:?need SUB (0..8)}
SEED=${SEED:-20250901}
ARM=${ARM:?need ARM (arm label, e.g. hier|rf|fixeddil)}
OUTARM="${OUTARM:-$ARM}"
STAGE2_EPOCHS=${STAGE2_EPOCHS:-600}
PRELOAD=${PRELOAD:-1}
NUM_WORKERS=${NUM_WORKERS:-0}

SUBJ=$(printf "%03d" $((SUB + 1)))
if [ -n "${GENOTYPE_JSON:-}" ]; then
  # Caller supplied the genotype marker explicitly (any arm).
  :
elif [ "$ARM" = "rf" ]; then
  GENOTYPE_JSON="$RUNS/outputs/rfsearch/bci42a/search_s${SUBJ}_seed${SEED}/genotype.json"
else
  GENOTYPE_JSON="$RUNS/outputs/hier/bci42a/hier_search_s${SUBJ}_seed${SEED}/genotype.json"
fi
RUN_DIR="$RUNS/outputs/$OUTARM/bci42a/train_s${SUBJ}_seed${SEED}_${ARM}"

# Preflight: the Stage-1 probe must exist, and so must the genotype marker.
[ -f "$RUN_DIR/final_summary.json" ] || { echo "!! $RUN_DIR has no Stage-1 summary" >&2; exit 1; }
[ -f "$RUN_DIR/best.pt" ] || { echo "!! $RUN_DIR has no best.pt" >&2; exit 1; }
[ -f "$GENOTYPE_JSON" ] || { echo "!! $GENOTYPE_JSON not found (used only to satisfy --stage2-only's path guard)" >&2; exit 1; }

EXTRA=""
[ "$PRELOAD" = 1 ] && EXTRA="$EXTRA --preload-data"

# --- cluster-specific environment (kept consistent with the existing jobs) --
source /gpfs/apps/gcc/9.1.0/env.sh
source /gpfs/home/W125221190/anaconda3/bin/activate
conda activate eeg
# ---------------------------------------------------------------------------

cd "$RUNS"
echo "=== stage2 arm=$ARM sub=$SUBJ seed=$SEED host=$(hostname) $(date) ==="
echo "=== run_dir=$RUN_DIR stage2_epochs=$STAGE2_EPOCHS ==="
srun python "$REPO/train_retrain.py" \
  --genotype-json "$GENOTYPE_JSON" \
  --data-root "$DATA" \
  --output-root "$RUNS/outputs/$OUTARM" \
  --log-root "$RUNS/logs/$OUTARM" \
  --dataset bci42a \
  --arm "$ARM" \
  --subject "$SUBJ" \
  --seed "$SEED" \
  --stage2-only \
  --stage2-epochs "$STAGE2_EPOCHS" \
  --batch-size 16 \
  --num-workers "$NUM_WORKERS" \
  $EXTRA
echo "=== stage2 arm=$ARM sub=$SUBJ rc=$? $(date) ==="
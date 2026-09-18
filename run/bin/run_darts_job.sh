#!/bin/bash
#SBATCH --job-name=darts
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH -o sh_log/darts/%j.log
#SBATCH -e sh_log/darts/%j.err
#SBATCH -p GPUFEE04
#SBATCH --constraint="Python"

# Slurm runs the spool copy under /var/spool/slurm/<jobid>/, not this file, so
# BASH_SOURCE cannot be used to find the repository.  SLURM_SUBMIT_DIR is where
# sbatch was invoked (the submit_*.sh drivers cd to run/ first).
RUNS="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO="${TDARTS_REPO:-$(cd "$RUNS/.." && pwd)}"
DATA="${DATA_ROOT:-$REPO/../FBNAS-master-main/data/bci42a/multiviewPython}"
SUB=${SUB:?need SUB (0..8)}
SEED=${SEED:-20250901}
# 200 epochs leaves alpha ~98% uniform: 15 alpha steps/epoch * 180 epochs *
# 3e-4 = 0.81 total logit displacement, while the argmax needs O(10) to be
# decisive.  DARTS on CIFAR spends ~26000 alpha steps; 1500 epochs here gives
# ~22000, which is the same order.
EPOCHS=${EPOCHS:-200}
OUTARM=${OUTARM:-darts}
# Architecture learning rate.  3e-4 (the DARTS default) needs ~22000 alpha
# steps to move the logits far enough apart to be decisive; 1e-3 gets there in
# a third of the budget.  Running both at the same epoch count makes the pair a
# clean ablation of the architecture step size alone.
ALPHA_LR=${ALPHA_LR:-3e-4}
# Where the architecture gradient comes from.  minibatch is the original DARTS
# schedule -- one validation batch per weight step, ~15 Adam steps per epoch --
# and stays the default so every archived search reproduces exactly.  fullval
# instead takes one accumulated step over the whole 57-trial validation set per
# epoch.  The step-frequency change is part of the method, not a knob to offset
# against ALPHA_LR.
ALPHA_UPDATE_MODE=${ALPHA_UPDATE_MODE:-minibatch}
# Which genotype genotype.json points at.  last is the historical behaviour (the
# final epoch's argmax); ema decodes the argmax of the averaged probabilities.
# The search writes genotype_last.json and genotype_ema.json either way, so this
# decides only which one genotype.json aliases -- both stay retrainable.
DECODE_MODE=${DECODE_MODE:-last}
EMA_DECAY=${EMA_DECAY:-0.9}
# SUB is a 0-based index into the nine BCI-IV-2a subjects, matching the
# subTorun index run_fbnas_subject.py uses, so both arms take the same SUB.
# The dataset spells its subjects 001..009, hence the +1: SUB=0 is subject 001
# and SUB=8 is subject 009.  Without the offset SUB=0 asked for a nonexistent
# "Subject000" and subject 009 was never submitted at all.
SUBJ=$(printf "%03d" $((SUB + 1)))

export PYTHONUNBUFFERED=1

# --- cluster-specific environment (edit per site) -----------------------
source /gpfs/apps/gcc/9.1.0/env.sh
source /gpfs/home/W125221190/anaconda3/bin/activate
conda activate eeg
# ------------------------------------------------------------------------

cd "$RUNS"
echo "=== DARTS sub=$SUBJ seed=$SEED host=$(hostname) $(date) ==="
# Running the script by absolute path puts $REPO on sys.path[0], so
# `from tdarts...` resolves with no PYTHONPATH and no install.
#
# --arm "" leaves the leaf as the plain search_s<subject>_seed<seed>: the arm is
# already spelled out by the run/outputs/darts root above.  tdarts/genotype.py
# recovers the seed from that leaf, so it has to keep carrying "seed<digits>".
srun python "$REPO/train_search.py" \
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
  --alpha-update-mode "$ALPHA_UPDATE_MODE" \
  --decode-mode "$DECODE_MODE" \
  --ema-decay "$EMA_DECAY"
echo "=== DARTS sub=$SUBJ done rc=$? $(date) ==="

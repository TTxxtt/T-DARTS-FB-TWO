#!/bin/bash
#SBATCH --job-name=fbnas
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH -o sh_log/fbnas/%j.log
#SBATCH -e sh_log/fbnas/%j.err
#SBATCH -p GPUFEE04
#SBATCH --constraint="Python"
# NOTE: no --time/--mem/--qos/--array -- the Prolog rejects/reaps them, and a
# job with null Features is scancelled outright.  --constraint="Python" is
# MANDATORY (see FBNAS-master-main/*.err).

# Slurm copies this script into /var/spool/slurm/<jobid>/ before running it, so
# BASH_SOURCE points at that spool copy and deriving RUNS from it would cd into
# /var/spool.  SLURM_SUBMIT_DIR is the directory sbatch was invoked from, which
# the submit_*.sh drivers set to run/ by cd-ing there first.  The BASH_SOURCE
# fallback only applies when the script is run directly rather than via sbatch.
RUNS="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO="${TDARTS_REPO:-$(cd "$RUNS/.." && pwd)}"
export TDARTS_REPO="$REPO"

export PYTHONUNBUFFERED=1
# Belt and braces: the wrapper sets sys.dont_write_bytecode too, which is what
# actually keeps bytecode out of the frozen FBNAS/ baseline.
export PYTHONDONTWRITEBYTECODE=1

# --- cluster-specific environment (edit per site) -----------------------
source /gpfs/apps/gcc/9.1.0/env.sh
source /gpfs/home/W125221190/anaconda3/bin/activate
conda activate eeg
# ------------------------------------------------------------------------

cd "$RUNS"
SUB=${SUB:?need SUB (0..8)}
echo "=== FBNAS sub=$SUB host=$(hostname) $(date) ==="
srun python py/run_fbnas_subject.py
echo "=== FBNAS sub=$SUB done rc=$? $(date) ==="

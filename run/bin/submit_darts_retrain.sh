#!/bin/bash
# Submit nine DARTS retrain runs, one subject each.  Run this only after
# submit_darts.sh has finished -- each job reads the search artefacts of its own
# subject and exits immediately if they are not there yet.
set -u

RUNS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RUNS"
mkdir -p sh_log/fbnas sh_log/darts outputs/fbnas outputs/darts logs/fbnas logs/darts

SEED="${SEED:-20250901}"
AGENDA="$RUNS/sh_log/darts_retrain_agenda.txt"
: > "$AGENDA"
for i in $(seq 0 8); do echo "$i" >> "$AGENDA"; done
echo "=== to run: $(wc -l < "$AGENDA") (SEED=$SEED) ==="
[ "${1:-run}" = "dry" ] && { cat "$AGENDA"; exit 0; }

submit_one() {
  local sub=$1
  local r04 r05 r02
  r04=$(squeue -u "$USER" -p GPUFEE04 -h -t RUNNING,PENDING 2>/dev/null | wc -l)
  r05=$(squeue -u "$USER" -p GPUFEE05 -h -t RUNNING,PENDING 2>/dev/null | wc -l)
  r02=$(squeue -u "$USER" -p GPUFEE02 -h -t RUNNING,PENDING 2>/dev/null | wc -l)
  local entry P cap used out
  for entry in "GPUFEE04 20 $r04" "GPUFEE05 2 $r05" "GPUFEE02 2 $r02"; do
    P=$(echo "$entry" | awk '{print $1}')
    cap=$(echo "$entry" | awk '{print $2}')
    used=$(echo "$entry" | awk '{print $3}')
    [ "$used" -lt "$cap" ] || continue
    out=$(SUB=$sub SEED="$SEED" sbatch -p "$P" bin/run_darts_retrain_job.sh 2>&1)
    if echo "$out" | grep -q "Submitted batch job"; then
      echo "  -> $P sub=$sub | $out"; return 0
    fi
  done
  return 1
}

want=30
while [ -s "$AGENDA" ]; do
  run=$(squeue -u "$USER" -o %.8T 2>/dev/null | grep -c RUNNING)
  pend=$(squeue -u "$USER" -o %.8T 2>/dev/null | grep -c PENDING)
  if [ $((run + pend)) -ge $want ]; then sleep 20; continue; fi
  line=$(head -1 "$AGENDA"); sed -i '1d' "$AGENDA"
  submit_one "$line" || { echo "$line" >> "$AGENDA"; sleep 20; }
done
echo "=== ALL DARTS RETRAIN SUBMITTED ==="

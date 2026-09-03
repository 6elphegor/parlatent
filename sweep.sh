#!/usr/bin/env bash
# Muon-LR sweep for the looped model and the depth-matched unrolled model.
# Runs PAR jobs concurrently on one GPU (models are tiny).  Usage: ./sweep.sh [PAR]
set -uo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
PAR=${1:-5}
STEPS=${STEPS:-2500}
LRS=${LRS:-"0.005 0.01 0.02 0.04 0.08"}
COMMON="--n_repeats 4 --steps $STEPS --warmup 200 --eval_interval 500 --batch_size 64 --compile"
jobs=()
for lr in $LRS; do
  jobs+=("loop4_lr$lr  --n_layers 4  --n_loops 4 --muon_lr $lr")
  jobs+=("deep16_lr$lr --n_layers 16 --n_loops 1 --muon_lr $lr")
done
run_one() {
  name=$1; shift
  mkdir -p runs/$name
  $PY train.py --name $name $COMMON "$@" > runs/$name/train.log 2>&1
  echo "done $name exit=$?"
}
export -f run_one; export PY COMMON
printf '%s\n' "${jobs[@]}" | xargs -P "$PAR" -I{} bash -c 'run_one {}'
echo SWEEP_DONE

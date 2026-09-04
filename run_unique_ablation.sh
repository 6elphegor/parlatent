#!/usr/bin/env bash
# Looped vs depth-matched unrolled, on puzzles with unique solutions (data/unique).
# Pair A (4L x 4 loops vs 16L) runs concurrently, then pair B (8L x 8 loops vs 64L).
set -uo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
COMMON="--muon_lr 0.01 --steps ${STEPS:-5000} --warmup 200 --eval_interval 500 --batch_size 64 --compile"
run() { name=$1; shift; mkdir -p runs/$name; $PY train.py --name $name $COMMON "$@" > runs/$name/train.log 2>&1; echo "done $name exit=$?"; }
run u_loop4  --n_layers 4  --n_loops 4 --n_repeats 4 &
run u_deep16 --n_layers 16 --n_loops 1 --n_repeats 4 &
wait
run u_loop8  --n_layers 8  --n_loops 8 --n_repeats 8 &
run u_deep64 --n_layers 64 --n_loops 1 --n_repeats 8 &
wait
echo ABLATION_DONE

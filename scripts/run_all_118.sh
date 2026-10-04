#!/bin/bash
# IEEE 118 experiments used in docs/RESEARCH.md (two jobs at a time, 2 LP workers each, on 4 cores)
cd "$(dirname "$0")/.."
mkdir -p results/case118
f() { grep --line-buffered -v Highs; }
(python3 scripts/run_model1.py --cfg case118 --epochs 100 --rl_steps 300 --workers 2 2>&1 | f > results/case118/model1.log
 python3 scripts/run_label_free.py --cfg case118 --steps 600 --workers 2 2>&1 | f > results/case118/label_free.log) &
(python3 scripts/run_value.py --cfg case118 --n_train 400 --epochs 60 --workers 2 2>&1 | f > results/case118/value.log
 python3 scripts/run_model2.py --cfg case118 --epochs 60 --n_train 800 --workers 2 2>&1 | f > results/case118/model2.log) &
wait

#!/bin/bash
# UC experiments, started as soon as their datasets exist (run detached: setsid nohup ...)
cd "$(dirname "$0")/.."
f() { grep --line-buffered -v Highs; }
mkdir -p results/uc1 results/uc12
until [ -f data/generated/uc1/test.npz ]; do sleep 30; done
python3 scripts/uc_model1.py --cfg uc1 --epochs 60 --rl_steps 200 --workers 2 --n_fix 200 2>&1 | f > results/uc1/model1.log
python3 scripts/uc_model2.py --cfg uc1 --epochs 60 --workers 2 2>&1 | f > results/uc1/model2.log
until [ -f data/generated/uc12/test.npz ]; do sleep 60; done
python3 scripts/uc_model1.py --cfg uc12 --epochs 80 --rl_steps 120 --workers 3 --n_fix 60 2>&1 | f > results/uc12/model1.log

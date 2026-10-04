#!/bin/bash
# B2 label-free Model 1, started when the B2 Model 1 run has finished (it provides the MILP-label reference rows).
cd "$(dirname "$0")/.."
until [ -f results/uc12/uc_model1_results.json ]; do sleep 60; done
python3 scripts/uc_label_free.py --cfg uc12 --epochs 80 --rl_steps 120 --scratch_steps 200 --workers 2 2>&1 \
  | grep --line-buffered -v Highs > results/uc12/label_free.log

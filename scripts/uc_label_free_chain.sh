#!/bin/bash
# Label-free Model 1 on B1, started after the B1 fixing study (CPU budget).
cd "$(dirname "$0")/.."
until [ -f results/uc1/uc_fixing_results.json ]; do sleep 60; done
python3 scripts/uc_label_free.py --cfg uc1 --epochs 60 --rl_steps 200 --scratch_steps 400 --workers 2 2>&1 \
  | grep --line-buffered -v Highs > results/uc1/label_free.log

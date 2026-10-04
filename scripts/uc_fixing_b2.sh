#!/bin/bash
# B2 fixing study, started when the B2 Model 1 run has saved its weights and results.
cd "$(dirname "$0")/.."
until [ -f results/uc12/uc_model1_results.json ]; do sleep 60; done
python3 scripts/uc_fixing.py --cfg uc12 --n_fix 60 --ratios 0.8,0.9,0.95 2>&1 | grep --line-buffered -v Highs > results/uc12/fixing.log

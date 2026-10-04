#!/bin/bash
# Full data generation used for the results in docs/RESEARCH.md (HiGHS MILPs: 2 workers is fastest on 4 cores)
set -e
W=${W:-2}
g() { python3 scripts/gen_data.py --workers $W "$@" 2>&1 | grep --line-buffered -v Highs; }
g --cfg case30     --split train --n 1500 --seed 1
g --cfg case30     --split val   --n 200  --seed 2
g --cfg case30     --split test  --n 400  --seed 3 --n_alt 2
g --cfg case30_raw --split train --n 1500 --seed 1
g --cfg case30_raw --split test  --n 400  --seed 3 --n_alt 2
g --cfg case118     --split train --n 800 --seed 1
g --cfg case118     --split val   --n 120 --seed 2
g --cfg case118     --split test  --n 200 --seed 3 --n_alt 2
g --cfg case118_raw --split test  --n 100 --seed 3 --n_alt 2

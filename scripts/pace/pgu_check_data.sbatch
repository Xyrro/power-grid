#!/usr/bin/env bash
# Verify the generated training set before any GPU job uses it, and measure the peak memory of loading it, so the
# training jobs' --mem requests are based on a measurement rather than a guess. Loading 50,000 instances is far too
# heavy for a login node (it gets killed there), hence a batch job.
#
#   python3 scripts/pace/pgu_guard.py submit --kind cpu --name checkdata --cpus 2 --mem 32G \
#       --time 00:30:00 scripts/pace/pgu_check_data.sbatch
#SBATCH -N 1 --ntasks=1
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/pace/trm_env.sh"
python3 - <<'PY'
import resource
import sys

import numpy as np

sys.path.insert(0, ".")
from otsl import trm

DIRS = ["data/generated/uc12_m1x", "data/generated/recursive/pace"]
tr = trm.train_data(50000, m1x=DIRS)
n = len(tr["load"])
print(f"instances loaded: {n}")
bad = 0
for k in ("load", "avail", "u_rel", "lmp_rel", "flow_rel", "sr", "u0", "y", "c_lab", "c_rel"):
    a = np.asarray(tr[k]).astype(float)
    nan, inf = int(np.isnan(a).sum()), int(np.isinf(a).sum())
    bad += nan + inf
    print(f"  {k:9s} {str(a.shape):20s} nan={nan} inf={inf}")

y = np.asarray(tr["y"])
print(f"labels: on-fraction {y.mean():.3f}, values {sorted(np.unique(y))}")
cl, cr = np.asarray(tr["c_lab"], float), np.asarray(tr["c_rel"], float)
g = (cl - cr) / np.abs(cl) * 100          # the LP relaxation is a lower bound, so this should be >= 0
print(f"label cost above the LP bound: median {np.median(g):.3f}%  mean {g.mean():.3f}%  max {g.max():.2f}%")
print(f"instances where the label is below the bound (should be 0): {(g < -1e-6).sum()}")
print(f"peak RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.1f} GB")
print("DATA OK" if bad == 0 and n >= 50000 and (g < -1e-6).sum() == 0 else "DATA PROBLEM")
PY

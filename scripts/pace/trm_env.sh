#!/usr/bin/env bash
# Environment for the TRM study on PACE ICE. Sourced by every job script in this directory.
#
# PLACEHOLDERS: module names and versions below are NOT verified for PACE ICE. Check the PACE ICE documentation
# (software / modules pages) and replace them before the first run.
set -euo pipefail
module purge 2>/dev/null || true
# PLACEHOLDER: the cluster's Python / Anaconda module (check the PACE ICE docs), e.g. export PYTHON_MODULE=<name>
module load "${PYTHON_MODULE:?PLACEHOLDER: set PYTHON_MODULE to the PACE ICE python module name (check the PACE ICE docs)}"
# CUDA comes with the PyTorch wheel; load a CUDA module only if the PACE ICE docs require it for GPU jobs:
if [ -n "${CUDA_MODULE:-}" ]; then module load "$CUDA_MODULE"; fi   # PLACEHOLDER (check the PACE ICE docs)

REPO=${REPO:-$HOME/power-grid}                 # clone of this repository (branch recursive-uc)
ENV_DIR=${ENV_DIR:-$HOME/envs/power-grid}      # PLACEHOLDER: a location with enough quota (PACE ICE scratch/project space)
if [ ! -d "$ENV_DIR" ]; then
  python3 -m venv "$ENV_DIR"
  "$ENV_DIR/bin/pip" install --upgrade pip
  # versions used in the CPU study: torch 2.14, scipy 1.17 (bundles HiGHS), highspy 1.12, numpy, pandas
  "$ENV_DIR/bin/pip" install torch numpy scipy pandas highspy pytest
fi
source "$ENV_DIR/bin/activate"
cd "$REPO"
export OTSL_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1   # HiGHS and the LP / MILP workers are single-threaded

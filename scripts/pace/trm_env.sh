#!/usr/bin/env bash
# Environment for every job script in this directory (both arms). Everything lives inside $PGU_ROOT, this project's
# dedicated folder (created by pgu_setup.sh): the repository clone, the Python environment, pip / matplotlib caches and
# logs. Nothing is written to your home folder, your conda base, or your other projects' folders.
#
# PLACEHOLDERS: module names are NOT verified for PACE ICE; set PYTHON_MODULE (and CUDA_MODULE only if needed) from
# `module avail` / the PACE ICE docs.
set -euo pipefail
: "${PGU_ROOT:?PGU_ROOT is not set (submit through scripts/pace/pgu_guard.py, which exports it)}"
[ -f "$PGU_ROOT/.pgu-root" ] || { echo "PGU_ROOT has no .pgu-root marker"; exit 1; }
module purge 2>/dev/null || true
module load "${PYTHON_MODULE:?set PYTHON_MODULE in pgu.conf / the environment (check the PACE ICE docs)}"
if [ -n "${CUDA_MODULE:-}" ]; then module load "$CUDA_MODULE"; fi

REPO=${REPO:-$PGU_ROOT/repo}
ENV_DIR=${ENV_DIR:-$PGU_ROOT/env/venv}
export PIP_CACHE_DIR=$PGU_ROOT/env/pip-cache MPLCONFIGDIR=$PGU_ROOT/env/mpl XDG_CACHE_HOME=$PGU_ROOT/env/cache
case "$ENV_DIR" in "$PGU_ROOT"/*) ;; *) echo "refusing: ENV_DIR must be inside PGU_ROOT"; exit 1;; esac
if [ ! -x "$ENV_DIR/bin/python" ]; then
  python3 -m venv "$ENV_DIR"
  "$ENV_DIR/bin/pip" install --upgrade pip
  # versions used in the CPU study: torch 2.14, scipy 1.17 (bundles HiGHS), highspy 1.12, numpy, pandas
  "$ENV_DIR/bin/pip" install torch numpy scipy pandas highspy pytest matplotlib
fi
source "$ENV_DIR/bin/activate"
cd "$REPO"
export OTSL_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1   # HiGHS and the LP / MILP workers are single-threaded

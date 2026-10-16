#!/usr/bin/env bash
# Environment for every job script in this directory (both arms), verified on PACE ICE (login-ice, Oct 2026).
#
# Everything this project creates lives inside $PGU_ROOT: the repository clone, the Python environment, the pip /
# matplotlib / torch caches, the data and the logs. Nothing is written to your home folder, your conda base, or any
# other project's folder.
#
# IMPORTANT: ~/.bashrc on this account runs `conda activate ~/scratch/envs/kneeoa` (another project). Slurm exports
# the submitting shell's environment, so that env would otherwise leak into these jobs and they would silently use its
# interpreter. The block below removes it from this process only; the other project is never modified.
set -euo pipefail
: "${PGU_ROOT:?PGU_ROOT is not set (submit through scripts/pace/pgu_guard.py, which exports it)}"
[ -f "$PGU_ROOT/.pgu-root" ] || { echo "PGU_ROOT has no .pgu-root marker"; exit 1; }

# --- leave any inherited conda / venv environment (this shell only) -------------------------------------------------
while [ -n "${CONDA_PREFIX:-}" ] && command -v conda >/dev/null 2>&1; do
  conda deactivate 2>/dev/null || break
done
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER VIRTUAL_ENV PYTHONPATH PYTHONHOME 2>/dev/null || true
PATH=$(printf '%s' "$PATH" | tr ':' '\n' | grep -v -E "/scratch/envs/|/\.conda/envs/|/anaconda3?/|/miniconda3?/" | paste -sd: -)
export PATH
module purge 2>/dev/null || true
module load "${PYTHON_MODULE:-python/3.11.9}"
if [ -n "${CUDA_MODULE:-}" ]; then module load "$CUDA_MODULE"; fi

# --- this project's own environment, inside PGU_ROOT ----------------------------------------------------------------
REPO=${REPO:-$PGU_ROOT/repo}
ENV_DIR=${ENV_DIR:-$PGU_ROOT/env/venv}
case "$ENV_DIR" in "$PGU_ROOT"/*) ;; *) echo "refusing: ENV_DIR must be inside PGU_ROOT"; exit 1;; esac
export PIP_CACHE_DIR=$PGU_ROOT/env/pip-cache MPLCONFIGDIR=$PGU_ROOT/env/mpl XDG_CACHE_HOME=$PGU_ROOT/env/cache
export TORCH_HOME=$PGU_ROOT/env/torch HF_HOME=$PGU_ROOT/env/hf   # never the shared ~/scratch/hf_cache
if [ ! -x "$ENV_DIR/bin/python" ]; then
  python3 -m venv "$ENV_DIR"                       # plain venv: no --system-site-packages, so nothing is inherited
  "$ENV_DIR/bin/pip" install --quiet --upgrade pip
  # versions used in the CPU study: torch 2.14, scipy 1.17 (bundles HiGHS), highspy 1.12, numpy, pandas
  "$ENV_DIR/bin/pip" install --quiet torch numpy scipy pandas highspy pytest matplotlib
fi
source "$ENV_DIR/bin/activate"
hash -r
# compare canonical paths: ~/scratch is a symlink on ICE, and a venv's bin/python3 points at the base interpreter
_want=$(cd "$ENV_DIR" && pwd -P)
_have=$(python3 -c 'import sys, os; print(os.path.realpath(sys.prefix))')
[ "$_have" = "$_want" ] || { echo "refusing: python3 prefix is $_have, not this project's env ($_want)"; exit 1; }
unset _want _have
cd "$REPO"
export OTSL_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1   # HiGHS and the LP / MILP workers are single-threaded

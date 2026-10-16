#!/usr/bin/env bash
# One-time setup of this project's dedicated folder on PACE ICE. Touches nothing outside PGU_ROOT:
# no ~/.bashrc edits, no conda base changes, no files in your other projects' folders.
#
#   export PGU_ROOT=~/scratch/power-grid-uc     # a NEW folder used only by this project
#   bash scripts/pace/pgu_setup.sh              # run from a clone of the repository, or let it clone one
set -euo pipefail
: "${PGU_ROOT:?set PGU_ROOT to a new, dedicated folder (for example ~/scratch/power-grid-uc)}"
PGU_ROOT=$(realpath -m "$PGU_ROOT")
case "$PGU_ROOT" in "$HOME"|"$HOME/"|"$HOME/scratch"|"$HOME/scratch/"|/|/tmp|/tmp/) echo "refusing: PGU_ROOT must be a dedicated sub-folder"; exit 1;; esac
if [ -e "$PGU_ROOT" ] && [ ! -f "$PGU_ROOT/.pgu-root" ] && [ -n "$(ls -A "$PGU_ROOT" 2>/dev/null)" ]; then
  echo "refusing: $PGU_ROOT exists, is not empty and is not marked as this project's folder"; exit 1
fi
mkdir -p "$PGU_ROOT"/{logs,env}
touch "$PGU_ROOT/.pgu-root"
[ -f "$PGU_ROOT/pgu.conf" ] || cp "$(dirname "$0")/pgu.conf.example" "$PGU_ROOT/pgu.conf"
[ -f "$PGU_ROOT/jobs.tsv" ] || printf '#jobid\tname\tkind\tgpus\ttime\tarray\tsubmitted\tscript\n' > "$PGU_ROOT/jobs.tsv"
if [ ! -d "$PGU_ROOT/repo/.git" ]; then
  git clone --branch recursive-uc https://github.com/Xyrro/power-grid.git "$PGU_ROOT/repo"
fi
echo "free space where PGU_ROOT lives:"; df -h "$PGU_ROOT" | tail -1
echo "Next: edit $PGU_ROOT/pgu.conf (partitions, GPU type, this project's share), then"
echo "  cd $PGU_ROOT/repo && PGU_ROOT=$PGU_ROOT python3 scripts/pace/pgu_guard.py status"

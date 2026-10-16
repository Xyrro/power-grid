# Running the recursive study on PACE ICE, safely

Your PACE account is shared with other projects (their files in your home and scratch, their jobs in the queue), and
the ICE limits are per user. Everything here is built so that this project cannot touch the others:

| rule | how it is enforced |
|---|---|
| one dedicated folder | every file lives under `PGU_ROOT` (repo clone, Python env, pip/matplotlib caches, data, logs). `pgu_setup.sh` refuses a non-empty folder without its `.pgu-root` marker; `trm_env.sh` refuses to run without it. No `~/.bashrc` or conda-base changes. |
| only our jobs are touched | every job is named `pgu-…` and recorded in `PGU_ROOT/jobs.tsv`; `pgu_guard.py cancel` only cancels ids that are recorded **and** still carry the prefix. Never use `scancel -u`. |
| the cluster limits are never exceeded | before each submission `pgu_guard.py` reads **all** of your jobs (this project's and the others') and refuses if the new job would take this project past its share or you past ICE's per-user limits: 32 GPU-hours of running capacity (GPUs × remaining walltime; pending jobs counted at their full request) and 50 queued + running GPU jobs (PACE notice, September 2026). |
| room is left for your other projects | this project's share is set in `PGU_ROOT/pgu.conf`: by default 16 of the 32 GPU-hours, 4 GPU jobs, 16 CPU tasks (8 at a time), at most 8 h per job. |
| nothing runs by surprise | every command has `--dry-run`, which prints the accounting and the exact `sbatch` line. |

Don't share your GT password or Duo codes with anyone or any tool; run these commands in your own PACE session.

## 1. One-time setup (login node)

```bash
export PGU_ROOT=~/scratch/power-grid-uc            # a NEW folder used only by this project
mkdir "$PGU_ROOT"                                  # fails if it already exists: then choose another name
touch "$PGU_ROOT/.pgu-root"
git clone --branch recursive-uc https://github.com/Xyrro/power-grid.git "$PGU_ROOT/repo"
bash "$PGU_ROOT/repo/scripts/pace/pgu_setup.sh"    # logs/, env/, pgu.conf, jobs.tsv inside PGU_ROOT only
cd "$PGU_ROOT/repo"
python3 scripts/pace/pgu_guard.py status           # read-only: partitions, GPU types, your limits, ALL your jobs, quota
```

Then edit `$PGU_ROOT/pgu.conf`: partition / QOS / account names and the GPU type string from the `status` output
(GRES column of `sinfo`), and `PYTHON_MODULE` from `module avail` (the guard passes it to every job). Lower the
`SHARE_*` values if your other projects need more.

Always submit from `$PGU_ROOT/repo` with `PGU_ROOT` exported.

## 2. Stage 1: data (CPU only, ≈ 90 core-hours)

```bash
G="python3 scripts/pace/pgu_guard.py submit"
$G --kind cpu --name base     --cpus 8 --mem 16G --time 04:00:00 --dry-run scripts/pace/trm_gen_base.sbatch
$G --kind cpu --name teachers --cpus 6 --mem 16G --time 06:00:00 --dependency afterok:<base id> scripts/pace/trm_teachers.sbatch
$G --kind cpu --name gen --array 0-15%8 --cpus 1 --mem 3G --time 07:30:00 \
   --dependency afterok:<teachers id> --export N_TOTAL=50000,N_TASKS=16 scripts/pace/pgu_gen_data.sbatch
$G --kind cpu --name extrap   --cpus 4 --mem 16G --time 08:00:00 scripts/pace/rloop_data_extrap.sbatch
```
Drop `--dry-run` once the printed line looks right. Re-submitting any of them resumes; finished pieces are skipped.

## 3. Stage 2: the deciding runs (≈ 40 GPU-hours in jobs of ≤ 8 h, at most two at a time)

Arm A go/no-go first (it is cheap and decides whether stage 3 is worth it):
```bash
$G --kind gpu --name trm-small --gpus 1 --cpus 8 --mem 48G --time 06:00:00 \
   --export TAG=gs_trm,PRESET=gpu_small,N_TRAIN=50000 scripts/pace/trm_train_gpu.sbatch
$G --kind gpu --name oneshot-small --gpus 1 --cpus 8 --mem 48G --time 02:00:00 \
   --export TAG=gs_oneshot,PRESET=gpu_small,N_TRAIN=50000,EXTRA="--mode oneshot" scripts/pace/trm_train_gpu.sbatch
```
then the deep control and the variants without LP features the same way (one job each, < 2 h), and the cheap one-shot
+ REINFORCE + safety-net pipeline (`RL=1`). Arm B afterwards, as chained 7-hour segments that resume from the
checkpoint (`--dependency afterany:<previous id>`):
```bash
$G --kind gpu --name rloop-B --gpus 1 --cpus 20 --mem 48G --time 07:00:00 \
   --export TAG=gpu_main,N_TRAIN=50000,EXTRA="--data_dir data/generated/recursive/pace" scripts/pace/rloop_train.sbatch
$G --kind gpu --name rloop-C --gpus 1 --cpus 20 --mem 48G --time 07:00:00 \
   --export TAG=gpu_nofb,N_TRAIN=50000,EXTRA="--no_feedback --data_dir data/generated/recursive/pace" \
   scripts/pace/rloop_train.sbatch
```
(≈ 4 segments for the run with LP feedback, 1 for the control without it). Evaluation is CPU-only:
`--kind cpu --array 0-7%8 --cpus 1 --mem 6G` with `trm_eval_cpu.sbatch` / `rloop_eval.sbatch`.

## 4. Watching and stopping

```bash
python3 scripts/pace/pgu_guard.py jobs             # this project's jobs and their states
python3 scripts/pace/pgu_guard.py status           # budget left, with your other projects' jobs included
python3 scripts/pace/pgu_guard.py cancel <jobid>   # only works on this project's recorded jobs
```
Logs: `$PGU_ROOT/logs/`. Results land in `$PGU_ROOT/repo/results/recursive/`; commit and push them to the
`recursive-uc` branch (or send the `results/recursive/*.md` and `*.json` files).

## Unverified items

The partition, QOS and account names, the GPU type strings, the walltime limit per partition, the module names and the
scratch quota were not visible from outside the cluster; take them from `pgu_guard.py status` and the PACE ICE docs.
If ICE's walltime limit is below 8 h, set `MAX_WALLTIME` accordingly; the training jobs checkpoint and resume.

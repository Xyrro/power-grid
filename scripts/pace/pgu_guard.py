#!/usr/bin/env python3
"""Guarded Slurm submission for this project on PACE ICE (standard library only).

The account this runs under is shared with other projects. This tool therefore:
  * keeps every file of this project inside one dedicated root folder (PGU_ROOT, marked by a `.pgu-root` file);
  * tags every job with a name prefix and records its id in PGU_ROOT/jobs.tsv, and only ever cancels jobs that are
    both recorded there and still carry the prefix (it never runs `scancel -u`);
  * before every submission, reads ALL of the user's jobs (this project's and others') and refuses if the new job would
    push this project past its configured share, or the user past the cluster's per-user limits
    (ICE, October 2026: QOS MaxTRESRunMins gres/gpu=1920, i.e. 32 GPU-hours of RUNNING capacity, and
    MaxSubmitPU 50, which counts EVERY queued or running job of the user, array tasks one by one);
  * supports --dry-run, which prints the accounting and the exact sbatch command without submitting.

Usage (from inside PGU_ROOT/repo):
  python3 scripts/pace/pgu_guard.py status
  python3 scripts/pace/pgu_guard.py submit --kind gpu --name trm-small --gpus 1 --time 06:00:00 [--dry-run] \
      [--cpus 8 --mem 48G] [--export TAG=x,PRESET=gpu_small] scripts/pace/trm_train_gpu.sbatch
  python3 scripts/pace/pgu_guard.py submit --kind cpu --name gen --array 0-199%8 --time 02:00:00 ... SCRIPT
  python3 scripts/pace/pgu_guard.py jobs
  python3 scripts/pace/pgu_guard.py cancel JOBID [JOBID ...]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass

DEFAULTS = {
    "PGU_ROOT": "",
    "JOB_PREFIX": "pgu-",
    "ACCOUNT": "",              # leave empty if ICE assigns it automatically
    "GPU_PARTITION": "",        # fill from `status` (sinfo) / the PACE ICE docs
    "CPU_PARTITION": "",
    "QOS": "",                  # fallback when GPU_QOS / CPU_QOS are empty
    "GPU_QOS": "",              # e.g. coe-ice for the coe-gpu partition
    "CPU_QOS": "",
    "GPU_GRES": "gpu",          # e.g. gpu:H100 or gpu:H200 -- the exact type string from `sinfo` on ICE
    "USER_GPU_HOURS_LIMIT": "32",   # cluster per-user running-capacity limit (GPU x remaining walltime)
    "USER_GPU_JOBS_LIMIT": "50",    # cluster per-user limit on queued + running GPU jobs
    "USER_JOBS_LIMIT": "50",        # QOS MaxSubmitPU: ALL queued + running jobs, array tasks counted one by one
    "SHARE_JOBS": "40",             # leave the rest of MaxSubmitPU free for the user's other projects
    "SHARE_GPU_HOURS": "16",        # this project's share: its queued + running GPU jobs at their full request
    "SHARE_GPU_JOBS": "4",          # this project's max GPU jobs, queued + running
    "SHARE_CPU_TASKS": "16",        # this project's max CPU jobs/array tasks, queued + running
    "SHARE_CPU_CONCURRENT": "8",    # max throttle (%k) allowed on a CPU array
    "MAX_WALLTIME": "08:00:00",         # cap for GPU jobs (ice-gpu allows 16:00:00)
    "MAX_WALLTIME_CPU": "18:00:00",     # cap for CPU jobs (ice-cpu allows 18:00:00)
    "PYTHON_MODULE": "",            # passed to the jobs (trm_env.sh loads it); from `module avail`
    "CUDA_MODULE": "",              # only if the PACE ICE docs require a CUDA module for GPU jobs
}


# ----------------------------------------------------------------------------------------------- config and paths
def load_config(path: str | None = None) -> dict:
    cfg = dict(DEFAULTS)
    root = os.environ.get("PGU_ROOT", "")
    path = path or (os.path.join(root, "pgu.conf") if root else "")
    if path and os.path.exists(path):
        for line in open(path):
            line = line.split("#", 1)[0].strip()
            if "=" in line:
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip().strip('"').strip("'")
    if os.environ.get("PGU_ROOT"):
        cfg["PGU_ROOT"] = os.environ["PGU_ROOT"]
    return cfg


def check_root(cfg: dict, script: str | None = None) -> str:
    root = os.path.realpath(os.path.expanduser(cfg["PGU_ROOT"])) if cfg["PGU_ROOT"] else ""
    if not root or not os.path.isfile(os.path.join(root, ".pgu-root")):
        sys.exit("refusing: PGU_ROOT is not set or has no .pgu-root marker (run scripts/pace/pgu_setup.sh first)")
    cwd = os.path.realpath(os.getcwd())
    if not (cwd == root or cwd.startswith(root + os.sep)):
        sys.exit(f"refusing: run from inside PGU_ROOT ({root}); current directory is {cwd}")
    if script:
        sp = os.path.realpath(script)
        if not sp.startswith(root + os.sep):
            sys.exit(f"refusing: job script {sp} is outside PGU_ROOT")
    return root


# ----------------------------------------------------------------------------------------------- parsing helpers
def parse_slurm_time(s: str) -> float | None:
    """Slurm duration -> hours. Accepts [D-]HH:MM:SS, MM:SS, M, D-HH, D-HH:MM. None for UNLIMITED/NOT_SET/INVALID."""
    s = (s or "").strip()
    if not s or s.upper() in {"UNLIMITED", "NOT_SET", "INVALID", "N/A", "INFINITE"}:
        return None
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
        parts = [int(x) for x in s.split(":")]
        while len(parts) < 3:
            parts.append(0)                      # D-HH, D-HH:MM
        h, m, sec = parts
    else:
        parts = [int(x) for x in s.split(":")]
        if len(parts) == 1:
            h, m, sec = 0, parts[0], 0           # minutes
        elif len(parts) == 2:
            h, m, sec = 0, parts[0], parts[1]    # MM:SS
        else:
            h, m, sec = parts
    return days * 24 + h + m / 60 + sec / 3600


_GPU_PATTERNS = [
    re.compile(r"gres/gpu(?::[\w.\-]+)?=(\d+)"),   # TRES strings: gres/gpu=1, gres/gpu:h100=1
    re.compile(r"gpu(?::[\w.\-]+)?:(\d+)"),        # gres strings: gpu:1, gpu:h100:1, gres:gpu:H100:2
]


def gpu_count(tres: str) -> int:
    """Largest GPU count mentioned in a TRES / gres string (typed and untyped entries name the same GPUs)."""
    tres = (tres or "").replace("gres:", "")
    best = 0
    for pat in _GPU_PATTERNS:
        for m in pat.finditer(tres):
            best = max(best, int(m.group(1)))
    return best


@dataclass
class Job:
    jobid: str
    name: str
    state: str          # RUNNING, PENDING, ...
    nodes: int
    gpus_per_node: int
    hours_left: float   # running: remaining; pending: requested limit
    hours_limit: float

    @property
    def gpus(self) -> int:
        return self.gpus_per_node * max(self.nodes, 1)

    @property
    def gpu_hours_committed(self) -> float:
        h = self.hours_left if self.state.startswith("R") else self.hours_limit
        return self.gpus * (h or 0.0)


def parse_squeue_lines(text: str) -> list[Job]:
    """Lines of `squeue -h -r -o '%i|%j|%T|%D|%b|%L|%l'` (one line per array task)."""
    jobs = []
    for line in text.splitlines():
        if not line.strip():
            continue
        f = line.split("|")
        if len(f) < 7:
            continue
        jobid, name, state, nodes, tres, left, limit = f[:7]
        lim = parse_slurm_time(limit) or 0.0
        lft = parse_slurm_time(left)
        jobs.append(Job(jobid.strip(), name.strip(), state.strip().upper(), int(nodes or 1),
                        gpu_count(tres), lft if lft is not None else lim, lim))
    return jobs


def read_jobs(user: str) -> list[Job]:
    out = subprocess.run(["squeue", "-a", "-u", user, "-h", "-r", "-o", "%i|%j|%T|%D|%b|%L|%l"],
                         capture_output=True, text=True, check=True).stdout
    jobs = parse_squeue_lines(out)
    # %b (TRES per node) misses GPUs requested per job (--gpus); fill those in from scontrol.
    for j in jobs:
        if j.gpus_per_node == 0:
            info = subprocess.run(["scontrol", "show", "job", "-o", j.jobid], capture_output=True, text=True).stdout
            m = re.search(r"\bTresPerJob=(\S+)", info)
            if m and gpu_count(m.group(1)) > 0:
                j.gpus_per_node = gpu_count(m.group(1))
                j.nodes = 1
    return jobs


# ----------------------------------------------------------------------------------------------- registry
def registry_path(root: str) -> str:
    return os.path.join(root, "jobs.tsv")


def read_registry(root: str) -> dict:
    reg = {}
    p = registry_path(root)
    if os.path.exists(p):
        for line in open(p):
            f = line.rstrip("\n").split("\t")
            if f and f[0] and not f[0].startswith("#"):
                reg[f[0]] = f
    return reg


def base_id(jobid: str) -> str:
    return jobid.split("_", 1)[0]


def is_ours(job: Job, reg: dict, prefix: str) -> bool:
    return job.name.startswith(prefix) and base_id(job.jobid) in reg


# ----------------------------------------------------------------------------------------------- accounting
@dataclass
class Usage:
    gpu_hours_all: float        # RUNNING jobs only: this is what Slurm's MaxTRESRunMins actually limits
    gpu_hours_ours: float       # this project, running + pending (so we never queue more than our share)
    gpu_jobs_all: int
    gpu_jobs_ours: int
    cpu_tasks_ours: int
    gpu_hours_ours_running: float = 0.0
    jobs_all: int = 0               # every queued/running job of the user (QOS MaxSubmitPU counts these)
    jobs_ours: int = 0


def usage(jobs: list[Job], reg: dict, prefix: str) -> Usage:
    active = list(jobs)                 # squeue lists only unfinished jobs
    g_all = [j for j in active if j.gpus > 0]
    g_run = [j for j in g_all if j.state.startswith("R")]
    g_our = [j for j in g_all if is_ours(j, reg, prefix)]
    c_our = [j for j in active if j.gpus == 0 and is_ours(j, reg, prefix)]
    ours = [j for j in active if is_ours(j, reg, prefix)]
    return Usage(sum(j.gpu_hours_committed for j in g_run), sum(j.gpu_hours_committed for j in g_our),
                 len(g_all), len(g_our), len(c_our),
                 sum(j.gpu_hours_committed for j in g_our if j.state.startswith("R")),
                 len(active), len(ours))


def array_size(spec: str | None) -> tuple[int, int | None]:
    """'0-199%8' -> (200, 8); '1,3,5' -> (3, None); None -> (1, None)."""
    if not spec:
        return 1, None
    throttle = None
    if "%" in spec:
        spec, t = spec.split("%", 1)
        throttle = int(t)
    n = 0
    for part in spec.split(","):
        step = 1
        if ":" in part:
            part, s = part.split(":", 1)
            step = int(s)
        if "-" in part:
            a, b = part.split("-", 1)
            n += len(range(int(a), int(b) + 1, step))
        elif part:
            n += 1
    return n, throttle


def check_budget(cfg: dict, u: Usage, kind: str, gpus: int, hours: float, n_tasks: int,
                 throttle: int | None, notes: list | None = None) -> list[str]:
    """Return a list of violations (empty = allowed). `notes` collects non-blocking remarks."""
    bad = []
    notes = notes if notes is not None else []
    # QOS MaxSubmitPU counts every queued or running job, array tasks individually, across all of the user's projects
    if u.jobs_all + n_tasks > int(cfg["USER_JOBS_LIMIT"]):
        bad.append(f"user limit: {u.jobs_all} jobs queued/running + {n_tasks} new > USER_JOBS_LIMIT "
                   f"{cfg['USER_JOBS_LIMIT']} (QOS MaxSubmitPU; it would block every project of yours)")
    if u.jobs_ours + n_tasks > int(cfg["SHARE_JOBS"]):
        bad.append(f"project share: {u.jobs_ours} of our jobs + {n_tasks} new > SHARE_JOBS {cfg['SHARE_JOBS']}")
    key = "MAX_WALLTIME" if kind == "gpu" else "MAX_WALLTIME_CPU"
    max_wall = parse_slurm_time(cfg.get(key) or cfg["MAX_WALLTIME"]) or 8.0
    if hours > max_wall + 1e-9:
        bad.append(f"walltime {hours:.2f} h exceeds {key} {max_wall:.2f} h")
    if kind == "gpu":
        if n_tasks != 1:
            bad.append("GPU jobs must not be arrays here; submit them one by one")
        new = gpus * hours
        share, cap = float(cfg["SHARE_GPU_HOURS"]), float(cfg["USER_GPU_HOURS_LIMIT"])
        # This project's share counts every GPU job of ours, queued or running, at its full request, whatever it
        # waits on: Slurm's own cap (MaxTRESRunMins) is per USER and shared with the user's other projects, and a
        # queued job of ours that later starts takes capacity those projects would otherwise get. So no job state
        # (pending, waiting on a dependency) exempts a job from the share; a resume chain is submitted one segment
        # at a time, each once the previous one has finished.
        if new > share + 1e-9:
            bad.append(f"project share: one job of {gpus} GPU x {hours:.1f} h = {new:.1f} GPU-h > SHARE_GPU_HOURS "
                       f"{cfg['SHARE_GPU_HOURS']}")
        elif u.gpu_hours_ours + new > share + 1e-9:
            bad.append(f"project share: {u.gpu_hours_ours:.1f} GPU-h of ours queued/running + {new:.1f} > "
                       f"SHARE_GPU_HOURS {cfg['SHARE_GPU_HOURS']} (headroom reserved for your other projects)")
        if share > cap - 1e-9:
            notes.append(f"SHARE_GPU_HOURS {share:g} reserves nothing below the per-user cap {cap:g}: "
                         "your other projects get no guaranteed GPU capacity while this project's jobs hold it")
        if u.gpu_hours_all + new > cap + 1e-9:
            notes.append(f"your running jobs hold {u.gpu_hours_all:.1f} GPU-h; this one waits for "
                         f"MaxGRESRunMinsPerUser ({cfg['USER_GPU_HOURS_LIMIT']} GPU-h)")
        if u.gpu_jobs_ours + 1 > int(cfg["SHARE_GPU_JOBS"]):
            bad.append(f"project share: {u.gpu_jobs_ours} GPU jobs already, SHARE_GPU_JOBS {cfg['SHARE_GPU_JOBS']}")
        if u.gpu_jobs_all + 1 > int(cfg["USER_GPU_JOBS_LIMIT"]):
            bad.append(f"user limit: {u.gpu_jobs_all} GPU jobs queued/running, USER_GPU_JOBS_LIMIT {cfg['USER_GPU_JOBS_LIMIT']}")
    else:
        if gpus:
            bad.append("CPU jobs must not request GPUs")
        if u.cpu_tasks_ours + n_tasks > int(cfg["SHARE_CPU_TASKS"]):
            bad.append(f"project share: {u.cpu_tasks_ours} CPU tasks queued/running + {n_tasks} new > "
                       f"SHARE_CPU_TASKS {cfg['SHARE_CPU_TASKS']} (use fewer, larger array tasks)")
        conc = int(cfg["SHARE_CPU_CONCURRENT"])
        if n_tasks > conc and (throttle is None or throttle > conc):
            bad.append(f"CPU arrays larger than {conc} tasks need a throttle %k with k <= SHARE_CPU_CONCURRENT {conc}")
    return bad


# ----------------------------------------------------------------------------------------------- commands
def cmd_status(cfg: dict) -> None:
    root = check_root(cfg)
    user = os.environ.get("USER", "")
    print(f"PGU_ROOT {root}   user {user}   prefix {cfg['JOB_PREFIX']}")
    for title, cmd in [("partitions (GPU types in GRES)", ["sinfo", "-o", "%P|%a|%l|%D|%G|%c|%m"]),
                       ("your associations", ["sacctmgr", "-nP", "show", "assoc", f"user={user}",
                                              "format=account,partition,qos,grptresrunmins,maxjobs,maxsubmit"]),
                       ("QOS limits", ["sacctmgr", "-nP", "show", "qos",
                                       "format=name,maxwall,maxtrespu,maxtresrunminspu,maxsubmitpu,maxjobspu"])]:
        print(f"\n== {title}")
        r = subprocess.run(cmd, capture_output=True, text=True)
        print((r.stdout or r.stderr).strip()[:6000])
    reg = read_registry(root)
    jobs = read_jobs(user)
    u = usage(jobs, reg, cfg["JOB_PREFIX"])
    print("\n== all your jobs (* = this project)")
    for j in jobs:
        mark = "*" if is_ours(j, reg, cfg["JOB_PREFIX"]) else " "
        print(f" {mark} {j.jobid:>14} {j.state:<10} gpus={j.gpus} left={j.hours_left:.2f}h  {j.name}")
    print(f"\nGPU-h running (Slurm's cap): all {u.gpu_hours_all:.1f} / {cfg['USER_GPU_HOURS_LIMIT']}, "
          f"this project {u.gpu_hours_ours_running:.1f}")
    print(f"GPU-h queued+running, this project: {u.gpu_hours_ours:.1f} / {cfg['SHARE_GPU_HOURS']}")
    print(f"GPU jobs: all {u.gpu_jobs_all} / {cfg['USER_GPU_JOBS_LIMIT']}, this project {u.gpu_jobs_ours} / {cfg['SHARE_GPU_JOBS']}")
    print(f"CPU jobs/tasks of this project: {u.cpu_tasks_ours} / {cfg['SHARE_CPU_TASKS']}")
    print(f"ALL jobs (QOS MaxSubmitPU): all {u.jobs_all} / {cfg['USER_JOBS_LIMIT']}, "
          f"this project {u.jobs_ours} / {cfg['SHARE_JOBS']}")
    print("\n== storage")
    for cmd in (["pace-quota"], ["du", "-sh", root], ["df", "-h", root]):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            print(r.stdout.strip()[:3000])


def build_sbatch(cfg: dict, a) -> list[str]:
    cmd = ["sbatch", "--parsable", "-J", cfg["JOB_PREFIX"] + a.name, "--comment=power-grid-uc", "-t", a.time]
    part = cfg["GPU_PARTITION"] if a.kind == "gpu" else cfg["CPU_PARTITION"]
    qos = (cfg.get("GPU_QOS") if a.kind == "gpu" else cfg.get("CPU_QOS")) or cfg["QOS"]
    if cfg["ACCOUNT"]:
        cmd += ["-A", cfg["ACCOUNT"]]
    if part:
        cmd += ["-p", part]
    if qos:
        cmd += ["-q", qos]
    if a.kind == "gpu":
        cmd += [f"--gres={cfg['GPU_GRES']}:{a.gpus}"]
    if a.cpus:
        cmd += [f"--cpus-per-task={a.cpus}"]
    if a.mem:
        cmd += [f"--mem={a.mem}"]
    if a.array:
        cmd += [f"--array={a.array}"]
    if a.dependency:
        cmd += [f"--dependency={a.dependency}"]
    exports = "ALL,PGU_ROOT=" + cfg["PGU_ROOT"]
    for k in ("PYTHON_MODULE", "CUDA_MODULE"):
        if cfg.get(k):
            exports += f",{k}={cfg[k]}"
    exports += ("," + a.export) if a.export else ""
    cmd += [f"--export={exports}", "-o", os.path.join(cfg["PGU_ROOT"], "logs", "%x-%j.out"), a.script]
    return cmd


def cmd_submit(cfg: dict, a) -> None:
    root = check_root(cfg, a.script)
    cfg["PGU_ROOT"] = root
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.\-]{0,40}", a.name):
        sys.exit("refusing: --name must be short and alphanumeric")
    hours = parse_slurm_time(a.time)
    if hours is None:
        sys.exit("refusing: --time must be a finite walltime")
    n_tasks, throttle = array_size(a.array)
    reg = read_registry(root)
    u = usage(read_jobs(os.environ.get("USER", "")), reg, cfg["JOB_PREFIX"])
    notes: list = []
    bad = check_budget(cfg, u, a.kind, a.gpus if a.kind == "gpu" else 0, hours, n_tasks, throttle, notes)
    cmd = build_sbatch(cfg, a)
    print(f"before: GPU-h running all {u.gpu_hours_all:.1f}/{cfg['USER_GPU_HOURS_LIMIT']}, project q+r "
          f"{u.gpu_hours_ours:.1f}/{cfg['SHARE_GPU_HOURS']}; GPU jobs all {u.gpu_jobs_all}/"
          f"{cfg['USER_GPU_JOBS_LIMIT']}; ALL jobs all {u.jobs_all}/{cfg['USER_JOBS_LIMIT']}, project "
          f"{u.jobs_ours}/{cfg['SHARE_JOBS']}")
    for nmsg in notes:
        print("note:", nmsg)
    print("command:", " ".join(shlex.quote(c) for c in cmd))
    if bad:
        print("REFUSED:\n  " + "\n  ".join(bad))
        sys.exit(2)
    if a.dry_run:
        print("dry run: nothing submitted")
        return
    os.makedirs(os.path.join(root, "logs"), exist_ok=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("sbatch failed: " + r.stderr.strip())
    jobid = base_id(r.stdout.strip().split(";")[0])
    with open(registry_path(root), "a") as f:
        f.write("\t".join([jobid, cfg["JOB_PREFIX"] + a.name, a.kind, str(a.gpus if a.kind == "gpu" else 0), a.time,
                           a.array or "", dt.datetime.now().isoformat(timespec="seconds"), a.script]) + "\n")
    print("submitted", jobid)


def cmd_jobs(cfg: dict) -> None:
    root = check_root(cfg)
    reg = read_registry(root)
    live = {base_id(j.jobid): j.state for j in read_jobs(os.environ.get("USER", ""))}
    for jid, f in reg.items():
        print(jid, live.get(jid, "finished/unknown"), *f[1:])


def cmd_cancel(cfg: dict, ids: list[str]) -> None:
    root = check_root(cfg)
    reg = read_registry(root)
    jobs = {j.jobid: j for j in read_jobs(os.environ.get("USER", ""))}
    for jid in ids:
        if not re.fullmatch(r"\d+(_\d+)?", jid):
            print(f"refusing to cancel {jid!r}: not a plain job id")
            continue
        b = base_id(jid)
        live = [j for j in jobs.values() if base_id(j.jobid) == b]
        if b not in reg or not live or not all(j.name.startswith(cfg["JOB_PREFIX"]) for j in live):
            print(f"refusing to cancel {jid}: not a recorded, live job of this project")
            continue
        subprocess.run(["scancel", jid], check=False)
        print("cancelled", jid)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sub.add_parser("jobs")
    c = sub.add_parser("cancel")
    c.add_argument("ids", nargs="+")
    s = sub.add_parser("submit")
    s.add_argument("--kind", choices=["gpu", "cpu"], required=True)
    s.add_argument("--name", required=True)
    s.add_argument("--time", required=True)
    s.add_argument("--gpus", type=int, default=1)
    s.add_argument("--cpus", type=int, default=0)
    s.add_argument("--mem", default="")
    s.add_argument("--array", default="")
    s.add_argument("--dependency", default="")
    s.add_argument("--export", default="")
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("script")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    if a.cmd == "status":
        cmd_status(cfg)
    elif a.cmd == "jobs":
        cmd_jobs(cfg)
    elif a.cmd == "cancel":
        cmd_cancel(cfg, a.ids)
    else:
        cmd_submit(cfg, a)


if __name__ == "__main__":
    main()

"""Work queue for scripts/agg_eval.py (docs/methods/agg.md): one worker per allocated core (each pinned by the job
script), claiming instances from the shared list (longest first) until none is left; any number of workers and jobs
can run at once. Every instance runs in its own single-threaded process.

    python3 scripts/agg_run.py --phase val --tag w0     # the 360 validation instances (the choices)
    python3 scripts/agg_run.py --phase test --tag w0 --acert s0+x:0.02,lag0.0003+x:0.4   # the test instances, once
    python3 scripts/agg_run.py --phase test --list      # the item list
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SHIFTS = ("load_hi", "load_lo", "ren_hi", "gen_out", "line_out", "night")
COST = {("uc12", "val"): 1.0, ("uc12", "tf"): 1.2, ("uc12", "shift"): 1.0, ("uc24", "tf"): 6.0, ("hard", "tf"): 4.0}


def items(phase):
    if phase.startswith("val"):
        return [("uc12", "val", k) for k in range(360)]
    out = [("uc12", "tf", k) for k in range(120)]
    if phase == "test2":                      # round 9: the learned pipelines only (12 h test_fresh and the shifts)
        return out + [("uc12", s, k) for s in SHIFTS for k in range(30)]
    out += [("uc12", s, k) for s in SHIFTS for k in range(30)]
    out += [("uc24", "tf", k) for k in range(40)]
    out += [("hard", "tf", k) for k in range(60)]
    return out


def cost(it):
    return COST.get((it[0], it[1]), COST[("uc12", "shift")])


def claim(folder, it, tag):
    os.makedirs(folder, exist_ok=True)
    try:
        fd = os.open(os.path.join(folder, f"{it[0]}_{it[1]}_{it[2]}"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, f"{tag} {socket.gethostname()} {time.time():.0f}\n".encode())
    os.close(fd)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("val", "val2", "val3", "test", "test2"), required=True,
                    help="val2 / val3: further validation passes, test2: a second test pass (round 9), each with its own "
                         "folder and reference solves")
    ap.add_argument("--tag", default="")
    ap.add_argument("--res", default="results/agg")
    ap.add_argument("--acert", default="", help="passed to agg_eval.py --acert (test: the validation choice)")
    ap.add_argument("--methods", default="", help="passed to agg_eval.py --methods")
    ap.add_argument("--script", default="scripts/agg_eval.py", help="the evaluation script (a frozen copy while another pass runs)")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    os.chdir(ROOT)
    order = sorted(items(a.phase), key=lambda x: (-cost(x), x))
    if a.list:
        print(f"{len(order)} items, first {order[:3]}")
        return
    res = os.path.join(a.res, a.phase)
    os.makedirs(res, exist_ok=True)
    tag = a.tag or f"j{os.environ.get('SLURM_JOB_ID', 'x')}"
    env = dict(os.environ, OTSL_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    man = os.path.join(res, f"manifest_{tag}.jsonl")
    t_all, n = time.time(), 0
    print(f"{tag}: {a.phase} on {socket.gethostname()}, CPUs {sorted(os.sched_getaffinity(0))}", flush=True)
    for it in order:
        if os.path.exists(os.path.join(res, "claims", f"{it[0]}_{it[1]}_{it[2]}")) or not claim(os.path.join(res, "claims"), it, tag):
            continue
        n += 1
        cmd = [sys.executable, a.script, "--bench", it[0], "--set", it[1], "--k", str(it[2]),
               "--out", os.path.join(res, f"{it[0]}_{tag}.jsonl")]
        if a.acert:
            cmd += ["--acert", a.acert]
        if a.methods:
            cmd += ["--methods", a.methods]
        t0, l0 = time.time(), os.getloadavg()[0]
        p = subprocess.run(cmd, env=env, cwd=ROOT, capture_output=True, text=True)
        rec = dict(bench=it[0], set=it[1], k=it[2], rc=p.returncode, start=t0, wall=time.time() - t0, loadavg0=l0,
                   loadavg1=os.getloadavg()[0], host=socket.gethostname(), cpus=sorted(os.sched_getaffinity(0)),
                   job=os.environ.get("SLURM_JOB_ID"), tag=tag)
        if p.returncode:
            rec["stderr_tail"] = p.stderr[-3000:]
        with open(man, "a") as f:
            f.write(json.dumps(rec) + "\n")
        with open(os.path.join(res, f"log_{tag}.txt"), "a") as f:
            f.write(f"==== {it} rc={p.returncode} {rec['wall']:.0f}s\n{p.stdout[-3000:]}\n" + (p.stderr[-3000:] if p.returncode else ""))
        print(f"[#{n} {time.time() - t_all:.0f}s] {it}: rc {p.returncode}, {rec['wall']:.0f}s", flush=True)
    print(f"{tag}: done, {n} items in {time.time() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()

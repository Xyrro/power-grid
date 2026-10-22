"""Step 2 of docs/unified_metrics.md: re-time every method on one machine. For each test instance, every evaluation
process that covers it runs back to back on the same core (this task's single CPU), in a random order seeded by the
instance; each process re-solves its own full MILP at 0.1 % (paired speed-up reference, so the reference is measured
several times per instance: a check of timing noise).

    python3 scripts/unified_retime.py --list                         # the item list and the commands of one item
    python3 scripts/unified_retime.py --task 3 --ntasks 12           # one array task: its share of the items
    python3 scripts/unified_retime.py --items uc12:tf:0 --skip armA  # explicit items (smoke tests)

Processes per instance (outputs under --res, one file per process kind and task; every script resumes from its own
file only, never from the original studies' records):
  fix     scripts/unified_fix_eval.py: full MILPs, fixing rules, no-learning baselines, early one-shot models
          (uc12 test_fresh and the six shifted sets, uc24)
  hard    scripts/uc_hard_eval.py: the congested benchmark's full MILPs and no-learning rules (its learned rules need
          data that exist only where that study ran, and are skipped by the script)
  armA    scripts/rec_trm_eval.py with the GPU round's test plan (uc12 test_fresh)
  armB    scripts/rec_rloop_eval.py, w2_rloop_rl (uc12 test_fresh; zero-shot on uc24, hard and line_out with 24 steps,
          the 12-hour validation choice); armB0: w1_rloop_nofb_rl, the control without LP feedback (uc12 test_fresh);
          armBs1-3: g3_plain_s1..3_rl, three more training seeds of armB's configuration (GPU-scale runs of the
          attention-residual comparison, docs/recursive/infiloop_pilot.md), with armB's settings (uc12 test_fresh)
A manifest line per process records the host, CPU model, start / end, exit code and load average.
"""
import argparse
import glob
import json
import os
import socket
import subprocess
import sys
import time
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SHIFTS = ("load_hi", "load_lo", "ren_hi", "gen_out", "line_out", "night")
SEL_B = "results/recursive/gpu_round/rloop_select_gpu.json"
PLAN_A = "results/recursive/gpu_round/trm_select_gpu.json"
N = {"uc12": 120, "shift": 30, "uc24": 40, "hard": 60}
COST = {"uc12": 8.5, "shift": 2.0, "uc24": 9.5, "hard": 7.0}        # rough minutes per item (PACE speed), for load balancing


def items_all():
    out = [("uc12", "tf", k) for k in range(N["uc12"])]
    out += [("uc12", s, k) for s in SHIFTS for k in range(N["shift"])]
    out += [("uc24", "tf", k) for k in range(N["uc24"])]
    out += [("hard", "tf", k) for k in range(N["hard"])]
    return out


def cost(it):
    return COST["shift"] if it[0] == "uc12" and it[1] != "tf" else COST[it[0]]


def assign(items, ntasks):
    """greedy longest-first assignment to the least loaded task (deterministic)"""
    load, out = [0.0] * ntasks, [[] for _ in range(ntasks)]
    for it in sorted(items, key=lambda x: (-cost(x), x)):
        j = min(range(ntasks), key=lambda t: (load[t], t))
        out[j].append(it)
        load[j] += cost(it)
    return out, load


def commands(it, res, tag, py):
    """{kind: argv} for one item"""
    bench, s, k = it
    cmd = {}
    if bench in ("uc12", "uc24"):
        cmd["fix"] = [py, "scripts/unified_fix_eval.py", "--bench", bench, "--set", s, "--k", str(k),
                      "--out", f"{res}/{bench}_fix_{tag}.jsonl"]
    if bench == "hard":
        cmd["hard"] = [py, "scripts/uc_hard_eval.py", "--only", str(k), "--out", f"{res}/hard_fix_{tag}.jsonl"]
    if bench == "uc12" and s == "tf":
        cmd["armA"] = [py, "scripts/rec_trm_eval.py", "--split", "test_fresh", "--only", str(k), "--plan", PLAN_A,
                       "--round", "step2", "--trm_device", "cpu", "--out", f"{res}/uc12_armA_{tag}.jsonl"]
    b_bench = {("uc12", "tf"): "uc12", ("uc24", "tf"): "uc24", ("hard", "tf"): "hard", ("uc12", "line_out"): "line_out"}
    bb = b_bench.get((bench, s))
    if bb:
        for kind, model in (("armB", "w2_rloop_rl"), ("armB0", "w1_rloop_nofb_rl"), ("armBs1", "g3_plain_s1_rl"),
                            ("armBs2", "g3_plain_s2_rl"), ("armBs3", "g3_plain_s3_rl")):
            if kind != "armB" and bb != "uc12":
                continue
            c = [py, "scripts/rec_rloop_eval.py", "--bench", bb, "--select", SEL_B, "--model", model,
                 "--skip", "m1x,guided,union", "--seeded", "0", "--start", str(k), "--n", str(k + 1),
                 "--out", f"{res}/{bb}_{kind}_{tag}.jsonl"]
            if bb != "uc12":
                c += ["--k_sel", "24", "--gaps", "0.001"]       # zero-shot: the 12-hour choice; reference only
            elif kind.startswith("armBs"):
                c += ["--gaps", "0.001"]                          # seeds: the paired reference only
            cmd[kind] = c
    return cmd


def cpus():
    return sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None


def cpu_model():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return ""


def claim(res, it, tag):
    """atomically claim an item for this worker (a file created with O_EXCL); False if another worker has it"""
    d = os.path.join(res, "claims")
    os.makedirs(d, exist_ok=True)
    try:
        fd = os.open(os.path.join(d, f"{it[0]}_{it[1]}_{it[2]}"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, f"{tag} {socket.gethostname()} {time.time():.0f}\n".encode())
    os.close(fd)
    return True


def audit(res, skip=()):
    """items whose every process finished with exit code 0 in some worker's manifest, and the rest"""
    ok = {}
    for p in glob.glob(os.path.join(res, "manifest_*.jsonl")):
        for line in open(p):
            r = json.loads(line)
            if r["rc"] == 0:
                ok.setdefault((r["bench"], r["set"], r["k"]), set()).add(r["kind"])
    done, missing = [], []
    for it in items_all():
        need = {k for k in commands(it, res, "x", "python3") if k not in skip}
        (done if need <= ok.get(it, set()) else missing).append(it)
    return done, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--ntasks", type=int, default=1)
    ap.add_argument("--items", default="", help="comma list bench:set:k (overrides --task / --ntasks)")
    ap.add_argument("--pull", action="store_true",
                    help="claim items from the shared list (longest first) until none is left; 12-hour items wait for "
                         "the marker <res>/prep/READY (written once the preparation job's outputs are checked)")
    ap.add_argument("--tag", default="", help="output tag with --items / --pull (default t<array task id> or tx)")
    ap.add_argument("--res", default="results/unified/retime")
    ap.add_argument("--skip", default="", help="comma list of process kinds to leave out (fix, hard, armA, armB, armB0)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--audit", action="store_true", help="print the items not complete yet (as an --items list)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.chdir(ROOT)
    py = sys.executable
    skip = {x for x in a.skip.split(",") if x}
    if a.audit:
        done, missing = audit(a.res, skip)
        print(f"{len(done)} items complete, {len(missing)} not")
        print(",".join(f"{b}:{s}:{k}" for b, s, k in missing))
        return
    if a.items:
        mine = [(x.split(":")[0], x.split(":")[1], int(x.split(":")[2])) for x in a.items.split(",") if x]
        tag = a.tag or "t" + (os.environ.get("SLURM_ARRAY_TASK_ID") or "x")
    elif a.pull:
        mine, tag = None, a.tag or f"j{os.environ.get('SLURM_JOB_ID', 'x')}"
    else:
        parts, load = assign(items_all(), a.ntasks)
        mine, tag = sorted(parts[a.task], key=lambda x: (x[0] != "uc12" or x[1] != "tf", x)), f"t{a.task}"
        if a.list:
            print(f"{len(items_all())} items, {a.ntasks} tasks, est. minutes per task: {[round(x) for x in load]}")
    if a.list:
        print(f"task {a.task}: {len(mine)} items, first {mine[:3]}")
        for kind, c in commands(mine[0], a.res, tag, "python3").items():
            print(f"  {kind}: {' '.join(c)}")
        return
    os.makedirs(a.res, exist_ok=True)
    man = os.path.join(a.res, f"manifest_{tag}.jsonl")
    done = set()
    if os.path.exists(man):
        for line in open(man):
            r = json.loads(line)
            if r["rc"] == 0:
                done.add((r["bench"], r["set"], r["k"], r["kind"]))
    env = dict(os.environ, OTSL_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    host, cpu = socket.gethostname(), cpu_model()
    print(f"{tag}: {'pull mode' if a.pull else f'{len(mine)} items'} on {host} ({cpu}), CPUs {cpus()}", flush=True)
    t_all = time.time()

    def run_item(j, it):
        cmd = {k_: v for k_, v in commands(it, a.res, tag, py).items() if k_ not in skip}
        order = sorted(cmd)
        rng_seed = zlib.crc32(f"{it[0]}/{it[1]}/{it[2]}/{a.seed}".encode())
        order = [order[i] for i in sorted(range(len(order)), key=lambda i: zlib.crc32(f"{rng_seed}/{order[i]}".encode()))]
        for kind in order:
            if (it[0], it[1], it[2], kind) in done:
                continue
            l0, t0 = os.getloadavg()[0], time.time()
            p = subprocess.run(cmd[kind], env=env, cwd=ROOT, capture_output=True, text=True)
            rec = dict(bench=it[0], set=it[1], k=it[2], kind=kind, rc=p.returncode, start=t0, wall=time.time() - t0,
                       loadavg0=l0, loadavg1=os.getloadavg()[0], host=host, cpu_model=cpu, order=order,
                       cpus=cpus(), job=os.environ.get("SLURM_JOB_ID"), tag=tag)
            if p.returncode != 0:
                rec["stderr_tail"] = p.stderr[-3000:]
            with open(man, "a") as f:
                f.write(json.dumps(rec) + "\n")
            with open(os.path.join(a.res, f"log_{tag}.txt"), "a") as f:
                f.write(f"==== {it} {kind} rc={p.returncode} {rec['wall']:.0f}s\n{p.stdout[-4000:]}\n")
                if p.returncode:
                    f.write(p.stderr[-4000:] + "\n")
            print(f"[{j} {time.time() - t_all:.0f}s] {it} {kind}: rc {p.returncode}, {rec['wall']:.0f}s", flush=True)

    if not a.pull:
        for j, it in enumerate(mine):
            run_item(f"{j + 1}/{len(mine)}", it)
    else:
        ready = os.path.join(a.res, "prep", "READY")
        claims = os.path.join(a.res, "claims")
        order = sorted(items_all(), key=lambda x: (-cost(x), x))           # longest first: a short tail at the end
        n = 0
        while True:
            took = waiting = False
            for it in order:
                if os.path.exists(os.path.join(claims, f"{it[0]}_{it[1]}_{it[2]}")):
                    continue
                if it[0] == "uc12" and not os.path.exists(ready):
                    waiting = True
                    continue
                if claim(a.res, it, tag):
                    n += 1
                    run_item(f"#{n}", it)
                    took = True
                    break                                                  # re-scan: READY may have appeared
            if took:
                continue
            if not waiting:
                break
            time.sleep(60)
    print(f"{tag}: done in {time.time() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()

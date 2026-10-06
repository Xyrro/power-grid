"""Hybrid study: the whole run sequence, one heavy job at a time (one core), with the validation-only selection rules
fixed here before any result was seen.

    python3 scripts/uc_hybrid_run.py --eval_at 08:55 [--eval_n 60]

Sequence (each step a subprocess; the log is results/uc12/hybrid_run.log):
 0. wait for data/generated/uc12/val_extra2.json (180 new validation instances, uc_gen.py --seed 41)
 1. BCE GNN seeds 1, 2 (uc_hybrid_train.py); probabilities and error-cost scores (uc_hybrid_prep.py)
 2. guard-aware tuning, eps = 1 %, guards in the check = adequacy + min up/down rows, seed 0, 180 validation
    instances (the size of the faithful runs):
      hg  probabilities of the BCE GNN
      he  error-cost scores on the BCE GNN
 3. selection (validation only, the paper's criterion): the family with the larger guarded validation fixed share
    -> its seed 0 on 360 instances (warm start from the 180 cuts + held-out check of the 180 thresholds on the new
    instances), its seeds 1, 2 on 180, then on 360, then its seed 0 at eps = 5 % (180)
 4. if time remains: the "all guards in the check" variant (adequacy + rows + LP relaxation) on 180; held-out check
    of the faithful LtF-BCE thresholds (eps = 1 %, tuned on 180) on the 180 new instances; the other family's seed 0
    on 360; the held-out check of the faithful LtF-kNN thresholds
 5. at --eval_at (or when 1-4 are done): test evaluation (uc_hybrid_eval.py) on the first --eval_n test_fresh
    instances, then the report.
Optional steps are skipped when their expected duration would cross --eval_at; a tuning job's budget ends at --eval_at
(an unconverged run is not evaluated).
"""
import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = "results/uc12"
LOG = os.path.join(OUT, "hybrid_run.log")
GUARD = "adeq+rows"


def log(s):
    line = f"[{dt.datetime.now().strftime('%H:%M:%S')}] {s}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def run(cmd):
    log("RUN " + " ".join(cmd))
    t0 = time.time()
    with open(LOG, "a") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    log(f"exit {r.returncode} after {time.time() - t0:.0f}s")
    return r.returncode


def minutes_left(deadline):
    return (deadline - time.time()) / 60


def tune(specs, deadline, est_min, budget_cap=60.0):
    """run tuning jobs (one process each) if est_min fits before the deadline; returns True if run"""
    left = minutes_left(deadline)
    if left < est_min:
        log(f"skip {specs}: {left:.0f} min left, needs ~{est_min:.0f}")
        return False
    for s in specs:
        left = minutes_left(deadline)
        if left < 3:
            log(f"skip {s}: deadline")
            return False
        run([sys.executable, "scripts/uc_hybrid_tune.py", "--job", s, "--budget_min", f"{min(budget_cap, left - 1):.1f}"])
    return True


def fixed_share(tag):
    p = os.path.join(OUT, f"hybrid_tune_{tag}.json")
    if not os.path.exists(p):
        return -1.0
    r = json.load(open(p))
    return r["val_fixed_share"] if r["converged"] else -1.0


def wall_min(tag):
    p = os.path.join(OUT, f"hybrid_tune_{tag}.json")
    return json.load(open(p))["wall_s"] / 60 if os.path.exists(p) else 30.0


def pair(fam, src, seed, eps="0.01"):
    """180 -> 360 (warm start) specs for one family / seed / eps"""
    e = {"0.01": "e1", "0.05": "e5"}[eps]
    key = src if fam == "hg" else f"harm_{src}"
    t180, t360 = f"{fam}_{src}_{e}_n180", f"{fam}_{src}_{e}_n360"
    return [f"{t180}:{key}:{eps}:{GUARD}:180", f"{t360}:{key}:{eps}:{GUARD}:360:{t180}"], t180, t360


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_at", default="09:15", help="local HH:MM at which the test evaluation starts at the latest")
    ap.add_argument("--eval_n", type=int, default=60)
    ap.add_argument("--skip_to", type=int, default=0)
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    now = dt.datetime.now()
    hh, mm = map(int, a.eval_at.split(":"))
    deadline = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if deadline < now:
        deadline += dt.timedelta(days=1)
    deadline = deadline.timestamp()
    log(f"driver start; evaluation at {a.eval_at} at the latest ({minutes_left(deadline):.0f} min)")
    # 0. data
    while not os.path.exists("data/generated/uc12/val_extra2.json"):
        time.sleep(30)
    log("val_extra2 ready")
    if a.skip_to <= 1:
        run([sys.executable, "scripts/uc_hybrid_train.py", "--seeds", "1,2"])
        run([sys.executable, "scripts/uc_hybrid_prep.py"])
    # 2. seed 0 of both families, 180 validation instances (the size of the faithful runs)
    if a.skip_to <= 2:
        for fam in ("hg", "he"):
            specs, _, _ = pair(fam, "bce_s0", 0)
            tune(specs[:1], deadline, 5)
    # 3. selection on validation (the paper's criterion)
    fs = {fam: fixed_share(f"{fam}_bce_s0_e1_n180") for fam in ("hg", "he")}
    best = max(fs, key=fs.get) if fs["hg"] != fs["he"] else "hg"
    other = "he" if best == "hg" else "hg"
    json.dump({"rule": "larger guarded validation fixed share at eps = 1 %, 180 instances, seed 0 (tie: hg)",
               "val_fixed_share": fs, "chosen": best}, open(os.path.join(OUT, "hybrid_selection.json"), "w"), indent=1)
    log(f"selection (validation): {fs} -> {best}")
    w180 = wall_min(f"{best}_bce_s0_e1_n180")
    specs, _, _ = pair(best, "bce_s0", 0)
    tune(specs[1:], deadline, 0.5 * w180)                       # seed 0 on 360 (warm start, held-out check)
    w360 = wall_min(f"{best}_bce_s0_e1_n360")
    for s in (1, 2):                                            # seeds 1, 2 on 180
        specs, _, _ = pair(best, f"bce_s{s}", s)
        tune(specs[:1], deadline, 0.7 * w180)
    for s in (1, 2):                                            # ... and on 360 (warm start)
        if fixed_share(f"{best}_bce_s{s}_e1_n180") > 0:
            specs, _, _ = pair(best, f"bce_s{s}", s)
            tune(specs[1:], deadline, 0.8 * w360)
    key = "bce_s0" if best == "hg" else "harm_bce_s0"
    tune([f"{best}_bce_s0_e5_n180:{key}:0.05:{GUARD}:180"], deadline, 0.7 * w180)
    # 4. optional
    tune([f"hn_bce_s0_e1_n180:bce_s0:0.01:all:180"], deadline, 15, budget_cap=25)
    if minutes_left(deadline) > 15:
        run([sys.executable, "scripts/uc_hybrid_tune.py", "--holdout", "ltfx_tune_bce_1:bce_s0:none:180"])
    specs, _, _ = pair(other, "bce_s0", 0)
    tune(specs[1:], deadline, 0.8 * w360)
    if minutes_left(deadline) > 15:
        run([sys.executable, "scripts/uc_hybrid_tune.py", "--holdout", "ltfx_tune_knn_1:knn:none:180"])
    # 5. test: the selected family (all seeds, both validation sizes), the other family's seed 0, eps = 5 %, the
    #    all-guards variant; the LP guard at test for the selected family's seed 0 on the largest validation set
    tags = []
    for s in (0, 1, 2):
        for n in (180, 360):
            tags.append(f"{best}_bce_s{s}_e1_n{n}")
    tags += [f"{other}_bce_s0_e1_n180", f"{other}_bce_s0_e1_n360", f"{best}_bce_s0_e5_n180", "hn_bce_s0_e1_n180"]
    tags = [t for t in tags if fixed_share(t) > 0]
    lp = f"{best}_bce_s0_e1_n360" if fixed_share(f"{best}_bce_s0_e1_n360") > 0 else f"{best}_bce_s0_e1_n180"
    log(f"test rules: {tags} (+lp: {lp})")
    run([sys.executable, "scripts/uc_hybrid_eval.py", "--n", str(a.eval_n), "--workers", "1", "--tags", ",".join(tags),
         "--lp_tags", lp])
    run([sys.executable, "scripts/uc_hybrid_report.py"])
    log("driver done")

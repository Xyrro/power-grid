"""hard benchmark: Learning to Fix's joint threshold tuning (Algorithm 1 + 2) on the validation instances.

    python scripts/uc_hard_tune.py --run lp --src lp --tuner faithful       # learning-free LtF: pi = LP relaxation
    python scripts/uc_hard_tune.py --run knn --src knn --tuner faithful     # LtF-kNN (eq. 13, k = 50)
    python scripts/uc_hard_tune.py --run hg_fin --src fin_ens --tuner guarded   # learned pipeline's thresholds

Sources: lp = u_rel of each validation instance (no model); knn = otsl.ltfx.KNNProb on the training scenarios with
their polished labels (no full MILP; written to data/generated/hard/probs/knn_probs.npz with the test probabilities);
otherwise data/generated/hard/probs/<src>_probs.npz. Tuners: faithful = otsl.uc24ltf.StartLtFTuner (the paper's
check on the plain fixings, relaxation warm starts), guarded = otsl.hard.HardTuner (adequacy guard + minimal min
up/down conflict release inside the check, the cuts and the verification). Settings of the uc24 runs: eps 1 %, Q = 20,
K_max = 10, relaxation MILPs 8 s, check MILPs 60 s, warm-start checks 15 s; C* = the validation MILP objective (0.1 %).
Output: results/hard/hard_tune_<run>.{json,log} (+ _cuts not kept). A converged run is skipped on rerun.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from otsl.hard import DATA_DIR, HardTuner, bench_system, load_split  # noqa: E402
from otsl.ltfx import KNNProb, fix_masks  # noqa: E402
from otsl.uc24ltf import StartLtFTuner  # noqa: E402

RES = "results/hard"


def knn_probs(sysm, v, te):
    """LtF-kNN probabilities of validation and test from the training scenarios with polished labels"""
    import glob
    from otsl.b3 import load_b3
    parts, ys = [], []
    for f in sorted(glob.glob(os.path.join(DATA_DIR, "train_[0-9][0-9].npz"))):
        pf = f.replace("train_", "polish_")
        if os.path.exists(pf):
            d = load_b3(f)
            parts.append({k: d[k] for k in ("load", "avail", "u0", "start")})
            ys.append(np.load(pf)["y"])
    tr = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    tr["u"] = np.concatenate(ys)
    knn = KNNProb(sysm, tr, k=50)
    out = dict(val=knn.predict(v).astype(np.float32), n_train=len(tr["u"]))
    if te:
        out["test"] = knn.predict(te).astype(np.float32)
    np.savez_compressed(os.path.join(DATA_DIR, "probs", "knn_probs.npz"), **out)
    return out["val"].astype(np.float64)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--tuner", default="faithful", choices=["faithful", "guarded"])
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--budget_min", type=float, default=120.0)
    ap.add_argument("--relax_tl", type=float, default=8.0)
    ap.add_argument("--check_tl", type=float, default=60.0)
    ap.add_argument("--n_val", type=int, default=0)
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    out = os.path.join(RES, f"hard_tune_{a.run}.json")
    if os.path.exists(out) and json.load(open(out))["converged"]:
        print(f"{a.run}: already converged, skipped", flush=True)
        sys.exit(0)
    import torch
    torch.set_num_threads(1)
    sysm, m = bench_system()
    v = load_split("val")
    n = a.n_val or len(v["obj"])
    v = {k: x[:n] for k, x in v.items()}
    if a.src == "lp":
        pi = v["u_rel"].astype(np.float64)
    elif a.src == "knn":
        os.makedirs(os.path.join(DATA_DIR, "probs"), exist_ok=True)
        pi = knn_probs(sysm, v, load_split("test"))[:n]
    else:
        pi = np.load(os.path.join(DATA_DIR, "probs", f"{a.src}_probs.npz"))["val"].astype(np.float64)[:n]
    assert len(pi) == len(v["obj"])
    logf = open(os.path.join(RES, f"hard_tune_{a.run}.log"), "w")

    def log(s):
        logf.write(s + "\n")
        logf.flush()
    log(f"hard tuning {a.run}: src {a.src} tuner {a.tuner} eps={a.eps} n_val={n} relax_tl={a.relax_tl} check_tl={a.check_tl}")
    t0, c0 = time.time(), time.process_time()
    cls = HardTuner if a.tuner == "guarded" else StartLtFTuner
    tu = cls(m, sysm, v, pi, a.eps, K_max=10, Q=20, check_tl=a.check_tl, relax_tl=a.relax_tl, master_tl=120.0, log=log,
             time_budget=a.budget_min * 60, start_tl=15.0)
    lo, hi = tu.run()
    conv = tu.history[-1].get("result") == "all instances feasible"
    ver = tu.verify(lo, hi)
    off, on = fix_masks(pi, lo, hi)
    if a.tuner == "guarded":
        po, pn = tu.guarded_masks(lo, hi)
    else:
        po, pn = off, on
    res = dict(run=a.run, src=a.src, tuner=a.tuner, eps=a.eps, converged=conv, lo=lo.tolist(), hi=hi.tolist(), n_val=n,
               val_fixed_share_pre_guard=float((off | on).mean()), val_fixed_share=float((po | pn).mean()),
               val_fixed_off=float(po.mean()), val_fixed_on=float(pn.mean()), n_iter=len(tu.history),
               n_cut_sets=len(tu.master.cuts), verify_rel_increase=ver.tolist(), verify_max=float(np.max(ver)),
               verify_ok=bool(np.all(ver <= a.eps * (1 + 1e-6))), n_verify_fail=int((ver > a.eps * (1 + 1e-6)).sum()),
               stats={k: v_ for k, v_ in tu.stats.items()}, history=tu.history, wall_s=time.time() - t0,
               cpu_s=time.process_time() - c0, cfg=dict(K_max=10, Q=20, relax_tl=a.relax_tl, check_tl=a.check_tl,
                                                       budget_s=a.budget_min * 60, start_tl=15.0))
    json.dump(res, open(out, "w"), indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))
    log(f"done: converged={conv} val fixed {res['val_fixed_share'] * 100:.2f}% after guards "
        f"({res['val_fixed_share_pre_guard'] * 100:.2f}% before), verify max {res['verify_max'] * 100:.3f}% "
        f"({res['n_verify_fail']} above eps), wall {res['wall_s']:.0f}s")
    print(f"{a.run}: converged={conv} val fixed {res['val_fixed_share'] * 100:.2f}% "
          f"(pre {res['val_fixed_share_pre_guard'] * 100:.2f}%) wall {res['wall_s']:.0f}s", flush=True)

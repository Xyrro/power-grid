"""m1x study, downstream: Learning to Fix's joint eps-tuning (otsl.hybrid.HybridTuner, guard-aware) on new
probability sources, on the same 360 validation instances as the hybrid reference (val + val_extra + val_extra2).

    python3 scripts/uc_m1x_tune.py --job RUN:SCORE:EPS:GUARDS [--budget_min 90]

  SCORE   harm:<src>   error-cost scores of the harm ensemble combo_harm_s0.pt on features of <src>'s probabilities,
                       mapped by otsl.hybrid.harm_to_score (mu, sd of log h over the first 180 validation instances,
                       as in uc_hybrid_prep.py) - the reference hybrid's score family ("he")
          prob:<src>   the probabilities themselves ("hg")
          <src> = a tag of data/generated/uc12_m1x/probs/<src>_probs.npz (validation probabilities, 360 instances)
  GUARDS  adeq+rows (the reference hybrid's check), none, all
Settings as the reference (scripts/uc_hybrid_tune.py): Q = 20, K_max = 10, relaxation MILPs 6 s, check MILPs 60 s.
Output: results/uc12/m1x_tune_<RUN>.json / .log / _cuts.jsonl; harm scores cached in
data/generated/uc12_m1x/probs/<src>_harm_probs.npz.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from otsl.combo import harm_from_file, harm_scores  # noqa: E402
from otsl.fixpolicy import FixFeaturizer  # noqa: E402
from otsl.hybrid import HybridTuner, harm_norm, harm_to_score  # noqa: E402
from otsl.ltfx import fix_masks  # noqa: E402
from otsl.uc import UCModel, load_rts_gmlc  # noqa: E402
from uc_hybrid_tune import parse_guards, val_data  # noqa: E402
from uc_m1x_train import val_sets  # noqa: E402

DATA, RES = "data/generated/uc12_m1x", "results/uc12"
N_VAL = 360


def harm_val(src, sysm):
    """harm scores of the 360 validation instances for probability source src (cached)"""
    path = os.path.join(DATA, "probs", f"{src}_harm_probs.npz")
    if os.path.exists(path):
        return np.load(path)["val"]
    p = np.load(os.path.join(DATA, "probs", f"{src}_probs.npz"))["val"].astype(np.float64)
    d = val_sets()
    ens = harm_from_file(os.path.join(RES, "combo_harm_s0.pt"))
    H = harm_scores(ens, FixFeaturizer(sysm, 12), p, d, list(range(N_VAL))).astype(np.float64)
    np.savez_compressed(path, val=H)
    return H


def scores(spec, sysm):
    kind, src = spec.split(":", 1)
    p = np.load(os.path.join(DATA, "probs", f"{src}_probs.npz"))["val"].astype(np.float64)
    if kind == "prob":
        return p, None
    H = harm_val(src, sysm)
    norm = harm_norm(H[:180])
    return harm_to_score(H, p, *norm), norm


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--budget_min", type=float, default=90.0)
    ap.add_argument("--relax_tl", type=float, default=6.0)
    ap.add_argument("--check_tl", type=float, default=60.0)
    ap.add_argument("--K_max", type=int, default=10)
    ap.add_argument("--Q", type=int, default=20)
    ap.add_argument("--harm_only", action="store_true", help="only compute and cache the harm scores")
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    torch.set_num_threads(1)
    run, spec, eps, gname = a.job.split(":", 1)[0], *a.job.split(":", 1)[1].rsplit(":", 2)
    eps = float(eps)
    guards = parse_guards(gname)
    sysm = load_rts_gmlc()
    pi, norm = scores(spec, sysm)
    if a.harm_only:
        sys.exit(0)
    t_cpu0, t0 = time.process_time(), time.time()
    m = UCModel(sysm, T=12, network=True)
    d = val_data(N_VAL)
    assert len(pi) == N_VAL == len(d["obj"])
    logf = open(os.path.join(RES, f"m1x_tune_{run}.log"), "w")

    def log(s):
        logf.write(s + "\n")
        logf.flush()
    cut_log = os.path.join(RES, f"m1x_tune_{run}_cuts.jsonl")
    open(cut_log, "w").close()
    log(f"m1x tuning {run}: score {spec} eps={eps} guards={guards} n_val={N_VAL} K_max={a.K_max} relax_tl={a.relax_tl} "
        f"check_tl={a.check_tl} Q={a.Q} harm norm={norm}")
    tu = HybridTuner(m, sysm, d, pi, eps, guards=guards, cut_log=cut_log, K_max=a.K_max, Q=a.Q, check_tl=a.check_tl,
                     relax_tl=a.relax_tl, master_tl=120.0, log=log, time_budget=a.budget_min * 60)
    lo, hi = tu.run()
    conv = tu.history[-1].get("result") == "all instances feasible"
    ver = tu.verify(lo, hi)
    log(f"verification: max rel. cost increase {np.max(ver) * 100:.4f}% (eps {eps * 100:g}%), instances above eps: "
        f"{int((ver > eps * (1 + 1e-6)).sum())}")
    off, on = fix_masks(pi, lo, hi)
    po, pn = tu.guarded_masks(lo, hi)
    res = dict(run=run, score=spec, eps=eps, guards=list(guards), converged=conv, lo=lo.tolist(), hi=hi.tolist(),
               n_val=N_VAL, harm_norm=norm, val_fixed_share_pre_guard=float((off | on).mean()),
               val_fixed_share=float((po | pn).mean()), val_fixed_off=float(po.mean()), val_fixed_on=float(pn.mean()),
               n_iter=len(tu.history), n_cut_sets=len(tu.master.cuts), verify_rel_increase=ver.tolist(),
               verify_max=float(np.max(ver)), verify_ok=bool(np.all(ver <= eps * (1 + 1e-6))), stats=tu.stats,
               history=tu.history, cfg=dict(K_max=a.K_max, Q=a.Q, relax_tl=a.relax_tl, check_tl=a.check_tl,
                                            budget_s=a.budget_min * 60),
               wall_s=time.time() - t0, cpu_s=time.process_time() - t_cpu0)
    with open(os.path.join(RES, f"m1x_tune_{run}.json"), "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o))
    log(f"done: converged={conv} val fixed {res['val_fixed_share'] * 100:.2f}% after guards "
        f"({res['val_fixed_share_pre_guard'] * 100:.2f}% before) wall {res['wall_s']:.0f}s cpu {res['cpu_s']:.0f}s")
    print(f"{run}: converged={conv} val fixed {res['val_fixed_share'] * 100:.2f}% wall {res['wall_s']:.0f}s", flush=True)

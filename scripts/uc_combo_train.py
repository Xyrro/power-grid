"""Seed replicates of the trainable components (B2, uc12), for the combined pipelines.

    python scripts/uc_combo_train.py --stage st   --seed 1   # self-trained BCE on the existing label pool
    python scripts/uc_combo_train.py --stage harm --seed 1   # error-cost (harm) model on the existing labels
    python scripts/uc_combo_train.py --stage rl   --seed 1   # label-free + plain REINFORCE (rl_lf)
    python scripts/uc_combo_train.py --stage lag  --seed 1   # label-free + Lagrangian PG + KL anchor (lag_D)

Every stage reproduces the original recipe exactly, only the seed changes:
  st    otsl.selftrain.train_bce_fixed on results/uc12/selftrain_pool.npz (round-3 labels), 80 epochs,
        network initialised with the same seed (as scripts/uc_selftrain.py, stage rounds)
  harm  scripts/uc_fixpolicy_train.py --labels comp (3-member ensemble, 20 epochs, hidden 32, lr 5e-4, wd 1e-2,
        w_reg 1); replicate s uses member seeds 3s, 3s+1, 3s+2 (replicate 0 = the saved policy)
  rl    scripts/uc_constrained.py --stage rl --name rl_lf (otsl.ucml.train_uc_reinforce, 120 steps, bs 16, 6 samples)
  lag   scripts/uc_constrained.py --stage lag --name lag_D --muprop 1 --window 1 --kl 0.05 --eps 0.15 --lam_max 5
Seed 0 of rl / lag is the existing checkpoint (constrained_rl_lf.pt / constrained_lag_D.pt); st and harm seed 0 are
retrained and compared with the saved models (reproduction check).
Output: results/uc12/combo_<stage>_s<seed>.pt and combo_train_<stage>_s<seed>.json
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from otsl.constrained import HourlyOracle, train_constrained  # noqa: E402
from otsl.fixpolicy import FixFeaturizer, HarmEnsemble, HarmModel, fixed_harm  # noqa: E402
from otsl.selftrain import train_bce_fixed  # noqa: E402
from otsl.ucml import UCFeaturizer, build_uc_model1, train_uc_reinforce  # noqa: E402
from uc_constrained import _ScoreAdapter, load_model1, make_repair, setup, strip  # noqa: E402
from uc_fixpolicy_train import harm_array  # noqa: E402

OUT = os.path.join("results", "uc12")


def dump(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["st", "harm", "rl", "lag"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--tag", default="", help="suffix for smoke tests")
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    torch.set_num_threads(1)
    cfg, tr_full, va_full, te, sysm, rep = setup()
    T = cfg["T"]
    tr, va = strip(tr_full), strip(va_full)
    s = a.seed
    name = f"combo_{a.stage}_s{s}{a.tag}"
    t0 = time.time()
    info = {"stage": a.stage, "seed": s}

    if a.stage == "st":
        pool = np.load(os.path.join(OUT, "selftrain_pool.npz"))
        feat = UCFeaturizer(sysm, tr, relax=True, sym=True)
        m1 = build_uc_model1(sysm, feat, T, "gnn", seed=s)
        train_bce_fixed(m1, tr, pool["Y"], epochs=a.epochs, seed=s)
        torch.save(m1.net.state_dict(), os.path.join(OUT, name + ".pt"))
        if s == 0:     # reproduction check against the saved round-3 model
            ref = load_model1(os.path.join(OUT, "selftrain_m3.pt"), sysm, tr, T)
            pa, pb = m1.predict(va), ref.predict(va)
            info["repro_max_abs_diff_val"] = float(np.abs(pa - pb).max())
            info["repro_decisions_differ_val"] = int(((pa > 0.5) != (pb > 0.5)).sum())
            print("reproduction vs selftrain_m3.pt:", info, flush=True)

    elif a.stage == "harm":
        P = np.load(os.path.join(OUT, "fixpolicy_probs.npz"))
        with open(os.path.join(OUT, "fixpolicy_labels.pkl"), "rb") as f:
            labels = pickle.load(f)
        with open(os.path.join(OUT, "fixpolicy_labels_comp.pkl"), "rb") as f:
            comp = pickle.load(f)
        ids_tr = sorted(k[1] for k in labels if k[0] == "tr" and k in comp)
        n_va = len(va_full["load"])
        G = sysm.G
        H_tr, _ = harm_array(labels, "tr", ids_tr, T, G, comp)
        H_va, _ = harm_array(labels, "va", range(n_va), T, G, comp)
        ff = FixFeaturizer(sysm, T)
        X_tr = np.stack([ff(P["bce_oof_tr"][i], tr_full, i) for i in ids_tr])
        X_va = np.stack([ff(P["bce_va"][i], va_full, i) for i in range(n_va)])
        members = []
        for ms in (3 * s, 3 * s + 1, 3 * s + 2):
            hm = HarmModel(X_tr.shape[-1], hidden=32, seed=ms)
            hm.fit(X_tr.reshape(-1, X_tr.shape[-1]), H_tr.reshape(-1), X_va, H_va, epochs=20, lr=5e-4, wd=1e-2,
                   w_reg=1.0, seed=ms, log=None)
            print(f"  [harm] member seed {ms}: best val harm-in-fixed-set {hm.best_val:.4f}", flush=True)
            members.append(hm)
        ens = HarmEnsemble(members)
        info["val_crit"] = {"ensemble": fixed_harm(ens.score(X_va), H_va),
                            "RACLearn": fixed_harm(np.minimum(P["bce_va"], 1 - P["bce_va"]), H_va),
                            "members": [m.best_val for m in members]}
        torch.save({"members": [{"state": m.net.state_dict(), "mu": m.mu, "sd": m.sd} for m in members],
                    "hidden": 32, "d_in": X_tr.shape[-1], "member_seeds": [3 * s, 3 * s + 1, 3 * s + 2]},
                   os.path.join(OUT, name + ".pt"))
        if s == 0:
            pol = torch.load(os.path.join(OUT, "fixpolicy_policy_comp.pt"), weights_only=False)
            ref = HarmEnsemble.from_states(pol["d_in"], pol["hidden"], pol["members"])
            ra, rb = ens.score(X_va), ref.score(X_va)
            info["repro_max_rel_diff_val"] = float(np.max(np.abs(ra - rb) / np.maximum(rb, 1e-12)))
            print("reproduction vs fixpolicy_policy_comp.pt:", info["repro_max_rel_diff_val"], flush=True)
        print(info["val_crit"], flush=True)

    else:
        m1 = load_model1(os.path.join(OUT, "constrained_lf_bce.pt"), sysm, tr, T)
        oracle = HourlyOracle(cfg, a.workers)
        if a.stage == "rl":
            _, hist = train_uc_reinforce(m1, tr, _ScoreAdapter(oracle), steps=a.steps, bs=a.bs, n_samples=6, lr=3e-4,
                                         seed=s, repair=rep, log_every=10)
        else:
            _, hist = train_constrained(m1, tr, oracle, tr["c_rel"], make_repair("mud", tr, sysm, rep), steps=a.steps,
                                        bs=a.bs, n_samples=5, lr=3e-4, seed=s, eps=0.15, lam0=1.0, lam_lr=1.0, window=1,
                                        kappa=1e-3, tau=1.0, muprop=True, beta=1.0, lam_max=5.0, kl=0.05)
        info["train_LPs"] = oracle.n
        info["hist"] = hist
        torch.save(m1.net.state_dict(), os.path.join(OUT, name + ".pt"))
        oracle.close()
    info["train_s"] = time.time() - t0
    dump(os.path.join(OUT, f"combo_train_{a.stage}_s{s}{a.tag}.json"), info)
    print(f"done {name} ({info['train_s']:.0f}s)", flush=True)

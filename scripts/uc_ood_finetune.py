"""Robustness study, step 4 (optional): can a few labelled shifted instances recover what a shift costs?

    OTSL_THREADS=1 taskset -c 2 python3 scripts/uc_ood_finetune.py --shift gen_out --stage gen     # 50 MILP labels
    OTSL_THREADS=1 taskset -c 2 python3 scripts/uc_ood_finetune.py --shift gen_out --stage train   # fine-tune BCE GNN
    OTSL_THREADS=1 taskset -c 2 python3 scripts/uc_ood_eval.py --shifts gen_out --n 30 \
        --bce results/uc12/ood_ft_bce_gen_out.pt --knn_extra data/generated/uc12_ood/ft_gen_out.npz \
        --rules "LtF kNN eps=1%,LtF BCE eps=1%,hybrid eps=1%,guarded error-cost 90%" \
        --out results/uc12/ood_ft_eval.jsonl --inst_dir data/generated/uc12_ood/ft_eval

* gen: 50 instances of the shift on *training* calendar days (day % 5 in {1, 3, 4}; never a test day), seed 201, each
  with its full MILP (60 s, 0.1 %) and LP relaxation; written instance by instance (resumable) and gathered into
  data/generated/uc12_ood/ft_<shift>.npz.
* train: the MILP-label BCE GNN (uc_model1_4.pt) fine-tuned on 40 of them plus 200 original training instances
  (against forgetting), canonical labels, AdamW 3e-4, 30 epochs, early stopping on the log-loss of the other 10 shifted
  instances. The input normalisation stays the original one. Output results/uc12/ood_ft_bce_<shift>.pt.
* kNN: the 50 labelled instances are added to its pool (its form of fine-tuning). Thresholds, the error-cost model and
  every other setting stay as fixed on the original validation set.
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from otsl.ood import OODModel, build_shift_scenario, scenario_to_dict, solve_milp  # noqa: E402
from otsl.uc import load_rts_gmlc  # noqa: E402
from otsl.ucdata import load  # noqa: E402
from otsl.ucml import canonical_labels  # noqa: E402
from uc_constrained import load_model1, strip  # noqa: E402

ROOT, OUTD, OUT = "data/generated/uc12", "data/generated/uc12_ood", "results/uc12"
DAYS = np.arange(366)
TRAIN_DAYS = DAYS[(DAYS % 5 != 0) & (DAYS % 5 != 2)]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shift", required=True)
    ap.add_argument("--stage", choices=["gen", "train"], required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--n_val", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=30)
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    torch.set_num_threads(1)
    sysm = load_rts_gmlc()
    specs = json.load(open(os.path.join(OUTD, "specs.json")))
    sp = specs["shifts"][a.shift]
    s0, s1 = sp["starts"]
    days = TRAIN_DAYS[TRAIN_DAYS + 1 < 366] if s1 + 12 > 24 else TRAIN_DAYS
    pdir = os.path.join(OUTD, f"ft_{a.shift}")
    if a.stage == "gen":
        os.makedirs(pdir, exist_ok=True)
        m = OODModel(sysm, T=12)
        rng = np.random.default_rng(201)
        jobs = [(int(rng.choice(days)), int(rng.integers(s0, s1 + 1)), int(rng.integers(1 << 31))) for _ in range(a.n)]
        t0 = time.time()
        for k, (day, start, seed) in enumerate(jobs):
            f = os.path.join(pdir, f"{k:03d}.npz")
            if os.path.exists(f):
                continue
            sc = build_shift_scenario(sysm, m, sp["kw"], dict(day=day, start=start, seed=seed), specs.get("line_pool"))
            rel = m.solve_dispatch(sc, None, relax=True)
            sol = solve_milp(m, sc, {}, 60.0, 1e-3)
            np.savez_compressed(f, **scenario_to_dict(sc, dict(u=sol["u"], obj=sol["obj"], time=sol["time"], u_rel=rel.u,
                                                               lmp_rel=rel.lmp, flow_rel=rel.flow, c_rel=rel.obj)))
            print(f"[{k + 1}/{a.n} {time.time() - t0:.0f}s] day {day} start {start} MILP {sol['time']:.1f}s", flush=True)
        inst = [dict(np.load(os.path.join(pdir, f"{k:03d}.npz"))) for k in range(a.n)]
        keys = ("load", "avail", "u0", "sr", "start", "day", "u", "obj", "time", "u_rel", "lmp_rel", "flow_rel", "c_rel")
        np.savez_compressed(os.path.join(OUTD, f"ft_{a.shift}.npz"), **{key: np.stack([x[key] for x in inst]) for key in keys})
        print("saved", os.path.join(OUTD, f"ft_{a.shift}.npz"))
    else:
        tr = load(os.path.join(ROOT, "train.npz"))
        ft = load(os.path.join(OUTD, f"ft_{a.shift}.npz"))
        m1 = load_model1(os.path.join(OUT, "uc_model1_4.pt"), sysm, strip(tr), 12)
        rng = np.random.default_rng(0)
        orig = rng.choice(len(tr["load"]), 200, replace=False)
        nv = a.n_val
        fields = ("load", "avail", "u0", "sr", "u_rel", "lmp_rel", "flow_rel")
        dtr = {key: np.concatenate([ft[key][nv:], tr[key][orig]]) for key in fields}
        ytr = np.concatenate([canonical_labels(sysm, ft["u"][nv:], ft["u0"][nv:]), canonical_labels(sysm, tr["u"][orig], tr["u0"][orig])])
        dva = {key: ft[key][:nv] for key in fields}
        yva = torch.as_tensor(canonical_labels(sysm, ft["u"][:nv], ft["u0"][:nv]), dtype=torch.float32)
        y = torch.as_tensor(ytr, dtype=torch.float32)
        opt = torch.optim.AdamW(m1.net.parameters(), lr=3e-4, weight_decay=1e-4)

        def vloss():
            m1.net.eval()
            with torch.no_grad():
                return float(F.binary_cross_entropy_with_logits(m1.logits(dva, np.arange(nv)), yva))
        before = vloss()
        best, state, hist = before, copy.deepcopy(m1.net.state_dict()), []
        print(f"epoch 0 (original model): val log-loss on shifted {best:.4f}", flush=True)
        t0 = time.time()
        for ep in range(a.epochs):
            m1.net.train()
            perm = rng.permutation(len(y))
            for i in range(0, len(y), 32):
                idx = perm[i:i + 32]
                loss = F.binary_cross_entropy_with_logits(m1.logits(dtr, idx), y[idx])
                opt.zero_grad(); loss.backward(); opt.step()
            vl = vloss()
            hist.append(vl)
            if vl < best:
                best, state = vl, copy.deepcopy(m1.net.state_dict())
            print(f"epoch {ep + 1}: train {loss.item():.4f} val {vl:.4f} ({time.time() - t0:.0f}s)", flush=True)
        m1.net.load_state_dict(state)
        torch.save(m1.net.state_dict(), os.path.join(OUT, f"ood_ft_bce_{a.shift}.pt"))
        json.dump({"shift": a.shift, "val_logloss_before": before, "val_hist": hist,
                   "best": best, "train_s": time.time() - t0, "n_shifted_train": a.n - nv, "n_orig": 200},
                  open(os.path.join(OUT, f"ood_ft_bce_{a.shift}.json"), "w"), indent=1)

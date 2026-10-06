"""m1x study: validation-only selection of the probability family for the downstream tuning (rule written before the
polished-label, temporal-GNN and MLP results were seen).

    python3 scripts/uc_m1x_select.py

Default family: GNN on the 500 MILP labels (the reference recipe; seeds ref_bce_s0..s2 + pol_n500_gnn_s3, s4).
Candidates (seed 0, 360 validation instances): pol_n1000_gnn, pol_n2000_gnn, lf_n4000_gnn, pol_n500_gnnt, pol_n500_mlp.
A candidate replaces the default only if
  (i)  its share fixable at 99.9 % precision exceeds the default's seed mean by more than the default's seed range
       (max - min over its 5 seeds), i.e. by more than seed noise, and
  (ii) its log-loss is not above the default's seed mean;
the best such candidate by (i) wins. Downstream runs (guard-aware LtF, eps = 1 %, adeq+rows, 360 val):
  A  harm:milp500_gnn_ens5                    (deep ensemble alone)
  B  harm:<winner>_ens5                       if a candidate won (5 seeds of it are trained),
     else harm on the ensemble with the disagreement filter if it raised fixable@99.9 % by > 0.02 on validation,
     else prob:milp500_gnn_ens5               (the "hg" family: tuning on the ensemble probabilities themselves)
--extended adds pol_n2000_gnnt (temporal GNN on 2,000 polished instances) to the candidates, same criteria. It was added
after the phase-2 validation results (both more data and the temporal head passed the bar), before any downstream
tuning or test result.
Output: results/uc12/m1x_select.json (the extended selection is stored under "extended")
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = "results/uc12"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extended", action="store_true")
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    tr = json.load(open(os.path.join(RES, "m1x_train.json")))
    v = lambda t, k: tr[t]["val"]["all360"][k]
    seeds = ["ref_bce_s0", "ref_bce_s1", "ref_bce_s2", "pol_n500_gnn_s3", "pol_n500_gnn_s4"]
    f = np.array([v(t, "fix999") for t in seeds])
    ll = np.array([v(t, "logloss") for t in seeds])
    bar, ll_bar = f.mean() + (f.max() - f.min()), ll.mean()
    cands = {c: (v(c + "_s0", "fix999"), v(c + "_s0", "logloss")) for c in
             ("pol_n1000_gnn", "pol_n2000_gnn", "lf_n4000_gnn", "pol_n500_gnnt", "pol_n500_mlp")
             + (("pol_n2000_gnnt",) if a.extended else ()) if c + "_s0" in tr}
    ok = {c: x for c, x in cands.items() if x[0] > bar and x[1] <= ll_bar}
    winner = max(ok, key=lambda c: ok[c][0]) if ok else None
    ens = tr.get("milp500_gnn_ens5", {}).get("val", {}).get("all360", {})
    dis_gain = max(ens.get("fix999_dis1", 0), ens.get("fix999_dis2", 0)) - ens.get("fix999", 0) if ens else None
    if winner:
        b = f"harm:{winner}_ens5"
    elif dis_gain is not None and dis_gain > 0.02:
        b = "harm:milp500_gnn_ens5_dis"
    else:
        b = "prob:milp500_gnn_ens5"
    out = dict(default_seeds=seeds, default_fix999=f.tolist(), default_logloss=ll.tolist(), bar_fix999=float(bar),
               bar_logloss=float(ll_bar), candidates=cands, passing=ok, winner=winner, disagreement_gain=dis_gain,
               run_A="harm:milp500_gnn_ens5", run_B=b)
    path = os.path.join(RES, "m1x_select.json")
    if a.extended:
        base = json.load(open(path))
        base.setdefault("winner_original", base["winner"])
        base.setdefault("run_B_original", base["run_B"])
        base.update({k: v for k, v in out.items() if k in ("winner", "run_B")}, extended=out)
        out = base
    json.dump(out, open(path, "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))

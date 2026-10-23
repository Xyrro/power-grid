"""Step 2 of docs/unified_metrics.md: checks that the cluster's inputs are the ones the saved results used, before and
after the preparation jobs. Writes results/unified/retime/checks_<what>.json and prints a summary; exits 1 on a failure.

    python3 scripts/unified_s2_checks.py inputs     # test sets and the shifted instances against the saved records,
                                                    # the ensemble's stored validation probabilities against its models
    python3 scripts/unified_s2_checks.py tune       # the thresholds tuned here: converged, verified on validation
    python3 scripts/unified_s2_checks.py early      # the retrained early one-shot models: checkpoints and their batch run

inputs:
  test sets   the LP relaxation cost stored in data/generated/{uc12/test_fresh, uc24/test, hard/test_*}.npz against every
              saved record of the same instance (scripts/unified_rescore.py's registry; exact up to 1e-6 relative)
  shifts      every shifted instance (data/generated/uc12_ood/<shift>/<k>.npz) against the robustness study's record
              (results/uc12/ood_eval.jsonl): day, start window, units and lines out identical; the intact network's LP
              relaxation cost (re-solved here) equal up to 1e-6; the reference MILP's cost within both runs' MIP gaps
  ensemble    the 5-member temporal-GNN ensemble's validation probabilities (data/generated/uc12_m1x/probs/
              pol_n2000_gnnt_ens5_probs.npz, the tuner's input) re-computed from the checkpoints on 12 instances
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
RES = "results/unified/retime"
TUNE = os.environ.get("S2_TUNE", os.path.join(RES, "tune"))
SHIFTS = ("load_hi", "load_lo", "ren_hi", "gen_out", "line_out", "night")


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-9)


def inputs():
    import numpy as np
    import unified_rescore as U
    from otsl.b3 import load_b3
    from otsl.hard import load_split
    from otsl.ood import OODModel, OODScenario
    from otsl.uc import load_rts_gmlc
    U.load_all()
    out, ok = {}, True
    # test sets
    sets = {"tf": load_b3("data/generated/uc12/test_fresh.npz")["c_rel"], "u24": load_b3("data/generated/uc24/test.npz")["c_rel"],
            "hard": load_split("test")["c_rel"]}
    for key, c in sets.items():
        bad, n = [], 0
        for k, v in enumerate(c):
            saved = [s for s, _ in U.REG.crel.get((key, k), [])]
            n += bool(saved)
            if saved and any(rel(float(v), s) > 1e-6 for s in saved):
                bad.append((k, float(v), saved[:3]))
        out[f"test_{key}"] = dict(instances=len(c), with_saved_records=n, disagreements=len(bad), examples=bad[:3])
        ok &= not bad and n == len(c)
    # shifted instances
    R = {}
    for r in map(json.loads, open("results/uc12/ood_eval.jsonl")):
        if r["rule"] == "full MILP":
            R[(r["shift"], r["k"])] = r
    sysm = load_rts_gmlc()
    m = OODModel(sysm, T=12)
    for s in SHIFTS:
        res = dict(present=0, identical_spec=0, intact_lp_equal=0, milp_consistent=0, problems=[])
        for k in range(30):
            p = os.path.join("data/generated/uc12_ood", s, f"{k:03d}.npz")
            if not os.path.exists(p):
                res["problems"].append(f"{k}: missing")
                continue
            res["present"] += 1
            z, r = np.load(p), R[(s, k)]
            spec = (int(z["day"]) == r["day"] and int(z["start"]) == r["start"] and list(map(int, z["g_out"])) == r["g_out"]
                    and list(map(int, z["l_out"])) == r["l_out"])
            res["identical_spec"] += spec
            intact = OODScenario(load=z["load"], avail=z["avail"], u0=z["u0"], sr=z["sr"])
            c0 = float(m.solve_dispatch(intact, None, relax=True).obj)
            eq = "c_rel_intact" not in r or rel(c0, r["c_rel_intact"]) <= 1e-6
            res["intact_lp_equal"] += eq
            # both MILP costs lie above both bounds (each run proved its own bound)
            o1, b1, o2, b2 = float(z["obj_full"]), float(z["bound_full"]), r["obj"], r["bound"]
            cons = o1 >= b2 * (1 - 1e-6) and o2 >= b1 * (1 - 1e-6)
            res["milp_consistent"] += cons
            if not (spec and eq and cons):
                res["problems"].append(dict(k=k, spec=spec, intact_lp=[c0, r.get("c_rel_intact")], milp=[o1, b1, o2, b2]))
        out[f"shift_{s}"] = res
        ok &= res["present"] == 30 and not res["problems"]
    # the ensemble's stored validation probabilities against its checkpoints
    import torch
    from uc_m1x_train import featurizer, load_net
    torch.set_num_threads(1)
    feat = featurizer(sysm)
    nets = [load_net(f"results/uc12/m1x_pol_n2000_gnnt_s{s}.pt", "gnnt", sysm, feat) for s in range(5)]
    stored = np.load("data/generated/uc12_m1x/probs/pol_n2000_gnnt_ens5_probs.npz")["val"].astype(np.float64)
    keys = ("load", "avail", "u0", "sr", "start", "u_rel", "lmp_rel", "flow_rel")
    parts = [load_b3(f"data/generated/uc12/{f}.npz") for f in ("val", "val_extra", "val_extra2")]   # the tuner's order
    d = {k: np.concatenate([q[k] for q in parts]) for k in keys}
    assert len(d["load"]) == len(stored) == 360, (len(d["load"]), len(stored))
    idx = list(range(0, 360, 30))
    di = {k: d[k][idx] for k in keys}
    p = np.mean([n.predict(di).astype(np.float64) for n in nets], 0)
    dmax = float(np.abs(p - stored[idx]).max())
    out["ensemble_val_probs"] = dict(instances=len(idx), max_abs_diff=dmax)
    ok &= dmax < 1e-4
    return out, ok


def tune():
    out, ok = {}, True
    old = {"ltfx_tune_knn_1.json": "results/uc12/ltfx_tune_knn_1.json", "ltfx_tune_knn_5.json": "results/uc12/ltfx_tune_knn_5.json",
           "ltfx_tune_knn_10.json": "results/uc12/ltfx_tune_knn_10.json", "ltfx_tune_bce_1.json": "results/uc12/ltfx_tune_bce_1.json",
           "m1x_tune_hgP.json": "results/uc12/m1x_tune_hg_pol_n2000_gnnt_ens5.json",
           "m1x_tune_hgPb.json": "results/uc12/m1x_tune_hg_pol_n2000_gnnt_ens5.json",     # other seeds: compared with the
           "m1x_tune_hgPc.json": "results/uc12/m1x_tune_hg_pol_n2000_gnnt_ens5.json"}     # original ensemble's tuning
    for f, o in old.items():
        p = os.path.join(TUNE, f)
        if not os.path.exists(p):
            out[f] = "missing"
            ok = False
            continue
        j, jo = json.load(open(p)), json.load(open(o))
        out[f] = dict(converged=j["converged"], verify_ok=j["verify_ok"], verify_max=j["verify_max"], n_val=j["n_val"],
                      val_fixed_share=j["val_fixed_share"], original_val_fixed_share=jo["val_fixed_share"],
                      wall_s=j.get("wall_s"))
        ok &= bool(j["converged"] and j["verify_ok"])
    return out, ok


def early():
    out, ok = {}, True
    d = os.path.join(RES, "models")
    for f in ("uc_model1_1.pt", "uc_model1_2.pt", "uc_model1_3.pt", "uc_model1_4.pt", "uc_model1_rl.pt"):
        out[f] = os.path.exists(os.path.join(d, f))
        ok &= out[f]
    p = os.path.join(d, "uc_model1_results.json")
    if os.path.exists(p):
        j = json.load(open(p))
        out["batch_rows"] = {r["method"]: dict(served=r.get("no_shed_no_shortfall_%"), gap_median=r.get("gap_median_%"))
                             for r in j["rows"]}
        out["train_s"] = {k: v for k, v in j["extra"].items() if k.startswith(("train_s", "rl_time"))}
    else:
        ok = False
    return out, ok


if __name__ == "__main__":       # S2_TUNE=results/unified/retime/tune2 checks the second tuning pass
    os.chdir(os.path.dirname(HERE))
    what = sys.argv[1]
    out, ok = {"inputs": inputs, "tune": tune, "early": early}[what]()
    os.makedirs(RES, exist_ok=True)
    suffix = "" if os.path.basename(TUNE) == "tune" or what != "tune" else "_" + os.path.basename(TUNE)
    json.dump(dict(ok=ok, **out), open(os.path.join(RES, f"checks_{what}{suffix}.json"), "w"), indent=1, default=str)
    print(json.dumps(dict(ok=ok, **out), indent=1, default=str)[:6000])
    sys.exit(0 if ok else 1)

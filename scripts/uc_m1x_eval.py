"""m1x study, test: the full MILP and every reduced MILP back to back per instance, on core 3 (paired timing).

    python3 scripts/uc_m1x_eval.py --runs RUN1,RUN2 [--start 0 --n 60]

Rules (thresholds, scores and guards fixed on validation before this script reads test_fresh):
  reference   hybrid he_bce_s0_e1_n360 (results/uc12/hybrid_tune_he_bce_s0_e1_n360.json: guard-aware LtF on the
              error-cost scores of the MILP-label BCE GNN, eps = 1 %, 360 val) - re-run here, same worker
              faithful LtF BCE eps=1% (results/uc12/ltfx_tune_bce_1.json), re-run here
  m1x         results/uc12/m1x_tune_<RUN>.json for every RUN in --runs: score from the source's member models
              (data/generated/uc12_m1x/sources.json; every member's forward pass is timed), error-cost features and
              harm ensemble when the score is harm:<src>, eq. (5), then the tuned guards in the worker (timed)
Per instance (uc_hybrid_eval._job, unchanged): full MILP (60 s, 0.1 %), the LP relaxation that feeds the GNN (timed),
then each rule's guards and reduced MILP (60 s, 0.1 %); an infeasible reduced MILP falls back to the full MILP.
Method time = reduced MILP + inference (all members) + error-cost features and ensemble + LP relaxation + guards.
Output: results/uc12/m1x_eval_<split>.jsonl (resume skips finished instances).
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import uc_hybrid_eval as HE  # noqa: E402
from otsl.combo import harm_from_file, harm_scores  # noqa: E402
from otsl.fixpolicy import FixFeaturizer  # noqa: E402
from otsl.hybrid import harm_to_score  # noqa: E402
from otsl.ltfx import fix_dict  # noqa: E402
from otsl.uc import load_rts_gmlc  # noqa: E402
from otsl.ucdata import load, scenario_from  # noqa: E402
from uc_hybrid_tune import parse_guards  # noqa: E402
from uc_m1x_train import featurizer, load_net  # noqa: E402

ROOT, DATA, RES = "data/generated/uc12", "data/generated/uc12_m1x", "results/uc12"
REF = "hybrid he_bce_s0_e1_n360"


def rules(runs):
    """{rule name: (kind, src, lo, hi, guards, norm)}"""
    r = json.load(open(os.path.join(RES, "hybrid_tune_he_bce_s0_e1_n360.json")))
    norm = json.load(open(os.path.join(RES, "hybrid_probs.json")))["harm_norm_logh_mean_sd"]["bce_s0"]
    out = {REF: ("harm", "ref_bce_s0", np.array(r["lo"]), np.array(r["hi"]), tuple(r["guards"]), norm)}
    r = json.load(open(os.path.join(RES, "ltfx_tune_bce_1.json")))
    out["faithful LtF BCE eps=1%"] = ("prob", "ref_bce_s0", np.array(r["lo"]), np.array(r["hi"]), (), None)
    for run in runs:
        r = json.load(open(os.path.join(RES, f"m1x_tune_{run}.json")))
        assert r["converged"], run
        kind, src = r["score"].split(":", 1)
        out[f"m1x {run}"] = (kind, src, np.array(r["lo"]), np.array(r["hi"]), tuple(r["guards"]), r["harm_norm"])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="")
    ap.add_argument("--split", default="test_fresh")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    torch.set_num_threads(1)
    sysm = load_rts_gmlc()
    feat = featurizer(sysm)
    d = load(os.path.join(ROOT, f"{a.split}.npz"))
    idx = list(range(a.start, min(a.start + a.n, len(d["load"]))))
    one = lambda i: {k: v[i:i + 1] for k, v in d.items() if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(d["load"])}
    R = rules([x for x in a.runs.split(",") if x])
    reg = json.load(open(os.path.join(DATA, "sources.json")))
    srcs = sorted({v[1] for v in R.values()})
    members = {s: [(load_net(pth, kind, sysm, feat), pth) for pth, kind in reg[s]] for s in srcs}
    harm = harm_from_file(os.path.join(RES, "combo_harm_s0.pt"))
    ff = FixFeaturizer(sysm, 12)
    harm_scores(harm, ff, members[srcs[0]][0][0].predict(one(idx[0])), one(idx[0]), [0])            # warm-up
    chk = np.load(os.path.join(RES, "hybrid_probs.npz"))
    jobs = []
    for i in idx:
        di = one(i)
        sc = scenario_from(d, i)
        P, Tinf, H, Th = {}, {}, {}, {}
        for s in srcs:
            t0 = time.time()
            P[s] = np.mean([m.predict(di)[0].astype(np.float64) for m, _ in members[s]], 0)
            Tinf[s] = time.time() - t0
            t0 = time.time()
            H[s] = harm_scores(harm, ff, P[s][None], d, [i])[0]
            Th[s] = time.time() - t0
        if a.split == "test_fresh":     # the reference source reproduces the hybrid study's inputs
            assert np.abs(P["ref_bce_s0"] - chk["bce_s0_tf"][i]).max() < 1e-6
            assert np.abs(H["ref_bce_s0"] / chk["harm_bce_s0_tf"][i] - 1).max() < 1e-4
        specs = []
        for name, (kind, s, lo, hi, guards, norm) in R.items():
            t0 = time.time()
            score = harm_to_score(H[s], P[s], *norm) if kind == "harm" else P[s]
            fix = fix_dict(score, lo, hi)
            pre = Tinf[s] + (Th[s] if kind == "harm" else 0.0) + time.time() - t0
            specs.append((name, fix, pre, True, guards))
        jobs.append((i, sc, specs))
    print(f"{len(R)} rules per instance:", list(R), flush=True)
    if a.dry:
        for nm, fix, pre, _, gu in jobs[0][2]:
            print(f"  {nm:45s} fixed (pre-guard) {len(fix) / (12 * sysm.G) * 100:5.1f}%  pre {pre * 1000:.1f} ms  guards {gu}")
        sys.exit(0)
    path = os.path.join(RES, f"m1x_eval_{a.split}.jsonl")
    done = set()
    if os.path.exists(path):
        done = {json.loads(line)["i"] for line in open(path)}
    jobs = [j for j in jobs if j[0] not in done]
    t0 = time.time()
    with mp.get_context("spawn").Pool(1, initializer=HE._init) as pool, open(path, "a") as f:
        for k, recs in enumerate(pool.imap(HE._job, jobs, chunksize=1)):
            for r in recs:
                f.write(json.dumps(r) + "\n")
            f.flush()
            print(f"[{k + 1}/{len(jobs)} {time.time() - t0:.0f}s] i={recs[0]['i']} full {recs[0]['time']:.1f}s, " +
                  ", ".join(f"{r['rule'].replace('hybrid ', '').replace('m1x ', '')[:22]} {r['time']:.1f}s" for r in recs[1:]),
                  flush=True)

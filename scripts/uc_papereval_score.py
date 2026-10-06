"""Re-score our per-instance records with the evaluation protocol of Learning to Fix (Fritz, Makrides, Fetanat &
Pinson, arXiv 2609.39396, Sec. IV-C and Table I), so our numbers can be put next to its Table I.

    python scripts/uc_papereval_score.py        # needs results/papereval_overhead.json (uc_papereval_overhead.py)
                                                 # and results/papereval_fresh_runs.jsonl (uc_papereval_run.py)

Paper metrics, per method m and instance i:
  * feasible: the reduced problem returned a solution. Our fallback re-solves (fixings in conflict with min up/down,
    the reduced MILP is infeasible, the full MILP is solved instead) count as infeasible for the method.
  * optimality gap (eq. 14): (C_m - DB) / C_m * 100 with DB the best dual bound of the full MILP of instance i, the
    same for every method. uc12: DB = obj * (1 - mip_gap) of the dataset's MILP run (scipy / HiGHS report
    mip_gap = (primal - dual) / primal); uc24: the stored dual bound (identical to obj * (1 - gap), checked).
    C_m includes the penalty costs of shedding / over-generation / reserve shortfall (soft constraints).
  * runtime T_m: total, including inference and every guard or repair. Recorded solver times + overheads measured by
    uc_papereval_overhead.py (scaled by the load factor of the earlier runs, see that script).
  * speed-up (eq. 15): T_MILP / T_m per instance; mean and max (and the median) over feasible instances. T_MILP is
    the full MILP solved back to back in the same worker (fixing runs) or the dataset's MILP run (end-to-end rows,
    uc24, instances without a back-to-back solve).
  * gap, runtime, speed-up and fixed share are computed over feasible instances only; served share (no shedding and
    no reserve shortfall in the method's solution) too.
Also kept for comparison with our earlier conventions: mean / median gap to the reference objective over all
instances (fallbacks included with their re-solve) and the speed-up as a ratio of mean times without overheads.
Seeded configurations: every statistic is computed per seed and averaged (range of the mean gap / mean speed-up kept).
Output: results/papereval_results.json, tables printed (copied into results/papereval_results.md).
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
R12 = os.path.join(ROOT, "results", "uc12")
OUT = os.path.join(ROOT, "results", "papereval_results.json")
T12, G = 12, 73

# ------------------------------------------------------------------------------------------- paper, Table I
PAPER = [  # method, feasible %, gap mean, gap max, runtime mean, runtime max, speed-up mean, max, fixed mean, max
    ("full MILP (Gurobi, 0.25 % MIP gap)", 100.00, 0.19, 0.25, 60.35, 947.21, 1.00, 1.00, 0.00, 0.00),
    ("cost-ranked kNN (k = 50, best of 50 LPs)", 100.00, 5.40, 89.99, 9.86, 18.91, 7.13, 184.73, 100.00, 100.00),
    ("kNN, tau = 0.5 (hard threshold)", 11.05, 78.84, 99.40, 0.16, 1.91, 421.17, 8175.31, 100.00, 100.00),
    ("kNN, [0.1, 0.9]", 100.00, 2.73, 96.04, 10.16, 138.56, 13.88, 214.68, 86.10, 91.94),
    ("kNN, [0.05, 0.95]", 100.00, 0.58, 46.62, 20.94, 215.25, 6.75, 91.67, 81.62, 88.18),
    ("kNN, [0.01, 0.99]", 100.00, 0.21, 1.13, 31.25, 393.99, 4.06, 60.59, 74.42, 81.35),
    ("kNN, worst-case thresholds", 99.81, 0.20, 0.59, 52.90, 858.13, 2.17, 52.29, 20.09, 40.06),
    ("kNN, suboptimality-constrained, eps = 10 %", 98.67, 1.56, 74.35, 1.28, 14.07, 60.33, 972.85, 87.46, 91.99),
    ("kNN, suboptimality-constrained, eps = 5 %", 98.86, 1.05, 45.45, 2.51, 43.76, 34.25, 1395.66, 82.25, 87.12),
    ("kNN, suboptimality-constrained, eps = 1 %", 99.81, 0.48, 6.63, 4.70, 96.28, 20.82, 240.81, 78.81, 84.37),
]

# ------------------------------------------------------------------------------------------- overheads
# components added to the recorded time of a rule (per instance, measured by uc_papereval_overhead.py):
#   rel = LP relaxation (GNN / error-cost features), gnn = GNN forward pass, harm = error-cost features + scoring,
#   rank = ranking / building the fixings, adeq = adequacy guard, rows = min up/down row release, knn = kNN search
OV = {
    "rac": ("rel", "gnn", "rank"), "raclp": ("rel", "gnn", "rank", "rows"),
    "harm": ("rel", "gnn", "harm", "rank", "adeq"), "harmlp": ("rel", "gnn", "harm", "rank", "adeq", "rows"),
    "asym": ("rel", "gnn", "rank", "adeq"), "asymlp": ("rel", "gnn", "rank", "adeq", "rows"),
    "gnnthr": ("rel", "gnn"), "gnnthrlp": ("rel", "gnn", "rows"),
    "knn": ("knn",), "knnlp": ("knn", "rows"), "none": (),
}


def stats(recs, db, t_full, ov, ref_obj, tag, proven=None):
    """recs: list of dicts (one per instance, same order as db / t_full / ov / ref_obj) with obj, shed, short, time
    (recorded), feasible, fixed (share 0..1). Returns the paper metrics + our earlier ones. proven: mask of instances
    whose reference MILP proved its 0.1 % gap (robustness check of the dual bound)."""
    feas = np.array([r["feasible"] for r in recs])
    n = len(recs)
    out = dict(method=tag, n=n, feasible=float(feas.mean() * 100))
    if feas.any():
        f = np.where(feas)[0]
        C = np.array([recs[k]["obj"] for k in f])
        gap = (C - db[f]) / C * 100
        tm = np.array([recs[k]["time"] for k in f]) + ov[f]
        sp = t_full[f] / tm
        fx = np.array([recs[k]["fixed"] for k in f]) * 100
        served = np.array([(recs[k]["shed"] < 1e-6) and (recs[k]["short"] < 1e-6) for k in f])
        out.update(gap_mean=float(gap.mean()), gap_median=float(np.median(gap)), gap_max=float(gap.max()),
                   time_mean=float(tm.mean()), time_median=float(np.median(tm)), time_max=float(tm.max()),
                   speedup_mean=float(sp.mean()), speedup_median=float(np.median(sp)), speedup_max=float(sp.max()),
                   speedup_min=float(sp.min()), fixed_mean=float(fx.mean()), fixed_max=float(fx.max()),
                   served=float(served.mean() * 100), overhead_mean=float(ov[f].mean()),
                   speedup_ratio_of_means=float(t_full[f].mean() / tm.mean()),
                   speedup_mean_no_overhead=float(np.mean(t_full[f] / (tm - ov[f]))),
                   gap_ref_mean_feasible=float(np.mean((C - ref_obj[f]) / ref_obj[f] * 100)))
        rng = np.random.default_rng(0)                 # instance bootstrap of the two headline means
        bi = rng.integers(0, len(f), (2000, len(f)))
        out.update(gap_mean_ci=[float(x) for x in np.percentile(gap[bi].mean(1), [2.5, 97.5])],
                   speedup_mean_ci=[float(x) for x in np.percentile(sp[bi].mean(1), [2.5, 97.5])])
        if proven is not None and np.asarray(proven)[f].any():
            pm = np.asarray(proven)[f]
            out.update(gap_mean_proven=float(gap[pm].mean()), gap_max_proven=float(gap[pm].max()),
                       n_proven=int(pm.sum()))
    # earlier conventions: all instances (fallback = the re-solve's result), gap to the reference objective,
    # speed-up = ratio of mean recorded times
    has = [k for k in range(n) if recs[k].get("obj_any") is not None]
    if has:
        Ca = np.array([recs[k]["obj_any"] for k in has])
        g0 = (Ca - ref_obj[has]) / ref_obj[has] * 100
        out.update(old_gap_mean=float(g0.mean()), old_gap_median=float(np.median(g0)),
                   old_speedup=float(t_full[has].mean() / np.mean([recs[k]["time_any"] for k in has])))
    return out


def seed_avg(rows, tag):
    """mean of every statistic over seeds; ranges for the headline statistics"""
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float)) and all(k in r for r in rows)]
    out = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    out["method"] = tag
    out["n"] = int(rows[0]["n"])
    for k in ("gap_mean_ci", "speedup_mean_ci"):          # seed-averaged bootstrap bounds (approximate)
        if all(k in r for r in rows):
            out[k] = [float(np.mean([r[k][j] for r in rows])) for j in (0, 1)]
    out["seeds"] = len(rows)
    for k in ("gap_mean", "speedup_mean", "feasible", "old_gap_mean"):
        if k in out:
            out[k + "_range"] = [float(min(r[k] for r in rows)), float(max(r[k] for r in rows))]
    return out


# ------------------------------------------------------------------------------------------- uc12 records
def read_jsonl(path):
    return [json.loads(x) for x in open(path)]


def rec_uc12(r):
    """normalise a uc12 record (combo / ltf / fixpolicy / papereval formats)"""
    fb = int(r.get("fallback", 0))
    ok = r.get("obj") is not None and r.get("status") != "infeasible" and fb == 0
    return dict(obj=r["obj"] if ok else None, shed=r.get("shed") if ok else None, short=r.get("short") if ok else None,
                time=r["time"], feasible=bool(ok), fixed=float(r.get("fixed_share", 0.0)),
                obj_any=r.get("obj"), time_any=r["time"])


class UC12:
    def __init__(self, split, n, overhead, extra_db=None):
        """extra_db: {instance: dual bound} of further full-MILP runs; DB = the best (largest) bound available"""
        d = np.load(os.path.join(ROOT, "data/generated/uc12", f"{split}.npz"))
        self.split, self.n = split, n
        self.obj, self.time_ds, self.opt = d["obj"], d["time"], d["opt"].copy()
        self.db = d["obj"] * (1 - d["gap"])
        self.n_db_tightened = 0
        for i, b in (extra_db or {}).items():
            if b > self.db[i] * (1 + 1e-9):
                self.db[i] = b
                self.n_db_tightened += 1
                self.opt[i] = True             # the other run proved its 0.1 % gap
        self.gap = (self.obj - self.db) / self.obj   # the reference run's gap to the best bound
        o = overhead[split]
        self.f = o["load_factor"]
        self.ov = {k: np.array(o["t_" + k]) for k in ("rel", "gnn", "harm", "rank", "adeq", "rows", "block", "lp")}
        self.o = o
        self.ov["knn"] = np.full(len(self.ov["rel"]), 0.005)          # replaced by the measured value if known

    def overhead(self, kind, scale=True):
        comps = OV[kind]
        v = np.zeros(len(self.ov["rel"]))
        for c in comps:
            v = v + self.ov[c]
        return v * (self.f if scale else 1.0)

    def row(self, recs_by_i, full_by_i, kind, tag, scale=True, idx=None):
        idx = list(range(self.n)) if idx is None else idx
        recs = [rec_uc12(recs_by_i[i]) for i in idx]
        t_full = np.array([full_by_i[i]["time"] for i in idx])
        return stats(recs, self.db[idx], t_full, self.overhead(kind, scale)[idx], self.obj[idx], tag, self.opt[idx])

    def full_row(self, full_by_i, tag, idx=None):
        idx = list(range(self.n)) if idx is None else idx
        recs = [rec_uc12(full_by_i[i]) for i in idx]
        t_full = np.array([full_by_i[i]["time"] for i in idx])
        return stats(recs, self.db[idx], t_full, np.zeros(len(idx)), self.obj[idx], tag, self.opt[idx])

    def dataset_full_row(self, idx):
        idx = np.asarray(idx)
        g = self.gap[idx] * 100
        t = self.time_ds[idx]
        return dict(method="full MILP, dataset reference run (0.1 % gap, 60 s limit)", n=len(idx), feasible=100.0,
                    gap_mean=float(g.mean()), gap_median=float(np.median(g)), gap_max=float(g.max()),
                    time_mean=float(t.mean()), time_median=float(np.median(t)), time_max=float(t.max()),
                    speedup_mean=1.0, speedup_median=1.0, speedup_max=1.0, fixed_mean=0.0, fixed_max=0.0,
                    proven_share=float(self.opt[idx].mean() * 100))


def group(recs, keyf):
    by = {}
    for r in recs:
        k = keyf(r)
        if k is not None:
            by.setdefault(k, {})[r["i"]] = r
    return by


def uc12_fixing(split, U, newruns=None):
    rows = []
    n = U.n
    # combo study (3 seeds where seeded)
    cb = group(read_jsonl(os.path.join(R12, f"combo_fix_{split}.jsonl")), lambda r: (r["config"], r["seed"], r["ratio"]))
    full = cb[("full", -1, 0.0)]
    rows.append(dict(U.full_row(full, "full MILP, back to back (combo run)"), family="full"))
    spec = [  # label, config, ratios, kind, seeds, family
        ("RACLearn-style (MILP-label BCE confidence)", "rac", (0.9, 0.95), "rac", (0,), "raclearn"),
        ("RACLearn-style + LP guard", "rac+lp", (0.95, 0.98), "raclp", (0,), "raclearn"),
        ("error-cost ranking + adequacy guard", "harm+guard", (0.9,), "harm", (0, 1, 2), "errorcost"),
        ("error-cost ranking + adequacy + LP guard", "harm+lp", (0.95,), "harmlp", (0, 1, 2), "errorcost"),
        ("combined: self-trained p + error cost + both guards", "combo st+harm+lp", (0.95, 0.98), "harmlp", (0, 1, 2),
         "combined"),
        ("combined without LP guard", "combo st+harm+guard", (0.95, 0.98), "harm", (0, 1, 2), "combined"),
        ("self-trained, asymmetric + adequacy guard", "st asym+guard", (0.95,), "asym", (0, 1, 2), "selftrained"),
        ("REINFORCE ranking (label-free REINFORCE p)", "rl rac", (0.95,), "rac", (0, 1, 2), "reinforce"),
        ("REINFORCE ranking (MILP-label REINFORCE p)", "milp_rl rac", (0.95,), "rac", (0,), "reinforce"),
    ]
    for label, cfg, ratios, kind, seeds, fam in spec:
        for q in ratios:
            per = []
            for s in seeds:
                per.append(U.row(cb[(cfg, s, q)], full, kind, f"{label}, {int(q * 100)} % target"))
            r = per[0] if len(per) == 1 else seed_avg(per, per[0]["method"])
            r.update(family=fam, source=f"combo_fix_{split}.jsonl", config=cfg, target=q)
            rows.append(r)
    # Learning to Fix reconstruction and the earlier single-seed rules (ltf study)
    tagf = "fresh" if split == "test_fresh" else "test"
    lt = group(read_jsonl(os.path.join(R12, f"ltf_eval_{tagf}.jsonl")), lambda r: r["rule"])
    fl = lt["full"]
    rows.append(dict(U.full_row(fl, "full MILP, back to back (ltf run)"), family="full"))
    ltf_spec = [
        ("LtF reconstruction (kNN-21, per-generator thresholds, tau = 1 %, as published)", "ltf:knn:budget:comp:0.01", "knn", "ltf"),
        ("LtF reconstruction, kNN, tau = 2 %", "ltf:knn:budget:comp:0.02", "knn", "ltf"),
        ("LtF reconstruction, kNN, tau = 2 % + LP guard", "ltf:knn:budget:comp:0.02+lp", "knnlp", "ltf"),
        ("LtF thresholds on MILP-label BCE GNN, tau = 1 %", "ltf:bce:budget:comp:0.01", "gnnthr", "ltf"),
        ("LtF thresholds on self-trained GNN, tau = 1 %", "ltf:st:budget:comp:0.01", "gnnthr", "ltf"),
        ("LtF thresholds on REINFORCE GNN, tau = 1 %", "ltf:rl:budget:comp:0.01", "gnnthr", "ltf"),
        ("LtF thresholds on REINFORCE GNN, tau = 2 %", "ltf:rl:budget:comp:0.02", "gnnthr", "ltf"),
        ("RACLearn-style, 90 %", "rac@0.9", "rac", "raclearn"),
        ("RACLearn-style, 95 %", "rac@0.95", "rac", "raclearn"),
        ("asymmetric + adequacy guard, 90 %", "asym@0.9", "asym", "asym"),
        ("asymmetric + adequacy guard, 95 %", "asym@0.95", "asym", "asym"),
        ("error-cost + adequacy guard, 90 %", "harm@0.9", "harm", "errorcost"),
        ("error-cost + adequacy guard, 95 %", "harm@0.95", "harm", "errorcost"),
        ("error-cost + adequacy + LP guard, 95 % target", "harm@0.95+lp", "harmlp", "errorcost"),
        ("self-trained asym + guard, 95 %", "st@0.95", "asym", "selftrained"),
        ("self-trained + LP guard, 95 % target", "st@0.95+lp", "asymlp", "selftrained"),
        ("REINFORCE ranking (MILP-label REINFORCE p), 95 %", "rl@0.95", "rac", "reinforce"),
        ("REINFORCE ranking, 97 %", "rl@0.97", "rac", "reinforce"),
        ("REINFORCE ranking + LP guard, 97 % target", "rl@0.97+lp", "raclp", "reinforce"),
    ]
    for label, rule, kind, fam in ltf_spec:
        if rule in lt:
            r = U.row(lt[rule], fl, kind, label)
            r.update(family=fam, source=f"ltf_eval_{tagf}.jsonl", config=rule)
            rows.append(r)
    if split == "test_fresh":
        lb = group(read_jsonl(os.path.join(R12, "ltf_eval_fresh_b.jsonl")), lambda r: r["rule"])
        per = [U.row(lb[f"ltf:knn_bs{s}:budget:comp:0.01"], lb["full"], "knn", "x") for s in (0, 1, 2)]
        r = seed_avg(per, "LtF reconstruction, kNN, tau = 1 %, 3 bootstrap calibration samples")
        r.update(family="ltf", source="ltf_eval_fresh_b.jsonl", config="ltf:knn_bs*:budget:comp:0.01")
        rows.append(r)
    if split == "test":   # the first fixing study (single seed; superseded where combo re-ran a rule)
        for fn in ("fixpolicy_eval_test.jsonl", "fixpolicy_eval_test_bc.jsonl"):
            fp = group(read_jsonl(os.path.join(R12, fn)), lambda r: (r["method"], r["ratio"]))
            ff = fp[("full", 0.0)]
            rows.append(dict(U.full_row(ff, f"full MILP, back to back ({fn.split('.')[0]})"), family="full"))
            kinds = {"rac": "rac", "rl": "rac", "asym": "asym", "harm_c_guard": "harm", "ltf": "gnnthr",
                     "rac_lp": "raclp", "harm_c_lp": "harmlp"}
            for (meth, q), recs in sorted(fp.items()):
                if meth == "full":
                    continue
                r = U.row(recs, ff, kinds[meth], f"[{fn.split('.')[0]}] {meth}, {int(round(q * 100))} %")
                r.update(family="fixpolicy", source=fn, config=meth, target=q)
                rows.append(r)
    if newruns is not None:
        nr = group(newruns, lambda r: r["rule"])
        fn_ = nr["full"]
        idx = [i for i in range(n) if i in fn_]
        r = U.full_row(fn_, "full MILP, back to back (papereval run)", idx)
        r.update(family="full")
        rows.append(r)
        knn_s = {i: rr["knn_s"] for i, rr in nr["knn_costrank"].items()}
        ov_knn = U.ov["knn"].copy()
        for i, v in knn_s.items():
            if i < len(ov_knn):
                ov_knn[i] = v
        U.ov["knn"] = ov_knn
        for mdl, kind, lab in (("knn50", "knn", "kNN (k = 50, IDW)"), ("bce", "gnnthr", "MILP-label BCE GNN")):
            for tau in (0.5, 0.1, 0.05, 0.01):
                name = f"{mdl}@{tau}"
                if name not in nr:
                    continue
                tl = "hard threshold 0.5" if tau == 0.5 else f"[{tau}, {1 - tau:.2f}]"
                r = U.row(nr[name], fn_, kind, f"{lab}, {tl}", scale=False, idx=idx)
                r.update(family="constant", source="papereval_fresh_runs.jsonl", config=name)
                rows.append(r)
        r = U.row(nr["knn_costrank"], fn_, "none", "cost-ranked kNN (k = 50, best of 50 LPs)", scale=False, idx=idx)
        r.update(family="costrank", source="papereval_fresh_runs.jsonl", config="knn_costrank")
        rows.append(r)
    return rows


# ------------------------------------------------------------------------------------------- end-to-end
def uc12_e2e(split, U):
    """end-to-end pipelines on all 120 instances; T_MILP = dataset MILP run; costs from combo_e2e arrays"""
    arr = np.load(os.path.join(R12, f"combo_e2e_{split}_arrays.npz"))
    o = U.o
    n = len(o["t_rel"])
    base = np.array(o["t_rel"]) + np.array(o["t_gnn"])
    tb, tl = np.array(o["t_block"]), np.array(o["t_lp"])
    idx = np.arange(n)
    rows = []

    def mk(tag, cost, shed, short, tm):
        recs = [dict(obj=float(cost[i]), shed=float(shed[i]), short=float(short[i]), time=float(tm[i]), feasible=True,
                     fixed=1.0, obj_any=float(cost[i]), time_any=float(tm[i])) for i in idx]
        return stats(recs, U.db[idx], U.time_ds[idx], np.zeros(n), U.obj[idx], tag, U.opt[idx])

    c, sh, so, _ = arr["milp__ref__0"]
    r = mk("MILP commitment through the dispatch LP (reference)", c, sh, so, U.time_ds[idx])
    r.update(family="full")
    rows.append(r)
    cal = {"rl_s0": 0.9, "rl_s1": 0.6, "rl_s2": 0.8}
    for label, keyf, nlp, ndec, fam in (
            ("framework one-shot: label-free + REINFORCE, top-1 + min up/down repair, 1 LP",
             lambda s: f"rl_s{s}__mud__0.5", None, 1, "e2e"),
            ("label-free + REINFORCE + block repair, 1 LP", lambda s: f"rl_s{s}__block__0.5", None, 1, "e2e"),
            ("REINFORCE, val threshold + block repair, 1 LP", lambda s: f"rl_s{s}__block__{cal[f'rl_s{s}']}", None, 1,
             "e2e"),
            ("combined, one LP: lag_D, threshold 0.6 + block repair", lambda s: f"lag_s{s}__block__0.6", None, 1, "e2e"),
            ("combined + screening (7 thresholds, best by LP)", None, "screen", 7, "e2e")):
        per = []
        for s in (0, 1, 2):
            if keyf is not None:
                c, sh, so, _ = arr[keyf(s)]
                k_lp = np.ones(n)
            else:
                stack = np.stack([arr[f"lag_s{s}__block__{th}"] for th in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)])  # [7,4,n]
                b = np.argmin(stack[:, 0, :], 0)
                c, sh, so = (stack[b, j, idx] for j in range(3))
                k_lp = np.array(o[f"screen_nlp_s{s}"])
            tm = (base + ndec * tb + k_lp * tl) * U.f
            per.append(mk(label, c, sh, so, tm))
        r = seed_avg(per, label)
        r.update(family=fam, n_lp=float(np.mean([np.mean(o[f"screen_nlp_s{s}"]) for s in (0, 1, 2)])) if nlp else 1.0)
        rows.append(r)
    if split == "test":   # our earlier kNN baseline (bus-level features, min up/down-repaired neighbours, 20 LPs)
        c, sh, so = np.load(os.path.join(R12, "constrained_test_final_arrays.npz"))["kNN-20 + LP check"]
        r = mk("earlier kNN-20 (repaired neighbours, best of 20 LPs)", c, sh, so, (0.005 + 20 * tl) * U.f)
        r.update(family="costrank", n_lp=20.0)
        rows.append(r)
    return rows


def costrank_all(newruns, U):
    """cost-ranked kNN on all 120 fresh instances against the dataset's MILP time (as the end-to-end rows)"""
    nr = group(newruns, lambda r: r["rule"])["knn_costrank"]
    idx = sorted(nr)
    recs = [rec_uc12(nr[i]) for i in idx]
    r = stats(recs, U.db[idx], U.time_ds[idx], np.zeros(len(idx)), U.obj[idx],
              "cost-ranked kNN (k = 50, best of 50 LPs), all instances, vs dataset MILP time", U.opt[idx])
    r.update(family="costrank", n_feasible_neighbours=float(np.mean([nr[i]["n_feasible"] for i in idx])),
             lp_s_mean=float(np.mean([np.mean(nr[i]["lp_s"]) for i in idx])))
    # best of the first k neighbours (k = 1, 5, 10, 20, 50): quality / time trade-off
    curve = {}
    for k in (1, 5, 10, 20, 50):
        recs_k = []
        for i in idx:
            cs = [c if c is not None else np.inf for c in nr[i]["costs"][:k]]
            b = int(np.argmin(cs))
            ok = np.isfinite(cs[b])
            recs_k.append(dict(obj=cs[b] if ok else None, shed=nr[i]["sheds"][b] if ok else None,
                               short=nr[i]["shorts"][b] if ok else None, time=float(sum(nr[i]["lp_s"][:k]) + nr[i]["knn_s"]),
                               feasible=bool(ok), fixed=1.0, obj_any=None, time_any=None))
        curve[k] = stats(recs_k, U.db[idx], U.time_ds[idx], np.zeros(len(idx)), U.obj[idx], f"best of {k}")
    r["best_of_k"] = curve
    return r


# ------------------------------------------------------------------------------------------- uc24
def uc24_rows(overhead):
    d = np.load(os.path.join(ROOT, "data/generated/uc24/test.npz"))
    o = overhead["uc24"]
    f = o["load_factor"]
    n = len(d["obj"])
    db, t_full, ref = d["bound"], d["time"], d["obj"]
    t_rel = d["t_rel"]
    t_gnn = np.array(o["t_gnn"]) * f
    t_adeq = np.array(o["t_adeq"]) * f
    rows = [dict(method="full MILP, dataset run (0.1 % gap, 300 s limit, HiGHS 1 thread)", n=n, feasible=100.0,
                 gap_mean=float(d["gap"].mean() * 100), gap_median=float(np.median(d["gap"]) * 100),
                 gap_max=float(d["gap"].max() * 100), time_mean=float(t_full.mean()), time_median=float(np.median(t_full)),
                 time_max=float(t_full.max()), speedup_mean=1.0, speedup_median=1.0, speedup_max=1.0, fixed_mean=0.0,
                 fixed_max=0.0, proven_share=float(d["opt"].mean() * 100), family="full")]
    res = {}
    for fn in ("b3_fix_test.jsonl", "b3_fix_test_extra.jsonl"):
        for line in open(os.path.join(ROOT, "results/uc24", fn)):
            r = json.loads(line)
            for k, v in r.items():
                if "|" in k:
                    res.setdefault(k, {})[r["i"]] = v
    labels = {"raclearn": "RACLearn-style confidence (imitation model), no guard",
              "best=bce_asym_g": "val-selected: imitation, OFF x 10 + adequacy + LP guards",
              "best=bce_g": "val-selected: imitation + adequacy + LP guards",
              "rl": "REINFORCE ranking, no guard", "rl_g": "REINFORCE ranking + adequacy + LP guards"}
    for key in sorted(res, key=lambda k: (k.split("|")[1], k)):
        q, rk = key.split("|")
        guarded = rk.endswith("_g")
        recs = []
        for i in range(n):
            v = res[key][i]
            ok = int(v["fallback"]) == 0 and v["obj"] is not None and np.isfinite(v["obj"])
            recs.append(dict(obj=v["obj"] if ok else None, shed=v["shed"], short=v["short"],
                             time=v["t_milp"] + v["t_guard"] + t_rel[i], feasible=ok, fixed=v["n_fixed"] / (24 * G),
                             obj_any=v["obj"], time_any=v["t_milp"] + v["t_guard"] + t_rel[i]))
        ov = t_gnn + (t_adeq if guarded else 0)
        r = stats(recs, db, t_full, ov, ref, f"{labels[rk]}, {q} target", d["opt"])
        r.update(family="uc24fix", config=key)
        rows.append(r)
    # end-to-end (recomputed by uc_papereval_overhead.py; costs checked against b3_e2e_results)
    base = t_rel + t_gnn
    for tag, ck, tm in (("REINFORCE + block repair, 1 LP", "block",
                         base + (np.array(o["t_block"]) + np.array(o["t_lp"])) * f),
                        ("REINFORCE + screening (7 thresholds + 8 samples, best by LP)", "screen",
                         base + (np.array(o["t_screen_block"]) + np.array(o["t_screen_lp"])) * f)):
        c, sh, so = (np.array(o[f"{x}_{ck}"]) for x in ("cost", "shed", "short"))
        recs = [dict(obj=c[i], shed=sh[i], short=so[i], time=tm[i], feasible=True, fixed=1.0, obj_any=c[i],
                     time_any=tm[i]) for i in range(n)]
        r = stats(recs, db, t_full, np.zeros(n), ref, tag, d["opt"])
        r.update(family="e2e")
        rows.append(r)
    return rows


# ------------------------------------------------------------------------------------------- output
COLS = [("feasible", "feas. %", 1), ("gap_mean", "gap mean %", 2), ("gap_median", "gap med. %", 3),
        ("gap_max", "gap max %", 2), ("time_mean", "time mean s", 2), ("time_max", "time max s", 1),
        ("speedup_mean", "speed-up mean", 2), ("speedup_median", "speed-up med.", 2), ("speedup_max", "speed-up max", 1),
        ("fixed_mean", "fixed mean %", 1), ("fixed_max", "fixed max %", 1), ("served", "served %", 1)]


def table(rows, cols=COLS, old=False):
    c2 = cols + ([("old_gap_mean", "old: gap to ref., mean %", 2), ("old_speedup", "old: ratio of mean times", 2)]
                 if old else [])
    lines = ["| method | n | " + " | ".join(h for _, h, _ in c2) + " |", "|---|---|" + "---|" * len(c2)]
    for r in rows:
        vals = []
        for k, _, p in c2:
            v = r.get(k)
            vals.append("–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{p}f}")
        lines.append(f"| {r['method']} | {r.get('n', '')} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


CURATED = {   # (benchmark key, part, method label prefix, short name)
    "test_fresh": [
        ("fixing_first60", "full MILP, dataset reference run", "full MILP (reference run)"),
        ("fixing_first60", "full MILP, back to back (papereval run)", "full MILP (back to back)"),
        ("fixing_first60", "cost-ranked kNN (k = 50, best of 50 LPs)", "cost-ranked kNN, k = 50"),
        ("fixing_first60", "kNN (k = 50, IDW), hard threshold 0.5", "kNN, tau = 0.5"),
        ("fixing_first60", "kNN (k = 50, IDW), [0.1, 0.90]", "kNN, [0.1, 0.9]"),
        ("fixing_first60", "kNN (k = 50, IDW), [0.05, 0.95]", "kNN, [0.05, 0.95]"),
        ("fixing_first60", "kNN (k = 50, IDW), [0.01, 0.99]", "kNN, [0.01, 0.99]"),
        ("fixing_first60", "MILP-label BCE GNN, hard threshold 0.5", "BCE GNN, tau = 0.5"),
        ("fixing_first60", "MILP-label BCE GNN, [0.1, 0.90]", "BCE GNN, [0.1, 0.9]"),
        ("fixing_first60", "MILP-label BCE GNN, [0.05, 0.95]", "BCE GNN, [0.05, 0.95]"),
        ("fixing_first60", "MILP-label BCE GNN, [0.01, 0.99]", "BCE GNN, [0.01, 0.99]"),
        ("fixing_first60", "LtF reconstruction (kNN-21, per-generator thresholds, tau = 1 %, as published)",
         "earlier LtF reconstruction (kNN, tau = 1 %)"),
        ("fixing_first60", "LtF thresholds on REINFORCE GNN, tau = 1 %", "earlier LtF thresholds on REINFORCE GNN"),
        ("fixing_first60", "RACLearn-style (MILP-label BCE confidence), 90 % target", "RACLearn-style, 90 %"),
        ("fixing_first60", "RACLearn-style (MILP-label BCE confidence), 95 % target", "RACLearn-style, 95 %"),
        ("fixing_first60", "RACLearn-style + LP guard, 95 % target", "RACLearn-style + LP guard, 95 %"),
        ("fixing_first60", "error-cost ranking + adequacy guard, 90 % target", "error-cost + adequacy guard, 90 %"),
        ("fixing_first60", "error-cost ranking + adequacy + LP guard, 95 % target", "error-cost + both guards, 95 %"),
        ("fixing_first60", "combined: self-trained p + error cost + both guards, 95 % target", "combined pipeline, 95 %"),
        ("fixing_first60", "combined: self-trained p + error cost + both guards, 98 % target", "combined pipeline, 98 %"),
        ("fixing_first60", "REINFORCE ranking (label-free REINFORCE p), 95 % target", "REINFORCE ranking, 95 % (3 seeds)"),
        ("fixing_first60", "REINFORCE ranking (MILP-label REINFORCE p), 95 % target", "REINFORCE ranking, 95 % (MILP-label model)"),
        ("e2e_all120", "full MILP, dataset reference run", "full MILP (reference run), 120 inst."),
        ("e2e_all120", "cost-ranked kNN (k = 50, best of 50 LPs), all instances", "cost-ranked kNN, k = 50, 120 inst."),
        ("e2e_all120", "label-free + REINFORCE + block repair, 1 LP", "end-to-end: REINFORCE + block repair, 1 LP"),
        ("e2e_all120", "combined, one LP", "end-to-end: combined, 1 LP"),
        ("e2e_all120", "combined + screening", "end-to-end: combined + screening (~6 LPs)"),
    ],
    "test": [
        ("fixing_first60", "full MILP, dataset reference run", "full MILP (reference run)"),
        ("fixing_first60", "full MILP, back to back (combo run)", "full MILP (back to back)"),
        ("fixing_first60", "LtF reconstruction (kNN-21, per-generator thresholds, tau = 1 %, as published)",
         "earlier LtF reconstruction (kNN, tau = 1 %)"),
        ("fixing_first60", "RACLearn-style (MILP-label BCE confidence), 90 % target", "RACLearn-style, 90 %"),
        ("fixing_first60", "RACLearn-style (MILP-label BCE confidence), 95 % target", "RACLearn-style, 95 %"),
        ("fixing_first60", "RACLearn-style + LP guard, 95 % target", "RACLearn-style + LP guard, 95 %"),
        ("fixing_first60", "error-cost ranking + adequacy guard, 90 % target", "error-cost + adequacy guard, 90 %"),
        ("fixing_first60", "error-cost ranking + adequacy + LP guard, 95 % target", "error-cost + both guards, 95 %"),
        ("fixing_first60", "combined: self-trained p + error cost + both guards, 95 % target", "combined pipeline, 95 %"),
        ("fixing_first60", "combined: self-trained p + error cost + both guards, 98 % target", "combined pipeline, 98 %"),
        ("fixing_first60", "REINFORCE ranking (label-free REINFORCE p), 95 % target", "REINFORCE ranking, 95 % (3 seeds)"),
        ("fixing_first60", "REINFORCE ranking (MILP-label REINFORCE p), 95 % target", "REINFORCE ranking, 95 % (MILP-label model)"),
        ("e2e_all120", "full MILP, dataset reference run", "full MILP (reference run), 120 inst."),
        ("e2e_all120", "earlier kNN-20", "earlier kNN-20 (repaired neighbours, 20 LPs)"),
        ("e2e_all120", "label-free + REINFORCE + block repair, 1 LP", "end-to-end: REINFORCE + block repair, 1 LP"),
        ("e2e_all120", "combined, one LP", "end-to-end: combined, 1 LP"),
        ("e2e_all120", "combined + screening", "end-to-end: combined + screening (~6 LPs)"),
    ],
}


def side_by_side(out):
    cols = [("feasible", 1), ("gap_mean", 2), ("gap_median", 3), ("gap_max", 2), ("time_mean", 2), ("time_max", 1),
            ("speedup_mean", 2), ("speedup_median", 2), ("speedup_max", 1), ("fixed_mean", 1), ("fixed_max", 1),
            ("served", 1)]
    hdr = ("| method | n | feasible % | gap mean % | gap median % | gap max % | runtime mean s | runtime max s | "
           "speed-up mean | speed-up median | speed-up max | fixed mean % | fixed max % | served % |")
    sep = "|---|---|" + "---|" * len(cols)

    def line(name, r):
        v = []
        for k, p in cols:
            x = r.get(k)
            v.append("–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{p}f}")
        return f"| {name} | {r.get('n', '')} | " + " | ".join(v) + " |"
    txt = ["\n## Side by side\n", "### Paper, Table I (kNN; Irish system, copper plate, 72 h; Gurobi, 8 CPUs; 525 test "
           "instances)\n", hdr, sep]
    for p in out["paper_table_I"]:
        txt.append(line(p["method"], dict(p, n=525)))
    names = {"test_fresh": "12-hour fresh test set (B2)", "test": "12-hour original test set (B2)"}
    for key, items in CURATED.items():
        txt += [f"\n### {names[key]}\n", hdr, sep]
        for part, prefix, short in items:
            hit = [r for r in out[key][part] if r["method"].startswith(prefix)]
            if hit:
                txt.append(line(short, hit[0]))
    txt += ["\n### 24-hour test set (B3)\n", hdr, sep]
    for r in out["uc24"]:
        txt.append(line(r["method"], r))
    return "\n".join(txt)


def pareto(out, thresholds=(2, 5, 10, 20, 50), min_feasible=95.0):
    """best mean gap among methods with mean speed-up >= x and feasibility >= min_feasible (paper metrics)"""
    sets = {"paper Table I (72 h, copper plate)": [r for r in out["paper_table_I"] if r["fixed_mean"] > 0],
            "B2 fresh, 12 h (fixing: 60, end-to-end: 120 inst.)":
                [r for part in ("fixing_first60", "e2e_all120") for r in out["test_fresh"][part]
                 if r.get("family") != "full"],
            "B2 original, 12 h": [r for part in ("fixing_first60", "e2e_all120") for r in out["test"][part]
                                  if r.get("family") not in ("full", "fixpolicy")],
            "B3, 24 h (40 inst.)": [r for r in out["uc24"] if r.get("family") != "full"]}
    res, lines = {}, ["| benchmark | " + " | ".join(f"speed-up >= {x}" for x in thresholds) + " |",
                      "|---|" + "---|" * len(thresholds)]
    for name, rows in sets.items():
        cells, res[name] = [], {}
        for x in thresholds:
            ok = [r for r in rows if r.get("speedup_mean", 0) >= x and r.get("feasible", 0) >= min_feasible
                  and "gap_mean" in r]
            if ok:
                b = min(ok, key=lambda r: r["gap_mean"])
                res[name][x] = dict(method=b["method"], gap_mean=b["gap_mean"], speedup_mean=b["speedup_mean"],
                                    feasible=b["feasible"])
                cells.append(f"{b['gap_mean']:.2f} % ({b['speedup_mean']:.1f}x; {b['method']})")
            else:
                cells.append("–")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return res, "\n".join(lines)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--overhead", default=os.path.join(ROOT, "results", "papereval_overhead.json"))
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--no_new", action="store_true", help="ignore results/papereval_fresh_runs.jsonl")
    a = ap.parse_args()
    OUT = a.out
    ovh = json.load(open(a.overhead))
    newp = os.path.join(ROOT, "results", "papereval_fresh_runs.jsonl")
    newruns = read_jsonl(newp) if os.path.exists(newp) and not a.no_new else None
    out = {"paper_table_I": [dict(zip(["method", "feasible", "gap_mean", "gap_max", "time_mean", "time_max",
                                       "speedup_mean", "speedup_max", "fixed_mean", "fixed_max"], p)) for p in PAPER]}
    extra = None
    if newruns is not None:     # dual bounds of the back-to-back full MILPs of this study (fresh set)
        extra = {r["i"]: r["obj"] * (1 - r["gap"]) for r in newruns if r["rule"] == "full" and r["obj"] is not None}
    for split in ("test_fresh", "test"):
        U = UC12(split, 60, ovh, extra if split == "test_fresh" else None)
        rows = [dict(U.dataset_full_row(np.arange(60)), family="full")]
        rows += uc12_fixing(split, U, newruns if split == "test_fresh" else None)
        U120 = UC12(split, 120, ovh, extra if split == "test_fresh" else None)
        e2e = [dict(U120.dataset_full_row(np.arange(120)), family="full")] + uc12_e2e(split, U120)
        if split == "test_fresh" and newruns is not None:
            e2e.append(costrank_all(newruns, U120))
        out[split] = {"fixing_first60": rows, "e2e_all120": e2e, "load_factor": U.f,
                      "full_milp_dataset_gap_first60": float(U.gap[:60].mean() * 100),
                      "db_tightened_by_new_run": U.n_db_tightened}
        print(f"\n## {split}: fixing, first 60 instances (load factor {U.f:.2f})\n")
        print(table(rows, old=True))
        print(f"\n## {split}: end-to-end, 120 instances\n")
        print(table(e2e, old=True))
    out["uc24"] = uc24_rows(ovh)
    print("\n## uc24\n")
    print(table(out["uc24"], old=True))
    out["pareto"], ptxt = pareto(out)
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    print("\nwrote", OUT)
    print(side_by_side(out))
    print("\n## Best mean gap at a mean speed-up of at least x (feasibility >= 95 %)\n")
    print(ptxt)

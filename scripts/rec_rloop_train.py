"""rloop (arm B) training: deep supervision with LP feedback, then cost-aware fine-tuning (REINFORCE, exact LP critic).

    # CPU (this container): one process, LPs in process or in worker processes pinned to given cores
    OTSL_THREADS=1 taskset -c 2 python3 scripts/rec_rloop_train.py --tag main --preset cpu
    OTSL_THREADS=1 taskset -c 2,3 python3 scripts/rec_rloop_train.py --tag main --preset cpu --lp_workers 1 --lp_cores 3
    python3 scripts/rec_rloop_train.py --tag main_rl --stage rl --init main            # fine-tune a trained model
    # GPU (scripts/pace/rloop_train.sbatch): network on cuda, LPs in a CPU worker pool, batches kept in flight
    python3 scripts/rec_rloop_train.py --tag gpu_main --preset gpu --device cuda --lp_workers 16 --inflight 4

Ablation switches: --no_feedback (LP signals zeroed, the schedule and repair changes kept), --no_recall,
--no_deep_sup (loss on the last outer step only, back-propagated through all steps), --no_counts (per-unit heads with
symmetry-aligned labels and the symmetry-rank input), --k_train, --decode round, --p_ext 0.

Checkpoints (resume: rerun the same command): data/generated/recursive/rloop/<tag>.pt (git-ignored); history:
results/recursive/rloop_train_<tag>.json. Validation monitor: the first --val_n instances of val (stored MILP dual
bound), K = cfg k_eval outer steps, through the same LP pool.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rec_rloop_common as C  # noqa: E402
from otsl import rloop  # noqa: E402


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--preset", default="cpu", choices=sorted(rloop.PRESETS))
    ap.add_argument("--stage", default="sup", choices=["sup", "rl"])
    ap.add_argument("--init", default=None, help="tag of the checkpoint to start from (stage rl)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--lp_workers", type=int, default=None)
    ap.add_argument("--lp_cores", default="", help="comma-separated CPU ids for the LP workers (empty: inherit)")
    ap.add_argument("--inflight", type=int, default=None)
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--data_dir", action="append", default=[],
                    help="extra polished-label chunks (extra_k.npz + polish_k.npz, uc12_m1x format); repeatable")
    ap.add_argument("--val_n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    for k in ("hidden", "n_latent", "n_cycles", "mp_layers", "k_train", "epochs", "batch", "k_eval", "rl_updates",
              "rl_batch", "rl_samples", "rl_k", "rl_log", "rl_ckpt", "accum"):
        ap.add_argument(f"--{k}", type=int, default=None)
    for k in ("lr", "ema", "p_ext", "rl_lr", "bce_w", "halt_w", "clip", "wd", "p_seed", "w_wrong", "w_grad", "rl_temp",
              "anchor_init"):
        ap.add_argument(f"--{k}", type=float, default=None)
    ap.add_argument("--decode", default=None, choices=["viterbi", "round"])
    ap.add_argument("--reward", default=None, choices=["improve", "abs"])
    ap.add_argument("--no_feedback", action="store_true")
    ap.add_argument("--no_recall", action="store_true")
    ap.add_argument("--no_deep_sup", action="store_true")
    ap.add_argument("--no_counts", action="store_true")
    ap.add_argument("--no_carry_z", action="store_true")
    ap.add_argument("--anchor", action="store_true", help="skip connection from the priced schedule to the logits")
    return ap.parse_args()


def main():
    a = parse()
    torch.set_num_threads(a.threads)
    os.chdir(C.ROOT)
    over = {k: getattr(a, k) for k in ("hidden", "n_latent", "n_cycles", "mp_layers", "k_train", "epochs", "batch",
                                        "k_eval", "rl_updates", "rl_batch", "rl_samples", "rl_k", "rl_log", "rl_ckpt", "accum",
                                        "lr", "ema", "p_ext",
                                        "rl_lr", "bce_w", "halt_w", "clip", "wd", "decode", "reward", "lp_workers",
                                        "inflight", "p_seed", "w_wrong", "w_grad", "rl_temp", "anchor_init")}
    ckpt_path = os.path.join(C.OUT, f"{a.tag}.pt")
    hist_path = os.path.join(C.RES, f"rloop_train_{a.tag}.json")
    init = None
    if a.stage == "rl":
        src = ckpt_path if os.path.exists(ckpt_path) else os.path.join(C.OUT, f"{a.init}.pt")
        init = torch.load(src, map_location="cpu", weights_only=False)
        cfg = dict(init["cfg"])
        cfg.update({k: v for k, v in over.items() if v is not None})
        if a.anchor:
            cfg["anchor"] = True
    else:
        cfg = C.model_cfg(a.preset, **over)
        cfg.update(feedback=not a.no_feedback, recall=not a.no_recall, deep_sup=not a.no_deep_sup,
                   counts=not a.no_counts, carry_z=not a.no_carry_z, sym_rank=a.no_counts, anchor=a.anchor)
    cfg.update(device=a.device, preset=a.preset, seed=a.seed, n_train=a.n_train, line_scale=1.0)
    log_f = open(os.path.join(C.OUT, f"{a.tag}.log"), "a")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    sysm = C.system("uc12")
    feats = C.feats_for(sysm, sym_rank=cfg.get("sym_rank", False))
    tr, y = C.train_data(a.n_train, a.data_dir)
    if len(y) < a.n_train:
        log(f"[{a.tag}] only {len(y)} training instances available")
    c_lab = C.label_costs(tr, y, f"train{a.n_train}")
    insts = C.insts_of(tr)
    va = C.val_data()
    vidx = list(range(a.val_n))
    vinsts = C.insts_of(va, vidx)
    db = va["db"][vidx]
    cores = [int(c) for c in a.lp_cores.split(",") if c != ""] or None
    pool = rloop.LPPool(workers=cfg["lp_workers"], cores=cores)
    device = torch.device(a.device)
    net = rloop.build_net(sysm, feats, cfg).to(device)
    n_par = sum(p.numel() for p in net.parameters())
    log(f"[{a.tag}] stage {a.stage} params {n_par} cfg {json.dumps(cfg, default=str)}")

    starts = None
    if cfg.get("p_seed", 0.0) > 0:             # monitor the seeded mode too (LP-rounding best of 5 on validation)
        lpr = [min((rloop.decode((x["u_rel"] > th).astype(np.int8), x, sysm, "round")[1] for th in
                    (0.001, 0.05, 0.2, 0.5, 0.8)), key=lambda s_: pool.run([(("va", i), x, s_, 1.0, "round")])[0]["cost"])
               for i, x in zip(vidx, vinsts)]
        starts = np.stack(lpr)

    def val_fn(model):
        out = val_one(model, None)
        if starts is not None:
            out["seeded"] = val_one(model, starts)
        return out

    def val_one(model, st0):
        t0 = time.time()
        K = cfg["k_eval"]
        steps = rloop.rollout_batch(model, feats, sysm, vinsts, K, pool, device, [("va", i) for i in vidx],
                                    mode=cfg.get("decode", "viterbi"), starts=st0)
        cost = np.array([[x[0] for x in st] for st in steps])          # [K(+1), n]
        served = np.array([[x[1] < 1e-6 and x[2] < 1e-6 for x in st] for st in steps])
        best = np.minimum.accumulate(cost, 0)
        g = lambda c: (c - db) / c * 100
        out = dict(gap_k0=float(g(cost[0]).mean()), gap_best=float(g(best[-1]).mean()),
                   gap_best_k=[round(float(g(best[k]).mean()), 4) for k in range(len(cost))],
                   served_k0=float(served[0].mean() * 100),
                   served_best=float(np.mean([served[np.argmin(cost[:, i]), i] for i in range(len(vidx))]) * 100),
                   med_best=float(np.median(g(best[-1]))), s=round(time.time() - t0, 1))
        return out

    hist = json.load(open(hist_path)) if os.path.exists(hist_path) else []

    def ckpt(it, net_, ema, opt, sched, rec):
        rec = dict(rec, stage=a.stage)
        hist.append(rec)
        os.makedirs(C.RES, exist_ok=True)
        json.dump(hist, open(hist_path, "w"), indent=1, default=float)
        C.save_ckpt(ckpt_path, net_, ema, opt, sched, cfg, rec, extra=dict(it=it, stage=a.stage, params=n_par))

    start, opt_state, ema_state = 0, None, None
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        if ck["extra"].get("stage") == a.stage:
            net.load_state_dict(ck["net"])
            ema_state, opt_state = ck["ema"], dict(opt=ck["opt"], sched=ck["sched"])
            start = ck["extra"]["it"] + 1
            log(f"[{a.tag}] resuming {a.stage} at {start}")
    elif init is not None:
        missing, unexpected = net.load_state_dict(init["ema"], strict=False)   # fine-tuning starts from the EMA
        assert not unexpected and set(missing) <= {"anchor"}, (missing, unexpected)
        log(f"[{a.tag}] initialised from {a.init}: {init['rec'].get('val')}")
    t0 = time.time()
    if a.stage == "sup":
        if start < cfg["epochs"]:
            rloop.train_supervised(net, feats, sysm, insts, y, c_lab, cfg, pool, device, log=log, val_fn=val_fn,
                                   ckpt=ckpt, start_epoch=start, opt_state=opt_state, ema_state=ema_state, seed=a.seed)
    else:
        if start < cfg["rl_updates"]:
            if start == 0 and init is not None:
                ema_state = dict(net.state_dict())
            rloop.train_reinforce(net, feats, sysm, insts, y, c_lab, cfg, pool, device, log=log, val_fn=val_fn,
                                  ckpt=ckpt, start_update=start, opt_state=opt_state, ema_state=ema_state, seed=a.seed)
    log(f"[{a.tag}] done in {time.time() - t0:.0f}s; LPs {pool.n_lp}, hits {pool.n_hit}, LP seconds {pool.t_lp:.0f}")
    pool.close()


if __name__ == "__main__":
    main()

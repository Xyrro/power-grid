"""Train a tiny recursive model (or a non-recursive control) for end-to-end 12-hour UC (arm A, docs/recursive/trm.md).

    python3 scripts/rec_trm_train.py --tag main --preset cpu                       # deep supervision (stage 1)
    python3 scripts/rec_trm_train.py --tag oneshot --preset cpu --mode oneshot
    python3 scripts/rec_trm_train.py --tag main_rl --stage rl --base main --rl_steps 150   # REINFORCE (stage 2)
    python3 scripts/rec_trm_train.py --tag paper --preset paper --device cuda      # TRM paper scale on a GPU

Every scale knob of otsl.trm.TRMConfig is a flag (unset flags keep the preset's value). Data: train.npz (500 MILP
labels) + teacher-polished extra instances of the m1x study (--n_train 2000), validation loss on val.npz (60).
Outputs: checkpoint <out_dir>/<tag>.pt (resumable; rerun the same command after an interruption), training log
results/recursive/trm_train_<tag>.json.
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
from otsl import trm  # noqa: E402
from otsl.uc import UCModel, UCScenario, load_rts_gmlc  # noqa: E402

OUT = os.path.join("data", "generated", "recursive")
RES = os.path.join("results", "recursive")


def grad_at_labels(sysm, tr, path, logf=print):
    """dispatch-LP sensitivities d cost / d u at every training label (cached; cost weighting of the loss)"""
    if os.path.exists(path):
        g = np.load(path)["grad"]
        if len(g) == len(tr["load"]):
            return g
    m = UCModel(sysm, T=tr["load"].shape[1])
    out = np.zeros(tr["y"].shape, np.float32)
    t0 = time.time()
    for i in range(len(out)):
        sc = UCScenario(load=tr["load"][i], avail=tr["avail"][i], u0=tr["u0"][i], sr=tr["sr"][i])
        _, out[i] = m.dispatch_gradient(sc, tr["y"][i].astype(float))
        if (i + 1) % 200 == 0:
            logf(f"[grad] {i + 1}/{len(out)} ({time.time() - t0:.0f}s)")
    np.savez_compressed(path, grad=out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--preset", default="cpu", choices=sorted(trm.PRESETS))
    ap.add_argument("--stage", default="ds", choices=["ds", "rl"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--out_dir", default=os.path.join(OUT, "models"))
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--val_every", type=int, default=500)
    # every TRMConfig field
    for f, typ in [("hidden", int), ("layers", int), ("expansion", float), ("unit_mix", str), ("time_mix", str),
                   ("heads", int), ("time_kernel", int), ("hops", int), ("sym", str), ("y0", str),
                   ("cost_weight", float), ("mode", str), ("n", int), ("T", int), ("n_sup", int), ("halt_tol", int),
                   ("halt_explore", float), ("halt_weight", float), ("batch", int), ("steps", int), ("lr", float),
                   ("weight_decay", float), ("warmup", int), ("lr_min_frac", float), ("ema", float), ("clip", float),
                   ("amp", str), ("seed", int)]:
        ap.add_argument(f"--{f}", type=typ, default=None)
    for f in ("elec", "lp_feats", "halt_train"):
        ap.add_argument(f"--{f}", type=int, default=None, help="0 / 1")
    ap.add_argument("--beta1", type=float, default=None)
    ap.add_argument("--beta2", type=float, default=None)
    # stage 2 (REINFORCE with the dispatch LP)
    ap.add_argument("--base", default=None, help="tag of the stage-1 checkpoint")
    ap.add_argument("--rl_steps", type=int, default=150)
    ap.add_argument("--rl_bs", type=int, default=16)
    ap.add_argument("--rl_samples", type=int, default=6)
    ap.add_argument("--rl_lr", type=float, default=1e-4)
    ap.add_argument("--rl_anchor", type=float, default=0.5)
    ap.add_argument("--rl_random_step", type=int, default=0)
    ap.add_argument("--rl_k", type=int, default=0, help="supervision steps of the fine-tuned answer (0: n_sup)")
    ap.add_argument("--workers", type=int, default=1, help="dispatch-LP worker processes (stage 2)")
    a = ap.parse_args()
    os.chdir(os.path.dirname(HERE))
    torch.set_num_threads(a.threads)
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    logpath = os.path.join(OUT, "logs", f"train_{a.tag}.log")
    os.makedirs(os.path.dirname(logpath), exist_ok=True)
    logfile = open(logpath, "a")

    def logf(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        logfile.write(line + "\n")
        logfile.flush()

    sysm = load_rts_gmlc()
    tr = trm.train_data(a.n_train)
    ckpt = os.path.join(a.out_dir, f"{a.tag}.pt")
    if a.stage == "ds":
        over = {k: getattr(a, k) for k in [f.name for f in trm.fields(trm.TRMConfig)] if hasattr(a, k)}
        for k in ("elec", "lp_feats", "halt_train"):
            if over.get(k) is not None:
                over[k] = bool(over[k])
        if a.beta1 is not None or a.beta2 is not None:
            b = trm.PRESETS[a.preset].get("betas", (0.9, 0.95))
            over["betas"] = (a.beta1 if a.beta1 is not None else b[0], a.beta2 if a.beta2 is not None else b[1])
        cfg = trm.make_config(a.preset, **over)
        va = trm.val_data()
        g = None
        if cfg.cost_weight > 0:
            g = grad_at_labels(sysm, tr, os.path.join(OUT, f"grad_lp_n{a.n_train}.npz"), logf)
        t0 = time.time()
        model, ema, feats, log = trm.train(cfg, sysm, tr, va, ckpt, device=a.device, log_every=a.log_every,
                                           val_every=a.val_every, logf=logf, grad_lp=g)
        rec = dict(tag=a.tag, stage="ds", preset=a.preset, cfg=cfg.to_json(), params=trm.n_params(model),
                   n_train=a.n_train, train_s=log["time_s"], threads=a.threads, device=a.device,
                   train=log["train"], val=log["val"])
    else:
        from otsl.ucml import DispatchOracle
        base = os.path.join(a.out_dir, f"{a.base}.pt")
        model, feats, cfg, _ = trm.load_model(base, sysm, a.device, which="ema")
        model.train()
        for p_ in model.parameters():
            p_.requires_grad_(True)
        oracle = DispatchOracle({"T": tr["load"].shape[1]}, workers=a.workers)
        t0 = time.time()
        model, ema, hist = trm.train_reinforce(model, cfg, sysm, feats, tr, oracle, ckpt, steps=a.rl_steps, bs=a.rl_bs,
                                               n_samples=a.rl_samples, lr=a.rl_lr, k_sup=a.rl_k or None,
                                               random_step=bool(a.rl_random_step), anchor=a.rl_anchor,
                                               device=a.device, logf=logf, base_ckpt=base)
        oracle.close()
        rec = dict(tag=a.tag, stage="rl", base=a.base, cfg=cfg.to_json(), params=trm.n_params(model),
                   rl=dict(steps=a.rl_steps, bs=a.rl_bs, samples=a.rl_samples, lr=a.rl_lr, anchor=a.rl_anchor,
                           random_step=a.rl_random_step, k=a.rl_k), train_s=hist[-1]["s"] if hist else 0.0,
                   threads=a.threads, hist=hist)
    with open(os.path.join(RES, f"trm_train_{a.tag}.json"), "w") as f:
        json.dump(rec, f, indent=1, default=float)
    logf(f"[done] {a.tag} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

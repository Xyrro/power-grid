#!/usr/bin/env python3
"""Run a plan of resumable jobs through pgu_guard.py, as many at a time as the project share allows.

    python3 scripts/pace/pgu_queue.py scripts/pace/plans/<plan>.json          # one pass: submit what can start now

The plan is a JSON list, in priority order, of entries
    {"name": "<job name without the pgu- prefix>", "done": "<repo-relative file that exists once finished>",
     "max_segments": <int>, "submit": [<pgu_guard.py submit arguments, without --name>]}
A pass skips an entry whose done-file exists, whose job is queued or running, or which has been submitted
max_segments times (reported as stuck); otherwise it submits the entry through the guard, which refuses anything over
the project share, and stops at the first refusal so later entries wait for capacity. Idempotent: run it again at any
time (e.g. every few minutes from a watcher); it never cancels anything.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def live_names():
    r = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-a", "-o", "%j"],
                       capture_output=True, text=True, check=True)
    return {x.strip() for x in r.stdout.splitlines() if x.strip()}


def submissions(prefix):
    path = os.path.join(os.environ["PGU_ROOT"], "jobs.tsv")
    n = {}
    if os.path.exists(path):
        for line in open(path):
            f = line.rstrip("\n").split("\t")
            if len(f) > 1 and f[1].startswith(prefix):
                n[f[1]] = n.get(f[1], 0) + 1
    return n


def main(plan_path, dry=False):
    plan = json.load(open(plan_path))
    live, subs = live_names(), submissions("pgu-")
    status, blocked = [], False
    for e in plan:
        full = "pgu-" + e["name"]
        if os.path.exists(e["done"]):
            status.append((e["name"], "done"))
        elif full in live:
            status.append((e["name"], "queued/running"))
        elif subs.get(full, 0) >= e.get("max_segments", 4):
            status.append((e["name"], f"STUCK after {subs[full]} segments"))
        elif blocked:
            status.append((e["name"], "waiting"))
        else:
            cmd = [sys.executable, os.path.join(HERE, "pgu_guard.py"), "submit", "--name", e["name"], *e["submit"]]
            if dry:
                cmd.append("--dry-run")
            r = subprocess.run(cmd, capture_output=True, text=True)
            last = (r.stdout.strip().splitlines() or [""])[-1]
            if r.returncode == 0 and not dry:
                status.append((e["name"], last))
            else:
                blocked = True
                out = (r.stdout + r.stderr).strip().splitlines()
                why = "dry run, accepted" if dry and r.returncode == 0 else " | ".join(out[-2:])
                status.append((e["name"], f"waiting: {why}"))
    for name, st in status:
        print(f"{name:24s} {st}")
    return status


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--dry-run"]
    if len(args) != 1:
        sys.exit(__doc__)
    main(args[0], dry="--dry-run" in sys.argv)

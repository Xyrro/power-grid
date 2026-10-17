"""Tests for scripts/pace/pgu_guard.py: Slurm parsing and the budget rules that protect the user's other projects."""
import importlib.util
import os
import sys

import pytest

_spec = importlib.util.spec_from_file_location(
    "pgu_guard", os.path.join(os.path.dirname(__file__), "..", "scripts", "pace", "pgu_guard.py"))
g = importlib.util.module_from_spec(_spec)
sys.modules["pgu_guard"] = g          # dataclasses look the module up by name
_spec.loader.exec_module(g)


def cfg(**kw):
    c = dict(g.DEFAULTS)
    c.update({k: str(v) for k, v in kw.items()})
    return c


@pytest.mark.parametrize("s, h", [("08:00:00", 8.0), ("1-02:00:00", 26.0), ("30:00", 0.5), ("90", 1.5),
                                  ("2-12", 60.0), ("0-01:30", 1.5)])
def test_parse_slurm_time(s, h):
    assert g.parse_slurm_time(s) == pytest.approx(h)


@pytest.mark.parametrize("s", ["UNLIMITED", "NOT_SET", "", "INVALID"])
def test_parse_slurm_time_unbounded(s):
    assert g.parse_slurm_time(s) is None


@pytest.mark.parametrize("tres, n", [("gres:gpu:H100:2", 2), ("gpu:1", 1), ("gres/gpu=4,gres/gpu:h100=4", 4),
                                     ("N/A", 0), ("cpu=8,mem=48G", 0), ("gres:gpu:h200:1", 1)])
def test_gpu_count(tres, n):
    assert g.gpu_count(tres) == n


SQUEUE = """\
1001|other-proj-train|RUNNING|1|gres:gpu:H100:1|5:00:00|8:00:00
1002|other-proj-eval|PENDING|1|gres:gpu:H100:2|8:00:00|8:00:00
2001|pgu-trm-small|RUNNING|1|gres:gpu:H100:1|2:00:00|6:00:00
2002_3|pgu-gen|RUNNING|1|N/A|1:00:00|2:00:00
2002_[4-9%4]|pgu-gen|PENDING|1|N/A|2:00:00|2:00:00
3001|pgu-not-ours|RUNNING|1|gres:gpu:H100:1|7:00:00|8:00:00
"""


def test_usage_separates_projects():
    jobs = g.parse_squeue_lines(SQUEUE)
    reg = {"2001": [], "2002": []}            # 3001 has the prefix but is not in the registry: not ours
    u = g.usage(jobs, reg, "pgu-")
    # Slurm's MaxTRESRunMins counts RUNNING jobs only (1001: 5, 2001: 2, 3001: 7); the pending 1002 does not count
    assert u.gpu_hours_all == pytest.approx(5 + 2 + 7)
    assert u.gpu_hours_ours == pytest.approx(2)          # ours, queued + running
    assert u.gpu_hours_ours_running == pytest.approx(2)
    assert u.gpu_jobs_all == 4 and u.gpu_jobs_ours == 1
    assert u.cpu_tasks_ours == 2


def test_pending_job_of_ours_counts_against_our_share_only():
    jobs = g.parse_squeue_lines(SQUEUE + "2003|pgu-pending|PENDING|1|gres:gpu:H100:1|8:00:00|8:00:00\n")
    u = g.usage(jobs, {"2001": [], "2002": [], "2003": []}, "pgu-")
    assert u.gpu_hours_all == pytest.approx(5 + 2 + 7)   # unchanged: pending holds no running capacity
    assert u.gpu_hours_ours == pytest.approx(2 + 8)      # but it does count against our own share


def test_budget_respects_project_share_and_user_limit():
    u = g.Usage(gpu_hours_all=26, gpu_hours_ours=2, gpu_jobs_all=4, gpu_jobs_ours=1, cpu_tasks_ours=0)
    c = cfg(SHARE_GPU_HOURS=16, USER_GPU_HOURS_LIMIT=32)
    assert g.check_budget(c, u, "gpu", 1, 6.0, 1, None) == []
    # Exceeding the RUNNING-capacity caps is not a refusal: Slurm (MaxGRESRunMinsPerUser) just holds the job.
    notes: list = []
    assert g.check_budget(c, u, "gpu", 1, 7.0, 1, None, notes) == []
    assert any("MaxGRESRunMinsPerUser" in n for n in notes)
    notes.clear()
    assert g.check_budget(c, g.Usage(10, 10, 2, 2, 0, 10.0), "gpu", 1, 8.0, 1, None, notes) == []
    assert any("SHARE_GPU_HOURS" in n for n in notes)
    # queued jobs of ours hold no running capacity
    assert g.check_budget(c, g.Usage(0, 30, 2, 2, 0, 0.0), "gpu", 1, 8.0, 1, None) == []
    assert any("MAX_WALLTIME" in b for b in g.check_budget(c, g.Usage(0, 0, 0, 0, 0), "gpu", 1, 9.0, 1, None))
    assert any("arrays" in b for b in g.check_budget(c, g.Usage(0, 0, 0, 0, 0), "gpu", 1, 1.0, 4, 2))


def test_budget_gpu_job_counts():
    c = cfg(SHARE_GPU_JOBS=4, USER_GPU_JOBS_LIMIT=50)
    assert any("SHARE_GPU_JOBS" in b for b in g.check_budget(c, g.Usage(0, 0, 10, 4, 0), "gpu", 1, 1.0, 1, None))
    assert any("USER_GPU_JOBS_LIMIT" in b for b in g.check_budget(c, g.Usage(0, 0, 50, 0, 0), "gpu", 1, 1.0, 1, None))


def test_budget_cpu_arrays():
    c = cfg(SHARE_CPU_TASKS=16, SHARE_CPU_CONCURRENT=8)
    z = g.Usage(0, 0, 0, 0, 0)
    assert g.check_budget(c, z, "cpu", 0, 4.0, 16, 8) == []
    assert any("throttle" in b for b in g.check_budget(c, z, "cpu", 0, 4.0, 16, None))
    assert any("SHARE_CPU_TASKS" in b for b in g.check_budget(c, z, "cpu", 0, 4.0, 200, 8))
    assert any("SHARE_CPU_TASKS" in b for b in g.check_budget(c, g.Usage(0, 0, 0, 0, 10), "cpu", 0, 1.0, 8, 4))


def test_array_size():
    assert g.array_size("0-199%8") == (200, 8)
    assert g.array_size("1,3,5") == (3, None)
    assert g.array_size("0-15:2%4") == (8, 4)
    assert g.array_size("") == (1, None)


def _fake_slurm(tmp_path, squeue_text):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (tmp_path / "squeue.txt").write_text(squeue_text)
    for name, body in {"squeue": f"cat {tmp_path / 'squeue.txt'}", "scontrol": "true",
                       "sbatch": f"echo \"$@\" >> {tmp_path / 'sbatch_calls.txt'}; echo 4242",
                       "scancel": f"echo \"$@\" >> {tmp_path / 'scancel_calls.txt'}"}.items():
        p = bin_dir / name
        p.write_text("#!/bin/sh\n" + body + "\n")
        p.chmod(0o755)
    return bin_dir


def test_cli_submit_and_cancel_only_own_jobs(tmp_path, monkeypatch, capsys):
    root = tmp_path / "pgu"
    (root / "repo" / "scripts").mkdir(parents=True)
    (root / ".pgu-root").write_text("")
    script = root / "repo" / "scripts" / "job.sbatch"
    script.write_text("#!/bin/sh\n")
    (root / "jobs.tsv").write_text("#header\n")
    cfgfile = root / "pgu.conf"
    cfgfile.write_text("MAX_WALLTIME=20:00:00\nSHARE_GPU_HOURS=32\nUSER_GPU_HOURS_LIMIT=32\nPGU_ROOT=%s\n" % root)
    bin_dir = _fake_slurm(tmp_path, SQUEUE)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setenv("PGU_ROOT", str(root))
    monkeypatch.setenv("USER", "someone")
    monkeypatch.chdir(root / "repo")
    # dry run: accounting printed, nothing submitted
    g.main(["--config", str(cfgfile), "submit", "--kind", "gpu", "--name", "trm", "--gpus", "1",
            "--time", "02:00:00", "--dry-run", str(script)])
    assert "dry run" in capsys.readouterr().out and not (tmp_path / "sbatch_calls.txt").exists()
    # a hard limit still refuses: MAX_WALLTIME is 20 h in this config
    with pytest.raises(SystemExit):
        g.main(["--config", str(cfgfile), "submit", "--kind", "gpu", "--name", "trm", "--gpus", "1",
                "--time", "21:00:00", str(script)])
    assert not (tmp_path / "sbatch_calls.txt").exists()
    # a 2 h job fits: submitted and recorded
    g.main(["--config", str(cfgfile), "submit", "--kind", "gpu", "--name", "trm", "--gpus", "1",
            "--time", "02:00:00", str(script)])
    call = (tmp_path / "sbatch_calls.txt").read_text()
    assert "-J pgu-trm" in call and "--comment=power-grid-uc" in call and f"PGU_ROOT={root}" in call
    assert "4242" in (root / "jobs.tsv").read_text()
    # cancelling another project's job (1001) or an unrecorded prefixed job (3001) is refused
    g.main(["cancel", "1001", "3001"])
    assert not (tmp_path / "scancel_calls.txt").exists()


def test_root_marker_required(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        g.check_root(cfg(PGU_ROOT=str(tmp_path)))
    (tmp_path / ".pgu-root").write_text("")
    assert g.check_root(cfg(PGU_ROOT=str(tmp_path))) == os.path.realpath(tmp_path)
    outside = tmp_path.parent / "elsewhere.sbatch"
    with pytest.raises(SystemExit):
        g.check_root(cfg(PGU_ROOT=str(tmp_path)), str(outside))


def test_total_job_limit_protects_other_projects():
    """QOS MaxSubmitPU counts every job, not only GPU ones: a big CPU array can lock the user out entirely."""
    many = "".join(f"{1000 + i}|other-proj|RUNNING|1|N/A|1:00:00|2:00:00\n" for i in range(45))
    u = g.usage(g.parse_squeue_lines(many), {}, "pgu-")
    assert u.jobs_all == 45 and u.jobs_ours == 0
    c = cfg(USER_JOBS_LIMIT=50, SHARE_JOBS=40, SHARE_CPU_TASKS=40, SHARE_CPU_CONCURRENT=40)
    assert any("USER_JOBS_LIMIT" in b for b in g.check_budget(c, u, "cpu", 0, 2.0, 10, 10))
    assert g.check_budget(c, u, "cpu", 0, 2.0, 4, 4) == []
    ours = g.Usage(0, 0, 0, 0, 0, 0.0, 40, 40)
    assert any("SHARE_JOBS" in b for b in g.check_budget(c, ours, "cpu", 0, 2.0, 1, 1))

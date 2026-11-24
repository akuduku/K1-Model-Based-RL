"""Step 08's sequence of short walking runs (mbrl/grad/sequence.py): the queue and its flags, the time plans for 1, 2
and 4 physical cores, the core count, and the schedule itself on fake runs (a crash, a run without time, a hang)."""

import json
import sys

from mbrl.grad import sequence as S


def jobs(*argv):
    return S.build_jobs(S.parser().parse_args(list(argv)))


def opts(job):
    """train.py's options of a job as {flag: value} (the last one wins, as in argparse)."""
    out, cmd = {}, job.cmd[4:]
    i = 0
    while i < len(cmd):
        if i + 1 < len(cmd) and not cmd[i + 1].startswith("--"):
            out[cmd[i]] = cmd[i + 1]
            i += 2
        else:
            out[cmd[i]] = True
            i += 1
    return out


def test_queue_order_and_each_runs_change():
    js = jobs()
    assert [j.id for j in js][:6] == ["base-s1", "clock-fixed-s1", "fixed-s1", "base-s2", "clock-fixed-s2", "fixed-s2"]
    assert len(js) == 17
    by = {j.id: opts(j) for j in js}
    for o in by.values():   # W5's settings, one core, float32, one evaluation at the end
        assert o["--task"] == "walk" and o["--device"] == "cpu" and o["--dtype"] == "float32" and o["--threads"] == "1"
        assert o["--robots"] == "16" and o["--eval-every-min"] == "0" and o["--eval-robots"] == "64"
        assert o["--action-noise"] == "0.05" and o["--command-scale"] == "0.25" and "--iterations" not in o
    assert "--term2" not in by["base-s1"] and by["base-s1"]["--delta-final"] == "0.01" and by["base-s1"]["--regress"] == "8"
    assert by["fixed-s2"]["--term2"] == "fixed" and "--gait-clock" not in by["fixed-s2"]
    assert by["clock-fixed-s3"]["--term2"] == "fixed" and by["clock-fixed-s3"]["--gait-clock"] == "4"
    assert "--delta-final" not in by["optimiser-s1"] and by["optimiser-s1"]["--regress"] == "32"
    assert by["window40-s1"]["--steps"] == "40" and by["window40-s1"]["--minutes"] == "80.0"
    assert by["base-s3"]["--seed"] == "3" and by["base-s3"]["--minutes"] == "40.0"
    assert by["base-s3"]["--run-name"] == "step08-base-s3"
    assert by["long-clock-fixed-s1"]["--minutes"] == "160.0" and by["long-clock-fixed-s1"]["--gait-clock"] == "4"
    assert by["long-clock-fixed-s1"]["--run-name"] == "step08-long-clock-fixed-s1"


def test_smoke_options_and_forced_crash():
    js = jobs("--only", "fixed-s1,base-s1", "--robots", "4", "--iterations", "2", "--cap-minutes", "3",
              "--no-wandb", "--crash", "fixed-s1")
    assert [j.id for j in js] == ["base-s1", "fixed-s1"]               # queue order, not the order asked
    a, b = opts(js[0]), opts(js[1])
    assert a["--robots"] == "4" and a["--iterations"] == "2" and a["--minutes"] == "3.0" and a["--no-wandb"] is True
    assert a["--dtype"] == "float32" and b["--dtype"] == "no_such_dtype"


def test_time_plans_for_1_2_and_4_cores():
    js = jobs()
    for slots, runs, end in ((1, 8, 360), (2, 13, 315), (4, 17, 345)):
        plan, skipped = S.simulate(js, slots, 360)
        assert len(plan) == runs and max(e for _, e, _, _ in plan) == end and len(skipped) == 17 - runs
    plan, _ = S.simulate(js, 2, 360)
    started = {j.id: (s, e) for s, e, _, j in plan}
    assert started["base-s1"] == (0, 45) and started["clock-fixed-s1"] == (0, 45)
    assert started["window40-s1"] == (225, 310) and "long-clock-fixed-s1" not in started
    plan, _ = S.simulate(js, 4, 360)
    started = {j.id: (s, e) for s, e, _, j in plan}
    assert started["long-clock-fixed-s1"] == (135, 300) and started["long-optimiser-s1"] == (180, 345)
    assert started["window40-s2"] == (175, 260)


CPUINFO = "\n\n".join(f"processor\t: {p}\nphysical id\t: 0\ncore id\t\t: {c}\nmodel name\t: x"
                      for p, c in ((0, 0), (1, 1), (2, 0), (3, 1)))      # 2 cores with 2 hyperthreads each


def test_physical_cores_from_cpuinfo():
    assert S.physical_cores(CPUINFO, allowed={0, 1, 2, 3}) == (2, [0, 1], 4)
    assert S.physical_cores(CPUINFO, allowed={2, 3}) == (2, [2, 3], 2)   # only the CPUs this process may use
    assert S.choose_slots(2, None) == 2 and S.choose_slots(8, None) == 4 and S.choose_slots(4, 2.0) == 2
    assert S.choose_slots(4, None, override=3) == 3


FAKE = r"""
import json, pathlib, sys, time
d = pathlib.Path(sys.argv[1]); d.mkdir(parents=True, exist_ok=True)
n, code, wait = int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
with open(d / "metrics.jsonl", "w") as f:
    for i in range(1, n + 1):
        f.write(json.dumps({"iteration": i, "Cost/seconds": 2.0, "Walk/feet_air": 0.01 * i, "Grad/fell": 0.1,
                            "Learn/step_taken": 0.3}) + "\n")
(d / f"model_{n}.pt").write_bytes(b"")
time.sleep(wait)
if code:
    sys.exit(code)
print(f"[grad] stopped: time budget of 1 min used up after iteration {n} (1.0 min, 10 steps, 0 falls); "
      f"model_{n}.pt  fell 0.5  stood_s 3.2  speed_error 0.25", flush=True)
"""


def fake(tmp_path, name, idea, seed, n, code, wait, cap=1.0):
    run_name = f"t-{name}"
    return S.Job(name, idea, "short", seed, cap, run_name,
                 [sys.executable, "-c", FAKE, str(tmp_path / "runs" / run_name), str(n), str(code), str(wait)])


def test_one_crash_ends_only_that_run_and_a_row_follows_every_run(tmp_path, monkeypatch):
    calls = []
    real = S.write_summary
    monkeypatch.setattr(S, "write_summary", lambda rows, out, header="": calls.append(len(rows)) or real(rows, out, header))
    ok = fake(tmp_path, "base-s1", "base", 1, 35, 0, 0.1)
    bad = fake(tmp_path, "fixed-s1", "fixed", 1, 3, 1, 1.0)
    rows = S.run_sequence([ok, bad], 2, 10.0, tmp_path / "out", tmp_path / "runs", poll=0.05)
    assert calls == [1, 2]                                              # the summary after each run, not at the end
    saved = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert [r["run"] for r in saved] == [r["run"] for r in rows] == ["base-s1", "fixed-s1"]
    a, b = saved
    assert a["status"] == "ok" and a["updates"] == 35 and a["fair_test"] and a["checkpoint"] == "model_35.pt"
    assert abs(a["feet_air@30"] - 0.28) < 1e-9 and a["feet_air@40"] is None    # mean of updates 26-30; 40 not reached
    assert a["eval_fell"] == 0.5 and a["eval_stood_s"] == 3.2 and a["s_per_update"] == 2.0
    assert abs(a["last10_feet_air"] - 0.305) < 1e-9 and a["last10_clock_reward"] is None
    assert b["status"] == "crashed (exit 1) at update 3" and not b["fair_test"] and b["checkpoint"] == "model_3.pt"
    md = (tmp_path / "out" / "summary.md").read_text()
    assert "crashed (exit 1) at update 3" in md and "screen (feet in the air at update 30" in md
    assert (tmp_path / "out" / "summary.csv").read_text().startswith("run,")


def test_a_run_without_enough_time_is_skipped_not_started(tmp_path):
    never = S.Job("long-fixed-s1", "fixed", "long", 1, 100.0, "t-never",
                  [sys.executable, "-c", "raise SystemExit(7)"])
    rows = S.run_sequence([never], 1, 10.0, tmp_path / "out", tmp_path / "runs", poll=0.05)
    assert rows[0]["status"].startswith("skipped: 10 min left, needs 105")


def test_a_run_that_cannot_start_is_marked_and_the_next_one_runs(tmp_path):
    broken = S.Job("base-s1", "base", "short", 1, 1.0, "t-broken", [str(tmp_path / "no_such_python")])
    ok = fake(tmp_path, "fixed-s1", "fixed", 1, 2, 0, 0.0)
    rows = S.run_sequence([broken, ok], 1, 10.0, tmp_path / "out", tmp_path / "runs", cpus=[99999], poll=0.05)
    assert rows[0]["status"].startswith("could not start: FileNotFoundError")
    assert rows[1]["run"] == "fixed-s1" and rows[1]["status"] == "ok"        # the pin to a missing CPU did not stop it


def test_a_hanging_run_is_stopped(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "KILL_MIN", 0.005)                           # 0.3 s after its cap
    hang = S.Job("base-s1", "base", "short", 1, 0.001, "t-hang", [sys.executable, "-c", "import time; time.sleep(60)"])
    rows = S.run_sequence([hang], 1, 10.0, tmp_path / "out", tmp_path / "runs", poll=0.05)
    assert rows[0]["status"].startswith("stopped by the sequence") and rows[0]["minutes"] < 0.5


def test_screen_needs_every_seed_above_every_base_seed():
    rows = [{"idea": "base", "kind": "short", "feet_air@30": v} for v in (0.010, 0.012, 0.011)]
    rows += [{"idea": "clock-fixed", "kind": "short", "feet_air@30": v} for v in (0.030, 0.040, 0.025)]
    rows += [{"idea": "fixed", "kind": "short", "feet_air@30": v} for v in (0.050, 0.011, 0.040)]
    lines = S.screen(rows)
    assert any(line.strip().startswith("clock-fixed") and "PASSES" in line for line in lines)
    assert any(line.strip().startswith("fixed") and "does not pass" in line for line in lines)

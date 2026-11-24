r"""Step 08: short walking runs of the thesis method (mbrl/grad/train.py), several at once, one physical core each.

One run = one process of `python -m mbrl.grad.train --task walk` with W5's settings (COMMON) and one change (IDEAS),
its own seed and a time cap.  The step shrinks 0.05 -> 0.01 over the cap (train.py --delta-final: the anneal runs
over the time budget), and the network is evaluated once at the end (rl/walk/evaluate.py WalkEval: the full walk
world, six fixed commands).  The queue (QUEUE) is filled in order into as many slots as the machine has physical
cores (at most MAX_SLOTS); a run starts only if its cap + EVAL_MIN fit before the deadline, else it is skipped.
A crash ends only that run.  After every run the summary is rewritten, <out>/summary.{md,csv,json}, and printed:
the training curve (feet in the air, falls per window, speed error at updates 10 .. 50), the last 10 updates, the
final evaluation, and the screen of Step 08 (feet in the air at update 30 against the base seeds).

    python -m mbrl.grad.sequence --dry-run --cores 2                     # the commands and the time plan, runs nothing
    python -m mbrl.grad.sequence --minutes 360 --out /kaggle/working/step08 --log-root /kaggle/working/runs
    python -m mbrl.grad.sequence --minutes 20 --robots 4 --iterations 2 --cap-minutes 3 --eval-robots 4 --no-wandb \
        --only base-s1,clock-fixed-s1,fixed-s1 --crash fixed-s1 --slots 2 --out /tmp/seq   # laptop smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

EVAL_MIN = 5.0      # kept free after a run's cap: its last update, the final evaluation, the save
KILL_MIN = 15.0     # a run still alive this long after its cap gets SIGINT (train.py then stops cleanly), SIGKILL 1 min later
MAX_SLOTS = 4
CURVE_AT = (10, 20, 30, 40, 50)
BLOCK = 5           # the curve at update u = the mean over updates u-4 .. u
FAIR_UPDATES = 30   # fewer updates: "not a fair test"
SCREEN_AT = 30

# W5's settings (the grad-walk Kaggle task): random network, learned term 2 (train.py's default for walking), step
# 0.05 -> 0.01, 8 fitting steps, action noise 0.05, requests x0.25 at the start, the full walk world, 75 inputs.
# Short-run settings: 16 robots, float32, one core each, a checkpoint every 5 min, one evaluation at the end.
COMMON = {"--task": "walk", "--robots": 16, "--steps": 20, "--device": "cpu", "--dtype": "float32", "--threads": 1,
          "--delta": 0.05, "--delta-final": 0.01, "--regress": 8, "--action-noise": 0.05, "--command-scale": 0.25,
          "--save-every-min": 5, "--eval-every-min": 0, "--eval-robots": 64}
IDEAS = {"base": {},                                                # the yardstick
         "fixed": {"--term2": "fixed"},                             # A: the fixed term 2
         "clock-fixed": {"--term2": "fixed", "--gait-clock": 4},    # C + A: the gait clock and the fixed term 2
         "optimiser": {"--delta-final": None, "--regress": 32},     # D: no anneal (0.05 fixed), the network fitted harder
         "window40": {"--steps": 40}}                               # B: 0.8 s windows, longer than one gait cycle
CAP = {"short": 40.0, "window40": 80.0, "long": 160.0}              # minutes of training per run
QUEUE = [("base", 1), ("clock-fixed", 1), ("fixed", 1), ("base", 2), ("clock-fixed", 2), ("fixed", 2),
         ("optimiser", 1), ("base", 3), ("clock-fixed", 3), ("fixed", 3), ("window40", 1), ("optimiser", 2),
         ("optimiser", 3), ("long-clock-fixed", 1), ("long-fixed", 1), ("window40", 2), ("long-optimiser", 1)]

# what the summary keeps of metrics.jsonl
CURVE = {"feet_air": "Walk/feet_air", "fell": "Grad/fell", "speed_error": "Walk/speed_error"}
LAST10 = {"fell": "Grad/fell", "feet_air": "Walk/feet_air", "speed_error": "Walk/speed_error",
          "requested": "Walk/requested_speed", "step_taken": "Learn/step_taken", "action_change": "Learn/action_change",
          "clock_reward": "Reward/gait_clock", "w_upright": "Value/s_upright",
          "w_speed": "Value/s_track_linear_velocity", "command_scale": "Curriculum/command_scale"}


@dataclass
class Job:
    id: str             # e.g. clock-fixed-s1
    idea: str           # an IDEAS key
    kind: str           # short | window40 | long
    seed: int
    cap: float          # minutes of training
    run_name: str       # train.py's --run-name (W&B name, folder under --log-root)
    cmd: list = field(default_factory=list)


def build_jobs(args) -> list[Job]:
    jobs = []
    for name, seed in QUEUE:
        long = name.startswith("long-")
        idea = name[len("long-"):] if long else name
        kind = "long" if long else ("window40" if idea == "window40" else "short")
        cap = args.cap_minutes or CAP[kind]
        run_name = f"{args.prefix}-{name}-s{seed}"
        opts = {**COMMON, **IDEAS[idea], "--robots": args.robots, "--eval-robots": args.eval_robots, "--seed": seed,
                "--minutes": cap, "--iterations": args.iterations or None, "--run-name": run_name,
                "--log-root": args.log_root, "--wandb-project": args.wandb_project, "--wandb-entity": args.wandb_entity}
        cmd = [args.python, "-u", "-m", "mbrl.grad.train"]
        for k, v in opts.items():
            if v is not None:
                cmd += [k, str(v)]
        if args.no_wandb:
            cmd.append("--no-wandb")
        job = Job(f"{name}-s{seed}", idea, kind, seed, float(cap), run_name, cmd)
        if args.crash == job.id:            # the smoke test's forced crash: train.py fails on an unknown precision
            job.cmd += ["--dtype", "no_such_dtype"]
        jobs.append(job)
    if args.only:
        want = [s.strip() for s in args.only.split(",") if s.strip()]
        unknown = set(want) - {j.id for j in jobs}
        if unknown:
            raise SystemExit(f"--only: unknown runs {sorted(unknown)}; known: {[j.id for j in jobs]}")
        jobs = [j for j in jobs if j.id in want]
    return jobs


# ------------------------------------------------------------------------------------------------------- the machine
def physical_cores(text: str | None = None, allowed: set | None = None) -> tuple[int, list, int]:
    """-> (physical cores, one logical CPU per physical core, logical CPUs) from /proc/cpuinfo, counting only the
    CPUs this process may run on.  Hyperthreads share a core: two runs on one core would each run about 2x slower."""
    if text is None:
        try:
            text = Path("/proc/cpuinfo").read_text()
        except OSError:
            text = ""
    if allowed is None:
        try:
            allowed = os.sched_getaffinity(0)
        except (AttributeError, OSError):
            allowed = None
    first, logical = {}, 0
    for block in text.strip().split("\n\n"):
        f = {}
        for line in block.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                f[k.strip()] = v.strip()
        if "processor" not in f:
            continue
        cpu = int(f["processor"])
        if allowed is not None and cpu not in allowed:
            continue
        logical += 1
        first.setdefault((f.get("physical id", "0"), f.get("core id", f["processor"])), cpu)
    if not first:
        n = os.cpu_count() or 1
        return n, list(range(n)), n
    return len(first), sorted(first.values()), logical


def cpu_quota() -> float | None:
    """CPUs granted by the container's cgroup (cpu.max, or v1 cfs quota/period); None = no limit found."""
    try:
        q, p = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        return None if q == "max" else int(q) / int(p)
    except (OSError, ValueError):
        pass
    try:
        q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        p = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        return None if q <= 0 else q / p
    except (OSError, ValueError):
        return None


def choose_slots(cores: int, quota: float | None, override: int = 0) -> int:
    if override:
        return override
    return max(1, min(cores, MAX_SLOTS, int(quota) if quota else MAX_SLOTS))


# ----------------------------------------------------------------------------------------------------- the time plan
def simulate(jobs: list[Job], slots: int, minutes: float) -> tuple[list, list]:
    """The queue as the scheduler fills it if every run takes its cap + EVAL_MIN: each run in order takes the slot
    that frees first and is skipped if it would end after the deadline.  -> ([(start, end, slot, job)], [skipped])"""
    free_at, plan, skipped = [0.0] * slots, [], []
    for job in jobs:
        slot = min(range(slots), key=lambda s: (free_at[s], s))
        start = free_at[slot]
        if start + job.cap + EVAL_MIN > minutes + 1e-9:
            skipped.append(job)
            continue
        free_at[slot] = start + job.cap + EVAL_MIN
        plan.append((start, free_at[slot], slot, job))
    return plan, skipped


def plan_text(jobs, slots, minutes) -> str:
    plan, skipped = simulate(jobs, slots, minutes)
    lines = [f"time plan: {slots} slot(s), {minutes:g} min, every run its cap + {EVAL_MIN:g} min"]
    lines += [f"  {s:5.0f} - {e:5.0f} min  slot {k}  {j.id} (cap {j.cap:g} min)" for s, e, k, j in plan]
    end = max((e for _, e, _, _ in plan), default=0.0)
    lines.append(f"  {len(plan)} runs, the last ends at about {end:.0f} min"
                 + (f"; skipped (no time): {', '.join(j.id for j in skipped)}" if skipped else ""))
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------------- the summary
def read_metrics(path: Path) -> list[dict]:
    rows = []
    try:
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:          # a line cut by a crash
                pass
    except OSError:
        pass
    return rows


def _mean(values):
    v = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return sum(v) / len(v) if v else None


def block_mean(metrics: list[dict], key: str, u: int) -> float | None:
    """The mean of `key` over updates u-BLOCK+1 .. u; None if the run did not reach update u."""
    if not metrics or max(int(r["iteration"]) for r in metrics) < u:
        return None
    return _mean(r.get(key) for r in metrics if u - BLOCK < int(r["iteration"]) <= u)


FINAL = re.compile(r"\[grad\] (?:stopped: .*|done) after iteration (\d+) .*; (model_\d+\.pt)(.*)$")


def parse_final(log: Path) -> dict:
    """train.py's last line: '... after iteration N (...); model_N.pt  fell 0.9  stood_s 8.1 ...' -> the Eval/ numbers"""
    try:
        text = log.read_text(errors="replace")
    except OSError:
        return {}
    found = None
    for line in text.splitlines():
        m = FINAL.search(line)
        if m:
            found = m
    if found is None:
        return {}
    return {k: float(v) for k, v in re.findall(r"(\w+) (-?(?:[\d.]+(?:e[-+]?\d+)?|nan|inf))", found.group(3))}


def summarize(job: Job, code, minutes: float, stopped: bool, log_root: Path, log: Path) -> dict:
    run_dir = Path(log_root) / job.run_name
    metrics = read_metrics(run_dir / "metrics.jsonl")
    updates = max((int(r["iteration"]) for r in metrics), default=0)
    if stopped:
        status = f"stopped by the sequence (cap + {KILL_MIN:g} min) at update {updates}"
    elif code == 0:
        status = "ok"
    else:
        status = f"crashed (exit {code}) at update {updates}"
    ckpts = sorted(run_dir.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    secs = sorted(float(r["Cost/seconds"]) for r in metrics if "Cost/seconds" in r)
    row = {"run": job.id, "idea": job.idea, "kind": job.kind, "seed": job.seed, "cap_min": job.cap, "status": status,
           "updates": updates, "minutes": round(minutes, 1), "s_per_update": secs[len(secs) // 2] if secs else None,
           "fair_test": updates >= FAIR_UPDATES, "checkpoint": ckpts[-1].name if ckpts else None,
           "run_dir": str(run_dir), "log": str(log)}
    for name, key in CURVE.items():
        for u in CURVE_AT:
            row[f"{name}@{u}"] = block_mean(metrics, key, u)
    last = metrics[-10:]
    for name, key in LAST10.items():
        row[f"last10_{name}"] = _mean(r.get(key) for r in last)
    row.update({f"eval_{k}": v for k, v in parse_final(log).items()})
    return row


def skipped_row(job: Job, left: float) -> dict:
    return {"run": job.id, "idea": job.idea, "kind": job.kind, "seed": job.seed, "cap_min": job.cap,
            "status": f"skipped: {left:.0f} min left, needs {job.cap + EVAL_MIN:g}", "updates": 0, "fair_test": False}


def _f(v, scale=1.0, digits=2):
    return "–" if v is None else f"{v * scale:.{digits}f}"


def screen(rows: list[dict]) -> list[str]:
    """Step 08's first answer: feet in the air at update SCREEN_AT (mean of updates 26-30), short runs only.  An idea
    passes if every one of its seeds is above every base seed AND its mean is >= 2x the base mean."""
    key = f"feet_air@{SCREEN_AT}"
    base = [r[key] for r in rows if r.get("idea") == "base" and r.get("kind") == "short" and r.get(key) is not None]
    lines = [f"screen (feet in the air at update {SCREEN_AT}, %): base seeds "
             + (", ".join(_f(v, 100) for v in base) if base else "none yet")]
    for idea in [i for i in IDEAS if i != "base"]:
        vals = [r[key] for r in rows if r.get("idea") == idea and r.get("kind") != "long" and r.get(key) is not None]
        if not vals:
            continue
        line = f"  {idea}: seeds {', '.join(_f(v, 100) for v in vals)}"
        if base:
            ratio = (sum(vals) / len(vals)) / max(sum(base) / len(base), 1e-12)
            above = min(vals) > max(base)
            line += (f"; x{ratio:.1f} the base mean; all seeds above all base seeds: {'yes' if above else 'no'}"
                     f" -> {'PASSES' if above and ratio >= 2.0 else 'does not pass'} ({len(vals)} vs {len(base)} seeds)")
        lines.append(line)
    return lines


def markdown(rows: list[dict], header: str) -> str:
    out = [f"# Step 08 sequence: summary", "", header, "",
           "| run | status | updates | min | s/update | fair test | eval falls | eval speed err | eval stood s |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['run']} | {r['status']} | {r.get('updates', 0)} | {_f(r.get('minutes'), 1, 0)} | "
                   f"{_f(r.get('s_per_update'), 1, 0)} | {'yes' if r.get('fair_test') else 'no'} | "
                   f"{_f(r.get('eval_fell'))} | {_f(r.get('eval_speed_error'))} | {_f(r.get('eval_stood_s'), 1, 1)} |")
    out += ["", f"Training curve, mean of updates u-{BLOCK - 1} .. u, at u = {', '.join(map(str, CURVE_AT))}:", "",
            "| run | feet in the air % | falls per window % | speed error m/s |", "|---|---|---|---|"]
    for r in rows:
        if r.get("updates"):
            cell = lambda n, s, d: " / ".join(_f(r.get(f"{n}@{u}"), s, d) for u in CURVE_AT)  # noqa: E731
            out.append(f"| {r['run']} | {cell('feet_air', 100, 1)} | {cell('fell', 100, 1)} | {cell('speed_error', 1, 2)} |")
    out += ["", "Last 10 updates:", "",
            "| run | falls % | feet air % | speed err / requested | step_taken | action change | clock reward | "
            "upright w | speed w | requests x |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("updates"):
            out.append(f"| {r['run']} | {_f(r.get('last10_fell'), 100, 1)} | {_f(r.get('last10_feet_air'), 100, 1)} | "
                       f"{_f(r.get('last10_speed_error'))} / {_f(r.get('last10_requested'))} | "
                       f"{_f(r.get('last10_step_taken'))} | {_f(r.get('last10_action_change'), 1, 4)} | "
                       f"{_f(r.get('last10_clock_reward'), 1, 3)} | {_f(r.get('last10_w_upright'))} | "
                       f"{_f(r.get('last10_w_speed'))} | {_f(r.get('last10_command_scale'))} |")
    out += [""] + screen(rows) + [""]
    return "\n".join(out)


def write_summary(rows: list[dict], out: Path, header: str = "") -> str:
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(rows, indent=1))
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    md = markdown(rows, header)
    (out / "summary.md").write_text(md)
    return md


# ------------------------------------------------------------------------------------------------------ the schedule
def log(msg: str):
    print(f"[seq] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def _pin(cpu):
    """The child's CPU: one physical core per run.  A failed pin only loses the pinning, never the run."""
    if cpu is None or not hasattr(os, "sched_setaffinity"):
        return None

    def pin():
        try:
            os.sched_setaffinity(0, {cpu})
        except OSError:
            pass
    return pin


def run_sequence(jobs: list[Job], slots: int, minutes: float, out: Path, log_root: Path, cpus: list | None = None,
                 poll: float = 5.0, header: str = "") -> list[dict]:
    """Fill the slots in queue order; one process per run; a row and the summary after every run."""
    out, log_root = Path(out), Path(log_root)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    deadline = t0 + 60.0 * minutes
    pending, running, rows, free = list(jobs), {}, [], list(range(slots))
    child_env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}

    def finish(row):
        rows.append(row)
        md = write_summary(rows, out, header)
        log(f"{row['run']}: {row['status']}, {row.get('updates', 0)} updates")
        print(md, flush=True)

    try:
        while pending or running:
            now = time.time()
            while free and pending:
                job = pending.pop(0)
                left = (deadline - now) / 60.0
                if job.cap + EVAL_MIN > left:
                    finish(skipped_row(job, left))
                    continue
                slot = free.pop(0)
                cpu = cpus[slot] if cpus and slot < len(cpus) else None
                logf = open(out / "logs" / f"{job.id}.log", "w")
                try:
                    proc = subprocess.Popen(job.cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=REPO, env=child_env,
                                            preexec_fn=_pin(cpu))
                except (OSError, subprocess.SubprocessError) as exc:     # only this run is lost, not the sequence
                    logf.close()
                    free.insert(0, slot)
                    finish({"run": job.id, "idea": job.idea, "kind": job.kind, "seed": job.seed, "cap_min": job.cap,
                            "status": f"could not start: {type(exc).__name__}: {exc}", "updates": 0, "fair_test": False})
                    continue
                running[slot] = {"job": job, "proc": proc, "start": now, "log": logf, "sigint": None}
                log(f"start {job.id} in slot {slot}" + (f" (CPU {cpu})" if cpu is not None else "")
                    + f", cap {job.cap:g} min, {left:.0f} min left: {shlex.join(map(str, job.cmd))}")
            for slot, r in list(running.items()):
                code = r["proc"].poll()
                elapsed = (now - r["start"]) / 60.0
                if code is None:
                    if r["sigint"] is None and elapsed > r["job"].cap + KILL_MIN:
                        log(f"{r['job'].id} still running {elapsed:.0f} min after its start: SIGINT")
                        r["proc"].send_signal(signal.SIGINT)
                        r["sigint"] = now
                    elif r["sigint"] is not None and now - r["sigint"] > 60.0:
                        r["proc"].kill()
                    continue
                r["log"].close()
                finish(summarize(r["job"], code, elapsed, r["sigint"] is not None, log_root, Path(r["log"].name)))
                del running[slot]
                free.append(slot)
                free.sort()
            if pending or running:
                time.sleep(poll)
    except KeyboardInterrupt:
        log("interrupted: SIGINT to the running runs")
        for r in running.values():
            r["proc"].send_signal(signal.SIGINT)
        for r in running.values():
            try:
                code = r["proc"].wait(timeout=120)
            except subprocess.TimeoutExpired:
                r["proc"].kill()
                code = r["proc"].wait()
            r["log"].close()
            finish(summarize(r["job"], code, (time.time() - r["start"]) / 60.0, True, log_root, Path(r["log"].name)))
        raise
    return rows


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, default=360.0, help="the whole sequence; no run starts that cannot finish")
    ap.add_argument("--out", default=str(REPO / "logs" / "step08"), help="summary files and one log per run")
    ap.add_argument("--log-root", default=str(REPO / "logs"), help="train.py's --log-root (run folders)")
    ap.add_argument("--slots", type=int, default=0, help="runs at once; 0 = the physical cores, at most 4")
    ap.add_argument("--cores", type=int, default=0, help="--dry-run: plan for this many physical cores")
    ap.add_argument("--dry-run", action="store_true", help="print the time plan and the commands, run nothing")
    ap.add_argument("--only", default="", help="comma-separated run ids (e.g. base-s1,fixed-s1), in queue order")
    ap.add_argument("--prefix", default="step08", help="run names: <prefix>-<run>-s<seed>")
    ap.add_argument("--robots", type=int, default=16)
    ap.add_argument("--eval-robots", type=int, default=64, help="the final evaluation (WalkEval); 0 = none")
    ap.add_argument("--iterations", type=int, default=0, help="0 = by time only (the cap)")
    ap.add_argument("--cap-minutes", type=float, default=0.0, help="replace every cap (smoke tests)")
    ap.add_argument("--crash", default="", help="smoke test: this run id is made to crash")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-project", default=None)
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--python", default=sys.executable)
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    cores, cpus, logical = physical_cores()
    quota = cpu_quota()
    if args.cores:
        cores, cpus = args.cores, cpus[:args.cores]
    slots = choose_slots(cores, quota, args.slots)
    header = (f"Machine: {logical} logical CPUs, {cores} physical cores, cgroup quota "
              f"{'none' if quota is None else f'{quota:g} CPUs'} -> {slots} slots.  Budget {args.minutes:g} min.")
    log(header)
    jobs = build_jobs(args)
    print(plan_text(jobs, slots, args.minutes), flush=True)
    if args.dry_run:
        for j in jobs:
            print(f"{j.id}: {shlex.join(map(str, j.cmd))}")
        return
    pin = cpus if len(cpus) >= slots else None
    rows = run_sequence(jobs, slots, args.minutes, Path(args.out), Path(args.log_root), cpus=pin, header=header)
    log(f"done: {sum(r['status'] == 'ok' for r in rows)} of {len(rows)} runs ok; summary in {args.out}")


if __name__ == "__main__":
    main()

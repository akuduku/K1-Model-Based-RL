r"""Score a run of the gradient learner (standing or walking): the row for the thesis table, by the same rule every
time.  The task comes from the run's config.json (runs from before W5 have none: standing).

    python -m mbrl.grad.score grad-stand-v1c                 # a run fetched to outputs/kaggle/<name>/
    python -m mbrl.grad.score logs/grad-stand-laptop         # or the folder with metrics.jsonl and deploy_<it>.bin
    python -m mbrl.grad.score grad-stand-v1c --update 155    # another checkpoint of the same run

Standing:
1. The checkpoint that counts (docs/EXPERIMENTS.md, rule 6): the best evaluation inside the run, i.e. the fewest
   falls with pushes, and among equals the lowest power.
2. That checkpoint again, on robots the run has not seen: `--seeds` x `-B` episodes of 10 s on the C++ pH plant with
   pushes and noise (Full Body rl/stand/evaluate.py), once without disturbances, and 20 s in MuJoCo, quiet and
   pushed (rl/stand/eval_mujoco.py).
3. Prints the numbers and the row for the table in docs/EXPERIMENTS.md.

Walking (W5): the same, per command of Full Body rl/walk/evaluate.py COMMANDS (stand, forward 0.5 and 1.0 m/s, back
0.5, sideways 0.5, turn 1 rad/s).  The checkpoint that counts: the fewest falls over the six commands, and among
equals the lowest speed error.  MuJoCo: rl/walk/eval_mujoco.py, 20 s per command, quiet and pushed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from itertools import accumulate
from pathlib import Path

import torch

from mbrl import paths

REPO = Path(__file__).resolve().parents[2]


def find_run(name: str) -> Path:
    """The folder that holds metrics.jsonl: a path, or the name of a run fetched from Kaggle."""
    for c in (Path(name), REPO / "outputs" / "kaggle" / name / "runs" / name, REPO / "logs" / name):
        if (c / "metrics.jsonl").exists():
            return c
        inner = sorted(c.glob("runs/*/metrics.jsonl"))
        if inner:
            return inner[0].parent
    raise FileNotFoundError(f"no metrics.jsonl for {name!r} (tried the path, outputs/kaggle/{name}/runs/{name}, logs/{name})")


def task_of(run: Path) -> str:
    config = run / "config.json"
    return json.loads(config.read_text()).get("task", "stand") if config.exists() else "stand"


def best_checkpoint(run: Path, task: str = "stand") -> tuple[dict, list]:
    """-> (the row of the best in-run evaluation that has a deploy file, all rows with their hours added).
    Standing: fewest falls, then the lowest power.  Walking: fewest falls, then the lowest speed error (none
    measured, because every robot fell early: last)."""
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line.strip()]
    for r, seconds in zip(rows, accumulate(r.get("Cost/seconds", 0.0) for r in rows)):
        r["hours"] = seconds / 3600
    scored = [r for r in rows if "Eval/fell" in r and (run / f"deploy_{r['iteration']}.bin").exists()]
    if not scored:
        raise ValueError(f"{run}: no evaluation with a deploy_<it>.bin beside it")
    if task == "walk":
        error = lambda r: r["Eval/speed_error"] if r["Eval/speed_error"] == r["Eval/speed_error"] else float("inf")  # noqa: E731
        return min(scored, key=lambda r: (r["Eval/fell"], error(r))), rows
    return min(scored, key=lambda r: (r["Eval/fell"], r["Eval/watts"])), rows


def on_ph(policy_file: Path, robots: int, seed: int, quiet: bool) -> dict:
    from rl.kick import policy as P
    from rl.stand.env import StandConfig, StandEnv
    from rl.stand.evaluate import play
    from rl.stand.evaluate import quiet as make_quiet

    cfg = StandConfig(num_envs=robots, device="cpu", dtype="float32", seed=seed)
    env = StandEnv(make_quiet(cfg) if quiet else cfg)
    policy, _ = P.load(str(policy_file), "cpu")
    with torch.no_grad():
        return play(env, lambda o: policy(o["policy"]))


def in_mujoco(policy_file: Path, push: float, seconds: float = 20.0) -> str:
    cmd = [sys.executable, str(paths.full_body_root() / "rl" / "stand" / "eval_mujoco.py"), "--policy", str(policy_file),
           "--seconds", str(seconds), "--push", str(push)]
    out = subprocess.run(cmd, capture_output=True, text=True)
    lines = [line for line in out.stdout.splitlines() if "max tilt" in line]
    return lines[-1] if lines else f"failed: {out.stderr.strip().splitlines()[-1] if out.stderr.strip() else 'no output'}"


# ---------------------------------------------------------------------------------------------------- walking (W5)
def walk_on_ph(policy_file: Path, robots: int, seed: int, quiet: bool) -> dict:
    """-> {command: rl/walk/evaluate.py play's numbers} for its six COMMANDS, 10 s each from standing."""
    from rl.walk.env import WalkConfig, WalkEnv
    from rl.walk.env import quiet as make_quiet
    from rl.walk.evaluate import COMMANDS, load, play

    cfg = WalkConfig(num_envs=robots, device="cpu", dtype="float32", seed=seed)
    env = WalkEnv(make_quiet(cfg) if quiet else cfg)
    act = load(str(policy_file), env)
    with torch.no_grad():
        return {c: play(env, act, c) for c in COMMANDS}


def walk_in_mujoco(policy_file: Path, push: float, seconds: float = 20.0) -> list[str]:
    """rl/walk/eval_mujoco.py: one robot per command in MuJoCo through the firmware's loop -> its lines."""
    cmd = [sys.executable, str(paths.full_body_root() / "rl" / "walk" / "eval_mujoco.py"), "--policy", str(policy_file),
           "--seconds", str(seconds), "--push", str(push)]
    out = subprocess.run(cmd, capture_output=True, text=True)        # exit code 1 means: a robot fell
    lines = [line.strip() for line in out.stdout.splitlines() if line.strip().startswith("command")]
    return lines or [f"failed: {out.stderr.strip().splitlines()[-1] if out.stderr.strip() else 'no output'}"]


def nanmean(xs) -> float:
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def score_walk(run: Path, chosen: dict, best: dict, rows: list, args) -> None:
    from rl.walk.evaluate import COMMANDS, command_name

    it = chosen["iteration"]
    policy_file = run / f"deploy_{it}.bin"
    print(f"run {run}")
    print("in-run evaluations (update: falls over the six commands, speed error, power):  "
          + "  ".join(f"{r['iteration']}: {100 * r['Eval/fell']:.1f} %, {r['Eval/speed_error']:.2f} m/s, "
                      f"{r['Eval/watts']:.0f} W" for r in rows if "Eval/fell" in r))
    print(f"best in-run evaluation: update {best['iteration']} ({100 * best['Eval/fell']:.1f} % falls, "
          f"{best['Eval/speed_error']:.2f} m/s speed error).  Scoring update {it}: "
          f"{chosen['Cost/executed_steps'] / 1e6:.2f} M environment steps, {chosen['hours']:.1f} h"
          + (f", requests x {chosen['Curriculum/command_scale']:g}" if "Curriculum/command_scale" in chosen else "")
          + ".", flush=True)
    per = {c: [] for c in COMMANDS}
    for seed in args.seeds:
        for c, r in walk_on_ph(policy_file, args.robots, seed, quiet=False).items():
            per[c].append(r)
        print(f"  pH plant, pushes and noise, seed {seed}: done", flush=True)
    print(f"pH plant, pushes and noise ({len(args.seeds)} seed{'s' if len(args.seeds) > 1 else ''} x {args.robots} "
          f"robots, 10 s per command):")
    print(f"  {'command':>8s} {'fell':>12s} {'speed error':>12s} {'yaw error':>10s} {'speed':>7s} {'power':>8s} {'J per m':>8s}")
    table, fell_all, n_all = {}, 0, 0
    for c in COMMANDS:
        fell = sum(round(r["fell"] * r["robots"]) for r in per[c])
        n = sum(r["robots"] for r in per[c])
        m = {k: nanmean([r[k] for r in per[c]]) for k in ("speed_error", "yaw_rate_error", "speed", "watts", "joule_per_m")}
        table[c] = (fell, n, m)
        fell_all, n_all = fell_all + fell, n_all + n
        print(f"  {command_name(c):>8s} {fell:4d} of {n:4d} {m['speed_error']:8.3f} m/s {m['yaw_rate_error']:6.3f}/s "
              f"{m['speed']:6.2f} {m['watts']:6.1f} W {m['joule_per_m']:8.1f}", flush=True)
    q = walk_on_ph(policy_file, args.robots, args.seeds[0], quiet=True)
    print(f"  pH plant, quiet (seed {args.seeds[0]}): "
          + ", ".join(f"{command_name(c)} {100 * r['fell']:.0f} % fell, {r['speed_error']:.2f} m/s" for c, r in q.items()))
    if not args.no_mujoco:
        for push in (0.0, 0.3):
            print(f"  MuJoCo, 20 s per command, push {push:g} m/s:")
            for line in walk_in_mujoco(policy_file, push):
                print(f"    {line}", flush=True)
    fwd = table[(0.5, 0.0, 0.0)][2]
    print("\nrow for docs/EXPERIMENTS.md (falls over all six commands; speed error, power and energy per metre at "
          "0.5 m/s forward):")
    print(f"| {run.name}, update {it} ({chosen['Cost/executed_steps'] / 1e6:.2f} M environment steps, "
          f"{chosen['hours']:.1f} h; scored with seeds {', '.join(map(str, args.seeds))} x {args.robots} robots x "
          f"6 commands) | {100 * fell_all / n_all:.1f} % | {fwd['speed_error']:.2f} m/s | {fwd['watts']:.0f} W | "
          f"{fwd['joule_per_m']:.0f} J/m |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="run name (outputs/kaggle/<name>) or the folder with metrics.jsonl")
    ap.add_argument("--update", type=int, default=None, help="score this checkpoint instead of the best in-run one")
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 8, 9])
    ap.add_argument("-B", "--robots", type=int, default=128)
    ap.add_argument("--no-mujoco", action="store_true")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    run = find_run(args.run)
    task = task_of(run)
    best, rows = best_checkpoint(run, task)
    if args.update is not None:
        chosen = next((r for r in rows if r["iteration"] == args.update), None)
        if chosen is None or not (run / f"deploy_{args.update}.bin").exists():
            raise SystemExit(f"{run} has no deploy_{args.update}.bin")
    else:
        chosen = best
    if task == "walk":
        return score_walk(run, chosen, best, rows, args)
    it = chosen["iteration"]
    policy_file = run / f"deploy_{it}.bin"
    print(f"run {run}")
    print("in-run evaluations (update: falls with pushes, power):  "
          + "  ".join(f"{r['iteration']}: {100 * r['Eval/fell']:.1f} %, {r['Eval/watts']:.2f} W" for r in rows if "Eval/fell" in r))
    print(f"best in-run evaluation: update {best['iteration']} ({100 * best['Eval/fell']:.1f} % falls, "
          f"{best['Eval/watts']:.2f} W).  Scoring update {it}: {chosen['Cost/executed_steps'] / 1e6:.2f} M environment "
          f"steps, {chosen['hours']:.1f} h.", flush=True)

    fell = episodes = 0
    watts, tilt, stood = [], [], []
    for seed in args.seeds:
        r = on_ph(policy_file, args.robots, seed, quiet=False)
        n = round(r["fell"] * r["robots"])
        fell, episodes = fell + n, episodes + r["robots"]
        watts.append(r["watts"]), tilt.append(r["tilt_deg"]), stood.append(r["stood_s"])
        print(f"  pH plant, pushes and noise, seed {seed}: {n} of {r['robots']} fell ({100 * r['fell']:.1f} %), "
              f"{r['watts']:.2f} W, tilt {r['tilt_deg']:.2f} deg", flush=True)
    q = on_ph(policy_file, args.robots, args.seeds[0], quiet=True)
    print(f"  pH plant, quiet: {100 * q['fell']:.1f} % fell, {q['watts']:.2f} W, tilt {q['tilt_deg']:.2f} deg", flush=True)
    mean = lambda x: sum(x) / len(x)  # noqa: E731
    print(f"with pushes and noise: {fell} of {episodes} fell ({100 * fell / episodes:.1f} %), stood {mean(stood):.2f} s, "
          f"{mean(watts):.1f} W, tilt {mean(tilt):.2f} deg")
    if not args.no_mujoco:
        for push in (0.0, 0.3):
            print(f"  MuJoCo, 20 s, push {push:g} m/s: {in_mujoco(policy_file, push)}", flush=True)
    print("\nrow for docs/EXPERIMENTS.md:")
    print(f"| {run.name}, update {it} ({chosen['Cost/executed_steps'] / 1e6:.2f} M environment steps, {chosen['hours']:.1f} h; "
          f"scored with seeds {', '.join(map(str, args.seeds))} = {episodes} episodes) | {100 * fell / episodes:.1f} % | "
          f"{mean(stood):.2f} s | {mean(watts):.1f} W | {mean(tilt):.2f}° |")
    print(f"quiet: {100 * q['fell']:.1f} % falls, {q['watts']:.1f} W, tilt {q['tilt_deg']:.2f}°")


if __name__ == "__main__":
    main()

r"""Score a standing run of the gradient learner: the row for the thesis table, by the same rule every time.

    python -m mbrl.grad.score grad-stand-v1c                 # a run fetched to outputs/kaggle/<name>/
    python -m mbrl.grad.score logs/grad-stand-laptop         # or the folder with metrics.jsonl and deploy_<it>.bin
    python -m mbrl.grad.score grad-stand-v1c --update 155    # another checkpoint of the same run

1. The checkpoint that counts (docs/EXPERIMENTS.md, rule 6): the best evaluation inside the run, i.e. the fewest
   falls with pushes, and among equals the lowest power.
2. That checkpoint again, on robots the run has not seen: `--seeds` x `-B` episodes of 10 s on the C++ pH plant with
   pushes and noise (Full Body rl/stand/evaluate.py), once without disturbances, and 20 s in MuJoCo, quiet and
   pushed (rl/stand/eval_mujoco.py).
3. Prints the numbers and the row for the table in docs/EXPERIMENTS.md.
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


def best_checkpoint(run: Path) -> tuple[dict, list]:
    """-> (the row of the best in-run evaluation that has a deploy file, all rows with their hours added)."""
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line.strip()]
    for r, seconds in zip(rows, accumulate(r.get("Cost/seconds", 0.0) for r in rows)):
        r["hours"] = seconds / 3600
    scored = [r for r in rows if "Eval/fell" in r and (run / f"deploy_{r['iteration']}.bin").exists()]
    if not scored:
        raise ValueError(f"{run}: no evaluation with a deploy_<it>.bin beside it")
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
    best, rows = best_checkpoint(run)
    if args.update is not None:
        chosen = next((r for r in rows if r["iteration"] == args.update), None)
        if chosen is None or not (run / f"deploy_{args.update}.bin").exists():
            raise SystemExit(f"{run} has no deploy_{args.update}.bin")
    else:
        chosen = best
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

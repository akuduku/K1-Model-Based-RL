r"""W5b: does air_time's smooth stand-in show the learner a step?  Two checks on the laptop, no training.

    python -m mbrl.grad.probe_step gait                        # on a real gait: the stand-in pays where Eric's count pays
    python -m mbrl.grad.probe_step lift --weights 0 0.1 0.5 1  # a standing robot's next actions, optimised directly
                                                               # through the pH model: does a foot come up?

gait (plausibility): Eric's walk_v9, used here only as a check (it is part of no training), walks --warmup s on the
differentiable plant with a fixed request; then --seconds s are recorded per foot and control step: the ground force
over half the weight, Eric's air_time count (air clock between 0.05 and 0.5 s, request above 0.2), the stand-in
(Full Body rl/walk/task.py STAND_INS: unload x [nearly without weight for under 0.5 s] x [request above 0.2]) and the
contact.
-> agreement numbers and a figure.

lift (the learner's question without the network): robots start in the sprint's home pose with a request of --vx m/s
forward; after --settle s of zero actions (the request has ramped up) their next --steps actions are free numbers,
optimised with Adam on the learner's own window (mbrl/grad/train.py `window`, one control step of memory): the
discounted differentiated reward of the robots still up, a fall charged at the fall, and term 2 at the end (the
state costs x 1 s, the fixed term 2).
Once per weight of the stand-in, from the same start and the same first guess.  -> per weight: how much weight came
off a foot, how long feet were in the air, how far the trunk went, falls, and the moves the stand-in must not pay for:
both feet up, a dropping trunk (crouch), tilt, a foot held up past 0.5 s.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from mbrl import paths
from mbrl.grad.env import POLICY_DT
from mbrl.grad.walk_env import NA, GradWalk


def per_foot(env: GradWalk) -> dict:
    """This control step, per foot (B,2): unload (0 = carries half the robot or more, 1 = none of it), Eric's count,
    the stand-in (both unweighted, after the request gate), the contact."""
    task, s, mem = env.task, env.sig, env.mem
    target = mem["target"]
    gate = ((torch.linalg.vector_norm(target[:, :2], dim=1) + target[:, 2].abs()) > 0.2).to(env.dtype)[:, None]
    clock, unloaded = mem["air_time"], mem["unloaded_time"]   # the foot clocks after this step, as the task used them
    low, high = task.p.air_time_range
    unload = (1.0 - s.contact_force[..., 2] / task.half_weight).clamp(0.0, 1.0)
    return {"unload": unload, "eric": ((clock > low) & (clock < high)).to(env.dtype) * gate,
            "stand": unload * (unloaded < high).to(env.dtype) * gate, "contact": s.contact, "clock": unloaded}


# ------------------------------------------------------------------------------------------------------- gait
def gait(args) -> dict:
    from rl.walk.backends import resolve_policy

    vx = 0.5 if args.vx is None else args.vx
    env = GradWalk(args.robots, seed=args.seed, command=(vx, 0.0, 0.0), joint_noise=args.joint_noise,
                   air_time_smooth=0.1)
    net = torch.jit.load(str(resolve_policy(args.policy)), map_location="cpu").eval()
    act = lambda: net(env.observe(noise=False).float()).to(env.dtype)  # noqa: E731
    keys = ("force", "eric", "stand", "contact", "height")
    rec = {k: [] for k in keys}
    worst = 0.0
    alive = torch.ones(env.B, dtype=torch.bool)
    with torch.no_grad():
        for _ in range(round(args.warmup / POLICY_DT)):
            env.between_steps()
            env.step(act())
        x0 = env.plant.q[:, 0].clone()
        n = round(args.seconds / POLICY_DT)
        for _ in range(n):
            env.between_steps()
            _, log = env.step(act())
            f = per_foot(env)
            rec["force"].append(env.sig.contact_force[..., 2] / env.task.half_weight)
            rec["eric"].append(f["eric"])
            rec["stand"].append(f["stand"])
            rec["contact"].append(f["contact"].to(env.dtype))
            rec["height"].append(env.sig.foot_height)
            w = 0.1 * POLICY_DT                               # the task's terms are weight x policy step
            worst = max(worst, float((f["eric"].sum(1) * w - log["terms"]["air_time"]).abs().max()),
                        float((f["stand"].sum(1) * w - log["terms"]["air_time_smooth"]).abs().max()))
            alive &= ~env.fallen()
        speed = float(((env.plant.q[:, 0] - x0) / (n * POLICY_DT))[alive].mean()) if bool(alive.any()) else float("nan")
    R = {k: torch.stack(v)[:, alive] for k, v in rec.items()}          # (T, robots up, 2)
    eric, stand, contact = R["eric"], R["stand"], R["contact"]
    in_air = 1.0 - contact
    paid = eric > 0.5
    total = max(float(stand.sum()), 1e-12)
    out = {
        "policy": args.policy, "request_mps": vx, "robots": int(alive.sum()), "fell": int((~alive).sum()),
        "speed_mps": speed, "feet_in_air": float(in_air.mean()),
        "eric_pays_share": float(eric.mean()), "stand_in_mean": float(stand.mean()),
        "stand_in_where_eric_pays": float(stand[paid].mean()) if bool(paid.any()) else float("nan"),
        "stand_in_on_the_ground": float(stand[contact > 0.5].mean()),
        "share_of_stand_in_where_eric_pays": float(stand[paid].sum()) / total,
        "share_of_stand_in_in_the_air_unpaid": float(stand[(in_air > 0.5) & ~paid].sum()) / total,
        "share_of_stand_in_on_the_ground": float(stand[contact > 0.5].sum()) / total,
        "correlation": float(torch.corrcoef(torch.stack((eric.flatten(), stand.flatten())))[0, 1]),
        "worst_difference_to_the_task_terms": worst,
    }
    print(f"[gait] {args.policy} at {vx:g} m/s request: walks {speed:.2f} m/s, {out['fell']} of {args.robots} fell, "
          f"feet in the air {out['feet_in_air']:.0%} of the time")
    print(f"[gait] Eric's count pays {out['eric_pays_share']:.0%} of foot-steps; the stand-in averages "
          f"{out['stand_in_mean']:.2f}: {out['stand_in_where_eric_pays']:.2f} where Eric's count pays, "
          f"{out['stand_in_on_the_ground']:.2f} on the ground; correlation {out['correlation']:.2f}")
    print(f"[gait] the stand-in's total: {out['share_of_stand_in_where_eric_pays']:.0%} where Eric's count pays, "
          f"{out['share_of_stand_in_in_the_air_unpaid']:.0%} in the air before 0.05 s, "
          f"{out['share_of_stand_in_on_the_ground']:.0%} on the ground (weight coming off before the lift)")
    print(f"[gait] the per-foot numbers equal the task's terms to {worst:.1e}")
    out["figure"] = str(gait_figure(R, args, out))
    return out


def gait_figure(R: dict, args, out: dict) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = min(R["force"].shape[0], round(3.0 / POLICY_DT))
    t = [i * POLICY_DT for i in range(T)]
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    for i, (ax, name) in enumerate(zip(axes, ("left foot", "right foot"))):
        c = R["contact"][:T, 0, i].tolist()
        for j in range(T):                                   # the foot on the ground: grey background
            if c[j] > 0.5:
                ax.axvspan(t[j] - POLICY_DT / 2, t[j] + POLICY_DT / 2, color="0.9", lw=0)
        ax.plot(t, R["force"][:T, 0, i].tolist(), color="tab:blue", lw=1.2, label="ground force / half the weight")
        ax.step(t, R["eric"][:T, 0, i].tolist(), where="mid", color="tab:orange", lw=1.5, label="Eric's air_time count")
        ax.plot(t, R["stand"][:T, 0, i].tolist(), color="tab:green", lw=1.5, ls="--", label="stand-in (W5b)")
        ax.set_ylabel(name)
        ax.set_ylim(-0.05, 2.2)
    axes[0].legend(loc="upper right", fontsize=8, ncol=3)
    axes[-1].set_xlabel("time [s] (grey: the foot touches the ground)")
    fig.suptitle(f"air_time and its smooth stand-in on {args.policy}'s gait, pH plant, {out['request_mps']:g} m/s "
                 f"(correlation {out['correlation']:.2f})", fontsize=10)
    fig.tight_layout()
    path = Path(args.out) / f"air_time_on_{args.policy}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ------------------------------------------------------------------------------------------------------- lift
class OpenLoop:
    """train.window's "policy" for fixed actions (K,B,NA): the forward pass asks for steps 0..K-1, the backward pass
    for K-1..0 again.  Zero times the observation keeps the action on the graph's way to the state, so window's hook
    sees d(objective)/d(action) as it does for a network: the probe follows the learner's own derivative."""

    def __init__(self, actions: torch.Tensor):
        self.actions, self.calls = actions, 0

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        K = self.actions.shape[0]
        t = self.calls if self.calls < K else 2 * K - 1 - self.calls
        self.calls += 1
        return self.actions[t].to(obs.dtype) + 0.0 * obs[:, :1]


def measure(env: GradWalk, start: dict, actions: torch.Tensor, args):
    """Open-loop actions (K,B,NA) from `start`, no derivative: what the feet and the trunk did.  -> (numbers, loads)."""
    env.set_state({k: v.clone() for k, v in start.items()})
    B, K = env.B, actions.shape[0]
    alive = torch.ones(B, dtype=torch.bool)
    z0, x0 = env.plant.q[:, 2].clone(), env.plant.q[:, 0].clone()
    unload_max = torch.zeros(B, dtype=env.dtype)
    stand_sum, air_steps, both_up, steps_up = (torch.zeros(B, dtype=env.dtype) for _ in range(4))
    tilt_max, clock_max = torch.zeros(B, dtype=env.dtype), torch.zeros(B, dtype=env.dtype)
    loads = []
    with torch.no_grad():
        for k in range(K):
            env.between_steps()
            env.step(actions[k])
            live = alive.to(env.dtype)
            f = per_foot(env)
            unload_max = torch.where(alive, torch.maximum(unload_max, f["unload"].max(1).values), unload_max)
            stand_sum += f["stand"].sum(1) * live
            off = (~f["contact"]) & alive[:, None]
            air_steps += off.to(env.dtype).sum(1)
            both_up += off.all(1).to(env.dtype)
            steps_up += live
            tilt_max = torch.where(alive, torch.maximum(tilt_max, env.sig.roll_pitch.abs().max(1).values), tilt_max)
            clock_max = torch.where(alive, torch.maximum(clock_max, f["clock"].max(1).values), clock_max)
            loads.append((env.sig.contact_force[..., 2] / env.task.half_weight)[0].tolist())
            alive = alive & ~env.fallen()
        q = env.plant.q
    n = steps_up.clamp_min(1.0)
    up = bool(alive.any())
    numbers = {"unload_max": float(unload_max.mean()), "stand_in_per_step": float((stand_sum / n).mean()),
               "feet_in_air": float((air_steps / (2 * n)).mean()), "both_feet_up": float((both_up / n).mean()),
               "distance_m": float((q[:, 0] - x0)[alive].mean()) if up else float("nan"),
               "trunk_drop_m": float((z0 - q[:, 2])[alive].mean()) if up else float("nan"),
               "tilt_deg": math.degrees(float(tilt_max.mean())), "longest_unloaded_s": float(clock_max.max()),
               "fell": int((~alive).sum())}
    return numbers, loads


def lift(args) -> dict:
    from types import SimpleNamespace

    from mbrl.grad.train import window

    vx = 0.3 if args.vx is None else args.vx
    wargs = SimpleNamespace(steps=args.steps, clip=1e9, action_noise=0.0, gamma=args.gamma,
                            terminal_seconds=args.terminal_seconds, terminal_kinetic=0.0, fall_weight=1.0)
    results = {}
    for w in args.weights:
        env = GradWalk(args.robots, seed=args.seed, command=(vx, 0.0, 0.0), joint_noise=args.joint_noise,
                       air_time_smooth=w)
        with torch.no_grad():
            for _ in range(round(args.settle / POLICY_DT)):
                env.between_steps()
                env.step(torch.zeros(env.B, NA, dtype=env.dtype))
        start = {k: v.detach().clone() for k, v in env.get_state().items()}
        gen = torch.Generator().manual_seed(args.seed)
        actions = (args.init_noise * torch.randn(args.steps, env.B, NA, generator=gen, dtype=env.dtype)).requires_grad_(True)
        opt = torch.optim.Adam([actions], lr=args.lr)
        t0, first, numbers, loads, objective = time.time(), None, None, None, float("nan")
        for it in range(args.iterations + 1):
            if it % args.print_every == 0 or it == args.iterations:
                numbers, loads = measure(env, start, actions.detach(), args)
                numbers["objective"] = objective                  # of the actions before the last Adam step
                first = first or numbers
                print(f"[lift] weight {w:g}  it {it:3d}  objective {objective:+.4f}  weight off a foot "
                      f"{numbers['unload_max']:.2f}  feet in the air {numbers['feet_in_air']:.0%}  both up "
                      f"{numbers['both_feet_up']:.0%}  trunk +{numbers['distance_m']:.3f} m  drop {numbers['trunk_drop_m']:+.3f} m  "
                      f"tilt {numbers['tilt_deg']:.1f} deg  fell {numbers['fell']}  ({time.time() - t0:.0f} s)", flush=True)
            if it < args.iterations:                         # the learner's window: objective and its derivative
                env.set_state({k: v.clone() for k, v in start.items()})
                out = window(env, OpenLoop(actions.detach()), wargs)
                objective = out["stats"]["objective"]
                opt.zero_grad()
                actions.grad = -out["g"].to(actions.dtype)   # Adam goes down: minus d(objective)/d(action)
                opt.step()
        results[f"{w:g}"] = {"first": first, "last": numbers, "loads_robot0": loads, "seconds": time.time() - t0}
    print("\n[lift] after the optimisation, per weight of the stand-in (0 = Eric's terms only, as in W5):")
    print("  weight  weight off a foot  feet in air  both up  trunk moved  trunk drop  tilt  longest unloaded  fell")
    for w, r in results.items():
        n = r["last"]
        print(f"  {w:>6}  {n['unload_max']:17.2f}  {n['feet_in_air']:11.0%}  {n['both_feet_up']:7.0%}  "
              f"{n['distance_m']:+9.3f} m  {n['trunk_drop_m']:+8.3f} m  {n['tilt_deg']:4.1f}  "
              f"{n['longest_unloaded_s']:14.2f} s  {n['fell']:4d}")
    lift_figure(results, args, vx)
    return {"request_mps": vx, "steps": args.steps, "iterations": args.iterations, "robots": args.robots,
            "results": {w: {k: v for k, v in r.items() if k != "loads_robot0"} for w, r in results.items()}}


def lift_figure(results: dict, args, vx: float) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(results), 1, figsize=(9, 2.2 * len(results)), sharex=True, squeeze=False)
    for ax, (w, r) in zip(axes[:, 0], results.items()):
        loads = torch.tensor(r["loads_robot0"])
        t = [(k + 1) * POLICY_DT for k in range(loads.shape[0])]
        ax.plot(t, loads[:, 0].tolist(), label="left foot", color="tab:blue")
        ax.plot(t, loads[:, 1].tolist(), label="right foot", color="tab:red")
        ax.axhline(0.0, color="0.6", lw=0.8)
        ax.set_ylabel(f"weight {w}")
        ax.set_ylim(-0.05, 2.2)
    axes[0, 0].legend(loc="upper right", fontsize=8)
    axes[-1, 0].set_xlabel("time in the window [s]   (ground force / half the robot's weight; 0 = the foot is up)")
    fig.suptitle(f"Optimised actions of a standing robot asked for {vx:g} m/s: the load on each foot", fontsize=10)
    fig.tight_layout()
    path = Path(args.out) / "lift_loads.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[lift] figure: {path}")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("gait", "lift"))
    ap.add_argument("--vx", type=float, default=None, help="the request forward [m/s] (gait 0.5, lift 0.3)")
    ap.add_argument("--robots", type=int, default=4)
    ap.add_argument("--joint-noise", type=float, default=0.02, help="start-pose noise, so the robots differ")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default=str(paths.REPO / "outputs" / "steps" / "W5b"))
    ap.add_argument("--policy", default="walk_v9", help="gait: the walk that makes the robot step (a check only)")
    ap.add_argument("--warmup", type=float, default=2.0, help="gait: seconds of walking before the record")
    ap.add_argument("--seconds", type=float, default=5.0, help="gait: seconds recorded")
    ap.add_argument("--weights", type=float, nargs="+", default=[0.0, 0.1], help="lift: weights of the stand-in")
    ap.add_argument("--steps", type=int, default=40, help="lift: control steps optimised (40 = 0.8 s)")
    ap.add_argument("--settle", type=float, default=0.2, help="lift: seconds of zero actions first")
    ap.add_argument("--iterations", type=int, default=30, help="lift: Adam steps")
    ap.add_argument("--lr", type=float, default=0.05, help="lift: Adam step (action units)")
    ap.add_argument("--init-noise", type=float, default=0.01, help="lift: the first guess, around zero actions")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--terminal-seconds", type=float, default=1.0)
    ap.add_argument("--print-every", type=int, default=5)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    out = gait(args) if args.mode == "gait" else lift(args)
    path = Path(args.out) / f"{args.mode}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(f"[{args.mode}] numbers: {path}")


if __name__ == "__main__":
    main()

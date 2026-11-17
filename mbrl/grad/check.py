r"""Gradients through the pH model: do they exist, are they right, how far do they stay usable?

The Python plant (Full Body ph_k1/plant.py) is written in PyTorch, so a rollout can be differentiated with autograd.
This script rolls a standing robot K control steps (20 plant steps of 1 ms each, PD targets -> motor driver -> volts),
takes d cost / d targets with autograd and compares it with central finite differences.

    python -m mbrl.grad.check --steps 1 5 20
    python -m mbrl.grad.check --steps 5 --push 0.3        # a push at the start: the contact mode changes
    python -m mbrl.grad.check --steps 5 20 --drop 0.02    # the feet land inside the window
    python -m mbrl.grad.check --task walk --steps 5 20 50 # while the robot walks: repeated steps, sliding feet

cost = copper energy [J] + tilt_weight x sum of (tilt^2) x dt    (smooth terms of the stand reward)

--task walk: the walk task on the differentiable plant (mbrl/grad/walk_env.py).  A released walk (walk_v9) walks
for --warmup seconds; from there its next K actions are replayed open loop and the cost is minus the sum of the
walk rewards that have a derivative.  d cost / d actions by autograd against central finite differences.  The
walk policy only makes the robot step: it is not part of any training.
"""

from __future__ import annotations

import argparse
import time

import torch

from mbrl import paths  # noqa: F401  (puts Full Body on sys.path)
from ph_k1 import firmware as fwm
from ph_k1.model import load_k1
from ph_k1.plant import K1PHPlant, NJ

STATE = ("q", "p", "xi", "lam")
SUBSTEPS = 20        # 50 Hz control on a 1 ms plant
POLICY_DT = 0.02


def get_state(plant) -> dict:
    return {k: getattr(plant, k).detach().clone() for k in STATE}


def set_state(plant, state: dict) -> None:
    for k in STATE:
        setattr(plant, k, state[k].clone())


def rollout(plant, state, targets, kp, kd, tilt_weight: float = 10.0, substeps: int = SUBSTEPS):
    """targets (K, B, 22) absolute joint targets.  -> (cost (B,), dict of parts).  Differentiable in targets."""
    set_state(plant, state)
    copper = torch.zeros(plant.B, dtype=plant.dtype, device=plant.device)
    tilt = torch.zeros_like(copper)
    for k in range(targets.shape[0]):
        for _ in range(substeps):
            volts = fwm.motor_volts(plant, targets[k], kp, kd)
            plant.step(volts)
            i = plant.current
            copper = copper + plant.h * (plant.motors.R * i * i).sum(-1)
        qx, qy = plant.q[:, 4], plant.q[:, 5]
        tilt = tilt + POLICY_DT * (qx * qx + qy * qy)
    return copper + tilt_weight * tilt, {"copper_J": copper, "tilt": tilt}


def settle(plant, home, kp, kd, seconds: float = 0.5):
    with torch.no_grad():
        for _ in range(int(seconds / plant.h)):
            plant.step(fwm.motor_volts(plant, home, kp, kd))
    return get_state(plant)


def walk_check(args) -> None:
    from mbrl.grad.walk_env import GradWalk
    from rl.walk.backends import resolve_policy

    env = GradWalk(1, seed=args.seed, command=tuple(args.command))
    net = torch.jit.load(str(resolve_policy(args.policy)), map_location="cpu").eval()
    act = lambda: net(env.observe(noise=False).float()).to(torch.float64)  # noqa: E731
    with torch.no_grad():
        for _ in range(round(args.warmup / POLICY_DT)):
            env.step(act())
        start = {k: v.clone() for k, v in env.get_state().items()}
        recorded, contacts, speed = [], [], []
        for _ in range(max(args.steps)):          # the walk's own next actions and what the feet do meanwhile
            a = act()
            recorded.append(a)
            env.step(a)
            contacts.append(env.sig.contact[0].tolist())
            speed.append(float(env.sig.base_lin_vel[0, 0]))
    print(f"{args.policy} walks at {sum(speed) / len(speed):.2f} m/s on command {tuple(args.command)} after {args.warmup:g} s; "
          f"trunk z {float(start['q'][0, 2]):.3f} m")

    def rollout(actions):
        """actions (K,1,22), open loop from the start state.  -> cost = minus the sum of the differentiable rewards."""
        env.set_state({k: v.clone() for k, v in start.items()})
        total = 0.0
        for a in actions:
            reward, _ = env.step(a)
            total = total + reward
        return -total.sum()

    for K in args.steps:
        changes = sum(a != b for i in range(1, K) for a, b in zip(contacts[i - 1], contacts[i]))
        actions = torch.stack(recorded[:K]).clone().requires_grad_(True)
        t0 = time.time()
        cost = rollout(actions)
        t_fwd = time.time() - t0
        t0 = time.time()
        (grad,) = torch.autograd.grad(cost, actions)
        t_bwd = time.time() - t0
        per_step = grad.flatten(1).norm(dim=1)
        print(f"\nK = {K:3d} control steps ({K * SUBSTEPS} plant steps), {changes} foot landings and lift-offs inside: "
              f"cost {float(cost.detach()):.5f}, forward {t_fwd:.1f} s, backward {t_bwd:.1f} s")
        print(f"  gradient: finite {bool(torch.isfinite(grad).all())}, norm {float(grad.norm()):.4g}, largest "
              f"{float(grad.abs().max()):.4g}; norm by control step: first {float(per_step[0]):.3g}, largest "
              f"{float(per_step.max()):.3g} (step {int(per_step.argmax())}), last {float(per_step[-1]):.3g}")
        flat = grad.flatten()
        half = args.entries // 2
        idx = torch.cat((flat.abs().topk(half).indices, torch.randint(0, flat.numel(), (args.entries - half,))))
        worst = 0.0
        with torch.no_grad():
            for j in idx.tolist():
                d = torch.zeros_like(flat)
                d[j] = args.eps
                d = d.reshape(actions.shape)
                fd = float((rollout(actions.detach() + d) - rollout(actions.detach() - d)) / (2 * args.eps))
                ag = float(flat[j])
                rel = abs(fd - ag) / max(abs(fd), abs(ag), 1e-12)
                worst = max(worst, rel)
                print(f"    step {j // NJ:2d} joint {j % NJ:2d}: autograd {ag:+.6e}  finite diff {fd:+.6e}  rel. error {rel:.2e}")
        print(f"  worst relative error {worst:.2e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=("stand", "walk"), default="stand")
    ap.add_argument("--policy", default="walk_v9", help="--task walk: the walk that makes the robot step")
    ap.add_argument("--command", type=float, nargs=3, default=(0.5, 0.0, 0.0), metavar=("VX", "VY", "WZ"))
    ap.add_argument("--warmup", type=float, default=2.0, help="--task walk: seconds of walking before the window")
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 5, 20], help="control steps K to test")
    ap.add_argument("--push", type=float, default=0.0, help="sideways trunk speed added before the rollout [m/s]")
    ap.add_argument("--drop", type=float, default=0.0,
                    help="start this many metres above the floor: the feet land inside the window (contact switches on)")
    ap.add_argument("--entries", type=int, default=6, help="how many gradient entries to check by finite differences")
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    if args.task == "walk":
        return walk_check(args)

    model = load_k1(dtype=torch.float64, device="cpu")
    plant = K1PHPlant(model=model, batch=1)
    kp, kd = fwm.gains(model)
    home = plant.theta.clone()
    state = settle(plant, home, kp, kd)
    if args.push:
        kin = plant.kinematics()
        nu = plant.velocity(kin)
        nu[:, 4] += args.push                                    # body-frame sideways speed of the trunk
        state["p"] = (kin.M @ nu[..., None]).squeeze(-1).detach()
    if args.drop:
        state["q"][:, 2] += args.drop      # lifted, at rest, no contact: lands after sqrt(2 drop / g)
        state["p"].zero_()
        state["xi"].zero_()
    print(f"settled: trunk z {float(state['q'][0, 2]):.4f} m, |p| {float(state['p'].norm()):.2e}, push {args.push} m/s, drop {args.drop} m "
          f"(lands after about {(2 * args.drop / 9.81) ** 0.5 * 1000:.0f} ms)" if args.drop else
          f"settled: trunk z {float(state['q'][0, 2]):.4f} m, |p| {float(state['p'].norm()):.2e}, push {args.push} m/s")

    for K in args.steps:
        noise = 0.02 * torch.randn(K, 1, NJ, dtype=torch.float64)
        targets = (home.expand(K, 1, NJ) + noise).clone().requires_grad_(True)
        t0 = time.time()
        cost, parts = rollout(plant, state, targets, kp, kd)
        t_fwd = time.time() - t0
        t0 = time.time()
        (grad,) = torch.autograd.grad(cost.sum(), targets)
        t_bwd = time.time() - t0
        finite = bool(torch.isfinite(grad).all())
        print(f"\nK = {K:3d} control steps ({K * SUBSTEPS} plant steps): cost {float(cost):.5f} "
              f"(copper {float(parts['copper_J']):.4f} J), forward {t_fwd:.1f} s, backward {t_bwd:.1f} s")
        print(f"  gradient: finite {finite}, norm {float(grad.norm()):.4g}, largest {float(grad.abs().max()):.4g}; "
              f"norm by control step: first {float(grad[0].norm()):.3g}, last {float(grad[-1].norm()):.3g}")
        # finite differences on the largest entries and on random ones
        flat = grad.flatten()
        half = args.entries // 2
        idx = torch.cat((flat.abs().topk(half).indices, torch.randint(0, flat.numel(), (args.entries - half,))))
        worst = 0.0
        with torch.no_grad():
            for j in idx.tolist():
                d = torch.zeros_like(flat)
                d[j] = args.eps
                d = d.reshape(targets.shape)
                cp, _ = rollout(plant, state, targets.detach() + d, kp, kd)
                cm, _ = rollout(plant, state, targets.detach() - d, kp, kd)
                fd = float((cp - cm).sum() / (2 * args.eps))
                ag = float(flat[j])
                rel = abs(fd - ag) / max(abs(fd), abs(ag), 1e-12)
                worst = max(worst, rel)
                k, joint = j // NJ, j % NJ
                print(f"    step {k:2d} joint {joint:2d}: autograd {ag:+.6e}  finite diff {fd:+.6e}  rel. error {rel:.2e}")
        print(f"  worst relative error {worst:.2e}")


if __name__ == "__main__":
    main()

r"""Where does a learned term 2's derivative go?  The two measurements of W3 attempt 2 (docs/EXPERIMENTS.md).

From a checkpoint of a run with --term2 value or weights (policy and learned term 2), a few standing windows:

1. The action gradient of each control step (median over robots), with the fixed term 2, with the learned one,
   and with the learned one when its input does not contain the last action; and the direction of the action
   gradient against the fixed term 2's (cosine).
2. At the end states: the slope of the learned V in the pH state (q, p) against the slope of minus the fixed
   term 2 (cosine per robot), and their sizes.

    python -m mbrl.grad.term2_diag logs/grad-stand-value-net-laptop/model_24.pt
    python -m mbrl.grad.term2_diag logs/grad-stand-weights-laptop/model_<it>.pt
"""

from __future__ import annotations

import argparse

import torch

from mbrl.grad.env import GradStand
from mbrl.grad.train import OBS_NOISE, window
from mbrl.grad.value import ValueLearner, WeightLearner
from mbrl.mpc.nets import MLP
from rl.stand import contract as C


def load(path: str):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    policy = MLP(C.OBS_DIM, 22).double()
    policy.load_state_dict(ck["actor_state_dict"])
    policy.double()
    if "value_learner" not in ck:
        raise SystemExit(f"{path} has no learned term 2 (a run with --term2 fixed)")
    state = ck["value_learner"]
    if "s_target" in state:
        learned = WeightLearner(state["names"], state["s_target"].tolist())
    else:
        hidden = tuple(state["net"][k].shape[0] for k in sorted(state["net"]) if k.endswith(".weight"))[:-1]
        learned = ValueLearner(C.OBS_DIM + 6, hidden=hidden)
    learned.load_state_dict(state)
    return policy, learned


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--robots", type=int, default=16)
    ap.add_argument("--windows", type=int, default=3)
    ap.add_argument("--seed", type=int, default=9)
    args = ap.parse_args()
    torch.set_num_threads(1)
    policy, learned = load(args.checkpoint)
    wargs = argparse.Namespace(steps=20, clip=3.0, action_noise=0.05, gamma=0.99, lam=0.95, terminal_seconds=1.0,
                               terminal_kinetic=1.0, fall_weight=1.0, value_reward="smooth")
    world = {"obs_noise": OBS_NOISE, "push_dv": 0.3, "delay_ms": (10.0, 70.0)}
    is_net = isinstance(learned, ValueLearner)
    print(f"{args.checkpoint}: learned term 2 = {'a value network' if is_net else 'weights ' + str(dict(zip(learned.names, [round(v, 3) for v in learned.s_target.tolist()])))}")

    def one(value, label, mask_action=False):
        env = GradStand(args.robots, seed=args.seed, **world)
        if mask_action:       # the last action as a constant: V's slope reaches the actions only through the physics
            orig = env.critic_obs
            env.critic_obs = lambda: torch.cat((orig()[:, :50], orig()[:, 50:72].detach(), orig()[:, 72:]), -1)
        torch.manual_seed(5)
        g = window(env, policy, wargs, value=value)["g"]
        n = g.norm(dim=-1).median(dim=1).values
        print(f"  {label:40s} first step {float(n[0]):.2e}  middle {float(n[len(n) // 2]):.2e}  last {float(n[-1]):.2e}")
        return g

    print("1. action gradient per control step (median over robots)")
    g_fixed = one(None, "fixed term 2")
    g_learned = one(learned, "learned term 2")
    cos = lambda a, b: float((a * b).sum() / (a.norm() * b.norm()))  # noqa: E731
    line = f"  direction against the fixed term 2 (cosine): learned {cos(g_learned, g_fixed):+.2f}"
    if is_net:
        g_masked = one(learned, "learned V, last action not an input", mask_action=True)
        line += f", without the action input {cos(g_masked, g_fixed):+.2f}"
    print(line)

    print("2. at the end states: slope of the learned V against the slope of minus the fixed term 2, in q and p")
    env = GradStand(args.robots, seed=args.seed, **world)
    torch.manual_seed(5)
    for _ in range(args.windows):
        window(env, policy, wargs)
        state = env.get_state()
        q = state["q"].detach().clone().requires_grad_(True)
        p = state["p"].detach().clone().requires_grad_(True)
        env.set_state({**state, "q": q, "p": p})
        gv = torch.cat(torch.autograd.grad(learned(learned.inputs(env)).sum(), (q, p), allow_unused=True), -1)
        gq, gp = torch.autograd.grad(-env.terminal_cost(1.0, 1.0).sum(), (q, p), allow_unused=True)
        gf = torch.cat((gq, torch.zeros_like(p) if gp is None else gp), -1)
        c = (gv * gf).sum(-1) / (gv.norm(dim=-1) * gf.norm(dim=-1)).clamp_min(1e-30)
        print(f"  window: cosine per robot median {float(c.median()):+.2f} (min {float(c.min()):+.2f}, max {float(c.max()):+.2f}); "
              f"slope size learned {float(gv.norm(dim=-1).median()):.2e}, fixed {float(gf.norm(dim=-1).median()):.2e}")
        env.set_state({k: (v.detach() if torch.is_tensor(v) else v) for k, v in env.get_state().items()})


if __name__ == "__main__":
    main()

r"""Improve a stand policy with gradients of the reward through the pH model (the test of 16 Nov 2026).

The stand task of Full Body rl/stand, rebuilt on the Python plant (ph_k1/plant.py + ph_k1/firmware.py, PyTorch,
float64) so that autograd reaches from the reward back to the policy's weights:

    observation (rl/stand/contract.py) -> policy network -> decode -> firmware chain (clamp, slew, 20 ms delay,
    motor driver) -> 20 plant steps of 1 ms -> the smooth reward terms of rl/stand/env.py

Every iteration rolls B robots K control steps from where the last iteration ended (states detached: truncated
backpropagation through time, as SHAC does, here without a value network), and takes one optimiser step on the
policy.  Quiet world: no sensor noise, no pushes.  The terms left out are the ones without a gradient
(alive, feet_grounded, undesired_contacts, termination).

    python -m mbrl.grad.refine --policy outputs/kaggle/mb-stand-v1/runs/mb-stand-v1/model_260.pt \
        --robots 8 --steps 10 --iterations 30 --run-name grad-stand-1

Writes logs/<run-name>/model_<it>.pt and deploy_<it>.bin (the firmware's file), to be scored on the C++ plant with
Full Body's  rl/stand/evaluate.py --policy <deploy.bin> -B 128.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from mbrl import paths
from mbrl.grad.env import GradStand
from mbrl.mpc.nets import MLP
from rl.stand import contract as C


def load_policy(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    policy = MLP(C.OBS_DIM, C.NUM_ACTIONS)
    policy.load_state_dict(ck["actor_state_dict"])
    value = None
    if "critic_state_dict" in ck:
        n_in = ck["critic_state_dict"]["mlp.0.weight"].shape[1]
        value = MLP(n_in, 1)
        value.load_state_dict(ck["critic_state_dict"])
        value = value.double().requires_grad_(False)
    return policy.double(), value, ck


def save(policy: MLP, ck: dict, run_dir: Path, it: int) -> Path:
    from rl.stand.export import export_bin

    out = dict(ck)
    out["actor_state_dict"] = {k: v.float() if v.is_floating_point() else v for k, v in policy.state_dict().items()}
    out["iter"] = it
    path = run_dir / f"model_{it}.pt"
    torch.save(out, path)
    export_bin(path, path.with_name(f"deploy_{it}.bin"))
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, help="model_<it>.pt to start from")
    ap.add_argument("--run-name", default="grad-stand")
    ap.add_argument("--robots", type=int, default=8)
    ap.add_argument("--steps", type=int, default=10, help="control steps per rollout (K)")
    ap.add_argument("--iterations", type=int, default=30)
    ap.add_argument("--minutes", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--last-layer", action="store_true", help="train only the last layer")
    ap.add_argument("--value", action="store_true",
                    help="add the checkpoint's value network at the end of the rollout (SHAC: reward beyond the horizon)")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--terminal-seconds", type=float, default=1.0,
                    help="term 2: pose, tilt and height error at the end of the window, charged for this long; 0 = off")
    ap.add_argument("--terminal-kinetic", type=float, default=1.0, help="term 2: cost per joule of kinetic energy at the end")
    ap.add_argument("--delta", type=float, default=0.03,
                    help="controlled step: RMS change of the action targets per update (action units)")
    ap.add_argument("--clip", type=float, default=3.0, help="one robot's action gradient at most this x the batch median")
    ap.add_argument("--regress", type=int, default=5, help="regression steps of the policy onto the action targets")
    ap.add_argument("--checkpoint", action="store_true", help="one memory block per control step (slower, small memory)")
    ap.add_argument("--raw-gradient", action="store_true",
                    help="the first learner of 16 Nov: one optimiser step on the raw gradient (it made the stand worse)")
    ap.add_argument("--warmup", type=int, default=50, help="control steps of the policy before the first rollout")
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    run_dir = paths.REPO / "logs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    policy, value, ck = load_policy(args.policy)
    if args.value and value is None:
        raise SystemExit('the checkpoint has no value network')
    params = list(policy.mlp[-1].parameters()) if args.last_layer else list(policy.mlp.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)
    env = GradStand(args.robots, seed=args.seed, checkpoint=args.checkpoint)
    with torch.no_grad():
        for _ in range(args.warmup):
            env.step(policy(env.observe()))
    print(f"[grad] {args.robots} robots x {args.steps} control steps per rollout, lr {args.lr}"
          f"{' (last layer)' if args.last_layer else ''}, start {args.policy}", flush=True)

    t_start, falls = time.time(), 0
    with open(run_dir / "metrics.jsonl", "a") as metrics:
        for it in range(1, args.iterations + 1):
            t0 = time.time()
            env.detach()
            total, objective, disc, watts, parts = 0.0, 0.0, 1.0, [], {}
            obs_list, act_list, grads = [], [], []

            def clip_and_keep(g):   # per robot: at most clip x the batch median norm; what flows on is the clipped one
                n = g.norm(dim=-1, keepdim=True)
                g = g * (args.clip * n.median() / n.clamp_min(1e-30)).clamp(max=1.0)
                grads.append(g.detach())
                return g

            for _ in range(args.steps):
                o = env.observe()
                a = policy(o)
                if not args.raw_gradient:
                    a.register_hook(clip_and_keep)
                    obs_list.append(o.detach())
                    act_list.append(a.detach())
                r, info = env.step(a)
                total = total + r.mean()
                objective = objective + disc * r.mean()
                disc *= args.gamma
                watts.append(info["copper_W"].mean())
                for k, v in info["terms"].items():
                    parts[k] = parts.get(k, 0.0) + float(v.mean())
            t_fwd = time.time() - t0
            v_end = 0.0
            if args.value:
                v_end = value(env.critic_obs()).mean()
                objective = objective + disc * v_end
            term2 = torch.zeros(())
            if args.terminal_seconds > 0 or args.terminal_kinetic > 0:
                term2 = env.terminal_cost(args.terminal_seconds, args.terminal_kinetic).mean()
                objective = objective - disc * term2
            opt.zero_grad()
            (-objective).backward()        # objective = -(term 1 + term 2) of the window, through the pH model
            clipped = 0.0
            if args.raw_gradient:
                gnorm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
                opt.step()
            else:
                # the hooks ran from the last step to the first: grads[::-1] is in window order.  They hold the
                # derivative of what was backpropagated, the loss -objective: the improving direction is its negative
                g = -torch.stack(grads[::-1])                                 # (K, B, 22), d objective / d action
                norms = g.norm(dim=-1)
                gnorm = float(norms.median())
                clipped = float((norms >= 0.999 * args.clip * norms.median(dim=1, keepdim=True).values).float().mean())
                target = torch.stack(act_list) + args.delta * g / g.square().mean().sqrt().clamp_min(1e-30)
                obs_b, target_b = torch.cat(obs_list), target.flatten(0, 1)
                for _ in range(args.regress):                                 # the network follows the improved actions
                    opt.zero_grad()
                    (policy(obs_b) - target_b).square().mean().backward()
                    opt.step()
            fell = env.fallen()
            if bool(fell.any()):
                falls += int(fell.sum())
                env.detach()
                env.reset()                     # simple: all robots start again
                with torch.no_grad():
                    for _ in range(args.warmup):
                        env.step(policy(env.observe()))
            row = {"iteration": it, "reward_step": float(total) / args.steps, "copper_W": float(torch.stack(watts).mean()),
                   "grad_norm": gnorm, "clipped": clipped, "term2": float(term2), "falls": falls, "value_end": float(v_end), "forward_s": t_fwd, "seconds": time.time() - t0,
                   **{f"term/{k}": v / args.steps for k, v in parts.items()}}
            metrics.write(json.dumps(row) + "\n")
            metrics.flush()
            print(f"[grad] it {it:3d}  reward/step {row['reward_step']:+.5f}  copper {row['copper_W']:6.2f} W  "
                  f"|grad| {gnorm:8.3g}  term2 {float(term2):.4f}  clipped {clipped:.0%}  falls {falls}  ({row['seconds']:.0f} s, forward {t_fwd:.0f} s)", flush=True)
            if it % args.save_every == 0 or it == args.iterations:
                save(policy, ck, run_dir, it)
            if args.minutes and (time.time() - t_start) / 60 >= args.minutes:
                save(policy, ck, run_dir, it)
                break


if __name__ == "__main__":
    main()

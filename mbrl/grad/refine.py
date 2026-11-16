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
from mbrl.mpc.nets import MLP
from ph_k1 import firmware as fwm
from ph_k1.model import load_k1
from ph_k1.plant import K1PHPlant, Ledger
from rl.stand import contract as C
from rl.stand.env import REWARDS

SMOOTH = ("orientation", "base_height", "base_lin_vel", "base_ang_vel", "joint_vel", "joint_acc", "action_rate",
          "similar_to_default", "joint_limits", "action_excess", "copper_loss")
POLICY_DT, SUBSTEPS = 0.02, 20
MIN_HEIGHT = 0.28


class GradStand:
    """rl/stand's task on the differentiable Python plant.  Quiet: no noise, no pushes, 20 ms delay."""

    def __init__(self, robots: int, seed: int = 1, joint_noise: float = 0.02):
        self.B, self.joint_noise = robots, joint_noise
        self.gen = torch.Generator().manual_seed(seed)
        self.model = load_k1(dtype=torch.float64, device="cpu")
        self.plant = K1PHPlant(model=self.model, batch=robots, timestep=POLICY_DT / SUBSTEPS)
        cfg = fwm.FirmwareConfig(gains=C.GAINS, slew=C.SLEW, delay_ms=(20.0, 20.0), policy_dt=POLICY_DT,
                                 delay_hold_prob=None)
        self.contract = C.Contract(torch.float64, "cpu")
        self.plant.reset(pose=self.contract.default_qpos)
        self.stand_height = float(self.plant.q[:, 2].mean())
        self.fw = fwm.K1Firmware(self.plant, cfg)
        m = self.model   # which contact points are the left and the right sole (as rl/stand/env.py)
        sole, cb = m.contact_is_sole, m.contact_body
        left, right = m.body_names.index("left_foot_link"), m.body_names.index("right_foot_link")
        self.left_sole = (sole & (cb == left)).to(torch.float64)
        self.right_sole = (sole & (cb == right)).to(torch.float64)
        self.contact = torch.ones(robots, m.ncontact, dtype=torch.float64)
        self.reset()

    def reset(self) -> None:
        B = self.B
        noise = self.joint_noise * (2 * torch.rand(B, C.NUM_ACTIONS, generator=self.gen, dtype=torch.float64) - 1)
        self.plant.reset(pose=self.contract.default_qpos + noise)
        self.fw.reset()
        z = lambda n: torch.zeros(B, n, dtype=torch.float64)  # noqa: E731
        self.actions, self.prev_actions = z(C.NUM_ACTIONS), z(C.NUM_ACTIONS)
        self.qd, self.prev_qd, self.gyro, self.v_body = z(C.NUM_ACTIONS), z(C.NUM_ACTIONS), z(3), z(3)

    def detach(self) -> None:
        """Cut the graph: the next rollout starts from these numbers."""
        pl, fw = self.plant, self.fw
        for k in ("q", "p", "xi", "lam", "ball"):
            setattr(pl, k, getattr(pl, k).detach())
        pl.ledger = Ledger.zeros(self.B, pl.dtype, pl.device)
        fw.ring, fw.last_target = fw.ring.detach().clone(), fw.last_target.detach()
        if fw.ring_kp is not None:
            fw.ring_kp, fw.ring_kd = fw.ring_kp.detach().clone(), fw.ring_kd.detach().clone()
        for k in ("actions", "prev_actions", "qd", "prev_qd", "gyro", "v_body"):
            setattr(self, k, getattr(self, k).detach())

    def observe(self) -> torch.Tensor:
        q = self.plant.q
        return self.contract.observe(q[:, 7:], self.qd, self.gyro, q[:, 3:7], self.actions)

    def critic_obs(self) -> torch.Tensor:
        """rl/stand/env.py's critic input: observation | trunk velocity | height error | feet on the ground."""
        q = self.plant.q
        feet = torch.stack(((self.contact @ self.left_sole) > 0, (self.contact @ self.right_sole) > 0), -1).to(torch.float64)
        return torch.cat((self.observe(), self.v_body, q[:, 2:3] - self.stand_height, feet), -1)

    def step(self, a: torch.Tensor):
        """-> (reward (B,) of the smooth terms, dict of logged numbers).  Differentiable in a."""
        pl = self.plant
        self.prev_actions, self.actions = self.actions, a
        copper0 = pl.ledger.copper.clone()
        info = self.fw.policy_step(self.contract.decode(a), SUBSTEPS)
        nu = pl.velocity()
        self.prev_qd = self.qd
        self.qd, self.v_body, self.gyro = nu[:, 6:], nu[:, 3:6], info.extra["gyro"]
        self.contact = info.contact_active.to(torch.float64)
        copper_w = (pl.ledger.copper - copper0) / POLICY_DT
        q, m = pl.q, self.model
        theta = q[:, 7:]
        g = self.contract.projected_gravity(q[:, 3:7])
        terms = {
            "orientation": g[:, :2].square().sum(-1),
            "base_height": (q[:, 2] - self.stand_height).square(),
            "base_lin_vel": self.v_body.square().sum(-1),
            "base_ang_vel": self.gyro.square().sum(-1),
            "joint_vel": self.qd.square().sum(-1),
            "joint_acc": ((self.qd - self.prev_qd) / POLICY_DT).square().sum(-1),
            "action_rate": (self.actions - self.prev_actions).square().sum(-1),
            "similar_to_default": (theta - self.contract.default_qpos).abs().sum(-1),
            "joint_limits": ((m.jnt_low - theta).clamp_min(0) + (theta - m.jnt_high).clamp_min(0)).sum(-1),
            "action_excess": ((self.actions.abs() - 1.0).clamp_min(0).sum(-1)).clamp(max=10.0),
            "copper_loss": copper_w,
        }
        reward = sum(REWARDS[k] * terms[k] for k in SMOOTH) * POLICY_DT
        return reward, {"copper_W": copper_w.detach(), "terms": {k: (REWARDS[k] * v * POLICY_DT).detach() for k, v in terms.items()}}

    def fallen(self) -> torch.Tensor:
        rp = self.contract.roll_pitch(self.plant.q[:, 3:7].detach())
        return (rp.abs() > C.TILT_ABORT).any(-1) | (self.plant.q[:, 2].detach() < MIN_HEIGHT)


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
    env = GradStand(args.robots, seed=args.seed)
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
            for _ in range(args.steps):
                r, info = env.step(policy(env.observe()))
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
            opt.zero_grad()
            (-objective).backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
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
                   "grad_norm": gnorm, "falls": falls, "value_end": float(v_end), "forward_s": t_fwd, "seconds": time.time() - t0,
                   **{f"term/{k}": v / args.steps for k, v in parts.items()}}
            metrics.write(json.dumps(row) + "\n")
            metrics.flush()
            print(f"[grad] it {it:3d}  reward/step {row['reward_step']:+.5f}  copper {row['copper_W']:6.2f} W  "
                  f"|grad| {gnorm:8.3g}  V {float(v_end):.3f}  falls {falls}  ({row['seconds']:.0f} s, forward {t_fwd:.0f} s)", flush=True)
            if it % args.save_every == 0 or it == args.iterations:
                save(policy, ck, run_dir, it)
            if args.minutes and (time.time() - t_start) / 60 >= args.minutes:
                save(policy, ck, run_dir, it)
                break


if __name__ == "__main__":
    main()

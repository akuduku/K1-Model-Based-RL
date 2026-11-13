r"""Learn a get-up policy from the model-based plans: DAgger warm start (imitate the plan), then PPO (track + stand).

Lecture 11, option 1 (distil the planner into a policy) followed by option 2 (RL in the model): the pH plant is the
simulator, the reward is mbrl/getup/track_env.py's (the get-up reward + tracking the plan, fading), the policy is the
robot's StandupAgent contract (72 -> 22, fixed gains), exported as deploy_<it>.bin for the firmware's standup agent.

    python -m mbrl.getup.train --plans plans/getup-supine-v1.pt,plans/getup-prone-v1.pt --hours 6 --run-name getup-track-v1
    python -m mbrl.getup.train --plans ... --device cpu --num-envs 16 --hours 0.01 --dagger-iters 3 --no-wandb  # smoke
    python -m mbrl.getup.train --plans ... --fit fits/getup-both-v1.json ...   # every robot on a fitted parameter set

The warm start (--dagger-iters, before PPO, skipped on --resume): rollouts of --steps-per-env steps where each robot
executes the plan's target with probability beta = 0.99^it, else the student's; the actor is fitted to the plan's
targets on all visited states (DAgger), the critic to the GAE returns of the same rollouts (its warm-up).  Then PPO
starts from that actor with --init-std.  Everything else is Full Body's harness (rl/common/harness.py): W&B, time
budget, checkpoints, videos, the no-noise evaluation (Eval/*), resume.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import torch  # noqa: E402

from .. import paths  # noqa: E402
from .track_env import NUM_ACTIONS, OBS_DIM, TrackConfig, TrackEnv  # noqa: E402
from rl.common import export as E  # noqa: E402
from rl.common import harness as H  # noqa: E402
from rl.getup import contract as C  # noqa: E402
from rl.getup.evaluate import NoNoiseEval  # noqa: E402


def export_bin(checkpoint, out_path) -> Path:
    """The firmware's StandupAgent file: 72 inputs -> 22 targets (the agent rl_core's stand-up policies use)."""
    return E.export_bin(checkpoint, out_path, OBS_DIM, NUM_ACTIONS, C.HIDDEN_DIMS, agent="the standup")


def dagger_warm_start(env: TrackEnv, iters: int, *, beta0: float = 1.0, decay: float = 0.99, epochs: int = 5,
                      lr: float = 1e-3, gamma: float = 0.99, lam: float = 0.95, log=print):
    """-> before_learn(runner, resumed) for harness.train."""

    def before_learn(runner, resumed: bool):
        if resumed or iters <= 0:
            return
        alg = runner.alg
        actor, critic = alg.actor, alg.critic
        params_a = list(actor.parameters())
        params_c = list(critic.parameters())
        opt_a = torch.optim.Adam(params_a, lr=lr)
        opt_c = torch.optim.Adam(params_c, lr=lr)
        T = runner.cfg["num_steps_per_env"] if hasattr(runner, "cfg") else 24
        t_start = time.time()
        obs = env.get_observations().to(env.device)
        for it in range(iters):
            beta = beta0 * decay ** it
            buf_obs, buf_teacher, buf_rew, buf_done, buf_val = [], [], [], [], []
            with torch.no_grad():
                for _ in range(T):
                    teacher = env.teacher_actions()
                    student = actor(obs)
                    use_teacher = torch.rand(env.num_envs, 1, device=env.device) < beta
                    a = torch.where(use_teacher, teacher, student)
                    actor.update_normalization(obs)
                    critic.update_normalization(obs)
                    buf_obs.append(obs)
                    buf_teacher.append(teacher)
                    buf_val.append(critic(obs).squeeze(-1))
                    obs, rew, dones, _ = env.step(a)
                    obs = obs.to(env.device)
                    buf_rew.append(rew)
                    buf_done.append(dones.to(env.dtype))
                last_val = critic(obs).squeeze(-1)
                adv = torch.zeros_like(last_val)
                ret = []
                for t in reversed(range(T)):
                    nxt = last_val if t == T - 1 else buf_val[t + 1]
                    nonterm = 1.0 - buf_done[t]
                    delta = buf_rew[t] + gamma * nxt * nonterm - buf_val[t]
                    adv = delta + gamma * lam * nonterm * adv
                    ret.insert(0, adv + buf_val[t])
            o = torch.cat(buf_obs)
            y = torch.cat(buf_teacher)
            r = torch.cat(ret)
            n = y.shape[0]
            for _ in range(epochs):
                perm = torch.randperm(n, device=env.device)
                for mb in perm.split(max(1, n // 4)):
                    loss_a = (actor(o[mb]) - y[mb]).square().mean()
                    opt_a.zero_grad()
                    loss_a.backward()
                    torch.nn.utils.clip_grad_norm_(params_a, 1.0)
                    opt_a.step()
                    loss_c = (critic(o[mb]).squeeze(-1) - r[mb]).square().mean()
                    opt_c.zero_grad()
                    loss_c.backward()
                    torch.nn.utils.clip_grad_norm_(params_c, 1.0)
                    opt_c.step()
            if it % 10 == 0 or it == iters - 1:
                log(f"[dagger] it {it:4d}  beta {beta:.3f}  actor loss {float(loss_a):.4f}  critic loss "
                    f"{float(loss_c):.3f}  standing {float(env.standing.float().mean()):.2f}  "
                    f"({(time.time() - t_start) / 60:.1f} min)")
        # rsl-rl's own optimizer starts fresh on the warm-started weights
        alg.optimizer.state.clear()
        log(f"[dagger] warm start done after {iters} iterations ({(time.time() - t_start) / 60:.1f} min)")

    return before_learn


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    H.add_arguments(ap, run_prefix="getup-track", wandb_project="Akos pH MBRL", iterations=100000, save_every=0,
                    video_every=0, save_every_min=10.0, video_every_min=30.0, init_std=0.5, entropy_coef=0.005,
                    eval_every_min=30.0, first_eval_min=3.0)
    ap.add_argument("--plans", required=True, help="comma list of plan .pt files (mbrl.getup.plan), one per kind")
    ap.add_argument("--lying-kinds", default="supine,prone")
    ap.add_argument("--fit", default=None, help="fit JSON of mbrl.fit: every robot draws an elite at each reset")
    ap.add_argument("--dagger-iters", type=int, default=300)
    ap.add_argument("--dagger-lr", type=float, default=1e-3)
    ap.add_argument("--track-final", type=float, default=0.2)
    ap.add_argument("--delay-ms", default="10,40")
    ap.add_argument("--push-dv", type=float, default=0.3)
    ap.add_argument("--eval-robots", type=int, default=64)
    ap.set_defaults(video_seconds=8.0)
    args = ap.parse_args()
    if args.log_root == str(H.REPO / "logs"):
        args.log_root = str(paths.REPO / "logs")
    lo, hi = (float(x) for x in args.delay_ms.split(","))
    plans = tuple(p.strip() for p in args.plans.split(",") if p.strip())
    kinds = tuple(k.strip() for k in args.lying_kinds.split(",") if k.strip())
    cfg = TrackConfig(num_envs=args.num_envs, device=args.device, dtype=args.dtype, seed=args.seed,
                      steps_per_env=args.steps_per_env, delay_ms=(lo, hi), lying_kinds=kinds, push_dv=args.push_dv,
                      plans=plans, track_final=args.track_final)
    fit = None
    if args.fit:
        from mbrl.model import FittedModel, attach
        fit = FittedModel.load(args.fit)

    def build(c):
        e = TrackEnv(c)
        if fit is not None:
            from mbrl.model import attach
            attach(e.plant, fit, None, "elites", seed=c.seed)
        return e

    env = build(cfg)
    print(f"[track] {args.num_envs} robots on {args.device}, plans {plans}, fit {args.fit}, "
          f"DAgger {args.dagger_iters} iterations, budget {max(args.hours, args.max_minutes / 60):.2f} h, "
          f"logs -> {Path(args.log_root) / args.run_name}", flush=True)

    def video_env():
        return build(TrackConfig(**{**cfg.__dict__, "num_envs": 1, "seed": cfg.seed + 1, "library_size": 32,
                                    "push_dv": 0.0, "obs_noise": {}, "device": "cpu"}))

    def eval_env():
        return build(TrackConfig(**{**cfg.__dict__, "num_envs": args.eval_robots, "seed": cfg.seed + 2,
                                    "library_size": 2 * args.eval_robots, "push_dv": 0.0, "device": "cpu"}))

    evaluate = NoNoiseEval(eval_env, env) if args.eval_robots > 0 else None

    def prepare(venv, it):
        venv.cfg.start = kinds[prepare.n % len(kinds)]
        prepare.n += 1
        venv._reset_idx(torch.arange(1, device=venv.device))
        venv.load_curriculum_state(env.curriculum_state())
    prepare.n = 0

    video = H.VideoMaker(video_env, args.video_seconds, cfg.policy_dt, Path(args.log_root) / args.run_name / "videos",
                         "track", prepare)
    H.train(env, args, tag="track", hidden_dims=C.HIDDEN_DIMS, export=export_bin, video=video, evaluate=evaluate,
            artifact="k1-getup-track", artifact_meta={"agent": "standup", "obs_dim": OBS_DIM,
                                                       "num_actions": NUM_ACTIONS, "plans": list(plans)},
            before_learn=dagger_warm_start(env, args.dagger_iters, lr=args.dagger_lr))


if __name__ == "__main__":
    main()

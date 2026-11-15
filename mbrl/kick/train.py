r"""Learn a walk-and-kick policy from the model-based plan: DAgger warm start (imitate the plan), then PPO.

    python -m mbrl.kick.train --plan plans/kick-v1.pt --num-envs 4096 --hours 6 --run-name kick-track-v1    # GPU
    python -m mbrl.kick.train --plan plans/kick-v1.pt --teacher-report -B 256                # how good is the plan?
    python -m mbrl.kick.train --plan ... --device cpu --num-envs 16 --hours 0.01 --dagger-iters 3 --no-wandb  # smoke

--teacher-report: the plan itself (mbrl/kick/controller.py, with the true ball) on the whole task, also the requests
that come mid-walk, which the planner did not search on: the upper bound of what imitating it can reach.

The warm start (--dagger-iters, before PPO, skipped on --resume): rollouts of --steps-per-env steps where each robot
executes the plan's action with probability beta = 0.99^it, else the student's; the actor is fitted to the plan's
actions on all visited states (DAgger), the critic to the GAE returns of the same rollouts.  It counts in the time
budget (--hours) and stops after --dagger-minutes at the latest.  Then PPO on mbrl/kick/track_env.py.

W&B project "Akos pH Kick MBRL": everything rl/kick/train.py logs (Kick/*, Energy/*, Motors/*, Eval/*, video with
the Hamiltonian figure) plus Track/* (action error to the plan, the tracking weight, good kicks, the plan's phases)
and the DAgger curve (Track/dagger).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import torch  # noqa: E402

from .. import paths  # noqa: E402
from .plan import DEFAULT_WALK  # noqa: E402
from .track_env import TrackConfig, TrackEnv  # noqa: E402
from rl.common import harness as H  # noqa: E402
from rl.kick import contract as C  # noqa: E402
from rl.kick.evaluate import KickEval, play, quiet, report, summarize  # noqa: E402
from rl.kick.export import export_bin  # noqa: E402
from rl.kick.train import KickVideo  # noqa: E402


def dagger_warm_start(env: TrackEnv, iters: int, *, minutes: float = 0.0, beta0: float = 1.0, decay: float = 0.99,
                      epochs: int = 5, lr: float = 1e-3, gamma: float = 0.99, lam: float = 0.95, history=None,
                      log=print):
    """-> before_learn(runner, resumed) for harness.train.  history: a list that gets one row per DAgger iteration."""

    def before_learn(runner, resumed: bool):
        if resumed or iters <= 0:
            return
        alg = runner.alg
        actor, critic = alg.actor, alg.critic
        params_a, params_c = list(actor.parameters()), list(critic.parameters())
        opt_a, opt_c = torch.optim.Adam(params_a, lr=lr), torch.optim.Adam(params_c, lr=lr)
        T = runner.cfg["num_steps_per_env"] if hasattr(runner, "cfg") else 24
        t_start = time.time()
        obs = env.get_observations().to(env.device)
        it = 0
        for it in range(iters):
            if minutes > 0 and (time.time() - t_start) / 60 > minutes:
                log(f"[dagger] time limit of {minutes:g} min reached at iteration {it}")
                break
            beta = beta0 * decay ** it
            buf_obs, buf_teacher, buf_rew, buf_done, buf_val = [], [], [], [], []
            kicks, good = [], []
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
                    obs, rew, dones, extras = env.step(a)
                    obs = obs.to(env.device)
                    buf_rew.append(rew)
                    buf_done.append(dones.to(env.dtype))
                    for key, store in (("Kick/kick_rate", kicks), ("Track/good_kick", good)):
                        v = extras["log"].get(key)
                        if v is not None and v.numel() > 0:
                            store.append(v.float().reshape(-1))
                last_val = critic(obs).squeeze(-1)
                adv = torch.zeros_like(last_val)
                ret = []
                for t in reversed(range(T)):
                    nxt = last_val if t == T - 1 else buf_val[t + 1]
                    nonterm = 1.0 - buf_done[t]
                    delta = buf_rew[t] + gamma * nxt * nonterm - buf_val[t]
                    adv = delta + gamma * lam * nonterm * adv
                    ret.insert(0, adv + buf_val[t])
            o, y, r = torch.cat(buf_obs), torch.cat(buf_teacher), torch.cat(ret)
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
            mean = lambda xs: float(torch.cat(xs).mean()) if xs else float("nan")  # noqa: E731
            row = {"iteration": it, "beta": beta, "actor_loss": float(loss_a), "critic_loss": float(loss_c),
                   "kick_rate": mean(kicks), "good_kick": mean(good), "minutes": (time.time() - t_start) / 60}
            if history is not None:
                history.append(row)
            if it % 10 == 0 or it == iters - 1:
                log(f"[dagger] it {it:4d}  beta {beta:.3f}  actor loss {row['actor_loss']:.4f}  critic loss "
                    f"{row['critic_loss']:.3f}  kicked {row['kick_rate']:.2f}  good kicks {row['good_kick']:.2f}  "
                    f"({row['minutes']:.1f} min)")
        alg.optimizer.state.clear()   # rsl-rl's own optimizer starts fresh on the warm-started weights
        env.load_curriculum_state({})  # the fade of the tracking weight starts with PPO
        log(f"[dagger] warm start done after {it + 1} iterations ({(time.time() - t_start) / 60:.1f} min)")

    return before_learn


def teacher_report(cfg: TrackConfig, robots: int, noise: bool, seed: int) -> dict:
    c = replace(cfg, num_envs=robots, seed=seed, energy_log_envs=0)
    env = TrackEnv(c if noise else quiet(c))
    with torch.no_grad():
        s = summarize(play(env, lambda o: env.teacher_actions()), env.cfg.full_power)
    print(f"the plan {Path(cfg.plan).name} as a controller on the pH plant ({robots} robots"
          f"{', with the training noise' if noise else ''})")
    print(report(s))
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    H.add_arguments(ap, run_prefix="kick-track", wandb_project="Akos pH Kick MBRL", iterations=100000, save_every=0,
                    video_every=0, save_every_min=10.0, video_every_min=30.0, init_std=0.3, entropy_coef=0.003,
                    eval_every_min=30.0, first_eval_min=3.0)
    ap.add_argument("--plan", required=True, help="the library of mbrl.kick.plan (kick-<tag>.pt)")
    ap.add_argument("--walk", default=str(DEFAULT_WALK), help="TorchScript walk policy (75 -> 22) the plan acts through")
    ap.add_argument("--no-phase", action="store_true", help="165 inputs: without the request clock (for an A/B)")
    ap.add_argument("--dagger-iters", type=int, default=300)
    ap.add_argument("--dagger-minutes", type=float, default=60.0)
    ap.add_argument("--dagger-lr", type=float, default=1e-3)
    ap.add_argument("--track-final", type=float, default=0.2)
    ap.add_argument("--delay-ms", default="0,70")
    ap.add_argument("--push-dv", type=float, default=0.25)
    ap.add_argument("--eval-robots", type=int, default=64)
    ap.add_argument("--teacher-report", action="store_true", help="evaluate the plan itself and exit")
    ap.add_argument("-B", "--report-robots", type=int, default=256)
    ap.add_argument("--report-noise", action="store_true")
    ap.set_defaults(video_seconds=6.0, first_video_min=3.0)
    args = ap.parse_args()
    if args.log_root == str(H.REPO / "logs"):
        args.log_root = str(paths.REPO / "logs")
    lo, hi = (float(x) for x in args.delay_ms.split(","))
    cfg = TrackConfig(num_envs=args.num_envs, device=args.device, dtype=args.dtype, seed=args.seed,
                      steps_per_env=args.steps_per_env, delay_ms=(lo, hi), push_dv=args.push_dv, plan=args.plan,
                      walk_teacher=args.walk, phase_input=not args.no_phase, track_final=args.track_final)
    if args.teacher_report:
        teacher_report(cfg, args.report_robots, args.report_noise, args.seed + 100)
        return

    env = TrackEnv(cfg)
    run_dir = Path(args.log_root) / args.run_name
    print(f"[track] {args.num_envs} robots on {args.device}, plan {args.plan} (speeds {env.lib_speeds.tolist()}, "
          f"good kicks in the search {env.lib_success}), {env.contract.obs_dim} inputs, DAgger {args.dagger_iters} "
          f"iterations, budget {max(args.hours, args.max_minutes / 60):.2f} h, logs -> {run_dir}", flush=True)

    video = KickVideo(lambda: TrackEnv(quiet(cfg, num_envs=1, device="cpu", seed=cfg.seed + 1, energy_log_envs=0)),
                      args.video_seconds, cfg.policy_dt, run_dir / "videos", "kick")
    evaluate = KickEval(cfg, args.eval_robots, env_cls=TrackEnv) if args.eval_robots > 0 else None
    obs_dim = env.contract.obs_dim
    history: list = []

    def upload_dagger(writer, it):  # once, when the W&B run exists: the warm start's curve
        if not history or args.no_wandb or upload_dagger.done:
            return
        try:
            import wandb

            if wandb.run is None:
                return
            cols = list(history[0])
            table = wandb.Table(columns=cols, data=[[r[c] for c in cols] for r in history])
            wandb.log({"Track/dagger": table,
                       "Track/dagger_actor_loss": wandb.plot.line(table, "iteration", "actor_loss", title="DAgger: actor loss"),
                       "Track/dagger_good_kick": wandb.plot.line(table, "iteration", "good_kick", title="DAgger: good kicks")},
                      step=it)
        except Exception as exc:  # noqa: BLE001
            print(f"[dagger] curve not uploaded: {type(exc).__name__}: {exc}", flush=True)
        upload_dagger.done = True
    upload_dagger.done = False

    H.train(env, args, tag="track", hidden_dims=C.HIDDEN_DIMS, export=lambda ck, out: export_bin(ck, out, obs_dim),
            video=video, evaluate=evaluate, artifact="k1-kick-track",
            artifact_meta={"agent": "kick_walk", "obs_dim": obs_dim, "num_actions": C.NUM_ACTIONS, "plan": args.plan},
            before_learn=dagger_warm_start(env, args.dagger_iters, minutes=args.dagger_minutes, lr=args.dagger_lr,
                                           history=history),
            on_log=upload_dagger)


if __name__ == "__main__":
    main()

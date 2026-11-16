r"""Train a skill with the model-based learner (plan in the pH model, learn policy and value): from scratch.

    python -m mbrl.mpc.train stand --robots 16 --candidates 16 --horizon 10 --minutes 30 --no-wandb     # laptop
    python -m mbrl.mpc.train stand --robots 256 --candidates 32 --horizon 12 --hours 6 --run-name mb-stand-v1   # T4
    python -m mbrl.mpc.train stand --resume logs/mb-stand-v1/model_400.pt --hours 3

Output in logs/<run-name>/: config.json, metrics.jsonl (one row per iteration), model_<it>.pt with deploy_<it>.bin
(the firmware's policy file, the same format as the PPO runs'), videos/.  W&B project "Akos pH MB".
The policy file is evaluated like a PPO one: Full Body rl/stand/evaluate.py --policy logs/<run>/deploy_<it>.bin.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import torch  # noqa: E402

from .. import paths  # noqa: E402
from .learner import LearnConfig, Learner  # noqa: E402


def stand_task(args):
    """World, model, export, evaluation and video for standing under pushes (Full Body rl/stand)."""
    from rl.common import harness as H
    from rl.stand.env import StandConfig, StandEnv
    from rl.stand.evaluate import StandEval
    from rl.stand.export import export_bin

    cfg = StandConfig(num_envs=args.robots, device=args.device, dtype=args.dtype, seed=args.seed)
    world = StandEnv(cfg)
    # the model: the same env, nothing random; gains and delay of every robot come with its snapshot
    model = StandEnv(replace(cfg, num_envs=args.robots * args.candidates, seed=args.seed + 1, obs_noise={},
                             push_dv=0.0, gain_dr=0.0, delay_hold_prob=None))

    def after_restore(m):
        m.episode_length_buf.zero_()   # no time limit inside the planner's short rollouts

    evaluate = StandEval(cfg, args.eval_robots, device="cpu", seconds=args.eval_seconds) if args.eval_robots > 0 else None
    video = None
    if args.video_every_min > 0:
        video = H.VideoMaker(lambda: StandEnv(replace(cfg, num_envs=1, device="cpu", seed=cfg.seed + 2)),
                             8.0, cfg.policy_dt, Path(args.log_root) / args.run_name / "videos", "stand")
    return world, model, export_bin, evaluate, video, after_restore


TASKS = {"stand": stand_task}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=sorted(TASKS))
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--robots", type=int, default=16, help="world robots (R)")
    ap.add_argument("--candidates", type=int, default=16, help="model copies per world robot (N)")
    ap.add_argument("--horizon", type=int, default=10, help="planner steps of 20 ms (H)")
    ap.add_argument("--sigma", type=float, default=0.3)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--policy-share", type=float, default=0.25)
    ap.add_argument("--steps", type=int, default=24, help="control steps per iteration (T)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--iterations", type=int, default=1_000_000)
    ap.add_argument("--minutes", type=float, default=0.0)
    ap.add_argument("--hours", type=float, default=0.0)
    ap.add_argument("--save-every-min", type=float, default=10.0)
    ap.add_argument("--eval-every-min", type=float, default=5.0)
    ap.add_argument("--eval-robots", type=int, default=32)
    ap.add_argument("--eval-seconds", type=float, default=None, help="default: one episode")
    ap.add_argument("--video-every-min", type=float, default=0.0)
    ap.add_argument("--resume", default=None, help="model_<it>.pt to continue from")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--log-root", default=str(paths.REPO / "logs"))
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-project", default="Akos pH MB")
    ap.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", "karl-jahnel-universit-t-leipzig"))
    args = ap.parse_args()
    args.run_name = args.run_name or time.strftime(f"mb-{args.task}-%Y%m%d-%H%M")

    world, model, export, evaluate, video, after_restore = TASKS[args.task](args)
    cfg = LearnConfig(candidates=args.candidates, horizon=args.horizon, sigma=args.sigma, temperature=args.temperature,
                      policy_share=args.policy_share, steps=args.steps, lr=args.lr, seed=args.seed)
    run = None
    if not args.no_wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.run_name,
                         config={"task": args.task, **vars(args)})
    run_dir = Path(args.log_root) / args.run_name
    learner = Learner(world, model, cfg, run_dir, export=export, evaluate=evaluate, video=video,
                      after_restore=after_restore, wandb_run=run)
    if args.resume:
        learner.load(args.resume)
    per_step = args.robots * args.candidates * args.horizon
    print(f"[mb] {args.task}: {args.robots} world robots, {args.candidates} x {args.horizon} planner "
          f"({per_step} model steps per control step) on {args.device}, logs -> {run_dir}", flush=True)
    learner.train(args.iterations, minutes=max(args.minutes, 60 * args.hours), save_every_min=args.save_every_min,
                  eval_every_min=args.eval_every_min, video_every_min=args.video_every_min)
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()

r"""Train a Full Body skill (stand, getup) on the fitted model: the parameter ensemble, optionally + the residual.

    python -m mbrl.train stand --fit fits/getup-v1.json -- --num-envs 4096 --iterations 3000
    python -m mbrl.train getup --fit fits/getup-v1.json --residual residual/getup-v1.pt -- --hours 6.75
    python -m mbrl.train stand --fit fits/getup-v1.json -- --device cpu --num-envs 16 --iterations 2 --no-wandb  # smoke

The flags after `--` go unchanged to Full Body's rl/<task>/train.py (the same harness: W&B, checkpoints, deploy.bin
for the robot, a video every 500 iterations, --max-minutes/--hours, --resume-wandb).  The only change is the env:
its plant is wrapped by mbrl.model.attach (every robot draws an ensemble member at every reset).
Added unless given: --log-root <this repo>/logs, --wandb-project "Akos pH MBRL", --run-name <task>-mb-<date>.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

from . import paths  # noqa: E402
from .model import FittedModel, attach  # noqa: E402
from .residual import ResidualEnsemble  # noqa: E402

ENV_CLASS = {"stand": "StandEnv", "getup": "GetUpEnv"}


def patch_env(task: str, fit: FittedModel | None, residual: ResidualEnsemble | None, mode: str):
    """Replace the task's env class in rl.<task>.train by one whose plant is the fitted model; returns the module."""
    T = importlib.import_module(f"rl.{task}.train")
    name = ENV_CLASS[task]
    Base = getattr(T, name)

    class ModelBasedEnv(Base):
        def __init__(self, cfg):
            super().__init__(cfg)
            res = None if residual is None else residual.to(self.device)
            attach(self.plant, fit, res, mode, seed=cfg.seed)

    ModelBasedEnv.__name__ = f"ModelBased{name}"
    setattr(T, name, ModelBasedEnv)
    return T


def with_defaults(rest: list[str], task: str) -> list[str]:
    rest = [x for x in rest if x != "--"]
    if "--log-root" not in rest:
        rest += ["--log-root", str(paths.REPO / "logs")]
    if "--wandb-project" not in rest:
        rest += ["--wandb-project", "Akos pH MBRL"]
    if "--run-name" not in rest:
        rest += ["--run-name", time.strftime(f"{task}-mb-%Y%m%d-%H%M")]
    return rest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=sorted(ENV_CLASS))
    ap.add_argument("--fit", default=None, help="fit JSON of mbrl.fit (none: today's plant)")
    ap.add_argument("--residual", default=None, help="residual .pt of mbrl.residual")
    ap.add_argument("--mode", default="elites", choices=["elites", "gaussian", "mean"])
    a, rest = ap.parse_known_args(argv)
    fit = FittedModel.load(a.fit) if a.fit else None
    res = ResidualEnsemble.load(a.residual) if a.residual else None
    T = patch_env(a.task, fit, res, a.mode)
    rest = with_defaults(rest, a.task)
    print(f"[mbrl] {a.task} on the fitted model: fit={a.fit} residual={a.residual} mode={a.mode}", flush=True)
    sys.argv = [T.__file__] + rest
    T.main()


if __name__ == "__main__":
    main()

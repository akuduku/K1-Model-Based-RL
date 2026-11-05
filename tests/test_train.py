import pytest
import torch

pytest.importorskip("tensordict")

import mbrl  # noqa: E402,F401
from mbrl.train import patch_env  # noqa: E402

from test_model import _fit  # noqa: E402


def test_stand_env_runs_on_the_fitted_model():
    T = patch_env("stand", _fit(), None, "elites")
    from rl.stand.env import StandConfig
    env = T.StandEnv(StandConfig(num_envs=4, device="cpu", seed=3))
    assert hasattr(env.plant, "mbrl")
    for _ in range(3):
        env.step(torch.zeros(4, 22))
    assert torch.isfinite(env.plant.q).all()

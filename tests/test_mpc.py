"""The model-based learner: networks in the firmware's layout, the planner against the bare policy, one iteration.

    python -m pytest tests/test_mpc.py -q
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

import mbrl  # noqa: F401

pytest.importorskip("tensordict")

from mbrl.mpc.learner import LearnConfig, Learner  # noqa: E402
from mbrl.mpc.nets import MLP, checkpoint  # noqa: E402
from mbrl.mpc.planner import Planner, index_snapshot  # noqa: E402
from rl.stand.env import StandConfig, StandEnv  # noqa: E402


def envs(R=4, N=6):
    cfg = StandConfig(num_envs=R, device="cpu", dtype="float32", seed=3)
    world = StandEnv(cfg)
    model = StandEnv(replace(cfg, num_envs=R * N, seed=4, obs_noise={}, push_dv=0.0, gain_dr=0.0, delay_hold_prob=None))
    return world, model


def test_checkpoint_is_the_firmwares_network(tmp_path):
    """A checkpoint exports to the robot's .bin, and the .bin computes what the network computes."""
    from rl.common.export import load_policy
    from rl.stand.export import export_bin

    torch.manual_seed(0)
    policy, value = MLP(72, 22), MLP(80, 1)
    policy.obs_normalizer.update(torch.randn(500, 72) * 2 + 1)
    path = tmp_path / "model_1.pt"
    torch.save(checkpoint(policy, value, 1), path)
    fw = load_policy(export_bin(path, tmp_path / "deploy_1.bin"))
    obs = torch.randn(5, 72)
    with torch.no_grad():
        want = policy(obs)
    assert torch.allclose(torch.as_tensor(fw(obs.numpy())).float(), want, atol=1e-4)


def test_planner_copies_states_and_beats_the_bare_policy():
    world, model = envs()
    torch.manual_seed(1)
    policy, value = MLP(72, 22, last_scale=0.01), MLP(world.get_observations()["critic"].shape[1], 1)
    for _ in range(20):                                    # some motion first
        world.step(0.3 * torch.randn(4, 22))
    planner = Planner(world, model, policy, value, candidates=6, horizon=5,
                      after_restore=lambda m: m.episode_length_buf.zero_())
    snap = index_snapshot(world.snapshot(), planner.rows)
    assert snap["plant"]["q"].shape[0] == 24 and torch.equal(snap["plant"]["q"][7], world.plant.q[1])
    q_before = world.plant.q.clone()
    action, info = planner.plan()
    assert torch.equal(world.plant.q, q_before)            # planning does not touch the world
    assert action.shape == (4, 22) and planner.model_steps == 4 * 6 * 5
    assert (info["return_planner"] >= info["return_mean"] - 1e-5).all()     # the weights favour the better copies
    assert (info["return_best"] >= info["return_planner"] - 1e-5).all()


def test_one_iteration_learns_and_saves(tmp_path):
    world, model = envs(R=4, N=4)
    from rl.stand.export import export_bin

    learner = Learner(world, model, LearnConfig(candidates=4, horizon=3, steps=6, epochs=1), tmp_path, export=export_bin,
                      after_restore=lambda m: m.episode_length_buf.zero_(), log=lambda *a: None)
    learner.train(2, save_every_min=0.0)
    assert learner.executed == 2 * 6 * 4 and learner.planner.model_steps == 2 * 6 * 4 * 4 * 3
    assert (tmp_path / "model_2.pt").exists() and (tmp_path / "deploy_2.bin").exists()
    rows = (tmp_path / "metrics.jsonl").read_text().strip().splitlines()
    assert len(rows) == 2 and "Plan/return_planner" in rows[0] and "Cost/model_steps" in rows[0]
    again = Learner(world, model, LearnConfig(candidates=4, horizon=3, steps=6), tmp_path / "b", log=lambda *a: None)
    again.load(tmp_path / "model_2.pt")
    assert again.iteration == 2 and again.executed == learner.executed
    obs = world.get_observations()
    assert torch.allclose(again.act(obs), learner.act(obs))

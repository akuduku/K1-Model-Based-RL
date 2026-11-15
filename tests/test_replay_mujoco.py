"""The MuJoCo replay: same windows and error formulas as mbrl/replay.py, floor placement, the paired statistics."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
import torch

import mbrl  # noqa: F401
from helpers import synthetic_log
from mbrl.params import ParamSpace
from mbrl.replay import Replay, make_windows, projected_gravity
from mbrl.replay_mujoco import errors, floor_only, nothing_moves, paired, plant_rows, put_on_floor, replay_mujoco
from ph_k1.model import load_k1
from rl.walk.backends import MuJoCoBackend


@pytest.fixture(scope="module")
def replay():
    space = ParamSpace()
    logs = [synthetic_log(space.neutral(1)[0])]
    return Replay(logs, make_windows(logs), space, 1)


def test_a_perfect_prediction_has_no_error(replay):
    rp = replay
    out = errors(rp, rp.obs_theta.cpu(), rp.obs_gyro.cpu(), rp.obs_grav.cpu())
    assert out["loss"] == pytest.approx(0.0, abs=1e-12)


def test_the_window_tilt_adds_up_to_the_replays_loss(replay):
    """plant_rows derives each window's tilt as the rest of mbrl/replay.py's loss."""
    rp = replay
    row = plant_rows(rp, rp.space.neutral(1)[0])
    assert np.mean(row["windows"]["tilt"]) == pytest.approx(row["tilt"], rel=1e-9)
    assert np.mean(row["windows"]["loss"]) == pytest.approx(row["loss"], rel=1e-9)


def test_mujoco_replays_a_moving_log_better_than_nothing(replay):
    delay = float(replay.space.delay_ms(replay.space.neutral(1))[0])
    out = replay_mujoco(replay, delay)
    assert out["diverged"] == 0.0
    assert out["loss"] < nothing_moves(replay)["loss"]


def test_floor_only_keeps_the_floor_and_removes_self_collision():
    b = MuJoCoBackend(load_k1())
    m = b.m
    floor_only(m)
    collide = lambda i, j: bool((m.geom_contype[i] & m.geom_conaffinity[j]) | (m.geom_contype[j] & m.geom_conaffinity[i]))  # noqa: E731
    active = [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
    floor = [g for g in active if m.geom_bodyid[g] == 0]
    robot = [g for g in active if m.geom_bodyid[g] != 0]
    assert floor and len(robot) > 10
    assert all(collide(floor[0], g) for g in robot)
    assert not any(collide(i, j) for i in robot for j in robot if i < j)


@pytest.mark.parametrize("start", (0.9, 0.3))
def test_put_on_floor_touches_from_above_and_below(start):
    b = MuJoCoBackend(load_k1())
    m, d = b.m, b.d
    d.qpos[2] = start
    put_on_floor(m, d)
    mujoco.mj_kinematics(m, d)
    margins = m.geom_margin.copy()
    m.geom_margin[:] = 1.0
    mujoco.mj_collision(m, d)
    m.geom_margin[:] = margins
    gaps = [d.contact[i].dist for i in range(d.ncon) if 0 in (m.geom_bodyid[d.contact[i].geom1], m.geom_bodyid[d.contact[i].geom2])]
    assert min(gaps) == pytest.approx(0.0, abs=1e-3)


def test_paired_statistics():
    a, b = np.arange(40.0), np.arange(40.0) + 1.0
    p = paired(a, b)
    assert p["mean"] == pytest.approx(-1.0)
    assert p["a_better"] == 1.0
    assert p["low"] == pytest.approx(-1.0) and p["high"] == pytest.approx(-1.0)
    noisy = paired(np.random.default_rng(0).normal(size=200), np.random.default_rng(1).normal(size=200))
    assert noisy["low"] < 0.0 < noisy["high"]
    assert noisy["a_better"] == pytest.approx(0.5, abs=0.1)


def test_gravity_of_mbrl_and_walk_agree():
    from rl.walk import contract as C

    quat = np.array((0.9, 0.2, -0.3, 0.1))
    quat /= np.linalg.norm(quat)
    np.testing.assert_allclose(projected_gravity(torch.as_tensor(quat)[None])[0].numpy(), C.projected_gravity(quat))

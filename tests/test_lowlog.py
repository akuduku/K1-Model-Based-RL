"""Full-rate logs as a RobotLog: the grid, which command is in force, the attitude from roll and pitch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import mbrl  # noqa: F401
from mbrl import data, lowlog, paths

REAL = paths.REPO.parents[1] / "Karls Repo" / "Repo_Eric" / "rec_kick_walk_20261113_222701.csv"


def records(n=400, m=60, seed=0):
    rng = np.random.default_rng(seed)
    t = 100.0 + np.cumsum(rng.uniform(0.0015, 0.0025, n))
    tc = np.linspace(t[0] + 0.0105, t[-1], m)
    rpy = np.stack((0.2 * np.sin(3 * t), 0.1 * np.cos(2 * t), 5.0 * t), -1)
    return dict(t=t, q=rng.normal(size=(n, 22)), dq=rng.normal(size=(n, 22)), gyro=rng.normal(size=(n, 3)),
                acc=np.tile((0.0, 0.0, 9.81), (n, 1)), rpy=rpy, tau=rng.normal(size=(n, 22)), tc=tc,
                q_cmd=np.arange(m)[:, None] * np.ones(22), kp=np.full((m, 22), 80.0), kd=np.full((m, 22), 4.0))


def test_grid_and_the_command_in_force():
    r = records()
    log = lowlog.from_records("x", **r)
    assert log.dt == 1e-3 and log.step_dt == 1e-3
    assert np.allclose(np.diff(log.t), 1e-3)
    first = r["tc"][0] - r["t"][0]
    assert not log.cmd_ok[log.t < first - 1e-9].any() and log.cmd_ok[log.t >= first].all()
    for g in (20, 137, len(log) - 1):                  # the newest command at or before the grid time
        want = np.searchsorted(r["tc"] - r["t"][0], log.t[g], side="right") - 1
        assert log.cmd[g, 0] == want
    assert (np.diff(log.cmd[log.cmd_change, 0]) > 0).all()
    assert log.kp.shape == (len(log), 22) and (log.kp[log.cmd_ok] == 80.0).all() and (log.kd[log.cmd_ok] == 4.0).all()
    assert log.tau is not None and log.tau.shape == (len(log), 22)


def test_no_torque_estimate_reads_as_none():
    r = records()
    r["tau"] = np.zeros_like(r["tau"])
    assert lowlog.from_records("x", **r).tau is None


def test_roll_and_pitch_give_gravity_without_yaw():
    roll, pitch = np.array([0.3, 0.0, -0.2]), np.array([0.0, 0.4, 0.1])
    g = data.gravity_in_trunk(lowlog.quat_from_roll_pitch(roll, pitch))
    want = np.stack((np.sin(pitch), -np.sin(roll) * np.cos(pitch), -np.cos(roll) * np.cos(pitch)), -1)
    np.testing.assert_allclose(g, want, atol=1e-12)


@pytest.mark.skipif(not Path(str(REAL) + ".k1low").exists(), reason="the robot log is not on this machine")
def test_the_real_run_matches_the_motor_checks_reader():
    from mbrl.motor_check import load_run

    log, run = lowlog.load(REAL, mahony=False), load_run(REAL)
    assert abs(log.t[-1] - (run.t[-1] - run.t[0])) < 2e-3
    assert np.allclose(log.kp[log.cmd_ok], run.kp[0]) and np.allclose(log.kd[log.cmd_ok], run.kd[0])
    g = 12345
    k = np.searchsorted(run.tc, run.t[0] + log.t[g], side="right") - 1
    np.testing.assert_allclose(log.cmd[g], run.q_cmd[k])
    i = np.searchsorted(run.t - run.t[0], log.t[g])
    assert np.abs(log.q[g] - run.q[i]).max() < 0.02 and np.abs(log.tau[g] - run.tau[i]).max() < 3.0
    up = log.acc / np.linalg.norm(log.acc, axis=1, keepdims=True)
    quiet = np.abs(np.linalg.norm(log.acc, axis=1) - data.G) < 0.5
    angle = np.degrees(np.arccos(np.clip((-up * log.grav).sum(1), -1, 1)))
    assert np.median(angle[quiet]) < 1.5       # the robot's roll and pitch agree with its accelerometer

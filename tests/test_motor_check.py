"""The motor check's two motor models on a synthetic run: the robot holds still, one command step on the left knee."""

from __future__ import annotations

import numpy as np
import pytest

import mbrl  # noqa: F401
from mbrl.motor_check import Run, legs_rmse, metrics, mujoco_torque, ph_torque, load_motors

KNEE = 13
T_STEP = 0.1


def still_run(step: float) -> Run:
    """500 Hz states, joints at 0 and still; commands at 200 Hz, the knee's target jumps by `step` at T_STEP."""
    t = np.arange(0.0, 0.3, 0.002)
    tc = np.arange(0.0, 0.3, 0.005)
    z = np.zeros((len(t), 22))
    q_cmd = np.zeros((len(tc), 22))
    q_cmd[tc >= T_STEP, KNEE] = step
    kp = np.full((len(tc), 22), 80.0)
    kd = np.full((len(tc), 22), 4.0)
    tau = np.where((t >= T_STEP)[:, None] & (np.arange(22) == KNEE), 80.0 * step, 0.0)
    return Run("synthetic", t, z, z.copy(), tau, tc, q_cmd, kp, kd)


@pytest.fixture(scope="module")
def mot():
    return load_motors()


def test_small_step_same_in_both(mot):
    """Within the voltage limit the pH driver's current loop settles in one step: both models give kp * step."""
    run = still_run(0.01)  # 0.8 N m on the knee
    mj = mujoco_torque(run, mot, 0.0)
    _, _, ph = ph_torque(run, mot, [0.0], [mot.coil_time], keep=True)
    after = run.t >= T_STEP + 0.004
    assert np.allclose(mj[after, KNEE], 0.8)
    assert np.allclose(ph[0, after, KNEE], 0.8, atol=1e-6)


def test_large_step_coil_lags(mot):
    """A 40 N m knee step: MuJoCo has it at once. The pH coil at full voltage rises towards the stall torque,
    tau_stall (1 - exp(-t / (L/R))), so 40 N m take -L/R ln(1 - 40 / tau_stall) (~18 ms with L/R = 50 ms)."""
    run = still_run(0.5)
    mj = mujoco_torque(run, mot, 0.0)
    _, _, ph = ph_torque(run, mot, [0.0, 0.0], [0.05, 0.002], keep=True)
    stall = mot.effort[KNEE] * mot.speed[KNEE] / mot.width[KNEE]
    rise = -0.05 * np.log(1 - 40.0 / stall)
    i10 = np.searchsorted(run.t, T_STEP + 0.01)
    assert mj[i10, KNEE] == pytest.approx(40.0)
    assert ph[0, i10, KNEE] == pytest.approx(stall * (1 - np.exp(-0.01 / 0.05)), rel=0.15)
    assert ph[1, i10, KNEE] == pytest.approx(40.0, rel=0.02)          # short L/R: almost at once
    after = np.searchsorted(run.t, T_STEP + rise + 0.006)
    assert ph[0, after, KNEE] == pytest.approx(40.0, rel=0.02)


def test_delay_shifts_the_response(mot):
    run = still_run(0.01)
    late = mujoco_torque(run, mot, 0.02)
    k = np.searchsorted(run.t, T_STEP + 0.01)
    assert late[k, KNEE] == 0.0 and late[-1, KNEE] == pytest.approx(0.8)
    # the scan's score: the right delay fits exactly, a wrong one does not
    assert legs_rmse(metrics(mujoco_torque(run, mot, 0.0), run.tau)) < 1e-9
    assert legs_rmse(metrics(late, run.tau)) > 0.01

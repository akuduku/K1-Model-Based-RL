"""The flat-feet start (mbrl/calib.py): it must find a sensor offset that was put into a log, turn a drifted attitude
back, and leave a log alone that has neither."""

from __future__ import annotations

import numpy as np
import pytest

import mbrl  # noqa: F401
from helpers import synthetic_low_log
from mbrl import calib, sim2real

FAKE = np.radians(2.0)
L_ROLL = 15          # Left_Ankle_Roll


@pytest.fixture(scope="module")
def feet():
    return calib.Feet()


def stance_log():
    from rl.stand.contract import DEFAULT_QPOS

    return synthetic_low_log("mujoco", seconds=3.0, still_s=1.5, gains="standup", pose=DEFAULT_QPOS)


@pytest.fixture(scope="module")
def logs():
    clean, bad = stance_log(), stance_log()
    bad.q[:, L_ROLL] += FAKE            # the encoder reads 2 deg too much; the firmware's target was relative to it
    bad.cmd[:, L_ROLL] += FAKE
    return clean, bad


def difference_deg(log, feet, off=None):
    """Median left - right sole roll [deg] in the still samples."""
    fixed = calib.apply(log, off) if off is not None else log
    idx = np.flatnonzero(calib.still_mask(log))[::20]
    t = np.array([calib.tilt_deg(feet.normals(fixed.quat[i], fixed.q[i])) for i in idx])
    return float(np.median(t[:, 0, 0] - t[:, 1, 0]))


def test_a_clean_log_stands_flat_and_gets_no_offsets(logs, feet):
    clean, _ = logs
    assert calib.still_mask(clean).mean() > 0.3
    assert abs(difference_deg(clean, feet)) < 0.05
    off = calib.parallel_soles(clean, feet=feet)
    assert np.abs(np.degrees(off.joint)).max() < 0.05


def test_parallel_soles_find_a_fake_ankle_offset(logs, feet):
    _, bad = logs
    assert abs(abs(difference_deg(bad, feet)) - 2.0) < 0.1            # the fake offset shows as a 2 deg difference
    off = calib.parallel_soles(bad, feet=feet)
    j = np.degrees(off.joint)
    assert abs(abs(j[L_ROLL] - j[21]) - 2.0) < 0.1                   # split over both ankles, 2 deg apart
    assert abs(difference_deg(bad, feet, off)) < 0.05                 # parallel again


def test_flat_start_turns_a_drifted_attitude_back(logs, feet):
    clean, _ = logs
    i = np.flatnonzero(calib.still_mask(clean))[50]
    rot, info = calib.flat_start(feet, clean.quat[i], clean.q[i])
    assert info["applied"] and info["turn_deg"] < 0.05                # flat already: (almost) no turn
    drifted = calib.qmul(clean.quat[i], calib.rot_xy(np.radians(1.5), np.radians(-0.8)))
    rot, info = calib.flat_start(feet, drifted, clean.q[i])
    assert info["applied"] and info["full_sole"]
    assert abs(info["roll_deg"] + 1.5) < 0.05 and abs(info["pitch_deg"] - 0.8) < 0.05
    n = feet.normals(calib.qmul(drifted, rot), clean.q[i])
    assert np.abs(calib.tilt_deg(n)).max() < 0.05
    rot, info = calib.flat_start(feet, calib.qmul(clean.quat[i], calib.rot_xy(np.radians(5.0), 0.0)), clean.q[i])
    assert not info["applied"] and np.allclose(rot, (1, 0, 0, 0))   # above the cap: left alone


def still_joints(log, start):
    """MuJoCo replaying a MuJoCo-made log: the median joint error of the still windows [deg]."""
    r = sim2real.analyse(log, 0.2, 7.0, sensitivity=False, start=start)
    return r["phase_rows"]["still"]["MuJoCo"]["joints"]


def test_the_fix_brings_the_replay_back(logs):
    """One-sided fake offset (left ankle only): the fix splits it over both ankles (a standing robot cannot tell
    which one is off), so the feet lie flat again but each ankle is 1 deg off: most, not all, of the error goes."""
    clean, bad = logs
    good, broken, fixed = still_joints(clean, "as-before"), still_joints(bad, "as-before"), still_joints(bad, "flat-feet")
    assert broken > 3 * good                                          # the fake offset makes MuJoCo rock on its own log
    assert fixed < 0.75 * broken


def test_an_even_offset_is_undone(logs):
    """The offset split evenly over the two ankles, as the fix assumes: the replay comes back to the clean log's."""
    clean, _ = logs
    even = stance_log()
    even.q[:, L_ROLL] += FAKE / 2
    even.cmd[:, L_ROLL] += FAKE / 2
    even.q[:, 21] -= FAKE / 2                                         # Right_Ankle_Roll
    even.cmd[:, 21] -= FAKE / 2
    good, broken, fixed = still_joints(clean, "as-before"), still_joints(even, "as-before"), still_joints(even, "flat-feet")
    assert broken > 3 * good
    assert fixed < 1.5 * good + 0.01


def test_tilt_change_measure():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=(5, 3)), rng.normal(size=(5, 3))
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    out = sim2real.aligned(a[:, None], a, b)[:, 0]
    assert np.allclose(out, b, atol=1e-12)
    # a prediction that is the robot's own motion, started from a turned attitude: no tilt error left
    obs_g = np.repeat(b[:, None], 4, 1)
    traj = {"theta": np.zeros((5, 4, 22)), "qd": np.zeros((5, 4, 22)), "gyro": np.zeros((5, 4, 3)),
            "grav": np.repeat(a[:, None], 4, 1), "tau": np.zeros((5, 4, 22))}
    obs = dict(traj, grav=obs_g)
    jw = np.full(22, 1 / 22)
    assert np.nanmax(sim2real.measures(traj, obs, jw, (a, b))["tilt"]) < 1e-5
    assert np.nanmin(sim2real.measures(traj, obs, jw)["tilt"]) > 1.0     # scored plainly, the start difference shows


def test_mujoco_resting_depth():
    from ph_k1.model import load_k1
    from rl.walk.backends import MuJoCoBackend

    from mbrl.replay_mujoco import floor_only, resting_depth

    b = MuJoCoBackend(load_k1(), delay_ms=7.0, gain_preset="standup", slew=False)
    floor_only(b.m)
    assert 0.0 <= resting_depth(b) < 1e-4                             # its floor is stiff: well below 0.1 mm


def test_a_robot_lying_still_is_not_standing():
    """The get-ups start lying on the floor: still, but the soles are not on the floor, so no calibration from there."""
    from dataclasses import replace

    log = stance_log()
    lying = calib.qmul(log.quat, calib.rot_xy(0.0, np.radians(85.0)))     # the same log, the trunk tipped over
    from mbrl.data import gravity_in_trunk
    tipped = replace(log, quat=lying, grav=gravity_in_trunk(lying))
    assert calib.still_mask(log).mean() > 0.3
    assert not calib.still_mask(tipped).any()
    assert calib.parallel_soles(tipped).kind == "none"

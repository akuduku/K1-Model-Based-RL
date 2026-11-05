import numpy as np

import mbrl  # noqa: F401
from mbrl import data


def _raw(T=600, tilt=0.4, jitter=True, seed=0):
    rng = np.random.default_rng(seed)
    t = 100.0 + np.arange(T) * 0.002 + (rng.uniform(-0.0009, 0.0009, T) if jitter else 0)
    q = np.zeros((T, 22))
    q[:, 13] = 0.5 * np.sin(2 * np.pi * 1.0 * (t - t[0]))           # knee: known speed
    up = np.array([np.sin(tilt), 0.0, np.cos(tilt)])                 # robot pitched by `tilt`
    acc = np.repeat((data.G * up)[None], T, 0)
    gyro = np.zeros((T, 3))
    cmd_t = np.where(np.arange(T) < 20, -1.0, t)
    cmd = np.zeros((T, 22))
    cmd[:, 0] = np.floor((t - t[0]) / 0.02) * 0.05                    # changes every 20 ms
    cmd[cmd_t < 0] = 0.0
    return t, q, gyro, acc, cmd_t, cmd


def test_resampling_speed_and_commands():
    L = data.from_arrays("x", *_raw())
    assert np.allclose(np.diff(L.t), 0.002)
    tt = L.t[50:-50]
    true_qd = 0.5 * 2 * np.pi * np.cos(2 * np.pi * tt)
    assert np.abs(L.qd[50:-50, 13] - true_qd).max() < 0.15         # no jitter spikes
    assert not L.cmd_ok[:15].any() and L.cmd_ok[30:].all()
    steps = np.diff(L.t[L.cmd_change])
    assert np.abs(np.median(steps) - 0.02) < 0.003


def test_tilt_filter_finds_a_static_tilt():
    L = data.from_arrays("x", *_raw(tilt=0.4, jitter=False))
    g_true = -np.array([np.sin(0.4), 0.0, np.cos(0.4)])
    assert np.abs(L.grav[-1] - g_true).max() < 1e-3


def test_hold_phase_before_the_policy_commands_is_not_used():
    t, q, gyro, acc, cmd_t, cmd = _raw()
    cmd[:] = 0.0
    cmd[300:, 13] = 0.3                     # the policy starts commanding at sample 300
    L = data.from_arrays("x", t, q, gyro, acc, cmd_t, cmd)
    first_ok = int(np.argmax(L.cmd_ok))
    assert abs(L.t[first_ok] - (t[300] - t[0])) < 0.004

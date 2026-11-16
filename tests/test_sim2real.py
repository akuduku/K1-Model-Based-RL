"""The comparison must not favour an engine: each one has to win clearly on a log it made itself."""

from __future__ import annotations

import numpy as np
import pytest

import mbrl  # noqa: F401
from helpers import synthetic_low_log
from mbrl import sim2real
from mbrl.params import ParamSpace
from mbrl.replay import Replay, make_windows


@pytest.fixture(scope="module")
def results():
    return {e: sim2real.analyse(synthetic_low_log(e), 0.2, 7.0, sensitivity=False) for e in ("ph", "mujoco")}


@pytest.mark.parametrize("engine, own, other", (("ph", "pH nominal", "MuJoCo"), ("mujoco", "MuJoCo", "pH nominal")))
def test_each_engine_wins_on_its_own_log(results, engine, own, other):
    res = results[engine]
    rows = res["rows"]
    assert res["windows"] >= 12 and res["has_torque"]
    for k in ("joints", "speed", "gyro", "tilt", "torque"):
        assert rows[own][k]["median"] < 0.5 * rows[other][k]["median"], k
    # deg, its own log replayed: MuJoCo ~0.01, the plant ~0.05 (its contact friction state is not in a log)
    assert rows[own]["joints"]["median"] < (0.03 if engine == "mujoco" else 0.1)
    assert rows[own]["joints"]["median"] < 0.2 * rows["nothing moves"]["joints"]["median"]
    p = res["paired"]["all"]["joints"]                               # pH - MuJoCo
    assert sim2real.verdict(p) == ("pH closer" if engine == "ph" else "MuJoCo closer")


def test_the_floors():
    log = synthetic_low_log("ph", seconds=1.0)
    rp = Replay([log], make_windows([log], 100), ParamSpace(), 1, 100, delay_ms=7.0)
    obs, jw = sim2real.observed(rp), rp.joint_w.cpu().numpy()
    still = sim2real.measures(sim2real.nothing_moves(rp), obs, jw)
    const = sim2real.measures(sim2real.constant_velocity(rp), obs, jw)
    assert np.nanmedian(const["joints"][:, 4]) < np.nanmedian(still["joints"][:, 4])   # after 5 ms speed helps
    assert np.isnan(still["torque"]).all()
    perfect = sim2real.measures(obs, obs, jw)
    assert all(np.nanmax(v) < 1e-5 for v in perfect.values())


def test_mean_interval_and_energy():
    mean, low, high = sim2real.mean_interval(np.r_[np.ones(40), np.nan])
    assert mean == low == high == 1.0
    traj = {"tau": np.ones((2, 10, 22)), "qd": -np.ones((2, 10, 22))}
    e = sim2real.energy(traj, 1e-3, np.full(22, 2.0), np.ones(22))
    assert np.allclose(e["work_pos"], 0.0) and np.allclose(e["work_net"], -8 * 10 * 1e-3)
    assert np.allclose(e["copper"], 8 * 10 * 2.0 * 1e-3)


def test_weight_sink_puts_the_weight_on_the_contact_springs():
    import torch

    from mbrl.state import set_state
    from ph_k1.model import load_k1
    from ph_k1.native import K1PHPlantNative

    pl = K1PHPlantNative(model=load_k1(dtype=torch.float64), batch=2, timestep=1e-3)
    pl.reset()
    theta, quat = pl.theta.clone(), pl.q[:, 3:7].clone()
    quat[1] = torch.tensor([0.9987503, 0.0, 0.0499792, 0.0])        # leaning 5.7 deg: fewer points touch
    set_state(pl, theta, torch.zeros(2, 22), quat, torch.zeros(2, 3), preload=True)
    z = pl._sole_points(pl.kinematics())[..., 2]
    force = (float(pl.params().k_n) * (-z).clamp_min(0)).sum(-1)
    assert torch.allclose(force, torch.full_like(force, pl.model.total_mass * 9.81), rtol=1e-9)

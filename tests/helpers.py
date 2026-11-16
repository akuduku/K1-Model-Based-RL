"""A synthetic robot log made by the plant itself with known parameters (for the self-consistency tests)."""

from __future__ import annotations

import math

import numpy as np
import torch

import mbrl  # noqa: F401
from mbrl import data
from mbrl.params import ParamSpace
from ph_k1.firmware import FirmwareConfig
from ph_k1.model import load_k1
from ph_k1.native import K1PHPlantNative

LEGS = list(range(10, 22))


def synthetic_log(theta_true: torch.Tensor, seconds: float = 1.2, seed: int = 0) -> data.RobotLog:
    sp = ParamSpace()
    model = load_k1(dtype=torch.float64)
    d = float(sp.delay_ms(theta_true[None])[0])
    cfg = FirmwareConfig(gains="standup", slew=False, delay_ms=(d, d), policy_dt=0.02)
    pl = K1PHPlantNative(model=model, batch=1, timestep=1e-3, firmware=cfg)
    pl.robot_params = sp.robot_params(theta_true[None])
    kps, kds = sp.gain_scales(theta_true[None])
    kp, kd = pl.gains()
    kp, kd = (kp * kps).contiguous(), (kd * kds).contiguous()
    pl.reset()
    pl.set_delay(d, d)
    rng = np.random.default_rng(seed)
    phase = torch.as_tensor(rng.uniform(0, 2 * math.pi, 22))
    base = pl.theta.clone()
    rec = {k: [] for k in ("t", "q", "gyro", "acc", "cmd")}
    for k in range(int(seconds / 0.02)):
        target = base.clone()
        target[0, LEGS] += 0.12 * torch.sin(2 * math.pi * 1.5 * k * 0.02 + phase[LEGS])
        for s in range(10):  # 2 ms samples, the logger's rate
            info = pl.policy_step(target, 2, kp=kp, kd=kd)
            rec["t"].append((k * 10 + s + 1) * 0.002)
            rec["q"].append(pl.theta[0].clone())
            rec["gyro"].append(info.extra["gyro"][0].clone())
            rec["acc"].append(info.extra["acc"][0].clone())
            rec["cmd"].append(target[0].clone())
    a = {k: np.asarray(torch.stack(v).numpy() if k != "t" else v) for k, v in rec.items()}
    return data.from_arrays("synthetic", a["t"], a["q"], a["gyro"], a["acc"], a["t"].copy(), a["cmd"],
                            data.LogMeta(gains="standup", slew=False, kind="synthetic"))


def _rpy(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat.T
    return np.stack((np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
                     np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)), np.zeros_like(w)), -1)


def synthetic_low_log(engine: str, seconds: float = 2.0, delay_ms: float = 7.0, seed: int = 0):
    """A full-rate "robot log" (mbrl/lowlog.py) made by one engine: "ph" or "mujoco".  A standing robot whose legs and
    arms follow small sines; a new command every 5 ms, a state every 1 ms, the motor drivers' torque as tau_est."""
    from mbrl import lowlog
    from mbrl.replay_mujoco import floor_only
    from ph_k1 import firmware as fw
    from ph_k1.model import NB
    from rl.walk.backends import MuJoCoBackend

    model = load_k1(dtype=torch.float64)
    kp, kd = (g.numpy().copy() for g in fw.gains(model, "firmware"))
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0, 2 * math.pi, 22)
    amp = np.zeros(22)
    amp[LEGS], amp[2:10] = 0.06, 0.2
    n = int(seconds * 1000)
    rec = {k: [] for k in ("q", "dq", "gyro", "quat", "tau", "tc", "cmd")}
    if engine == "ph":
        cfg = FirmwareConfig(gains="firmware", pos_clamp=False, slew=False, target_clamp=False, overspeed_guard=False,
                             delay_ms=(delay_ms, delay_ms), policy_dt=1e-3)
        pl = K1PHPlantNative(model=model, batch=1, timestep=1e-3, firmware=cfg)
        pl.reset()
        pl.set_delay(delay_ms, delay_ms)
        base = pl.theta[0].numpy().copy()
        K = lambda a: torch.as_tensor(a, dtype=torch.float64)[None]  # noqa: E731
    else:
        import mujoco

        b = MuJoCoBackend(model, delay_ms=delay_ms, gain_preset="firmware", slew=False)
        floor_only(b.m)
        base = b.d.qpos[7:].copy()
        ring = [base.copy()] * (b.lag + 1)
    target = base.copy()
    for k in range(n):
        if k % 5 == 0:
            target = base + amp * np.sin(2 * math.pi * 1.5 * k * 1e-3 + phase)
            rec["tc"].append(k * 1e-3)
            rec["cmd"].append(target.copy())
        if engine == "ph":
            info = pl.policy_step(K(target), 1, kp=K(kp), kd=K(kd))
            row = (pl.theta[0].numpy().copy(), info.extra["nu"][0, NB:].numpy().copy(), info.extra["gyro"][0].numpy().copy(),
                   pl.q[0, 3:7].numpy().copy(), info.torque[0].numpy().copy())
        else:
            ring = ring[1:] + [target.copy()]
            d = b.d
            q, v = d.qpos[7:], d.qvel[6:]
            tau = kp * (ring[0] - q) - kd * v
            cap = np.minimum(np.maximum((b.v_max - b.k * v * np.sign(tau)) * b.k / b.R, 0.0), b.tau_max)
            d.ctrl[:] = np.clip(tau, -cap, cap)
            mujoco.mj_step(b.m, d)
            row = (d.qpos[7:].copy(), d.qvel[6:].copy(), d.qvel[3:6].copy(), d.qpos[3:7].copy(), d.ctrl.copy())
        for key, v in zip(("q", "dq", "gyro", "quat", "tau"), row):
            rec[key].append(v)
    a = {k: np.asarray(v) for k, v in rec.items()}
    t = (np.arange(n) + 1) * 1e-3
    grav = data.gravity_in_trunk(a["quat"])
    m = len(a["tc"])
    return lowlog.from_records(f"synthetic-{engine}", t, a["q"], a["dq"], a["gyro"], -data.G * grav, _rpy(a["quat"]),
                               a["tau"], a["tc"], a["cmd"], np.tile(kp, (m, 1)), np.tile(kd, (m, 1)))

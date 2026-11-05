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

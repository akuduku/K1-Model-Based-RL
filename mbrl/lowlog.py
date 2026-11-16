r"""A robot run with its full-rate file (<rec>.csv + <rec>.csv.k1low, k1_motion_test --rec --low-rec) as a RobotLog.

The .k1low file has every LowState (~500 Hz: q, dq, tau_est, gyro, accelerometer, the robot's own roll/pitch/yaw) and
every LowCmd (~200 Hz: the target and the gains the motor drivers received, after the firmware's safety filter).

Output: a RobotLog (mbrl/data.py) on a uniform 1 ms grid, the plant's step, so mbrl/replay.py can feed the command in
force at every plant step:
    q, qd (the motors' own velocity, not a difference of q), gyro, acc: linear interpolation of the LowStates;
    cmd, kp, kd: the newest LowCmd at or before each grid time (zero-order hold), cmd_ok where there is one;
    quat, grav: from the robot's roll and pitch (yaw-free, as the CSV logs' tilt filter gives it);
    tau: tau_est, None when the file has none (the converted old get-up logs);
    step_dt = dt: one replay step per grid sample;
    extra: "twist" (T,3) the commanded vx, vy, wz from the run log, "mahony" (T,3) gravity from data.tilt_filter on
    the same gyro and accelerometer (a check of the filter the CSV logs depend on), "task".

    log = lowlog.load("rec_kick_walk_20261113_222701.csv")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import paths
from .data import LogMeta, RobotLog, gravity_in_trunk, tilt_filter

DT = 1e-3


def quat_from_roll_pitch(roll: np.ndarray, pitch: np.ndarray) -> np.ndarray:
    """(T,4) w x y z of R = R_y(pitch) R_x(roll): the trunk's orientation without its yaw."""
    cr, sr, cp, sp = np.cos(roll / 2), np.sin(roll / 2), np.cos(pitch / 2), np.sin(pitch / 2)
    return np.stack((cp * cr, cp * sr, sp * cr, -sp * sr), -1)


def from_records(name: str, t, q, dq, gyro, acc, rpy, tau, tc, q_cmd, kp, kd, meta: LogMeta | None = None,
                 dt: float = DT, twist_t=None, twist=None, mahony: bool = True) -> RobotLog:
    """load() for arrays already in memory. t (N,) LowState times, tc (M,) LowCmd times, on one clock."""
    f = lambda a: np.asarray(a, dtype=np.float64)  # noqa: E731
    t, q, dq, gyro, acc, rpy, tc, q_cmd, kp, kd = map(f, (t, q, dq, gyro, acc, rpy, tc, q_cmd, kp, kd))
    keep = np.r_[True, np.diff(t) > 0]
    t, q, dq, gyro, acc, rpy = (a[keep] for a in (t, q, dq, gyro, acc, rpy))
    tau = None if tau is None else f(tau)[keep]
    t0 = t[0]
    tg = np.arange(0.0, t[-1] - t0 + 1e-9, dt)
    tr = t - t0
    interp = lambda a: np.stack([np.interp(tg, tr, a[:, j]) for j in range(a.shape[1])], 1)  # noqa: E731
    rp = interp(np.unwrap(rpy[:, :2], axis=0))
    quat = quat_from_roll_pitch(rp[:, 0], rp[:, 1])
    k = np.searchsorted(tc - t0, tg, side="right") - 1          # the newest command at or before each grid time
    ok = k >= 0
    kc = np.clip(k, 0, None)
    change = np.ceil((tc - t0) / dt - 1e-9).astype(int)       # the first grid sample each command is in force at
    change = np.unique(change[(change >= 0) & (change < len(tg))])
    has_tau = tau is not None and bool(np.abs(tau).max() > 0)
    extra = {}
    if twist is not None:
        j = np.clip(np.searchsorted(f(twist_t) - t0, tg, side="right") - 1, 0, len(twist) - 1)
        extra["twist"] = f(twist)[j]
    g_i, a_i = interp(gyro), interp(acc)
    if mahony:
        extra["mahony"] = gravity_in_trunk(tilt_filter(g_i, a_i, dt))
    return RobotLog(name, meta or LogMeta(gains="log", slew=False, kind="low"), dt, tg, interp(q), interp(dq), g_i, a_i,
                    q_cmd[kc], ok, change, quat, gravity_in_trunk(quat), step_dt=dt, kp=kp[kc], kd=kd[kc],
                    tau=interp(tau) if has_tau else None, extra=extra)


def load(rec: str | Path, name: str | None = None, dt: float = DT, mahony: bool = True) -> RobotLog:
    paths.k1_debug_root()
    from k1_debug.logs import low_beside, read_k1low, read_rec

    rec = Path(rec)
    low_path = low_beside(rec)
    if low_path is None:
        raise FileNotFoundError(f"{rec}: no {rec.name}.k1low beside it (the full-rate file)")
    run, low = read_rec(rec), read_k1low(low_path)
    st, cm = low.state, low.command
    if not len(cm) or not len(st):
        raise ValueError(f"{low_path}: no LowState or no LowCmd records")
    if np.abs(cm["dq"]).max() > 0 or np.abs(cm["tau"]).max() > 0:
        raise ValueError(f"{low_path}: the commands carry a velocity or torque feed-forward (not modelled)")
    twist_t = twist = None
    if all(run.has(c) for c in ("cmd_vx", "cmd_vy", "cmd_wz")) and len(run.t):
        twist_t, twist = run.t, np.stack([run.col(c) for c in ("cmd_vx", "cmd_vy", "cmd_wz")], 1)
    task = run.meta.get("task", "low")
    log = from_records(name or rec.name.removesuffix(".csv"), st["t"], st["q"], st["dq"], st["gyro"], st["acc"],
                       st["rpy"], st["tau_est"], cm["t"], cm["q"], cm["kp"], cm["kd"],
                       LogMeta(gains="log", slew=False, kind=task, note=run.meta.get("policy", "")), dt, twist_t, twist,
                       mahony)
    log.extra["task"] = task
    log.extra["t_offset"] = float(st["t"][0] - run.t[0]) if len(run.t) else 0.0   # grid time -> k1_debug's "time"
    return log

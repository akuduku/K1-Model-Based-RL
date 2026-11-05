r"""Robot logs -> uniform, cleaned arrays the replay can start from and compare against.

Input: the CSV the K1 logger writes at ~500 Hz (Karls Repo/Logs/.../getup_*.csv):
    t, q0..q21, gx gy gz (gyro, rad/s, trunk frame), ax ay az (accelerometer, m/s^2, trunk frame), cmd_t, cmd0..cmd21
`cmd*` are the policy's joint targets (rad, MJCF order); cmd_t = -1 before the command stream starts.

Output (RobotLog), on a uniform grid of dt = 2 ms (the raw stamps jitter down to 0.06 ms, which makes one-sample
q-dot spikes of hundreds of rad/s; resampling removes them):
    q, qd (smoothed central difference), gyro, acc, cmd (zero-order hold of the last command), cmd_ok,
    cmd_change (grid indices where a new command arrived: the policy steps, every ~20 ms),
    quat (trunk orientation, yaw-free, from a Mahony tilt filter of gyro + accelerometer), grav (unit gravity in
    the trunk frame = the policy's projected gravity).

cmd_ok is False before the policy really commands (see from_arrays, hold_tol).

KNOWN lists the logs this repository knows, with the agent settings their replay needs (gain preset of
ph_k1.firmware.GAIN_PRESETS, slew on/off).  New logs: add an entry, or load(path, meta=LogMeta(...)).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import paths

G = 9.81


@dataclass(frozen=True)
class LogMeta:
    gains: str = "standup"   # ph_k1.firmware.GAIN_PRESETS key of the agent that ran
    slew: bool = False       # did that agent use the firmware slew limiter
    kind: str = "getup"
    note: str = ""


# name -> (path below paths.logs_root(), settings)
KNOWN = {
    "getup_2236": ("30102026/laufen-schuss-20261031T132350Z-1-001/laufen-schuss/karl/logs/getup_2236.csv",
                   LogMeta(note="back get-up, 25 Oct 2026, K1-13 Phobos")),
    "getup_front_2250": ("30102026/laufen-schuss-20261031T132350Z-1-001/laufen-schuss/karl/logs/getup_front_2250.csv",
                         LogMeta(note="front get-up, 25 Oct 2026, K1-13 Phobos")),
}


@dataclass
class RobotLog:
    name: str
    meta: LogMeta
    dt: float
    t: np.ndarray          # (T,) seconds from the first sample
    q: np.ndarray          # (T, 22)
    qd: np.ndarray         # (T, 22)
    gyro: np.ndarray       # (T, 3)
    acc: np.ndarray        # (T, 3)
    cmd: np.ndarray        # (T, 22) held target
    cmd_ok: np.ndarray     # (T,) bool: a command was active
    cmd_change: np.ndarray  # (P,) grid indices of the policy steps
    quat: np.ndarray = field(default=None)  # (T, 4) w x y z, trunk -> world, yaw-free
    grav: np.ndarray = field(default=None)  # (T, 3) unit gravity in the trunk frame

    def __len__(self):
        return len(self.t)


# ------------------------------------------------------------------ quaternions (w, x, y, z), numpy
def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw])


def quat_rotate_inv(q, v):
    """R^T v for the rotation R of q (body -> world)."""
    w, x, y, z = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    return R.T @ v


def quat_from_up(up_body):
    """The yaw-free rotation R with R up_body = e_z (up_body: where 'up' points in the trunk frame)."""
    u = up_body / max(np.linalg.norm(up_body), 1e-9)
    ez = np.array([0.0, 0.0, 1.0])
    axis = np.cross(u, ez)
    s, c = np.linalg.norm(axis), float(u @ ez)
    if s < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0]) if c > 0 else np.array([0.0, 1.0, 0.0, 0.0])
    ang = np.arctan2(s, c)
    return np.concatenate(([np.cos(ang / 2)], np.sin(ang / 2) * axis / s))


def tilt_filter(gyro: np.ndarray, acc: np.ndarray, dt: float, kp: float = 5.0) -> np.ndarray:
    """Mahony filter without yaw correction: integrate the gyro, pull the estimated 'up' towards the accelerometer.
    The correction fades out when |acc| is far from g (impacts, fast motion)."""
    T = len(gyro)
    q = np.zeros((T, 4))
    q[0] = quat_from_up(acc[0])
    ez = np.array([0.0, 0.0, 1.0])
    for k in range(T - 1):
        a = acc[k]
        na = np.linalg.norm(a)
        w = gyro[k].copy()
        if na > 1e-6:
            trust = np.exp(-((na - G) / 2.0) ** 2)
            v = quat_rotate_inv(q[k], ez)      # estimated 'up' in the trunk frame
            w += kp * trust * np.cross(a / na, v)
        rv = w * dt
        ang = np.linalg.norm(rv)
        dq = np.array([1.0, 0, 0, 0]) if ang < 1e-12 else np.concatenate(([np.cos(ang / 2)], np.sin(ang / 2) * rv / ang))
        qn = quat_mul(q[k], dq)
        q[k + 1] = qn / np.linalg.norm(qn)
    return q


def gravity_in_trunk(quat: np.ndarray) -> np.ndarray:
    """R^T (0, 0, -1) per row (the contracts' projected gravity)."""
    w, x, y, z = quat.T
    return np.stack((-2 * (x * z - w * y), -2 * (y * z + w * x), -(1 - 2 * (x * x + y * y))), -1)


# ------------------------------------------------------------------ loading
def _moving_average(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return x
    pad = n // 2
    xp = np.concatenate((np.repeat(x[:1], pad, 0), x, np.repeat(x[-1:], pad, 0)))
    c = np.cumsum(np.concatenate((np.zeros((1,) + x.shape[1:]), xp)), 0)
    return (c[n:] - c[:-n])[: len(x)] / n


def from_arrays(name: str, t_raw, q_raw, gyro_raw, acc_raw, cmd_t_raw, cmd_raw, meta: LogMeta | None = None,
                dt: float = 0.002, smooth: int = 5, filter_kp: float = 5.0, hold_tol: float = 0.02) -> RobotLog:
    """The same as load() for arrays already in memory (tests, logs from other loggers).

    hold_tol: the logger writes the start pose as the command until the policy really commands (the get-up logs:
    ~1 s), and in that phase the joints move although the logged command does not (the driver had other targets).
    Commands count only from the first one that differs from that start pose by more than hold_tol rad."""
    order = np.argsort(t_raw, kind="stable")
    t_raw, q_raw, gyro_raw, acc_raw = t_raw[order], q_raw[order], gyro_raw[order], acc_raw[order]
    cmd_t_raw, cmd_raw = cmd_t_raw[order], cmd_raw[order]
    keep = np.concatenate(([True], np.diff(t_raw) > 0))
    t_raw, q_raw, gyro_raw, acc_raw, cmd_t_raw, cmd_raw = (a[keep] for a in (t_raw, q_raw, gyro_raw, acc_raw,
                                                                             cmd_t_raw, cmd_raw))
    t0 = t_raw[0]
    tg = np.arange(0.0, t_raw[-1] - t0 + 1e-9, dt)
    tr = t_raw - t0
    interp = lambda a: np.stack([np.interp(tg, tr, a[:, j]) for j in range(a.shape[1])], 1)  # noqa: E731
    q = interp(q_raw)
    gyro, acc = interp(gyro_raw), interp(acc_raw)
    j = np.clip(np.searchsorted(tr, tg, side="right") - 1, 0, len(tr) - 1)
    cmd, cmd_ok = cmd_raw[j], cmd_t_raw[j] >= 0
    if cmd_ok.any():
        first = int(np.argmax(cmd_ok))
        moved = cmd_ok & (np.abs(cmd - cmd[first]).max(1) > hold_tol)
        active = int(np.argmax(moved)) if moved.any() else len(cmd)
        cmd_ok[:active] = False
    changed = np.concatenate(([False], np.any(np.abs(np.diff(cmd_raw, axis=0)) > 1e-9, 1))) & (cmd_t_raw >= 0)
    change_idx = np.unique(np.clip(np.ceil(tr[changed] / dt - 1e-9).astype(int), 0, len(tg) - 1))
    qd = np.gradient(_moving_average(q, smooth), dt, axis=0)
    quat = tilt_filter(gyro, acc, dt, filter_kp)
    return RobotLog(name, meta or LogMeta(), dt, tg, q, qd, gyro, acc, cmd, cmd_ok, change_idx, quat,
                    gravity_in_trunk(quat))


def load(path: str | Path, name: str | None = None, meta: LogMeta | None = None, **kw) -> RobotLog:
    path = Path(path)
    d = np.genfromtxt(path, delimiter=",", names=True)
    col = lambda *names: np.stack([d[n] for n in names], 1)  # noqa: E731
    return from_arrays(name or path.stem, d["t"], col(*[f"q{i}" for i in range(22)]), col("gx", "gy", "gz"),
                       col("ax", "ay", "az"), d["cmd_t"], col(*[f"cmd{i}" for i in range(22)]), meta, **kw)


def load_known(names=None, root: Path | None = None) -> list[RobotLog]:
    root = Path(root) if root else paths.logs_root()
    out = []
    for n in (KNOWN if names is None else names):
        rel, meta = KNOWN[n]
        p = root / rel
        if not p.exists():
            raise FileNotFoundError(f"log {n}: {p} (set MBRL_LOGS to the folder that holds 30102026/...)")
        out.append(load(p, n, meta))
    return out

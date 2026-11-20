r"""Flat feet at the start of a replay window (sim2real's start "flat-feet").

While the K1 stands still on a flat floor, both soles lie flat on it.  The logged joint angles with the robot's own
roll and pitch should therefore put both soles flat, but on rec_kick_walk_20261113_222701 they do not: the left sole
comes out 2.8 deg off (median over the still parts), the right 0.3 deg.  A simulation started from such a state puts
the lowest sole corner on the floor while the other corners hang up to 8 mm above it, and the simulated robot first
rocks onto its feet.  That rocking was most of the error of both engines while the robot stood (report of 16 Nov 2026).

Split into the two feet's common tilt and their difference, the seven still bouts of that run show two errors:
  * the difference is steady: the left sole is rolled about 2.3 deg against the right in every bout, although both
    stand on one floor: an offset in the leg chain (the ankles' parallel linkage and its encoders);
  * the common tilt moves from bout to bout (roll -0.2 ... +2.0 deg) together with the robot's own roll estimate while
    the leg angles stay: the attitude estimate drifts (the IMU's and the Mahony filter's agree within 0.2 deg, so both
    carry it).

So the start gets two corrections, the same for both engines:
    off = parallel_soles(log)           # constant: ankle offsets that make the two soles parallel (split evenly)
    log2 = apply(log, off)              # angles AND commands shifted: the drivers' error target - angle is unchanged
    rot = flat_start(feet, quat, q)     # per window: when a whole sole touches, the trunk turned (at most 3 deg) until
                                        # the touching corners lie flat: the legs are trusted, the attitude is not
The replay then scores the tilt as its change since the window start, so the estimate's slow drift drops out.

A sample is "still" when the robot stands upright, no twist is commanded and every leg joint moves slower than
`speed` for `hold_s` around it (a robot lying still on the floor does not count: its soles are not on the floor).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from . import paths  # noqa: F401
from .data import RobotLog

LEGS = list(range(10, 22))
SIDES = ("left", "right")
TOUCH_MM = 5.0          # a sole corner touches the floor when it is within this of the lowest of the eight
CAP_DEG = 3.0           # larger turns are not a drifting estimate: then the foot is rolling over an edge


@dataclass
class Offsets:
    kind: str                                   # "none" or "parallel"
    joint: np.ndarray                           # (22,) added to the logged angles and commands [rad]
    samples: int = 0                            # still samples used
    info: dict = field(default_factory=dict)

    @staticmethod
    def none() -> "Offsets":
        return Offsets("none", np.zeros(22))

    def summary(self) -> dict:
        """JSON-ready, in degrees."""
        return {"kind": self.kind, "samples": self.samples, "joint_deg": np.degrees(self.joint).round(4).tolist(),
                **self.info}


class Feet:
    """MuJoCo kinematics of the K1 (the model both engines use): the soles' normals and corners for logged states."""

    def __init__(self):
        import mujoco

        from ph_k1.model import load_k1, load_mj

        k1 = load_k1()
        self.mujoco = mujoco
        self.m = m = load_mj(k1.xml_path)
        self.d = mujoco.MjData(m)
        names = list(k1.joint_names)
        mj_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, m.njnt)]
        if mj_names != names:
            raise ValueError("MuJoCo's joint order differs from the plant's")
        self.ankle = {s: (names.index(f"{s.title()}_Ankle_Pitch"), names.index(f"{s.title()}_Ankle_Roll"))
                      for s in SIDES}
        jid = {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{s.title()}_Ankle_Roll") for s in SIDES}
        self.body = {s: int(m.jnt_bodyid[jid[s]]) for s in SIDES}       # the foot: the ankle roll joint's body
        # the sole: the foot's box (its frame is the foot's, flat at the zero pose); its four bottom corners
        self.sole = {}
        for s in SIDES:
            g = [g for g in range(m.ngeom) if m.geom_bodyid[g] == self.body[s] and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX]
            if len(g) != 1:
                raise ValueError(f"{s} foot: {len(g)} boxes")
            self.sole[s] = g[0]
        self.corners = np.array([[sx, sy, -1.0] for sx in (-1, 1) for sy in (-1, 1)])

    def _pose(self, quat: np.ndarray, q: np.ndarray) -> None:
        d = self.d
        d.qpos[:3] = 0.0
        d.qpos[3:7] = quat
        d.qpos[7:] = q
        self.mujoco.mj_kinematics(self.m, d)

    def normals(self, quat: np.ndarray, q: np.ndarray) -> np.ndarray:
        """(2, 3) the soles' normals (the feet's z axes) in the world, left and right.  quat w x y z, q (22,)."""
        self._pose(quat, q)
        return np.stack([self.d.xmat[self.body[s]].reshape(3, 3)[:, 2] for s in SIDES])

    def corner_heights(self, quat: np.ndarray, q: np.ndarray) -> np.ndarray:
        """(2, 4) the heights of the soles' bottom corners [m] (the trunk at height 0)."""
        self._pose(quat, q)
        out = []
        for s in SIDES:
            g = self.sole[s]
            R = self.d.geom_xmat[g].reshape(3, 3)
            out.append((self.d.geom_xpos[g] + (self.corners * self.m.geom_size[g]) @ R.T)[:, 2])
        return np.array(out)


def tilt_deg(n: np.ndarray) -> np.ndarray:
    """(..., 3) sole normals -> (..., 2) roll, pitch of the sole against the floor [deg] (0, 0 = flat)."""
    return np.degrees(np.stack((np.arctan2(-n[..., 1], n[..., 2]), np.arctan2(n[..., 0], n[..., 2])), -1))


def still_mask(log: RobotLog, speed: float = 0.1, hold_s: float = 0.5, max_tilt_deg: float = 20.0) -> np.ndarray:
    """(T,) True where the robot stands still: upright (its trunk tilted less than max_tilt_deg: a robot lying on the
    floor is still too, but its soles are not on the floor), no twist commanded, and every leg joint slower than
    `speed` [rad/s], all for hold_s around."""
    upright = -log.grav[:, 2] > np.cos(np.radians(max_tilt_deg))
    quiet = (np.abs(log.qd[:, LEGS]).max(1) < speed) & upright
    tw = log.extra.get("twist")
    if tw is not None:
        quiet &= np.abs(tw).max(1) <= 1e-3
    n = max(1, int(round(hold_s / log.dt)))
    held = np.convolve(quiet.astype(float), np.ones(n), mode="same")
    return held >= n - 0.5


def bouts(mask: np.ndarray, t: np.ndarray, min_s: float = 0.3) -> list[tuple[int, int]]:
    """[start, end) index pairs of the True stretches of at least min_s."""
    ch = np.flatnonzero(np.r_[True, mask[1:] != mask[:-1], True])
    return [(int(a), int(b)) for a, b in zip(ch[:-1], ch[1:]) if mask[a] and t[b - 1] - t[a] >= min_s]


# ------------------------------------------------------------------ rotations (w x y z)
def rot_xy(roll: float, pitch: float) -> np.ndarray:
    """w x y z of R_x(roll) R_y(pitch)."""
    cr, sr, cp, sp = np.cos(roll / 2), np.sin(roll / 2), np.cos(pitch / 2), np.sin(pitch / 2)
    return np.array([cr * cp, sr * cp, cr * sp, -sr * sp])


def qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Quaternion products, rows (w x y z), broadcast over leading axes."""
    aw, ax, ay, az = np.moveaxis(np.asarray(a, dtype=float), -1, 0)
    bw, bx, by, bz = np.moveaxis(np.asarray(b, dtype=float), -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def _solve(f, x0: np.ndarray, iters: int = 4, h: float = 1e-5) -> np.ndarray:
    """Newton on f: R^2 -> R^2 with a finite-difference Jacobian."""
    x = np.asarray(x0, dtype=float).copy()
    for _ in range(iters):
        r = f(x)
        J = np.stack([(f(x + h * e) - r) / h for e in np.eye(2)], 1)
        x = x - np.linalg.solve(J, r)
    return x


# ------------------------------------------------------------------ the constant part: the soles parallel
def parallel_fix(feet: Feet, quat: np.ndarray, q: np.ndarray) -> np.ndarray:
    """(2 feet, 2) the (pitch, roll) ankle offsets [rad] that turn each sole onto the two soles' mean normal."""
    n = feet.normals(quat, q)
    mid = n.sum(0) / np.linalg.norm(n.sum(0))
    out = np.zeros((2, 2))
    for i, s in enumerate(SIDES):
        jp, jr = feet.ankle[s]

        def residual(x, i=i, jp=jp, jr=jr):
            qq = q.copy()
            qq[jp] += x[0]
            qq[jr] += x[1]
            return feet.normals(quat, qq)[i, :2] - mid[:2]
        out[i] = _solve(residual, np.zeros(2))
    return out


def parallel_soles(log: RobotLog, every: int = 10, min_samples: int = 20, feet: Feet | None = None,
                   t_range: tuple[float, float] | None = None) -> Offsets:
    """Constant ankle offsets that make the two soles parallel in the still samples (median over them); none without
    enough of them.  Their common tilt is left alone: that is flat_start's job, window by window."""
    mask = still_mask(log)
    if t_range is not None:
        mask &= (log.t >= t_range[0]) & (log.t < t_range[1])
    idx = np.flatnonzero(mask)[::every]
    if len(idx) < min_samples:
        return Offsets.none()
    feet = feet or Feet()
    fix = np.array([parallel_fix(feet, log.quat[i], log.q[i]) for i in idx])     # (n, 2, 2)
    med = np.median(fix, 0)
    joint = np.zeros(22)
    for i, s in enumerate(SIDES):
        joint[list(feet.ankle[s])] = med[i]
    off = Offsets("parallel", joint, len(idx))
    off.info = {"t_range": list(t_range) if t_range else None, "still_fraction": float(still_mask(log).mean()),
                "spread_iqr_deg": np.degrees(np.percentile(fix, 75, 0) - np.percentile(fix, 25, 0)).round(3).tolist(),
                "bouts": bout_table(log, feet, off)}
    return off


def bout_table(log: RobotLog, feet: Feet, off: Offsets | None = None, every: int = 20) -> list[dict]:
    """Per still bout: the soles' tilt as logged (left, right, their mean and difference) and the robot's own roll
    and pitch [deg]; with `off`: also the difference left after it."""
    rows = []
    fixed = apply(log, off) if off is not None else None
    for a, b in bouts(still_mask(log), log.t):
        idx = np.arange(a, b, every)
        t = np.array([tilt_deg(feet.normals(log.quat[i], log.q[i])) for i in idx])          # (n, 2 feet, 2)
        L, R = np.median(t[:, 0], 0), np.median(t[:, 1], 0)
        g = log.grav[idx]
        row = {"t0": float(log.t[a]), "t1": float(log.t[b - 1]), "left": L.round(3).tolist(), "right": R.round(3).tolist(),
               "common": ((L + R) / 2).round(3).tolist(), "difference": (L - R).round(3).tolist(),
               "imu": [float(np.degrees(np.median(np.arctan2(-g[:, 1], -g[:, 2])))),
                       float(np.degrees(np.median(np.arcsin(np.clip(g[:, 0], -1, 1)))))]}
        if fixed is not None:
            tf = np.array([tilt_deg(feet.normals(fixed.quat[i], fixed.q[i])) for i in idx])
            row["difference_after"] = (np.median(tf[:, 0], 0) - np.median(tf[:, 1], 0)).round(3).tolist()
        rows.append(row)
    return rows


def apply(log: RobotLog, off: Offsets) -> RobotLog:
    """A copy of `log` with the joint offsets added to the angles and to the commanded targets (the drivers' error
    target - angle, and so the torque they computed, is unchanged)."""
    if off.kind == "none":
        return log
    return replace(log, q=log.q + off.joint, cmd=log.cmd + off.joint, extra=dict(log.extra, offsets=off.summary()))


# ------------------------------------------------------------------ the per-window part: the standing soles flat
def flat_start(feet: Feet, quat: np.ndarray, q: np.ndarray, touch_mm: float = TOUCH_MM,
               cap_deg: float = CAP_DEG, rounds: int = 4) -> tuple[np.ndarray, dict]:
    """The trunk-fixed turn (w x y z) that puts the sole corners which touch the floor into one plane, and what was
    done.  A corner touches when it is within touch_mm of the lowest of the eight; after each turn the touching set is
    found again.  The fit is least squares on the touching corners' heights (a small pull towards no turn keeps it
    defined when they lie on one edge).  The turn is applied only when a whole sole (its four corners) touches and it
    is at most cap_deg: a foot on an edge (heel strike, toe-off) is not lying flat, and turning the trunk to level
    that edge made walking windows worse (0.03 deg in the joints on rec_kick_walk_20261113_222701)."""
    x = np.zeros(2)
    lam, h_fd = 1e-6, 1e-6

    def heights(x):
        return feet.corner_heights(qmul(quat, rot_xy(x[0], x[1])), q).reshape(-1)
    touch = None
    for _ in range(rounds):
        h = heights(x)
        touch = h < h.min() + touch_mm * 1e-3
        for _ in range(2):                                  # Gauss-Newton on the touching corners' spread
            h = heights(x)
            J = np.stack([(heights(x + h_fd * e) - h) / h_fd for e in np.eye(2)], 1)[touch]
            r = h[touch] - h[touch].mean()
            Jc = J - J.mean(0)
            x = x - np.linalg.solve(Jc.T @ Jc + lam * np.eye(2), Jc.T @ r + lam * x)
    h = heights(x)
    angle = float(np.degrees(np.hypot(*x)))
    t2 = touch.reshape(2, 4)
    full = bool(t2.all(1).any())                            # a whole sole lies on the floor
    ok = full and angle <= cap_deg
    feet_on = [SIDES[i] for i in range(2) if t2[i].any()]
    info = {"feet": "both" if len(feet_on) == 2 else feet_on[0], "corners": int(touch.sum()), "full_sole": full,
            "turn_deg": angle, "applied": ok, "roll_deg": float(np.degrees(x[0])), "pitch_deg": float(np.degrees(x[1])),
            "spread_mm": float((h[touch].max() - h[touch].min()) * 1e3)}
    return (rot_xy(x[0], x[1]) if ok else np.array([1.0, 0.0, 0.0, 0.0])), info

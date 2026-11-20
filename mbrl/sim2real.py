r"""Which physics predicts the real K1 better?  Full-rate robot logs replayed on MuJoCo and on the pH plant.

    python -m mbrl.sim2real <rec.csv>... [--start flat-feet|as-before] [--horizon 0.4] [--delay-ms 7]
    python -m mbrl.sim2real <rec.csv> --view | --save run.rrd      # also in Rerun (mbrl/sim2real_view.py)

Each run log needs its full-rate file beside it (<rec>.csv.k1low).  Writes into <out>/<run>/<start>/: report.md,
results.json and six diagrams (horizon.png, groups.png, paired.png, torque.png, energy.png, sensitivity.png).

Start modes (how both engines are put into the robot's state at a window start):
    as-before   the logged joints and the robot's own roll and pitch, the lowest point on the floor (until 20 Nov
                2026).  The soles then do not lie flat (left 2.8 deg off while standing), so both simulated robots
                first rock onto their feet: in standing windows both did worse than "nothing moves".
    flat-feet   mbrl/calib.py: constant ankle offsets make the two soles parallel; in windows that start while the
                robot stands still the trunk is turned (at most 3 deg) until the sole corners that touch the floor lie
                in one plane: the legs are trusted, the drifting attitude estimate is not.  MuJoCo also starts at its own resting depth (as the plant starts
                with its weight on its springs).  Tilt is scored as its change since the window start.
The report of a flat-feet run puts the as-before numbers beside its own when <out>/<run>/as-before/ exists.

The method is mbrl/replay.py's: cut the run into windows; put both engines into the robot's measured state at the
window start; feed the commands and gains the robot's motor drivers received (LowCmd, one per plant step, after an
actuation delay); compare what the engines predict with what the robot did.  New against mbrl/replay_mujoco.py:
the error at every horizon from 1 ms to the window's end, the torque and the energy during the free replay against
the motors' own torque estimate, and a second floor.

Rows:
    nothing moves       the joints stay at their start
    constant velocity   the joints, the gyro and the tilt keep their start speed: no forces at all
    MuJoCo              not fitted to the robot, no self-collision (the plant's replay has none either)
    pH nominal          not fitted to the robot

Measures, per window and horizon step:
    joints   weighted RMS error of the 22 joint angles [deg] (weights of mbrl/replay.py: legs 1, arms 0.5, head 0.25)
    speed    RMS error of the 12 leg joint speeds [rad/s]
    gyro     error of the trunk's angular velocity [rad/s]
    tilt     angle between the predicted and the measured change of the gravity direction since the window start
             [deg] (when the engines start from the robot's own attitude, as in as-before: the plain tilt error)
    torque   RMS error of the hip and knee torques against tau_est [N m] (ankles apart: their motors act through a
             parallel linkage, tau_est is not the joint torque)

Window groups: still (no twist, every leg joint slower than 0.15 rad/s through the window), walking (a twist is
commanded), no twist, legs moving (on kick_walk: the walk to the ball and the kick, coming to a stop; a get-up).

Limits: the trunk's height and linear velocity are not measured (mbrl/state.py guesses them from the lowest contact
point); windows overlap in time, so the intervals resample blocks of windows; one run is one run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import calib, lowlog, paths
from .params import GROUPS, ParamSpace
from .replay import Replay, make_windows
from .replay_mujoco import COLOURS, GRID, INK, MUTED, mujoco_trajectories, paired

START_MODES = ("flat-feet", "as-before")
ROWS = ("nothing moves", "constant velocity", "MuJoCo", "pH nominal")
ENGINES = ("MuJoCo", "pH nominal")
STILL_SPEED = 0.15            # rad/s: a window is still when every leg joint stays slower (and no twist is commanded)
BLOCKS = (1, 4, 10, 20, 40)   # bootstrap block lengths of the interval check
STILL_GROUP, WALK_GROUP, KICK_GROUP = "still", "walking (twist commanded)", "no twist, legs moving"
STYLE = {"nothing moves": dict(color=INK, ls=":"), "constant velocity": dict(color=MUTED, ls="--"),
         "MuJoCo": dict(color=COLOURS["MuJoCo"], ls="-"), "pH nominal": dict(color=COLOURS["pH nominal"], ls="-")}
MEASURES = {"joints": "joint angles [deg]", "speed": "leg joint speeds [rad/s]", "gyro": "gyro [rad/s]",
            "tilt": "tilt [deg]", "torque": "hip and knee torque [N m]"}
LEGS = list(range(10, 22))
HIP_KNEE = [10, 11, 12, 13, 16, 17, 18, 19]
MARKS_S = (0.02, 0.05, 0.1, 0.2, 0.4)
SENS_DELAYS = (0.0, 3.0, 7.0, 12.0, 20.0)


# ------------------------------------------------------------------ trajectories (W, H, ...) as numpy
def observed(rp: Replay) -> dict:
    n = lambda x: x.cpu().numpy()  # noqa: E731
    return {"theta": n(rp.obs_theta), "qd": n(rp.obs_qd), "gyro": n(rp.obs_gyro), "grav": n(rp.obs_grav),
            "tau": n(rp.obs_tau)}


def measured_quat0(rp: Replay):
    """The robot's own attitude at the window starts (rp.quat0 may carry the flat-feet turn, which only the engines
    get)."""
    return getattr(rp, "quat0_meas", rp.quat0)


def start_gravity(rp: Replay, engine: bool) -> np.ndarray:
    """(W,3) gravity in the trunk frame at the window starts: the engines' start, or the robot's measured one."""
    from .replay import projected_gravity

    return projected_gravity((rp.quat0 if engine else measured_quat0(rp)).cpu()).numpy()


def nothing_moves(rp: Replay) -> dict:
    from .replay import projected_gravity

    W, H = rp.W, rp.H
    g0 = projected_gravity(measured_quat0(rp).cpu()).numpy()
    return {"theta": np.repeat(rp.th0.cpu().numpy()[:, None], H, 1), "qd": np.zeros((W, H, 22)),
            "gyro": np.zeros((W, H, 3)), "grav": np.repeat(g0[:, None], H, 1), "tau": np.full((W, H, 22), np.nan)}


def constant_velocity(rp: Replay) -> dict:
    """Joints at their start speed; the trunk turns at its start rate (gravity in the trunk frame: g' = -w x g)."""
    from .replay import projected_gravity

    W, H = rp.W, rp.H
    t = (np.arange(H) + 1) * rp.step_dt
    th0, qd0, w = rp.th0.cpu().numpy(), rp.qd0.cpu().numpy(), rp.gyro0.cpu().numpy()
    g0 = projected_gravity(measured_quat0(rp).cpu()).numpy()
    rate = np.linalg.norm(w, axis=1, keepdims=True)
    axis = -w / np.maximum(rate, 1e-12)                                    # (W,3)
    ang = (rate * t[None])[..., None]                                      # (W,H,1)
    a, g = axis[:, None], g0[:, None]
    grav = g * np.cos(ang) + np.cross(a, g) * np.sin(ang) + a * (a * g).sum(-1, keepdims=True) * (1 - np.cos(ang))
    return {"theta": th0[:, None] + qd0[:, None] * t[None, :, None], "qd": np.repeat(qd0[:, None], H, 1),
            "gyro": np.repeat(w[:, None], H, 1), "grav": grav, "tau": np.full((W, H, 22), np.nan)}


def ph_trajectories(rp: Replay) -> dict:
    o = rp.run(rp.space.neutral(1), keep_traj=True)
    return {k: o[k][0].cpu().numpy() for k in ("theta", "qd", "gyro", "grav", "tau", "base", "foot_force")}


def all_trajectories(rp: Replay, delay_ms: float, sink: bool = False) -> dict:
    rp.delay_ms = delay_ms
    return {"nothing moves": nothing_moves(rp), "constant velocity": constant_velocity(rp),
            "MuJoCo": mujoco_trajectories(rp, delay_ms, sink=sink), "pH nominal": ph_trajectories(rp)}


def flat_turns(rp: Replay, feet: calib.Feet) -> list[dict]:
    """Turn the start attitude of the windows that start while the robot stands still (only the engines' start:
    rp.quat0; the robot's stays in rp.quat0_meas) until the sole corners that touch the floor lie in one plane
    (calib.flat_start).  Only there: the evidence for a drifting attitude estimate comes from the still bouts, and a
    moving robot's soles really do rock (on its own logs the turn could only add error).  Returns what was done."""
    q0 = rp.quat0.cpu().numpy()
    th0 = rp.th0.cpu().numpy()
    still = {li: calib.still_mask(L) for li, L in enumerate(rp.logs)}
    rots, infos = [], []
    for k, w in enumerate(rp.windows):
        rot, info = calib.flat_start(feet, q0[k], th0[k])
        info["still_start"] = bool(still[w.log][w.start])
        if not info["still_start"]:
            rot, info["applied"] = np.array([1.0, 0.0, 0.0, 0.0]), False
        rots.append(rot)
        infos.append(info)
    rp.quat0_meas = rp.quat0.clone()
    rp.quat0 = torch.as_tensor(calib.qmul(q0, np.array(rots)), dtype=rp.dtype, device=rp.device)
    return infos


def aligned(grav: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """grav (W,H,3) turned per window by the smallest rotation that takes a (W,3) onto b (W,3)."""
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    v, c = np.cross(a, b), (a * b).sum(-1)
    K = np.zeros((len(a), 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -v[:, 2], v[:, 1], -v[:, 0]
    K = K - K.transpose(0, 2, 1)
    R = np.eye(3) + K + K @ K / (1.0 + c)[:, None, None]
    return np.einsum("wij,whj->whi", R, grav)


# ------------------------------------------------------------------ measures
def measures(traj: dict, obs: dict, joint_w: np.ndarray, g0: tuple | None = None) -> dict:
    """{measure: (W,H)} and {"group/<name>": (W,H)} [deg]; NaN where the engine diverged or nothing was measured.
    g0 = (prediction's start gravity, robot's start gravity), each (W,3): the tilt is then scored as the change since
    the window start (the predicted gravity turned by the start difference first).  None: the plain tilt error."""
    e = traj["theta"] - obs["theta"]
    out = {"joints": np.degrees(np.sqrt((e * e * joint_w).sum(-1))),
           "speed": np.sqrt(((traj["qd"] - obs["qd"])[..., LEGS] ** 2).mean(-1)),
           "gyro": np.linalg.norm(traj["gyro"] - obs["gyro"], axis=-1)}
    grav = traj["grav"] if g0 is None or np.allclose(g0[0], g0[1]) else aligned(traj["grav"], g0[0], g0[1])
    cos = (grav * obs["grav"]).sum(-1) / np.maximum(
        np.linalg.norm(grav, axis=-1) * np.linalg.norm(obs["grav"], axis=-1), 1e-12)
    out["tilt"] = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    out["tilt"][~np.isfinite(cos)] = np.nan
    out["torque"] = np.sqrt(((traj["tau"] - obs["tau"])[..., HIP_KNEE] ** 2).mean(-1))
    for g, js in GROUPS.items():
        out[f"group/{g}"] = np.degrees(np.sqrt((e[..., list(js)] ** 2).mean(-1)))
    return out


def quartiles(x: np.ndarray) -> dict:
    """Over the windows, per horizon step: median and the quartiles."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # a measure nobody has (torque of a floor row): all NaN
        q = np.nanpercentile(x, (25, 50, 75), axis=0)
    return {"q25": q[0].tolist(), "median": q[1].tolist(), "q75": q[2].tolist()}


def window_means(m: dict) -> dict:
    """{measure: (W,)} the mean over the horizon (NaN for a diverged window)."""
    return {k: v.mean(1) for k, v in m.items()}


def torque_per_joint(traj: dict, obs: dict) -> np.ndarray:
    """(22,) RMSE against tau_est over all windows and steps."""
    with np.errstate(all="ignore"):
        return np.sqrt(np.nanmean((traj["tau"] - obs["tau"]) ** 2, axis=(0, 1)))


def energy(traj: dict, dt: float, R: np.ndarray, k: np.ndarray) -> dict:
    """Per window, hips and knees: the motors' positive work, their net work and the copper loss [J]."""
    tau, qd = traj["tau"][..., HIP_KNEE], traj["qd"][..., HIP_KNEE]
    p = tau * qd
    return {"work_pos": np.clip(p, 0, None).sum((1, 2)) * dt, "work_net": p.sum((1, 2)) * dt,
            "copper": (tau ** 2 * (R / k ** 2)[HIP_KNEE]).sum((1, 2)) * dt}


def mean_interval(x, block: int = 4, draws: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of the finite entries and its 95 % interval (moving block bootstrap, as replay_mujoco.paired)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < block:
        return (float(x.mean()) if n else float("nan"),) * 3
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block + 1, (draws, int(np.ceil(n / block))))
    index = (starts[..., None] + np.arange(block)).reshape(draws, -1)[:, :n]
    low, high = np.percentile(x[index].mean(1), (2.5, 97.5))
    return float(x.mean()), float(low), float(high)


def phases(rp: Replay) -> dict:
    """{name: (W,) bool}: all, still, walking (a twist commanded at the window start), no twist, legs moving (kicks,
    stops, get-ups).  Without a twist in the log: all, still, moving."""
    out = {"all": np.ones(rp.W, bool)}
    legs = np.maximum(np.abs(rp.qd0.cpu().numpy()[:, LEGS]).max(1),
                      np.abs(rp.obs_qd.cpu().numpy()[:, :, LEGS]).max((1, 2)))
    quiet = legs < STILL_SPEED
    tw = [rp.logs[w.log].extra.get("twist") for w in rp.windows]
    if all(t is not None for t in tw):
        moving = np.array([np.abs(t[w.start]).max() > 1e-3 for t, w in zip(tw, rp.windows)])
        out[STILL_GROUP] = ~moving & quiet
        out[WALK_GROUP] = moving
        out[KICK_GROUP] = ~moving & ~quiet
    else:
        out[STILL_GROUP] = quiet
        out["moving"] = ~quiet
    return out


def still_drift(traj: dict, obs: dict, rp: Replay, mask: np.ndarray) -> dict:
    """In the still windows, per leg joint group (hip, knee, ankle): the median |error| at the window's end [deg],
    its median sign against the robot's own PD error (target - angle; > 0: the engine's joint ended nearer the target
    than the robot's, i.e. it gives way less under the load), and how often that happened."""
    if mask.sum() < 4:
        return {}
    end = rp.H - 1
    e = np.degrees(traj["theta"][mask, end] - obs["theta"][mask, end])                 # (n, 22)
    p = np.degrees(rp.target.cpu().numpy()[mask, end] - obs["theta"][mask, end])      # the robot's PD error
    out = {}
    for name, js in (("hip", [10, 11, 12, 16, 17, 18]), ("knee", [13, 19]), ("ankle", [14, 15, 20, 21])):
        ee, pp = e[:, js].ravel(), p[:, js].ravel()
        big = np.abs(pp) > 0.2
        toward = np.sign(ee[big]) == np.sign(pp[big])
        out[name] = {"abs_end_deg": float(np.nanmedian(np.abs(ee))),
                     "toward_target": float(np.mean(toward)) if big.any() else float("nan"),
                     "robot_pd_error_deg": float(np.nanmedian(np.abs(pp)))}
    return out


# ------------------------------------------------------------------ one run
def prepare_start(log, start: str, feet: calib.Feet | None = None) -> tuple:
    """The log as the replay sees it for a start mode, and what was done to it: (log, info)."""
    if start not in START_MODES:
        raise ValueError(f"start {start!r}: one of {START_MODES}")
    info = {"mode": start}
    if start == "as-before":
        return log, info
    feet = feet or calib.Feet()
    T = float(log.t[-1])
    off = calib.parallel_soles(log, feet=feet)
    halves = [calib.parallel_soles(log, feet=feet, t_range=r) for r in ((0.0, T / 2), (T / 2, T + 1.0))]
    info["parallel"] = off.summary()
    info["parallel_halves_deg"] = [np.degrees(h.joint).round(3).tolist() for h in halves]
    return calib.apply(log, off), info


def turn_summary(infos: list[dict], groups: dict) -> dict:
    """The per-window flat-feet turns: how big, how often applied, per window group."""
    turn = np.array([i["turn_deg"] for i in infos])
    applied = np.array([i["applied"] for i in infos])
    out = {}
    for name, mask in groups.items():
        if mask.sum():
            t = turn[mask]
            out[name] = {"windows": int(mask.sum()), "applied": float(applied[mask].mean()),
                         "turn_median_deg": float(np.median(t)), "turn_p90_deg": float(np.percentile(t, 90)),
                         "turn_max_deg": float(t.max())}
    return out


def analyse(log, horizon_s: float, delay_ms: float, sensitivity: bool = True, max_windows: int | None = None,
            keep: dict | None = None, start: str = "flat-feet") -> dict:
    """keep: a dict that gets the replay, the trajectories and the per-step measures (the Rerun view needs them)."""
    H = round(horizon_s / log.step_dt)
    feet = calib.Feet() if start == "flat-feet" else None
    log, start_info = prepare_start(log, start, feet)
    windows = make_windows([log], H, max_count=max_windows)
    if len(windows) < 8:
        raise ValueError(f"{log.name}: only {len(windows)} windows")
    space = ParamSpace()
    rp = Replay([log], windows, space, 1, H, delay_ms=delay_ms)
    turns = flat_turns(rp, feet) if start == "flat-feet" else None
    obs = observed(rp)
    jw = rp.joint_w.cpu().numpy()
    has_tau = log.tau is not None
    traj = all_trajectories(rp, delay_ms, sink=start == "flat-feet")
    g_rob = start_gravity(rp, False)
    g_eng = start_gravity(rp, True)
    m = {r: measures(traj[r], obs, jw, (g_eng if r in ENGINES else g_rob, g_rob)) for r in ROWS}
    wm = {r: window_means(m[r]) for r in ROWS}
    groups = phases(rp)
    if turns is not None:
        start_info["turns"] = turn_summary(turns, groups)
    start_info["mujoco_resting_depth_mm"] = float(traj["MuJoCo"].get("resting_depth_mm", 0.0))
    if keep is not None:
        keep.update(rp=rp, traj=traj, obs=obs, measures=m, log=log, turns=turns, start=start_info)
    names = [k for k in MEASURES if has_tau or k != "torque"]
    marks = [(s, round(s / log.step_dt) - 1) for s in MARKS_S if round(s / log.step_dt) <= H]
    res = {"run": log.name, "task": log.extra.get("task", log.meta.kind), "windows": rp.W, "horizon_s": H * log.step_dt,
           "step_s": log.step_dt, "delay_ms": delay_ms, "seconds": float(log.t[-1]), "has_torque": has_tau,
           "start": start_info, "window_t": [float(log.t[w.start]) for w in windows],
           "window_group": [next(g for g in groups if g != "all" and groups[g][i]) if len(groups) > 1 else "all"
                            for i in range(rp.W)],
           "t_ms": ((np.arange(H) + 1) * log.step_dt * 1e3).tolist(), "curves": {}, "rows": {}, "paired": {},
           "groups": {}, "diverged": {r: float(np.isnan(wm[r]["joints"]).mean()) for r in ROWS}}
    for r in ROWS:
        res["curves"][r] = {k: quartiles(m[r][k]) for k in names}
        res["rows"][r] = {k: {"median": float(np.nanmedian(wm[r][k])), "mean": float(np.nanmean(wm[r][k])),
                              **{f"at_{int(s * 1e3)}ms": float(np.nanmedian(m[r][k][:, i])) for s, i in marks}}
                          for k in names if not (k == "torque" and r not in ENGINES)}
        res["groups"][r] = {g: {f"at_{int(s * 1e3)}ms": mean_interval(m[r][f"group/{g}"][:, i]) for s, i in marks
                                if s in (0.1, 0.4)} for g in GROUPS}
    res["phase_rows"] = {}
    for ph_name, mask in groups.items():
        if mask.sum() < 8:
            continue
        res["phase_rows"][ph_name] = {r: {k: float(np.nanmedian(wm[r][k][mask])) for k in names
                                          if not (k == "torque" and r not in ENGINES)} for r in ROWS}
        res["phase_curves"] = res.get("phase_curves", {})
        res["phase_curves"][ph_name] = {r: {k: np.nanmedian(m[r][k][mask], 0).tolist() for k in names
                                            if not (k == "torque" and r not in ENGINES)} for r in ROWS}
        res["paired"][ph_name] = {"windows": int(mask.sum())}
        for k in names:
            res["paired"][ph_name][k] = paired(wm["pH nominal"][k][mask], wm["MuJoCo"][k][mask])
    res["window_diff"] = {k: (wm["pH nominal"][k] - wm["MuJoCo"][k]).tolist() for k in names}
    res["window_means"] = {r: {k: wm[r][k].tolist() for k in names if not (k == "torque" and r not in ENGINES)}
                           for r in ROWS}
    res["block_check"] = {k: {str(b): mean_interval(res["window_diff"][k], block=b) for b in BLOCKS}
                          for k in ("joints", "tilt")}
    if STILL_GROUP in groups:
        res["still_drift"] = {r: still_drift(traj[r], obs, rp, groups[STILL_GROUP]) for r in ENGINES}
    if "mahony" in log.extra:
        cos = np.clip((log.extra["mahony"] * log.grav).sum(1), -1, 1)
        a = np.degrees(np.arccos(cos))
        res["tilt_filter_deg"] = {"median": float(np.median(a)), "p95": float(np.percentile(a, 95)), "max": float(a.max())}
    if has_tau:
        from .motor_check import load_motors, metrics

        mot = load_motors()
        res["torque_joint"] = {r: torque_per_joint(traj[r], obs).tolist() for r in ENGINES}
        # the shape and the size apart: correlation, and the slope a of tau_est ~ a x the engine's torque
        res["torque_fit"] = {r: {k: v for k, v in metrics(traj[r]["tau"].reshape(-1, 22), obs["tau"].reshape(-1, 22)).items()
                                 if k in ("corr", "slope", "bias")} for r in ENGINES}
        res["joint_names"] = list(rp.plant.model.joint_names)
        en = {"robot": energy(obs, log.step_dt, mot.R, mot.k)}
        en.update({r: energy(traj[r], log.step_dt, mot.R, mot.k) for r in ENGINES})
        res["energy"] = {who: {k: v.tolist() for k, v in e.items()} for who, e in en.items()}
        res["energy_total"] = {who: {k: float(np.nansum(np.where(np.isfinite(en["MuJoCo"][k] + en["pH nominal"][k]), v, np.nan)))
                                     for k, v in e.items()} for who, e in en.items()}
    if sensitivity:
        res["sensitivity"] = sensitivity_scan(log, windows, H, delay_ms, has_tau, start, feet)
        if has_tau:
            slope = float(np.median([res["torque_fit"][r]["slope"][j] for r in ENGINES for j in HIP_KNEE]))
            res["what_if"] = what_ifs(rp, obs, jw, groups, delay_ms, start == "flat-feet", round(slope, 2))
    return res


def what_ifs(rp: Replay, obs: dict, jw: np.ndarray, groups: dict, delay_ms: float, flat: bool, slope: float,
             friction: float = 1.0) -> dict:
    """Not part of the comparison: what moves the engines' error, above all while the robot stands?  Per window group
    the median joint, tilt and torque error (a) with the motors' torque scaled by tau_est's slope (both engines:
    'the motors give only that much'), (b) MuJoCo with Coulomb friction in the leg joints (the pH plant has no such
    term)."""
    g0 = (start_gravity(rp, True), start_gravity(rp, False))

    def table(t: dict) -> dict:
        out = {}
        for g, mask in groups.items():
            if mask.sum() >= 8:
                out[g] = {r: {k: float(np.nanmedian(v[mask])) for k, v in window_means(measures(t[r], obs, jw, g0)).items()
                              if k in ("joints", "tilt", "torque")} for r in t}
        return out
    kp, kd = rp.base_kp.clone(), rp.base_kd.clone()
    rp.base_kp, rp.base_kd = kp * slope, kd * slope
    scaled = table({"MuJoCo": mujoco_trajectories(rp, delay_ms, sink=flat), "pH nominal": ph_trajectories(rp)})
    rp.base_kp, rp.base_kd = kp, kd
    rubbing = table({"MuJoCo": mujoco_trajectories(rp, delay_ms, sink=flat, leg_friction=friction)})
    return {"slope": slope, "friction_nm": friction,
            "rows": {f"motor torque × {slope:.2f} (tau_est's slope), both engines": scaled,
                     f"MuJoCo with {friction:g} N m gear friction in the legs (the pH plant has none)": rubbing}}


def sensitivity_scan(log, windows, H: int, delay_ms: float, has_tau: bool, start: str = "as-before",
                     feet: calib.Feet | None = None, every: int = 4) -> dict:
    """The headline numbers on every 4th window: against the actuation delay, with the trunk's start velocity set to
    zero instead of the stance-point guess, and (flat-feet) without the per-window turn."""
    sub = windows[::every]
    rp = Replay([log], sub, ParamSpace(), 1, H, delay_ms=delay_ms)
    flat = start == "flat-feet"
    if flat:
        flat_turns(rp, feet or calib.Feet())
    obs, jw = observed(rp), rp.joint_w.cpu().numpy()
    keys = ("joints", "tilt") + (("torque",) if has_tau else ())

    def score(stance: bool, d: float) -> dict:
        rp.stance_velocity = stance
        rp.delay_ms = d
        t = {"MuJoCo": mujoco_trajectories(rp, d, sink=flat), "pH nominal": ph_trajectories(rp)}
        g0 = (start_gravity(rp, True), start_gravity(rp, False))
        return {r: {k: float(np.nanmedian(window_means(measures(t[r], obs, jw, g0))[k])) for k in keys} for r in ENGINES}

    out = {"windows": len(sub), "delays_ms": list(SENS_DELAYS), "delay": [score(True, d) for d in SENS_DELAYS],
           "stance": score(True, delay_ms), "zero_velocity": score(False, delay_ms)}
    if flat:
        rp.quat0 = rp.quat0_meas.clone()          # the parallel soles, but the robot's own attitude at every start
        out["no_turn"] = score(True, delay_ms)
    rp.stance_velocity = True
    return out


# ------------------------------------------------------------------ diagrams
def _axes(ax, title=None, ylabel=None, xlabel=None):
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    if title:
        ax.set_title(title, loc="left", color=INK, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED)
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED)


def plots(out: Path, res: dict, m1: dict | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.asarray(res["t_ms"])
    names = list(next(iter(res["curves"].values())))
    # 1. error against the horizon
    fig, axes = plt.subplots(1, len(names), figsize=(4.4 * len(names), 3.8), squeeze=False)
    for ax, k in zip(axes[0], names):
        for r in ROWS:
            c = res["curves"][r].get(k)
            if c is None or not np.isfinite(c["median"]).any():
                continue
            ax.plot(t, c["median"], lw=1.8, label=r, **STYLE[r])
            if r in ENGINES:
                ax.fill_between(t, c["q25"], c["q75"], color=STYLE[r]["color"], alpha=0.15, lw=0)
        top = max(np.nanmax(res["curves"][r][k]["q75"]) for r in ENGINES)
        ax.set_ylim(0, 1.6 * top)
        _axes(ax, MEASURES[k], xlabel="prediction horizon [ms]")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(f"{res['run']}: error against the robot, median of {res['windows']} windows (band: quartiles)",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "horizon.png", dpi=150)
    plt.close(fig)
    # 2. joint groups at 100 ms and 400 ms
    marks = list(next(iter(res["groups"]["MuJoCo"].values())))
    fig, axes = plt.subplots(1, len(marks), figsize=(5.6 * len(marks), 3.8), squeeze=False, sharey=False)
    groups = list(GROUPS)
    x = np.arange(len(groups))
    for ax, mk in zip(axes[0], marks):
        for i, r in enumerate(ENGINES):
            v = np.array([res["groups"][r][g][mk] for g in groups])
            ax.bar(x + (i - 0.5) * 0.38, v[:, 0], 0.34, color=STYLE[r]["color"], label=r,
                   yerr=(v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]), ecolor=INK, error_kw=dict(lw=1))
        for j, g in enumerate(groups):
            b = res["groups"]["constant velocity"][g][mk][0]
            ax.plot([j - 0.42, j + 0.42], [b, b], color=MUTED, ls="--", lw=1.4, label="constant velocity" if j == 0 else None)
        ax.set_xticks(x, groups, color=MUTED)
        _axes(ax, f"after {mk[3:-2]} ms", "joint RMSE [deg], mean and 95 % interval")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "groups.png", dpi=150)
    plt.close(fig)
    # 3. paired differences
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 3.6), squeeze=False)
    for ax, k in zip(axes[0], names):
        d = np.asarray(res["window_diff"][k], dtype=float)
        d = d[np.isfinite(d)]
        lim = np.percentile(np.abs(d), 98) if len(d) else 1.0
        ax.hist(np.clip(d, -lim, lim), bins=41, range=(-lim, lim), color="#9a9892")
        p = res["paired"]["all"][k]
        ax.axvline(0, color=INK, lw=1)
        ax.axvline(p["mean"], color=COLOURS["pH nominal"], lw=1.8)
        ax.axvspan(p["low"], p["high"], color=COLOURS["pH nominal"], alpha=0.2, lw=0)
        _axes(ax, f"{MEASURES[k]}\npH closer in {100 * p['a_better']:.0f} % of windows",
              xlabel="pH − MuJoCo per window (left of 0: pH closer)")
    fig.tight_layout()
    fig.savefig(out / "paired.png", dpi=150)
    plt.close(fig)
    # 4. torque per joint
    if res["has_torque"]:
        fig, ax = plt.subplots(figsize=(9.5, 3.8))
        jn = [n.lstrip("A").replace("_", " ").lower() for n in res["joint_names"]]
        x = np.arange(len(LEGS))
        for i, r in enumerate(ENGINES):
            ax.bar(x + (i - 0.5) * 0.38, np.asarray(res["torque_joint"][r])[LEGS], 0.34, color=STYLE[r]["color"],
                   label=f"{r}, free replay")
        if m1:
            for i, (key, r) in enumerate((("mujoco", "MuJoCo"), ("ph", "pH nominal"))):
                v = np.asarray(m1["per_joint"][key]["rmse"])[LEGS]
                ax.scatter(x + (i - 0.5) * 0.38, v, color=INK, s=14, zorder=3,
                           label="motor model alone (robot's motion forced)" if i == 0 else None)
        ax.set_xticks(x, [jn[j] for j in LEGS], rotation=30, ha="right", color=MUTED)
        _axes(ax, "Torque error against the motors' own estimate (ankles: linkage, not comparable)", "RMSE [N m]")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "torque.png", dpi=150)
        plt.close(fig)
        # 5. energy per window
        kinds = (("work_pos", "positive motor work [J]"), ("copper", "copper loss [J]"))
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
        for ax, (k, label) in zip(axes, kinds):
            rob = np.asarray(res["energy"]["robot"][k])
            for r in ENGINES:
                ax.scatter(rob, res["energy"][r][k], s=9, alpha=0.55, color=STYLE[r]["color"], label=r, lw=0)
            top = float(np.nanpercentile(rob, 99)) * 1.3
            ax.plot([0, top], [0, top], color=INK, lw=1)
            ax.set_xlim(0, top)
            ax.set_ylim(0, top)
            _axes(ax, f"{label} per window, hips and knees", "engine", "robot (from tau_est)")
        axes[0].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "energy.png", dpi=150)
        plt.close(fig)
    # 6. sensitivity
    s = res.get("sensitivity")
    if s:
        keys = list(s["stance"]["MuJoCo"])
        fig, axes = plt.subplots(1, len(keys), figsize=(4.4 * len(keys), 3.6), squeeze=False)
        for ax, k in zip(axes[0], keys):
            for r in ENGINES:
                ax.plot(s["delays_ms"], [d[r][k] for d in s["delay"]], marker="o", ms=4, label=r, **STYLE[r])
                ax.scatter([res["delay_ms"]], [s["zero_velocity"][r][k]], marker="x", s=50, color=STYLE[r]["color"],
                           label=f"{r}, trunk velocity 0", zorder=3)
            ax.set_ylim(0, None)
            _axes(ax, MEASURES[k], xlabel="actuation delay [ms]")
        axes[0, 0].legend(frameon=False, fontsize=8)
        fig.suptitle(f"How the choices move the result (median of {s['windows']} windows)", x=0.01, ha="left",
                     fontsize=10, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(out / "sensitivity.png", dpi=150)
        plt.close(fig)


# ------------------------------------------------------------------ report
def verdict(p: dict) -> str:
    if p["high"] < 0:
        return "pH closer"
    if p["low"] > 0:
        return "MuJoCo closer"
    return "no difference shown"


def report(res: dict) -> str:
    names = [k for k in MEASURES if k in res["rows"]["MuJoCo"]]
    unit = {"joints": "deg", "speed": "rad/s", "gyro": "rad/s", "tilt": "deg", "torque": "N m"}
    st = res.get("start", {"mode": "as-before"})
    flat = st["mode"] == "flat-feet"
    how = ("the sole corners that touch the floor laid flat (see 'The start' below)" if flat else
           "the robot's own roll and pitch and the lowest point on the floor (the start used until 20 Nov 2026)")
    L = [f"# MuJoCo and the pH plant against the real robot: {res['run']}", "",
         f"Task {res['task']}, {res['seconds']:.0f} s of robot data, {res['windows']} windows of "
         f"{res['horizon_s']:.1f} s (a new one every 0.1 s). Start **{st['mode']}**: both engines start from the "
         f"robot's measured joints, {how}. They get the commands and gains the motor drivers received, "
         f"{res['delay_ms']:g} ms late. Lower is better.", "",
         "![horizon](horizon.png)", "",
         "## Error over the window", "",
         "Median over the windows of each window's mean error." + (" Tilt: the error of its change since the window "
                                                                  "start." if flat else ""), "",
         "| Model | " + " | ".join(f"{k} [{unit[k]}]" for k in names) + " | Diverged |", "|---|" + "---|" * (len(names) + 1)]
    for r in ROWS:
        row = res["rows"][r]
        L.append(f"| {r} | " + " | ".join(f"{row[k]['median']:.3f}" if k in row else "–" for k in names)
                 + f" | {100 * res['diverged'][r]:.0f} % |")
    for ph_name, rows in res["phase_rows"].items():
        if ph_name == "all":
            continue
        L += ["", f"Only the windows \"{ph_name}\" ({res['paired'][ph_name]['windows']}):", "",
              "| Model | " + " | ".join(f"{k} [{unit[k]}]" for k in names) + " |", "|---|" + "---|" * len(names)]
        L += [f"| {r} | " + " | ".join(f"{rows[r][k]:.3f}" if k in rows[r] else "–" for k in names) + " |" for r in ROWS]
    if flat:
        L += report_start(st)
    if res.get("before"):
        L += report_before(res, names, unit)
    marks = [k for k in res["rows"]["MuJoCo"]["joints"] if k.startswith("at_")]
    L += ["", "## Joint angle error at fixed horizons [deg]", "", "Median over the windows.", "",
          "| Model | " + " | ".join(f"{k[3:-2]} ms" for k in marks) + " |", "|---|" + "---|" * len(marks)]
    for r in ROWS:
        L.append(f"| {r} | " + " | ".join(f"{res['rows'][r]['joints'][k]:.2f}" for k in marks) + " |")
    L += ["", "![groups](groups.png)", "", "## pH nominal against MuJoCo, window by window", "",
          "Negative: pH is closer to the robot. The interval resamples blocks of four windows.", "",
          "| Windows | Measure | pH closer in | Mean pH − MuJoCo | 95 % interval | Reading |", "|---|---|---|---|---|---|"]
    for ph_name, block in res["paired"].items():
        for k in names:
            p = block[k]
            L.append(f"| {ph_name} ({block['windows']}) | {k} [{unit[k]}] | {100 * p['a_better']:.0f} % | {p['mean']:+.3f} | "
                     f"{p['low']:+.3f} … {p['high']:+.3f} | {verdict(p)} |")
    if "block_check" in res:
        L += ["", "Does the interval depend on the block length? (all windows; neighbouring windows overlap, so blocks "
              "of windows are resampled together)", "", "| Measure | " + " | ".join(f"block {b}" for b in BLOCKS) + " |",
              "|---|" + "---|" * len(BLOCKS)]
        for k, bc in res["block_check"].items():
            L.append(f"| {k} | " + " | ".join(f"{bc[str(b)][1]:+.3f} … {bc[str(b)][2]:+.3f}" for b in BLOCKS) + " |")
    L += ["", "![paired](paired.png)", ""]
    if res.get("still_drift"):
        L += ["## While the robot stands still: where the engines go", "",
              "At the end of the still windows, per leg joint group: the median joint error, and how often the "
              "engine's joint ended nearer the commanded target than the robot's (it gives way less under the "
              "weight). Above 50 %: the engine is stiffer or stronger than the robot (the motors' torque, gear "
              "friction).", "",
              "| Group | robot's PD error [deg] | MuJoCo: error [deg] | MuJoCo: nearer the target | pH: error [deg] | "
              "pH: nearer the target |", "|---|---|---|---|---|---|"]
        for g in ("hip", "knee", "ankle"):
            mj, ph = res["still_drift"]["MuJoCo"].get(g), res["still_drift"]["pH nominal"].get(g)
            if mj and ph:
                L.append(f"| {g} | {mj['robot_pd_error_deg']:.2f} | {mj['abs_end_deg']:.2f} | {100 * mj['toward_target']:.0f} % | "
                         f"{ph['abs_end_deg']:.2f} | {100 * ph['toward_target']:.0f} % |")
        L.append("")
    if res["has_torque"]:
        jn, tj, tf = res["joint_names"], res["torque_joint"], res.get("torque_fit")
        L += ["## Torque against the motors' own estimate", "",
              "RMSE over all windows and steps, free replay (the table above has the median window, so its numbers are "
              "lower: a few windows with large torques carry this one). Correlation: the shape; slope a of "
              "tau_est ≈ a × the engine's torque: the size (below 1: the robot's estimate is smaller than the "
              "engine's torque). Ankle motors act through a parallel linkage, so their rows are not a model error.", ""]
        if tf:
            L += ["| Joint | MuJoCo RMSE [N m] | pH RMSE [N m] | MuJoCo corr. | pH corr. | MuJoCo slope | pH slope |",
                  "|---|---|---|---|---|---|---|"]
            L += [f"| {jn[j].lstrip('A')} | {tj['MuJoCo'][j]:.2f} | {tj['pH nominal'][j]:.2f} | "
                  f"{tf['MuJoCo']['corr'][j]:.2f} | {tf['pH nominal']['corr'][j]:.2f} | {tf['MuJoCo']['slope'][j]:.2f} | "
                  f"{tf['pH nominal']['slope'][j]:.2f} |" for j in LEGS]
        else:
            L += ["| Joint | MuJoCo [N m] | pH nominal [N m] |", "|---|---|---|"]
            L += [f"| {jn[j].lstrip('A')} | {tj['MuJoCo'][j]:.2f} | {tj['pH nominal'][j]:.2f} |" for j in LEGS]
        e = res["energy_total"]
        L += ["", "![torque](torque.png)", "", "## Energy, hips and knees, summed over the windows", "",
              "The robot's values use tau_est. Windows overlap, so the sums are not the run's energy; the ratio to the "
              "robot is what counts. With tau_est about 0.8 × the engines' torque (slopes above), the robot's column "
              "is too small by about that much: these ratios say more about tau_est than about the physics.", "",
              "| | Positive motor work [J] | Net motor work [J] | Copper loss [J] |", "|---|---|---|---|"]
        for who in ("robot",) + ENGINES:
            row = f"| {who} | {e[who]['work_pos']:.0f} | {e[who]['work_net']:.0f} | {e[who]['copper']:.0f} |"
            if who != "robot":
                row = (f"| {who} | {e[who]['work_pos']:.0f} (×{e[who]['work_pos'] / e['robot']['work_pos']:.2f}) | "
                       f"{e[who]['work_net']:.0f} | {e[who]['copper']:.0f} (×{e[who]['copper'] / e['robot']['copper']:.2f}) |")
            L.append(row)
        L += ["", "![energy](energy.png)", ""]
    if res.get("what_if"):
        L += report_what_if(res)
    s = res.get("sensitivity")
    if s:
        keys = list(s["stance"]["MuJoCo"])
        L += ["## How the choices move the result", "", f"Median of {s['windows']} windows (every 4th).", "",
              "| Setting | " + " | ".join(f"{r}: {k}" for k in keys for r in ENGINES) + " |",
              "|---|" + "---|" * (len(keys) * 2)]
        for d, sc in zip(s["delays_ms"], s["delay"]):
            L.append(f"| delay {d:g} ms | " + " | ".join(f"{sc[r][k]:.3f}" for k in keys for r in ENGINES) + " |")
        L.append(f"| delay {res['delay_ms']:g} ms, trunk velocity 0 | "
                 + " | ".join(f"{s['zero_velocity'][r][k]:.3f}" for k in keys for r in ENGINES) + " |")
        if "no_turn" in s:
            L.append(f"| delay {res['delay_ms']:g} ms, soles parallel but no per-window turn | "
                     + " | ".join(f"{s['no_turn'][r][k]:.3f}" for k in keys for r in ENGINES) + " |")
        L += ["", "![sensitivity](sensitivity.png)", ""]
    if "tilt_filter_deg" in res:
        f = res["tilt_filter_deg"]
        L += ["## Check of the tilt filter", "",
              f"The old CSV logs have no attitude, so mbrl/data.py estimates it from gyro and accelerometer. On this run "
              f"that estimate is {f['median']:.1f} deg (median), {f['p95']:.1f} deg (95 %) and at most {f['max']:.1f} deg "
              "away from the robot's own roll and pitch.", ""]
    L += ["## Limits", "",
          "- One run. The intervals are over windows of this run, not over runs or robots.",
          "- The trunk's height and linear velocity are guessed from the lowest contact point (mbrl/state.py); the "
          "plant starts with the robot's weight on its floor springs" + (", MuJoCo at its own resting depth." if flat else "."),
          "- Resolution of the method (tests/test_sim2real.py): an engine replaying a log it made itself is off by "
          "about 0.01 deg (MuJoCo) and 0.05 deg (pH plant, whose contact friction state is not in a log). Differences "
          "of that size say nothing.",
          "- Both engines run without self-collision and with nominal parameters; nothing is fitted to this run"
          + (" (the start's ankle offsets and turns are a calibration of the robot's sensors, the same for both)."
             if flat else "."),
          "- tau_est is the motors' estimate from their current, not a torque sensor.", ""]
    return "\n".join(L)


def report_what_if(res: dict) -> list[str]:
    """The what-ifs beside the run's own numbers, per window group: joints and tilt."""
    wi = res["what_if"]
    gs = [g for g in (STILL_GROUP, WALK_GROUP, KICK_GROUP) if g in res["phase_rows"]]
    short = {STILL_GROUP: "still", WALK_GROUP: "walking", KICK_GROUP: "no twist, moving"}
    L = ["## What-ifs: why do the engines move while the robot stands? (not part of the comparison)", "",
         "Medians of the window means; joints / tilt [deg]. Each line changes one thing for the named engine(s).", "",
         "| Setting | engine | " + " | ".join(f"{short.get(g, g)}: joints | {short.get(g, g)}: tilt" for g in gs) + " |",
         "|---|---|" + "---|---|" * len(gs)]
    for r in ENGINES:
        L.append(f"| as run | {r} | " + " | ".join(f"{res['phase_rows'][g][r]['joints']:.3f} | {res['phase_rows'][g][r]['tilt']:.3f}"
                                                for g in gs) + " |")
    for name, tab in wi["rows"].items():
        for r in ENGINES:
            if all(r in tab.get(g, {}) for g in gs):
                L.append(f"| {name} | {r} | " + " | ".join(f"{tab[g][r]['joints']:.3f} | {tab[g][r]['tilt']:.3f}" for g in gs) + " |")
    L += ["", "Reading: if the motors simply gave less torque, scaling it would bring the joints closer; if the robot's "
          "joints stick (gear friction), a model without friction drifts whichever way its own equilibrium lies, and "
          "friction holds it.", ""]
    return L


def report_start(st: dict) -> list[str]:
    """The flat-feet start: the constant ankle offsets, the per-bout check, the per-window turns."""
    par = st.get("parallel", {})
    L = ["", "## The start: flat feet", ""]
    if par.get("kind") == "parallel":
        j = par["joint_deg"]
        h = st.get("parallel_halves_deg", [])
        L += ["While the robot stands still, both soles lie flat on the floor, but the logged joints and the robot's "
              "own attitude do not put them flat. Per still bout (seconds), the soles' tilt as logged [deg, roll / "
              "pitch]:", "",
              "| Bout | left | right | common (both feet) | left − right | robot's own roll / pitch | left − right after "
              "the offsets |", "|---|---|---|---|---|---|---|"]
        for b in par.get("bouts", []):
            f = lambda v: f"{v[0]:+.2f} / {v[1]:+.2f}"  # noqa: E731
            L.append(f"| {b['t0']:.1f}–{b['t1']:.1f} | {f(b['left'])} | {f(b['right'])} | {f(b['common'])} | "
                     f"{f(b['difference'])} | {f(b['imu'])} | {f(b.get('difference_after', [np.nan, np.nan]))} |")
        L += ["", "The difference between the feet is steady (an offset in the leg chain): constant ankle offsets make "
              f"the soles parallel, from {par['samples']} still samples: left ankle pitch {j[14]:+.2f}, roll {j[15]:+.2f}, "
              f"right ankle pitch {j[20]:+.2f}, roll {j[21]:+.2f} deg (angles and commands shift together, so the "
              "drivers' torque is unchanged)."
              + (f" From the first half alone: {h[0][14]:+.2f} / {h[0][15]:+.2f} / {h[0][20]:+.2f} / {h[0][21]:+.2f}; "
                 f"from the second: {h[1][14]:+.2f} / {h[1][15]:+.2f} / {h[1][20]:+.2f} / {h[1][21]:+.2f}." if len(h) == 2 else ""),
              "", "The common tilt moves from bout to bout with the robot's own roll estimate while the legs stay: the "
              "attitude estimate drifts. So at the start of every window that begins while the robot stands still, "
              "the trunk is turned until the sole corners that "
              f"touch the floor (within {calib.TOUCH_MM:g} mm of the lowest) lie in one plane; a turn above "
              f"{calib.CAP_DEG:g} deg, or with no whole sole on the floor (a foot on an edge), is not applied.", ""]
    else:
        L += ["No still part in this run: no ankle offsets.", ""]
    if st.get("turns"):
        L += ["| Windows | count | turn applied | turn median [deg] | 90 % [deg] | max [deg] |", "|---|---|---|---|---|---|"]
        for g, t in st["turns"].items():
            L.append(f"| {g} | {t['windows']} | {100 * t['applied']:.0f} % | {t['turn_median_deg']:.2f} | "
                     f"{t['turn_p90_deg']:.2f} | {t['turn_max_deg']:.1f} |")
    L += ["", f"MuJoCo's resting depth in its floor: {st.get('mujoco_resting_depth_mm', 0.0):.4f} mm (its floor is "
          "stiff: starting it just touching was already its resting state).", ""]
    return L


def report_before(res: dict, names: list[str], unit: dict) -> list[str]:
    """The same windows with the start used until 20 Nov (res['before']: that run's results) beside this one."""
    b = res["before"]
    L = ["", "## Before and after the start fix", "",
         f"Same windows, same measures; 'before' = start as-before ({b.get('note', '')}). Median of the window means.", ""]
    for ph_name in res["phase_rows"]:
        if ph_name not in b["phase_rows"]:
            continue
        L += [f"**{ph_name}** ({res['paired'][ph_name]['windows']} windows)", "",
              "| Measure | nothing moves | MuJoCo before | MuJoCo after | pH before | pH after |", "|---|---|---|---|---|---|"]
        for k in names:
            nb, na = b["phase_rows"][ph_name], res["phase_rows"][ph_name]
            if k not in na["MuJoCo"] or k not in nb["MuJoCo"]:
                continue
            still = f"{na['nothing moves'][k]:.3f}" if k in na["nothing moves"] else "–"
            L.append(f"| {k} [{unit[k]}] | {still} | {nb['MuJoCo'][k]:.3f} | {na['MuJoCo'][k]:.3f} | "
                     f"{nb['pH nominal'][k]:.3f} | {na['pH nominal'][k]:.3f} |")
        L.append("")
    pb, pa = b["paired"].get("all", {}).get("joints"), res["paired"]["all"]["joints"]
    if pb:
        L += [f"pH − MuJoCo, joints, all windows: before {pb['mean']:+.3f} ({pb['low']:+.3f} … {pb['high']:+.3f}), "
              f"after {pa['mean']:+.3f} ({pa['low']:+.3f} … {pa['high']:+.3f}) deg.", ""]
    return L


def main(argv=None) -> list[dict]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rec", nargs="+", help="run logs (.csv) with their .k1low beside them")
    ap.add_argument("--start", choices=START_MODES, default="flat-feet",
                    help="how the engines start a window (as-before: the start used until 20 Nov 2026)")
    ap.add_argument("--horizon", type=float, default=0.4, help="window length [s]")
    ap.add_argument("--delay-ms", type=float, default=7.0, help="actuation delay (motor check: 6-8 ms)")
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--no-sensitivity", action="store_true")
    ap.add_argument("--view", action="store_true", help="open the run with both engines in the Rerun viewer")
    ap.add_argument("--save", help="write the Rerun recording to this .rrd file instead (one run log)")
    ap.add_argument("--no-camera", action="store_true", help="Rerun: leave the camera and depth images out (small file)")
    ap.add_argument("--out", default=str(paths.REPO / "outputs" / "sim2real"))
    a = ap.parse_args(argv)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    done = []
    for rec in a.rec:
        log = lowlog.load(rec)
        keep = {} if a.view or a.save else None
        res = analyse(log, a.horizon, a.delay_ms, not a.no_sensitivity, a.max_windows, keep, start=a.start)
        run_dir = Path(a.out) / Path(rec).stem
        out = run_dir / a.start
        out.mkdir(parents=True, exist_ok=True)
        before = run_dir / "as-before" / "results.json"
        if a.start != "as-before" and before.exists():
            b = json.loads(before.read_text())
            if b.get("windows") == res["windows"] and b.get("window_t") == res["window_t"]:
                res["before"] = {"phase_rows": b["phase_rows"], "paired": b["paired"], "rows": b["rows"],
                                 "phase_curves": b.get("phase_curves"), "note": "same code, --start as-before"}
        m1_path = run_dir / "motor_check.json"
        plots(out, res, json.loads(m1_path.read_text()) if m1_path.exists() else None)
        (out / "results.json").write_text(json.dumps(res))
        (out / "report.md").write_text(report(res))
        p = res["paired"]["all"]["joints"]
        print(f"[sim2real] {log.name}, start {a.start}: {res['windows']} windows; joints MuJoCo "
              f"{res['rows']['MuJoCo']['joints']['median']:.3f} deg, pH {res['rows']['pH nominal']['joints']['median']:.3f} deg; "
              f"pH − MuJoCo {p['mean']:+.3f} ({p['low']:+.3f} … {p['high']:+.3f}); wrote {out}", flush=True)
        if keep is not None:
            from .sim2real_view import show

            show(Path(rec).resolve(), log, res, keep, report(res), a.save, spawn=a.save is None, camera=not a.no_camera)
        done.append(res)
    return done


if __name__ == "__main__":
    main()

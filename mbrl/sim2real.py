r"""Which physics predicts the real K1 better?  Full-rate robot logs replayed on MuJoCo and on the pH plant.

    python -m mbrl.sim2real <rec.csv>... [--horizon 0.4] [--delay-ms 7] [--out outputs/sim2real]
    python -m mbrl.sim2real <rec.csv> --view | --save run.rrd      # also in Rerun (mbrl/sim2real_view.py)

Each run log needs its full-rate file beside it (<rec>.csv.k1low).  Writes into <out>/<run>/: report.md, results.json
and six diagrams (horizon.png, groups.png, paired.png, torque.png, energy.png, sensitivity.png).

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
    tilt     angle between the predicted and the measured gravity direction [deg]
    torque   RMS error of the hip and knee torques against tau_est [N m] (ankles apart: their motors act through a
             parallel linkage, tau_est is not the joint torque)

Limits: the trunk's height and linear velocity are not measured (mbrl/state.py guesses them from the lowest contact
point); windows overlap in time, so the intervals resample blocks of windows; one run is one run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import lowlog, paths
from .params import GROUPS, ParamSpace
from .replay import Replay, make_windows
from .replay_mujoco import COLOURS, GRID, INK, MUTED, mujoco_trajectories, paired

ROWS = ("nothing moves", "constant velocity", "MuJoCo", "pH nominal")
ENGINES = ("MuJoCo", "pH nominal")
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


def nothing_moves(rp: Replay) -> dict:
    from .replay import projected_gravity

    W, H = rp.W, rp.H
    g0 = projected_gravity(rp.quat0.cpu()).numpy()
    return {"theta": np.repeat(rp.th0.cpu().numpy()[:, None], H, 1), "qd": np.zeros((W, H, 22)),
            "gyro": np.zeros((W, H, 3)), "grav": np.repeat(g0[:, None], H, 1), "tau": np.full((W, H, 22), np.nan)}


def constant_velocity(rp: Replay) -> dict:
    """Joints at their start speed; the trunk turns at its start rate (gravity in the trunk frame: g' = -w x g)."""
    from .replay import projected_gravity

    W, H = rp.W, rp.H
    t = (np.arange(H) + 1) * rp.step_dt
    th0, qd0, w = rp.th0.cpu().numpy(), rp.qd0.cpu().numpy(), rp.gyro0.cpu().numpy()
    g0 = projected_gravity(rp.quat0.cpu()).numpy()
    rate = np.linalg.norm(w, axis=1, keepdims=True)
    axis = -w / np.maximum(rate, 1e-12)                                    # (W,3)
    ang = (rate * t[None])[..., None]                                      # (W,H,1)
    a, g = axis[:, None], g0[:, None]
    grav = g * np.cos(ang) + np.cross(a, g) * np.sin(ang) + a * (a * g).sum(-1, keepdims=True) * (1 - np.cos(ang))
    return {"theta": th0[:, None] + qd0[:, None] * t[None, :, None], "qd": np.repeat(qd0[:, None], H, 1),
            "gyro": np.repeat(w[:, None], H, 1), "grav": grav, "tau": np.full((W, H, 22), np.nan)}


def ph_trajectories(rp: Replay) -> dict:
    o = rp.run(rp.space.neutral(1), keep_traj=True)
    return {k: o[k][0].cpu().numpy() for k in ("theta", "qd", "gyro", "grav", "tau")}


def all_trajectories(rp: Replay, delay_ms: float) -> dict:
    rp.delay_ms = delay_ms
    return {"nothing moves": nothing_moves(rp), "constant velocity": constant_velocity(rp),
            "MuJoCo": mujoco_trajectories(rp, delay_ms), "pH nominal": ph_trajectories(rp)}


# ------------------------------------------------------------------ measures
def measures(traj: dict, obs: dict, joint_w: np.ndarray) -> dict:
    """{measure: (W,H)} and {"group/<name>": (W,H)} [deg]; NaN where the engine diverged or nothing was measured."""
    e = traj["theta"] - obs["theta"]
    out = {"joints": np.degrees(np.sqrt((e * e * joint_w).sum(-1))),
           "speed": np.sqrt(((traj["qd"] - obs["qd"])[..., LEGS] ** 2).mean(-1)),
           "gyro": np.linalg.norm(traj["gyro"] - obs["gyro"], axis=-1)}
    cos = (traj["grav"] * obs["grav"]).sum(-1) / np.maximum(
        np.linalg.norm(traj["grav"], axis=-1) * np.linalg.norm(obs["grav"], axis=-1), 1e-12)
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
    """{name: (W,) bool} from the commanded twist at the window start; one phase 'all' without a twist."""
    out = {"all": np.ones(rp.W, bool)}
    tw = [rp.logs[w.log].extra.get("twist") for w in rp.windows]
    if all(t is not None for t in tw):
        moving = np.array([np.abs(t[w.start]).max() > 1e-3 for t, w in zip(tw, rp.windows)])
        out["walking (a twist is commanded)"] = moving
        out["no twist commanded"] = ~moving
    return out


# ------------------------------------------------------------------ one run
def analyse(log, horizon_s: float, delay_ms: float, sensitivity: bool = True, max_windows: int | None = None,
            keep: dict | None = None) -> dict:
    """keep: a dict that gets the replay, the trajectories and the per-step measures (the Rerun view needs them)."""
    H = round(horizon_s / log.step_dt)
    windows = make_windows([log], H, max_count=max_windows)
    if len(windows) < 8:
        raise ValueError(f"{log.name}: only {len(windows)} windows")
    space = ParamSpace()
    rp = Replay([log], windows, space, 1, H, delay_ms=delay_ms)
    obs = observed(rp)
    jw = rp.joint_w.cpu().numpy()
    has_tau = log.tau is not None
    traj = all_trajectories(rp, delay_ms)
    m = {r: measures(traj[r], obs, jw) for r in ROWS}
    wm = {r: window_means(m[r]) for r in ROWS}
    if keep is not None:
        keep.update(rp=rp, traj=traj, obs=obs, measures=m)
    names = [k for k in MEASURES if has_tau or k != "torque"]
    marks = [(s, round(s / log.step_dt) - 1) for s in MARKS_S if round(s / log.step_dt) <= H]
    res = {"run": log.name, "task": log.extra.get("task", log.meta.kind), "windows": rp.W, "horizon_s": H * log.step_dt,
           "step_s": log.step_dt, "delay_ms": delay_ms, "seconds": float(log.t[-1]), "has_torque": has_tau,
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
    for ph_name, mask in phases(rp).items():
        if mask.sum() < 8:
            continue
        res["phase_rows"][ph_name] = {r: {k: float(np.nanmedian(wm[r][k][mask])) for k in names
                                          if not (k == "torque" and r not in ENGINES)} for r in ROWS}
        res["paired"][ph_name] = {"windows": int(mask.sum())}
        for k in names:
            res["paired"][ph_name][k] = paired(wm["pH nominal"][k][mask], wm["MuJoCo"][k][mask])
    res["window_diff"] = {k: (wm["pH nominal"][k] - wm["MuJoCo"][k]).tolist() for k in names}
    if "mahony" in log.extra:
        cos = np.clip((log.extra["mahony"] * log.grav).sum(1), -1, 1)
        a = np.degrees(np.arccos(cos))
        res["tilt_filter_deg"] = {"median": float(np.median(a)), "p95": float(np.percentile(a, 95)), "max": float(a.max())}
    if has_tau:
        from .motor_check import load_motors

        mot = load_motors()
        res["torque_joint"] = {r: torque_per_joint(traj[r], obs).tolist() for r in ENGINES}
        res["joint_names"] = list(rp.plant.model.joint_names)
        en = {"robot": energy(obs, log.step_dt, mot.R, mot.k)}
        en.update({r: energy(traj[r], log.step_dt, mot.R, mot.k) for r in ENGINES})
        res["energy"] = {who: {k: v.tolist() for k, v in e.items()} for who, e in en.items()}
        res["energy_total"] = {who: {k: float(np.nansum(np.where(np.isfinite(en["MuJoCo"][k] + en["pH nominal"][k]), v, np.nan)))
                                     for k, v in e.items()} for who, e in en.items()}
    if sensitivity:
        res["sensitivity"] = sensitivity_scan(log, windows, H, delay_ms, has_tau)
    return res


def sensitivity_scan(log, windows, H: int, delay_ms: float, has_tau: bool, every: int = 4) -> dict:
    """The headline numbers on every 4th window: against the actuation delay, and with the trunk's start velocity
    set to zero instead of the stance-point guess."""
    sub = windows[::every]
    rp = Replay([log], sub, ParamSpace(), 1, H, delay_ms=delay_ms)
    obs, jw = observed(rp), rp.joint_w.cpu().numpy()
    keys = ("joints", "tilt") + (("torque",) if has_tau else ())

    def score(stance: bool, d: float) -> dict:
        rp.stance_velocity = stance
        rp.delay_ms = d
        t = {"MuJoCo": mujoco_trajectories(rp, d), "pH nominal": ph_trajectories(rp)}
        return {r: {k: float(np.nanmedian(window_means(measures(t[r], obs, jw))[k])) for k in keys} for r in ENGINES}

    out = {"windows": len(sub), "delays_ms": list(SENS_DELAYS), "delay": [score(True, d) for d in SENS_DELAYS],
           "stance": score(True, delay_ms), "zero_velocity": score(False, delay_ms)}
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
    L = [f"# MuJoCo and the pH plant against the real robot: {res['run']}", "",
         f"Task {res['task']}, {res['seconds']:.0f} s of robot data, {res['windows']} windows of "
         f"{res['horizon_s']:.1f} s (a new one every 0.1 s). Both engines start from the robot's measured state and get "
         f"the commands and gains the motor drivers received, {res['delay_ms']:g} ms late. Lower is better.", "",
         "![horizon](horizon.png)", "",
         "## Error over the window", "",
         "Median over the windows of each window's mean error.", "",
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
    L += ["", "![paired](paired.png)", ""]
    if res["has_torque"]:
        jn, tj = res["joint_names"], res["torque_joint"]
        L += ["## Torque against the motors' own estimate", "",
              "RMSE over all windows and steps, free replay (the table above has the median window, so its numbers are "
              "lower: a few windows with large torques carry this one). Ankle motors act through a parallel linkage, so their rows are not "
              "a model error.", "", "| Joint | MuJoCo [N m] | pH nominal [N m] |", "|---|---|---|"]
        L += [f"| {jn[j].lstrip('A')} | {tj['MuJoCo'][j]:.2f} | {tj['pH nominal'][j]:.2f} |" for j in LEGS]
        e = res["energy_total"]
        L += ["", "![torque](torque.png)", "", "## Energy, hips and knees, summed over the windows", "",
              "The robot's values use tau_est. Windows overlap, so the sums are not the run's energy; the ratio to the "
              "robot is what counts.", "", "| | Positive motor work [J] | Net motor work [J] | Copper loss [J] |",
              "|---|---|---|---|"]
        for who in ("robot",) + ENGINES:
            ratio = "" if who == "robot" else " (×{:.2f}, ×{:.2f})"
            row = f"| {who} | {e[who]['work_pos']:.0f} | {e[who]['work_net']:.0f} | {e[who]['copper']:.0f} |"
            if who != "robot":
                row = (f"| {who} | {e[who]['work_pos']:.0f} (×{e[who]['work_pos'] / e['robot']['work_pos']:.2f}) | "
                       f"{e[who]['work_net']:.0f} | {e[who]['copper']:.0f} (×{e[who]['copper'] / e['robot']['copper']:.2f}) |")
            L.append(row)
        L += ["", "![energy](energy.png)", ""]
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
          "plant starts with the robot's weight on its floor springs.",
          "- Resolution of the method (tests/test_sim2real.py): an engine replaying a log it made itself is off by "
          "about 0.01 deg (MuJoCo) and 0.05 deg (pH plant, whose contact friction state is not in a log). Differences "
          "of that size say nothing.",
          "- Both engines run without self-collision and with nominal parameters; nothing is fitted to this run.",
          "- tau_est is the motors' estimate from their current, not a torque sensor.", ""]
    return "\n".join(L)


def main(argv=None) -> list[dict]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rec", nargs="+", help="run logs (.csv) with their .k1low beside them")
    ap.add_argument("--horizon", type=float, default=0.4, help="window length [s]")
    ap.add_argument("--delay-ms", type=float, default=7.0, help="actuation delay (motor check: 6-8 ms)")
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--no-sensitivity", action="store_true")
    ap.add_argument("--view", action="store_true", help="open the run with both engines in the Rerun viewer")
    ap.add_argument("--save", help="write the Rerun recording to this .rrd file instead (one run log)")
    ap.add_argument("--out", default=str(paths.REPO / "outputs" / "sim2real"))
    a = ap.parse_args(argv)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    done = []
    for rec in a.rec:
        log = lowlog.load(rec)
        keep = {} if a.view or a.save else None
        res = analyse(log, a.horizon, a.delay_ms, not a.no_sensitivity, a.max_windows, keep)
        out = Path(a.out) / Path(rec).stem
        out.mkdir(parents=True, exist_ok=True)
        m1_path = out / "motor_check.json"
        plots(out, res, json.loads(m1_path.read_text()) if m1_path.exists() else None)
        (out / "results.json").write_text(json.dumps(res))
        (out / "report.md").write_text(report(res))
        p = res["paired"]["all"]["joints"]
        print(f"[sim2real] {log.name}: {res['windows']} windows; joints MuJoCo {res['rows']['MuJoCo']['joints']['median']:.2f} "
              f"deg, pH {res['rows']['pH nominal']['joints']['median']:.2f} deg; pH − MuJoCo {p['mean']:+.3f} "
              f"({p['low']:+.3f} … {p['high']:+.3f}); wrote {out}", flush=True)
        if keep is not None:
            from .sim2real_view import show

            show(Path(rec).resolve(), log, res, keep, report(res), a.save, spawn=a.save is None)
        done.append(res)
    return done


if __name__ == "__main__":
    main()

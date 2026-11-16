r"""Which physics predicts the real robot better?  The logged get-ups replayed on MuJoCo and on the pH plant.

    python -m mbrl.replay_mujoco [--fit fits/getup-both-v1.json] [--held-out fits/getup-v1.json] [--out outputs/engines]

Writes report.md and groups.png.  Both engines get the same windows, the same start states and the same commands
(mbrl/replay.py): 0.4 s from a logged state, then the prediction against the log.  So the pH rows here are the numbers
of mbrl.report.

The start state comes from Replay's own estimate (mbrl/state.py) on the pH plant: joints, gyro, tilt and the trunk's
velocity from the stance leg.  Only the height is MuJoCo's own: its collision shapes differ from the plant's contact
points, so the robot is put on the floor with MuJoCo's geometry.  MuJoCo uses the like-for-like setup of
rl/walk/backends.py and has no coil currents to start.

How to read the rows:
    nothing moves     the joints stay at their start: what any model has to beat
    MuJoCo            not fitted to the robot, no self-collision: the plant's replay has none either
    MuJoCo + self     the same with the model's self-collision: only to show how much that matters
    pH nominal        not fitted to the robot: the fair comparison with MuJoCo
    pH fit, same log  fitted to this log: the score flatters it
    pH fit, other log fitted to the other log only: the honest score of the fit

The mean loss can hang on one window, so the median stands beside it.  The paired table compares pH nominal with MuJoCo
window by window, joints and tilt apart.  Its confidence interval resamples blocks of windows, because neighbouring
windows overlap in time.

Limit: this is get-up, 0.4 s at a time, on two runs.  It says nothing direct about walking.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch

from . import paths
from .data import KNOWN, load_known
from .params import GROUPS, ParamSpace
from .replay import POLICY_DT, Replay, group_rmse_deg, make_windows, projected_gravity
from .state import set_state
from ph_k1.dynamics import point_jacobian
from ph_k1.model import load_k1
from rl.walk.backends import MuJoCoBackend

COLOURS = {"MuJoCo": "#eb6834", "pH nominal": "#2a78d6", "pH fit, same log": "#1baf7a", "pH fit, other log": "#6250d6"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def put_on_floor(m, d) -> None:
    """Lift or lower the robot until its lowest collision shape just touches the floor."""
    import mujoco

    margins = m.geom_margin.copy()
    m.geom_margin[:] = 1.0  # report the distance of everything within a metre
    mujoco.mj_kinematics(m, d)
    mujoco.mj_collision(m, d)
    floor = {g for g in range(m.ngeom) if m.geom_bodyid[g] == 0}
    gaps = [d.contact[i].dist for i in range(d.ncon) if d.contact[i].geom1 in floor or d.contact[i].geom2 in floor]
    m.geom_margin[:] = margins
    if gaps:
        d.qpos[2] -= min(gaps)


def start_states(rp: Replay, theta: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """(W,29) qpos and (W,28) qvel in MuJoCo's convention, from Replay's estimate of each window's start."""
    plant = rp.plant
    n = rp.B // rp.W
    rep = lambda x: rp._rep(x, n)  # noqa: E731
    set_state(plant, rep(rp.th0), rep(rp.qd0), rep(rp.quat0), rep(rp.gyro0), stance_velocity=rp.stance_velocity)
    kin = plant.kinematics()
    nu = plant.velocity(kin)
    J = point_jacobian(kin, torch.tensor([0]), kin.pos[:, 0][:, None])[:, 0]  # the trunk origin's world velocity
    v_world = (J @ nu[..., None]).squeeze(-1)
    qvel = torch.cat((v_world, rep(rp.gyro0), rep(rp.qd0)), -1)
    return plant.q[:rp.W].cpu().numpy().copy(), qvel[:rp.W].cpu().numpy().copy()


def floor_only(m) -> None:
    """Geoms of the robot collide with the floor but no longer with each other."""
    collides = (m.geom_contype != 0) | (m.geom_conaffinity != 0)
    robot = m.geom_bodyid != 0
    m.geom_contype[collides & robot], m.geom_conaffinity[collides & robot] = 2, 0
    m.geom_contype[collides & ~robot], m.geom_conaffinity[collides & ~robot] = 0, 2


def replay_mujoco(rp: Replay, delay_ms: float, self_collision: bool = False) -> dict:
    """Replay.run's output for MuJoCo: the same errors over the same windows."""
    import mujoco

    theta = ParamSpace().neutral(1)
    qpos0, qvel0 = start_states(rp, theta)
    b = MuJoCoBackend(load_k1(), delay_ms=delay_ms, gain_preset="standup", slew=rp.cfg.slew)
    m, d = b.m, b.d
    if not self_collision:
        floor_only(m)
    W, H = rp.W, rp.H
    hist, prev, target = rp.hist.cpu().numpy(), rp.prev_cmd.cpu().numpy(), rp.target.cpu().numpy()
    th_pred = np.full((W, H, 22), np.nan)
    gy_pred = np.full((W, H, 3), np.nan)
    gr_pred = np.full((W, H, 3), np.nan)
    for w in range(W):
        mujoco.mj_resetData(m, d)
        d.qpos[:] = qpos0[w]
        d.qvel[:] = qvel0[w]
        put_on_floor(m, d)
        mujoco.mj_forward(m, d)
        # oldest first: after the first append, ring[0] is the command held `lag` plant steps before the window
        b.ring = deque([hist[w, k].copy() for k in range(b.lag - 1, -1, -1)], maxlen=b.lag + 1)
        b.last_target = torch.as_tensor(prev[w], dtype=torch.float64)[None]
        for k in range(H):
            b.step(target[w, k])
            if not b.healthy():
                break
            r = b.read()
            th_pred[w, k], gy_pred[w, k] = r.theta, r.gyro
            q = torch.as_tensor(r.quat)[None]
            gr_pred[w, k] = projected_gravity(q)[0].numpy()
    T = lambda x: torch.as_tensor(x, dtype=torch.float64)  # noqa: E731
    return errors(rp, T(th_pred), T(gy_pred), T(gr_pred))


def mujoco_trajectories(rp: Replay, delay_ms: float, self_collision: bool = False) -> dict:
    """What Replay.run(keep_traj=True) keeps, from MuJoCo: theta, qd, gyro, grav, tau (W,H,...), NaN from the step a
    window diverges.  For full-rate logs (rp.low): the logged command and gains go straight to the motor driver, one
    command per replay step; the PD law and its torque limit are MuJoCoBackend.step's."""
    import mujoco

    qpos0, qvel0 = start_states(rp, ParamSpace().neutral(1))
    b = MuJoCoBackend(load_k1(), delay_ms=delay_ms, gain_preset="standup", slew=rp.cfg.slew)
    m, d = b.m, b.d
    if not self_collision:
        floor_only(m)
    if abs(m.opt.timestep - rp.h) > 1e-12:
        raise ValueError("MuJoCo's step and the plant's differ")
    W, H, sub = rp.W, rp.H, rp.substeps
    hist, target = rp.hist.cpu().numpy(), rp.target.cpu().numpy()
    kp, kd = rp.base_kp.cpu().numpy(), rp.base_kd.cpu().numpy()
    out = {"theta": np.full((W, H, 22), np.nan), "qd": np.full((W, H, 22), np.nan), "gyro": np.full((W, H, 3), np.nan),
           "grav": np.full((W, H, 3), np.nan), "tau": np.full((W, H, 22), np.nan)}
    for w in range(W):
        mujoco.mj_resetData(m, d)
        d.qpos[:] = qpos0[w]
        d.qvel[:] = qvel0[w]
        put_on_floor(m, d)
        mujoco.mj_forward(m, d)
        ring = deque([hist[w, k].copy() for k in range(b.lag - 1, -1, -1)], maxlen=b.lag + 1)
        for k in range(H):
            for _ in range(sub):
                ring.append(target[w, k])
                q, v = d.qpos[7:], d.qvel[6:]
                tau = kp[w] * (ring[0] - q) - kd[w] * v
                available = np.maximum((b.v_max - b.k * v * np.sign(tau)) * b.k / b.R, 0.0)
                cap = np.minimum(available, b.tau_max)
                d.ctrl[:] = np.clip(tau, -cap, cap)
                mujoco.mj_step(m, d)
            if not b.healthy():
                break
            out["theta"][w, k], out["qd"][w, k], out["gyro"][w, k] = d.qpos[7:], d.qvel[6:], d.qvel[3:6]
            out["grav"][w, k] = projected_gravity(torch.as_tensor(d.qpos[3:7])[None])[0].numpy()
            out["tau"][w, k] = d.ctrl
    return out


def errors(rp: Replay, th_pred, gy_pred, gr_pred) -> dict:
    """mbrl/replay.py's formulas, for one set of predictions (W,H,...)."""
    e_th = (th_pred - rp.obs_theta.cpu()) ** 2
    e_gy = ((gy_pred - rp.obs_gyro.cpu()) ** 2).sum(-1)
    e_gr = ((gr_pred - rp.obs_grav.cpu()) ** 2).sum(-1)
    per_joint = e_th.mean(1)
    loss = (per_joint * rp.joint_w.cpu()).sum(-1) + rp.w_gyro * e_gy.mean(1) + rp.w_grav * e_gr.mean(1)
    bad = ~torch.isfinite(loss)
    keep = ~bad
    joint_mse = per_joint[keep].mean(0, keepdim=True)
    return {"loss": float(loss[keep].mean()), "groups": {g: float(v[0]) for g, v in group_rmse_deg(joint_mse).items()},
            "gyro": float(e_gy.mean(1)[keep].mean()), "tilt": float(e_gr.mean(1)[keep].mean()),
            "diverged": float(bad.float().mean()), "windows": {
                "loss": loss.tolist(), "joints": (per_joint * rp.joint_w.cpu()).sum(-1).tolist(),
                "tilt": e_gr.mean(1).tolist()}}


def nothing_moves(rp: Replay) -> dict:
    """The joints stay at their start, the gyro at zero, the tilt where it is: no physics at all."""
    th = rp.th0.cpu()[:, None].expand(-1, rp.H, -1)
    g0 = projected_gravity(rp.quat0.cpu())[:, None].expand(-1, rp.H, -1)
    return errors(rp, th, torch.zeros(rp.W, rp.H, 3, dtype=torch.float64), g0)


def plant_rows(rp: Replay, theta: torch.Tensor) -> dict:
    """Replay.run for one parameter vector, as errors() reports it."""
    o = rp.run(theta[None], keep_traj=True)
    joints = ((o["theta"][0].cpu() - rp.obs_theta.cpu()) ** 2).mean(1)
    joints = (joints * rp.joint_w.cpu()).sum(-1)
    gyro = ((o["gyro"][0].cpu() - rp.obs_gyro.cpu()) ** 2).sum(-1).mean(1)
    loss = o["loss_windows"][0].cpu()
    tilt = (loss - joints - rp.w_gyro * gyro) / rp.w_grav  # the rest of mbrl/replay.py's loss
    return {"loss": float(o["loss"][0]), "groups": {g: float(o["group_rmse_deg"][g][0]) for g in GROUPS},
            "gyro": float(o["gyro_mse"][0]), "tilt": float(o["tilt_err"][0]), "diverged": float(o["diverged"][0]),
            "windows": {"loss": loss.tolist(), "joints": joints.tolist(), "tilt": tilt.tolist()}}


def paired(a, b, block: int = 4, draws: int = 20000, seed: int = 0) -> dict:
    """a - b per window: its mean, a 95 % interval from a moving block bootstrap, the share of windows where a is lower."""
    d = np.asarray(a) - np.asarray(b)
    d = d[np.isfinite(d)]
    n = len(d)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block + 1, (draws, int(np.ceil(n / block))))
    index = (starts[..., None] + np.arange(block)).reshape(draws, -1)[:, :n]
    low, high = np.percentile(d[index].mean(1), (2.5, 97.5))
    return {"mean": float(d.mean()), "low": float(low), "high": float(high), "a_better": float((d < 0).mean()), "n": n}


def plot(path: Path, table: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logs = list(table)
    groups = list(GROUPS)
    fig, axes = plt.subplots(1, len(logs), figsize=(6.2 * len(logs), 4.2), sharey=True, squeeze=False)
    for ax, name in zip(axes[0], logs):
        rows = [(k, v) for k, v in table[name].items() if k in COLOURS]
        width = 0.8 / len(rows)
        x = np.arange(len(groups))
        for i, (label, row) in enumerate(rows):
            ax.bar(x + (i - (len(rows) - 1) / 2) * width, [row["groups"][g] for g in groups], width * 0.86,
                   color=COLOURS[label], label=label)
        base = table[name]["nothing moves"]["groups"]
        for k, g in enumerate(groups):
            ax.plot([k - 0.42, k + 0.42], [base[g], base[g]], color=INK, lw=1.5, label="nothing moves" if k == 0 else None)
        ax.set_xticks(x, groups, color=MUTED)
        ax.set_title(name, loc="left", color=INK)
        ax.tick_params(colors=MUTED, length=0)
        ax.grid(True, axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    axes[0, 0].set_ylabel("joint RMSE over 0.4 s [deg]", color=MUTED)
    entries = {}
    for ax in axes[0]:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            entries.setdefault(label, handle)
    fig.legend(entries.values(), entries.keys(), frameon=False, fontsize=9, ncols=len(entries), loc="upper right")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", default=str(paths.REPO / "fits" / "getup-both-v1.json"), help="fitted to both logs")
    ap.add_argument("--held-out", default=str(paths.REPO / "fits" / "getup-v1.json"), help="fitted to one log")
    ap.add_argument("--logs", default=",".join(KNOWN))
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--out", default=str(paths.REPO / "outputs" / "engines"))
    a = ap.parse_args(argv)
    space = ParamSpace()
    neutral = space.neutral(1)[0]
    both = json.loads(Path(a.fit).read_text())
    one = json.loads(Path(a.held_out).read_text())
    one_logs = set(one.get("train", {}).get("logs", []))
    delay = float(space.delay_ms(neutral[None])[0])
    table = {}
    windows = {}
    for name in a.logs.split(","):
        logs = load_known([name])
        ws = make_windows(logs, a.horizon)
        rp = Replay(logs, ws, space, 1, a.horizon)
        windows[name] = len(ws)
        rows = {"nothing moves": nothing_moves(rp), "MuJoCo": replay_mujoco(rp, delay),
                "MuJoCo + self": replay_mujoco(rp, delay, self_collision=True),
                "pH nominal": plant_rows(rp, neutral),
                "pH fit, same log": plant_rows(rp, torch.tensor(both["mean"], dtype=torch.float64))}
        if name not in one_logs:
            rows["pH fit, other log"] = plant_rows(rp, torch.tensor(one["mean"], dtype=torch.float64))
        table[name] = rows
        print(f"[engines] {name}: {len(ws)} windows, MuJoCo loss {rows['MuJoCo']['loss']:.4f}, "
              f"pH nominal {rows['pH nominal']['loss']:.4f}", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(table, indent=1))
    plot(out / "groups.png", table)
    L = ["# MuJoCo and the pH plant against the real robot", "",
         f"Real get-up logs, {a.horizon * POLICY_DT:.1f} s predictions from logged states, nominal delay {delay:g} ms. "
         "Lower is better. RMSE of the joint angles per group in degrees; tilt is the squared error of the gravity "
         "direction, as in mbrl.report.", "", "![groups](groups.png)", ""]
    for name, rows in table.items():
        L += [f"## {name} ({windows[name]} windows)", "",
              "| Model | Mean loss | Median loss | " + " | ".join(GROUPS) + " | Tilt | Diverged |",
              "|---|---|---|" + "---|" * len(GROUPS) + "---|---|"]
        for label, r in rows.items():
            g = " | ".join(f"{r['groups'][k]:.2f}" for k in GROUPS)
            L.append(f"| {label} | {r['loss']:.4f} | {np.nanmedian(r['windows']['loss']):.4f} | {g} | {r['tilt']:.4f} | "
                     f"{r['diverged']:.2f} |")
        L.append("")
    L += ["## pH nominal against MuJoCo, window by window", "",
          "Negative: pH is closer to the robot. An interval that spans 0 means these windows cannot separate the two.", "",
          "| Log | Measure | pH closer in | Mean pH − MuJoCo | 95 % interval |", "|---|---|---|---|---|"]
    pairs = {}
    for name, rows in table.items():
        for key, label in (("loss", "Loss"), ("joints", "Joints only"), ("tilt", "Tilt only")):
            p = paired(rows["pH nominal"]["windows"][key], rows["MuJoCo"]["windows"][key])
            pairs[f"{name}/{key}"] = p
            L.append(f"| {name} | {label} | {100 * p['a_better']:.0f} % of {p['n']} | {p['mean']:+.3f} | "
                     f"{p['low']:+.3f} … {p['high']:+.3f} |")
    L.append("")
    (out / "paired.json").write_text(json.dumps(pairs, indent=1))
    (out / "report.md").write_text("\n".join(L))
    print(f"[engines] wrote {out}", flush=True)


if __name__ == "__main__":
    main()

r"""Sim-to-real step M1: the motor models of MuJoCo and of the pH plant against the K1's own torque estimate.

Input: a robot run log (k1_motion_test --rec) with its full-rate file beside it (<rec>.k1low, --low-rec): every
LowState (~500 Hz: q, dq, tau_est) and every LowCmd (~200 Hz: the target, kp and kd the motor drivers got).

Teacher forcing: the body is NOT simulated. Both motor models get the robot's measured joint angles and velocities and
the logged commands (shifted by an actuation delay d) and predict the joint torque; the reference is the robot's
tau_est (the motors' estimate from their current).

  MuJoCo, as RL_Core trains the K1 (rl_core/plant/torque_speed.py):
      tau = clip(kp (q_cmd - q) - kd dq, +-curve(|dq|)), at once; curve = effort up to the knee speed, then linear
      to 0 at the no-load speed (servo.MOTOR_CURVE).
  pH plant (Full Body ph_k1.firmware.motor_volts + ph_k1.motors):
      the same PD, capped by the DC motor line at this speed (it may brake harder than it drives), the driver's
      one-step current loop, the coil  L di/dt = v - R i - s k dq  with |v| <= V_NOM, tau = s k i. Small torque
      changes settle in one step; at the voltage limit the torque rises like tau_stall (1 - exp(-t / (L/R))), so a
      step of D N m takes -L/R ln(1 - D / tau_stall), e.g. 40 N m on the knee ~18 ms (L/R = ELECTRICAL_TIME = 50 ms,
      a placeholder, not identified).

Fitted per model (the rest is nominal): the delay d (it also absorbs the motors' measurement lag), and for
"pH, fitted coil" also L/R. The score is the RMSE over the 12 leg joints; every joint is reported.

    python -m mbrl.motor_check <rec.csv> [--save run.rrd] [--no-view]

Opens the run in k1_debug's Rerun view (robot, camera, full-rate plots) with these tabs added: measured vs MuJoCo vs
pH torque per joint, their error over time (0.5 s RMS), and the summary table. Writes outputs/sim2real/<run>/motor_check.json.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import paths

H = 1e-3                                  # the pH plant's step: the driver's current loop runs at it
LEGS = np.arange(10, 22)                  # firmware order: left leg 10-15, right leg 16-21
DELAYS_MS = np.arange(0.0, 62.0, 2.0)
COIL_TIMES = (0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08)   # L/R candidates [s]; 0.05 = ELECTRICAL_TIME
ROLL_S = 0.5                              # window of the error-over-time plots
COLORS = {"measured": (40, 40, 40), "mujoco": (31, 78, 140), "ph": (181, 101, 29), "ph_fit": (230, 160, 40)}
LABELS = {"measured": "robot tau_est", "mujoco": "MuJoCo", "ph": "pH (nominal coil)", "ph_fit": "pH (fitted coil)"}


@dataclass
class Run:
    """One run's full-rate data on k1_debug's timeline (seconds since the run log's first row)."""
    name: str
    t: np.ndarray        # (N,) LowState times
    q: np.ndarray        # (N, 22)
    dq: np.ndarray
    tau: np.ndarray      # tau_est
    tc: np.ndarray       # (M,) LowCmd times
    q_cmd: np.ndarray    # (M, 22)
    kp: np.ndarray
    kd: np.ndarray


@dataclass
class Motors:
    """Per joint (22): MuJoCo's torque-speed curve and the pH plant's DC motor."""
    effort: np.ndarray
    speed: np.ndarray    # no-load speed
    width: np.ndarray    # no-load speed - knee speed
    k: np.ndarray
    R: np.ndarray
    tau_max: np.ndarray
    v_max: float
    coil_time: float     # nominal L/R


def load_run(rec: str | Path) -> Run:
    paths.k1_debug_root()
    from k1_debug.logs import low_beside, read_k1low, read_rec
    rec = Path(rec)
    low_path = low_beside(rec)
    if low_path is None:
        raise SystemExit(f"{rec}: no {rec.name}.k1low beside it (the motor check needs the full-rate file)")
    log, low = read_rec(rec), read_k1low(low_path)
    t0 = float(log.t[0])
    st, cm = low.state, low.command
    if not len(cm):
        raise SystemExit(f"{low_path}: no LowCmd records")
    if np.abs(cm["dq"]).max() > 0 or np.abs(cm["tau"]).max() > 0:
        raise SystemExit("the commands carry a velocity / torque feed-forward: not modelled here")
    keep = np.r_[True, np.diff(st["t"]) > 0]
    st = st[keep]
    f = lambda a: np.asarray(a, dtype=np.float64)  # noqa: E731
    return Run(rec.stem, f(st["t"]) - t0, f(st["q"]), f(st["dq"]), f(st["tau_est"]),
               f(cm["t"]) - t0, f(cm["q"]), f(cm["kp"]), f(cm["kd"]))


def load_motors() -> Motors:
    from ph_k1 import motors as M
    from ph_k1 import servo
    from ph_k1.model import load_k1
    model = load_k1()
    bank = M.build_motors(model)
    curve = np.array([servo.MOTOR_CURVE[M.motor_type(n)] for n in model.joint_names])  # (22, 3) effort, speed, knee
    n = lambda x: x.detach().cpu().numpy().astype(np.float64)  # noqa: E731
    return Motors(effort=np.asarray(servo.EFFORT, dtype=np.float64), speed=curve[:, 1],
                  width=np.maximum(curve[:, 1] - curve[:, 2], 1e-6), k=n(bank.k), R=n(bank.R),
                  tau_max=n(bank.tau_max), v_max=float(bank.v_max), coil_time=M.ELECTRICAL_TIME)


def command_index(tc: np.ndarray, t: np.ndarray) -> np.ndarray:
    """The newest command at or before each time (-1: none yet)."""
    return np.searchsorted(tc, t, side="right") - 1


def pd(run: Run, k: np.ndarray, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
    """kp (q_cmd - q) - kd dq with the commands k (NaN where there is none)."""
    out = run.kp[k] * (run.q_cmd[k] - q) - run.kd[k] * dq
    return np.where((k >= 0)[..., None], out, np.nan)


def mujoco_torque(run: Run, mot: Motors, delay: float) -> np.ndarray:
    """(N, 22) at the LowState samples: PD on the command of t - delay, clipped to the torque-speed curve."""
    tau = pd(run, command_index(run.tc, run.t - delay), run.q, run.dq)
    lim = mot.effort * np.clip((mot.speed - np.abs(run.dq)) / mot.width, 0.0, 1.0)
    return np.clip(tau, -lim, lim)


def ph_torque(run: Run, mot: Motors, delays, coil_times, keep: bool = False):
    """The pH driver + coil for every (delay, L/R) candidate at once, on a 1 ms grid along the measured motion.

    Returns the per-candidate, per-joint sum of squared errors against tau_est and the sample count (at the LowState
    samples after the first command), and with keep=True also the (C, N, 22) torque at the LowState samples."""
    delays, coil_times = np.asarray(delays, float), np.asarray(coil_times, float)
    tg = np.arange(run.t[0], run.t[-1], H)
    G, C = len(tg), len(delays)
    q = np.stack([np.interp(tg, run.t, run.q[:, j]) for j in range(22)], 1)
    dq = np.stack([np.interp(tg, run.t, run.dq[:, j]) for j in range(22)], 1)
    kidx = np.stack([command_index(run.tc, tg - d) for d in delays])           # (C, G)
    L = coil_times[:, None] * mot.R                                            # (C, 22)
    k, R, V, cap_max = mot.k, mot.R, mot.v_max, mot.tau_max
    # the LowState sample(s) nearest to each grid step
    at = np.clip(np.rint((run.t - tg[0]) / H).astype(int), 0, G - 1)
    order = np.argsort(at, kind="stable")
    starts = np.searchsorted(at[order], np.arange(G + 1))
    i = np.broadcast_to(run.tau[0] / k, (C, 22)).copy()                       # start at the measured torque
    sse, cnt = np.zeros((C, 22)), np.zeros((C, 22))
    out = np.full((C, len(run.t), 22), np.nan) if keep else None
    rows = np.arange(C)
    for n in range(G):
        kn = kidx[:, n]
        ok = kn >= 0
        kc = np.maximum(kn, 0)
        tau_d = run.kp[kc] * (run.q_cmd[kc] - q[n]) - run.kd[kc] * dq[n]      # (C, 22)
        avail = np.maximum((V - k * dq[n] * np.sign(tau_d)) * k / R, 0.0)
        cap = np.minimum(avail, cap_max)
        tau_d = np.clip(tau_d, -cap, cap)
        i_star = tau_d / k
        ki = np.abs(k * i)
        s = np.where(ki > cap_max, cap_max / np.maximum(ki, 1e-300), 1.0)
        v = np.clip(L * (i_star - i) / H + R * i_star + s * k * dq[n], -V, V)
        i_new = (L * i / H + v - s * k * dq[n]) / (L / H + R)
        i = np.where(ok[:, None], i_new, i)
        torque = s * k * i
        for m in order[starts[n]:starts[n + 1]]:
            e = np.where(ok[:, None], torque - run.tau[m], 0.0)
            sse += e * e
            cnt += ok[:, None]
            if keep:
                out[rows[ok], m] = torque[ok]
    return sse, cnt, out


def metrics(pred: np.ndarray, meas: np.ndarray) -> dict:
    """Per joint over the samples where pred is defined: RMSE, mean error (bias), correlation, and the slope a of
    tau_est ~ a * pred (1 = the measured torque has the model's size)."""
    ok = np.isfinite(pred)
    out = {"rmse": [], "bias": [], "corr": [], "slope": []}
    for j in range(pred.shape[1]):
        p, m = pred[ok[:, j], j], meas[ok[:, j], j]
        e = p - m
        out["rmse"].append(float(np.sqrt(np.mean(e * e))) if len(e) else float("nan"))
        out["bias"].append(float(np.mean(e)) if len(e) else float("nan"))
        c = np.corrcoef(p, m)[0, 1] if len(e) > 2 and p.std() > 0 and m.std() > 0 else float("nan")
        out["corr"].append(float(c))
        out["slope"].append(float(np.dot(p, m) / np.dot(p, p)) if len(e) and np.dot(p, p) > 0 else float("nan"))
    return out


def legs_rmse(m: dict) -> float:
    r = np.asarray(m["rmse"])[LEGS]
    return float(np.sqrt(np.nanmean(r * r)))


def rolling_rms(t: np.ndarray, e: np.ndarray, window: float = ROLL_S) -> np.ndarray:
    """Centered RMS of e (N, 22) over `window` seconds (NaN ignored)."""
    e2 = np.where(np.isfinite(e), e * e, 0.0)
    w = np.isfinite(e).astype(float)
    c2 = np.vstack([np.zeros((1, e.shape[1])), np.cumsum(e2, 0)])
    cw = np.vstack([np.zeros((1, e.shape[1])), np.cumsum(w, 0)])
    lo = np.searchsorted(t, t - window / 2)
    hi = np.searchsorted(t, t + window / 2)
    n = cw[hi] - cw[lo]
    return np.where(n > 0, np.sqrt((c2[hi] - c2[lo]) / np.maximum(n, 1)), np.nan)


def check(run: Run, mot: Motors) -> dict:
    """Fit the delay (and the pH coil time) and return every series and the summary."""
    # MuJoCo: best delay
    mj_scan = []
    for d in DELAYS_MS:
        mj_scan.append(legs_rmse(metrics(mujoco_torque(run, mot, d * 1e-3), run.tau)))
    d_mj = float(DELAYS_MS[int(np.argmin(mj_scan))])
    # pH: every (delay, L/R) in one pass
    cand = [(d, ct) for ct in COIL_TIMES for d in DELAYS_MS]
    sse, cnt, _ = ph_torque(run, mot, [c[0] * 1e-3 for c in cand], [c[1] for c in cand])
    score = np.sqrt((sse[:, LEGS] / np.maximum(cnt[:, LEGS], 1)).mean(1))
    nominal = [n for n, c in enumerate(cand) if c[1] == mot.coil_time]
    best_nom = nominal[int(np.argmin(score[nominal]))]
    best_fit = int(np.argmin(score))
    chosen = [cand[best_nom], cand[best_fit]]
    _, _, series = ph_torque(run, mot, [c[0] * 1e-3 for c in chosen], [c[1] for c in chosen], keep=True)
    pred = {"mujoco": mujoco_torque(run, mot, d_mj * 1e-3), "ph": series[0], "ph_fit": series[1]}
    ideal = pd(run, command_index(run.tc, run.t), run.q, run.dq)  # k1_debug's tau_pd: no delay, no limit
    met = {name: metrics(p, run.tau) for name, p in pred.items()}
    met["ideal_pd"] = metrics(ideal, run.tau)
    ph_scan = {f"{ct * 1e3:g} ms": [float(score[n]) for n, c in enumerate(cand) if c[1] == ct] for ct in COIL_TIMES}
    return {
        "run": run.name, "duration_s": float(run.t[-1] - run.t[0]),
        "states": len(run.t), "commands": len(run.tc),
        "delay_ms": {"mujoco": d_mj, "ph": chosen[0][0], "ph_fit": chosen[1][0]},
        "coil_time_ms": {"ph": chosen[0][1] * 1e3, "ph_fit": chosen[1][1] * 1e3},
        "legs_rmse": {name: legs_rmse(m) for name, m in met.items()},
        "per_joint": met,
        "scan": {"delays_ms": DELAYS_MS.tolist(), "mujoco": mj_scan, "ph_by_coil_time": ph_scan},
        "_pred": pred,
    }


def summary_markdown(res: dict, names: list[str]) -> str:
    d, ct, lr = res["delay_ms"], res["coil_time_ms"], res["legs_rmse"]
    lines = [f"# Motor check (M1): {res['run']}",
             f"{res['duration_s']:.1f} s, {res['states']} LowState, {res['commands']} LowCmd. Teacher forced: the models "
             "see the robot's measured q, dq and the logged commands; reference = the motors' tau_est.", "",
             "| model | delay | L/R | legs RMSE [N m] |", "|---|---|---|---|",
             f"| MuJoCo (RL_Core torque-speed curve, instant) | {d['mujoco']:g} ms | - | {lr['mujoco']:.2f} |",
             f"| pH, nominal coil | {d['ph']:g} ms | {ct['ph']:g} ms | {lr['ph']:.2f} |",
             f"| pH, fitted coil | {d['ph_fit']:g} ms | {ct['ph_fit']:g} ms | {lr['ph_fit']:.2f} |",
             f"| ideal PD (no delay, no limit; k1_debug's tau_pd) | 0 | - | {lr['ideal_pd']:.2f} |", "",
             "## Per joint: RMSE [N m] (correlation)", "",
             "| joint | MuJoCo | pH nominal | pH fitted | slope | mean abs tau_est |", "|---|---|---|---|---|---|"]
    pj = res["per_joint"]
    for j in list(LEGS) + list(range(10)):
        lines.append(f"| {names[j]} | " + " | ".join(
            f"{pj[m]['rmse'][j]:.2f} ({pj[m]['corr'][j]:.2f})" for m in ("mujoco", "ph", "ph_fit"))
            + f" | {pj['mujoco']['slope'][j]:.2f} | {res['mean_abs_tau'][j]:.2f} |")
    scan = res["scan"]["ph_by_coil_time"]
    lines += ["", "**slope** a of tau_est ~ a x MuJoCo's torque: 1 = the robot's estimate has the model's size. Below 1 "
              "the motors report less torque than the PD law asks for (estimate scale, friction or lower real gains: "
              "not separable here). The ankle rolls (~0.1-0.2) are driven through the ankle's parallel linkage: their "
              "tau_est is not comparable with a joint PD torque.", "",
              "**Delay** = the best shift of the commands for that model; it also absorbs the motors' measurement lag. "
              "**L/R** (ELECTRICAL_TIME of ph_k1.motors, a placeholder): legs RMSE at the best delay per candidate: "
              + ", ".join(f"{k}: {min(v):.2f}" for k, v in scan.items()) + "."]
    return "\n".join(lines)


def to_rerun(rec: Path, run: Run, res: dict, save: str | None, spawn: bool) -> None:
    import rerun as rr
    import rerun.blueprint as rrb
    from k1_debug import view
    from k1_debug.logs import camera_beside, low_beside, read_k1low
    from k1_debug.low import joint_paths

    view.build(rec, save=save, spawn=spawn)   # robot, camera, k1_debug's own plots; rr.init + spawn/save inside
    jp = joint_paths()
    series = {"measured": run.tau, **res["_pred"]}
    for j, p in enumerate(jp):
        for name, val in series.items():
            path = f"motor/torque/{p}/{name}"
            rr.log(path, rr.SeriesLines(colors=[COLORS[name]], names=[LABELS[name]],
                                        widths=[2.0 if name == "measured" else 1.2]), static=True)
            ok = np.isfinite(val[:, j])
            view._scalars(path, run.t[ok], val[ok, j])
        for name in ("mujoco", "ph", "ph_fit"):
            path = f"motor/error/{p}/{name}"
            rr.log(path, rr.SeriesLines(colors=[COLORS[name]], names=[LABELS[name]]), static=True)
            err = rolling_rms(run.t, series[name] - run.tau)[:, j]
            ok = np.isfinite(err)
            view._scalars(path, run.t[ok], err[ok])
    rr.log("motor/summary", rr.TextDocument(summary_markdown(res, [p.replace("/", " ") for p in jp]),
                                            media_type=rr.MediaType.MARKDOWN), static=True)

    def grid(kind: str, joints, name: str, cols: int = 2):
        return rrb.Grid(*[rrb.TimeSeriesView(origin=f"motor/{kind}/{jp[j]}", name=jp[j].split("/")[1].replace("_", " "))
                          for j in joints], grid_columns=cols, name=name)
    low = read_k1low(low_beside(rec))
    cam = camera_beside(rec)
    robot = rrb.Spatial3DView(origin="world", name="robot")
    left = rrb.Vertical(robot, rrb.Horizontal(rrb.Spatial2DView(origin="world/camera/left", name="left camera"),
                                              rrb.Spatial2DView(origin="world/camera/right", name="right camera")),
                        row_shares=[3, 2]) if cam else robot
    rr.send_blueprint(rrb.Blueprint(rrb.Horizontal(
        left,
        rrb.Vertical(
            rrb.Tabs(rrb.TextDocumentView(origin="motor/summary", name="motor check: summary"),
                     grid("torque", range(10, 16), "torque: left leg"),
                     grid("torque", range(16, 22), "torque: right leg"),
                     grid("torque", range(0, 10), "torque: arms, head", 3),
                     grid("error", range(10, 22), "error (0.5 s RMS): legs", 3),
                     *view._low_tabs(low)),
            rrb.TextLogView(origin="events", name="events"), row_shares=[5, 1]),
        column_shares=[2, 3]), collapse_panels=True))


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("rec", help="the robot's run log (rec_*.csv) with <rec>.k1low beside it")
    ap.add_argument("--save", help="write a .rrd file instead of opening the viewer")
    ap.add_argument("--no-view", action="store_true", help="numbers only")
    ap.add_argument("--out", help="folder for motor_check.json (default outputs/sim2real/<run>)")
    a = ap.parse_args(argv)
    rec = Path(a.rec).resolve()
    run, mot = load_run(rec), load_motors()
    res = check(run, mot)
    res["mean_abs_tau"] = np.nanmean(np.abs(run.tau), 0).tolist()
    paths.k1_debug_root()
    from k1_debug.low import joint_paths
    names = [p.replace("/", " ") for p in joint_paths()]
    print(summary_markdown(res, names))
    out = Path(a.out) if a.out else paths.REPO / "outputs" / "sim2real" / run.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "motor_check.json").write_text(json.dumps({k: v for k, v in res.items() if k != "_pred"}, indent=1))
    print(f"\nwrote {out / 'motor_check.json'}")
    if not a.no_view:
        to_rerun(rec, run, res, a.save, spawn=a.save is None)
    return res


if __name__ == "__main__":
    main()

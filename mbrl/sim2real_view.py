r"""The sim-to-real comparison (mbrl/sim2real.py) in Rerun: the real robot, MuJoCo and the pH plant on one timeline.

    python -m mbrl.sim2real <rec.csv> --view                         # opens the Rerun viewer
    python -m mbrl.sim2real <rec.csv> --save run.rrd [--no-camera]   # a recording (open it with `rerun run.rrd`)

Built on k1_debug's view of the run log (the real robot posed from the log, camera, full-rate plots; k1_debug itself
is not changed), all on its "time" line:
    3D      MuJoCo (orange) and the pH plant (blue) with the real robot (grey, drawn see-through; the camera follows
            its trunk).  The run is cut into windows that
            follow each other; at each window start both are put into the robot's state, then they run free on the
            logged commands.  Each shows its own trunk (anchored at the real trunk's place and heading at the window
            start, then where its own floor and motors take it: rocking, sliding, sinking).  Arrows: each engine's
            floor force on each foot (1 cm per 20 N).
    tabs    summary (the report); per window (each engine's mean joint error and pH - MuJoCo, one point per shown
            window: click in the plot to move the time cursor there); prediction error over time; start (the soles'
            tilt over the run as logged and after the parallel-soles offsets, the per-window turns); energy (hips and
            knees: motor power and copper loss of robot, MuJoCo, pH); per leg joint the angle and the torque (legs
            only: the arms and the head are in k1_debug's own full-rate plots).
            The error, energy and joint plots show 2 s around the time cursor and zoom together.
Each recording carries how it was made (recording property "sim2real": run, start mode, delay, model, commits).
"""

from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

from . import paths

RGB = {"robot": (40, 40, 40), "MuJoCo": (235, 104, 52), "pH nominal": (42, 120, 214), "nothing moves": (150, 150, 150),
       "constant velocity": (200, 170, 60)}
KEY = {"MuJoCo": "mujoco", "pH nominal": "ph", "nothing moves": "still", "constant velocity": "constant",
       "robot": "robot"}
GHOST_EVERY = 10          # one ghost pose per 10 ms
FORCE_SCALE = 0.01 / 20   # arrow length per newton: 1 cm per 20 N (the robot's weight, ~193 N, is ~10 cm)
SPAN_S = 1.0              # the scrolling plots show the time cursor +- this


def chain(starts: np.ndarray, length: int) -> list[int]:
    """Indices of windows that follow each other without overlap."""
    out, end = [], -1
    for i, s in enumerate(starts):
        if s >= end:
            out.append(i)
            end = s + length
    return out


def commit(repo: Path) -> str:
    """The repository's short commit, '+changes' when its working tree differs from it."""
    try:
        head = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return head + ("+changes" if dirty else "") if head else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def yaw_quat(yaw: np.ndarray) -> np.ndarray:
    """(N,) -> (N, 4) w x y z of a turn about the vertical."""
    return np.stack((np.cos(yaw / 2), np.zeros_like(yaw), np.zeros_like(yaw), np.sin(yaw / 2)), -1)


def show(rec: Path, log, res: dict, keep: dict, report_md: str, save: str | None, spawn: bool,
         camera: bool = True) -> None:
    """log: the run as loaded (before any start correction); keep: what sim2real.analyse kept (rp, traj, obs,
    measures, the corrected log, the turns, the start)."""
    import rerun as rr
    import rerun.blueprint as rrb
    import rerun.blueprint.components as rrbc

    paths.k1_debug_root()
    from k1_debug import view
    from k1_debug.logs import camera_beside, low_beside, read_k1low, read_rec
    from k1_debug.low import joint_paths
    from k1_debug.robot import Robot, rest_on_ground, trunk_positions

    from . import calib
    from .motor_check import load_motors
    from .sim2real import HIP_KNEE

    view.build(rec, save=save, spawn=spawn, **({} if camera else {"camera_every": 10 ** 9, "depth_every": 0}))
    rp, traj, obs, m = keep["rp"], keep["traj"], keep["obs"], keep["measures"]
    st = keep.get("start", res.get("start", {"mode": "as-before"}))
    H = rp.H
    starts = np.array([w.start for w in rp.windows])
    pick = chain(starts, H)
    offset = log.extra.get("t_offset", 0.0)
    idx = (starts[pick][:, None] + np.arange(H)[None] + 1).reshape(-1)        # grid index of every kept step
    t = log.t[idx] + offset
    t_start = log.t[starts] + offset                                          # every window's start on "time"
    flat = lambda a: a[pick].reshape(len(pick) * H, *a.shape[2:])  # noqa: E731
    jp = joint_paths()

    rr.send_property("sim2real", rr.AnyValues(
        run=res["run"], start_mode=st["mode"], delay_ms=float(res["delay_ms"]), horizon_s=float(res["horizon_s"]),
        windows=int(res["windows"]), windows_in_3d=len(pick), model=str(rp.plant.model.xml_path),
        commit_model_based_rl=commit(paths.REPO), commit_full_body=commit(paths.FULL_BODY),
        commit_k1_debug=commit(paths.k1_debug_root()), made=datetime.now().isoformat(timespec="seconds")))

    def style(path: str, who: str, width: float = 1.2, points: bool = False) -> None:
        if points:
            rr.log(path, rr.SeriesPoints(colors=[RGB[who]], names=[who], marker_sizes=[3.0]), static=True)
        else:
            rr.log(path, rr.SeriesLines(colors=[RGB[who]], names=[who], widths=[width]), static=True)

    def series(path: str, who: str, times: np.ndarray, values: np.ndarray, width: float = 1.2,
               points: bool = False) -> None:
        style(path, who, width, points)
        ok = np.isfinite(values)
        if ok.any():
            view._scalars(path, times[ok], values[ok])

    # ---------------------------------------------------------------- joints, torques, errors over the shown windows
    for j, p in enumerate(jp):
        if j < 10:                       # the tabs show the legs; the arms and head stay in k1_debug's own plots
            continue
        series(f"sim/q/{p}/robot", "robot", t, np.degrees(flat(obs["theta"])[:, j]), 2.0)
        series(f"sim/tau/{p}/robot", "robot", t, flat(obs["tau"])[:, j], 2.0)
        for who in ("MuJoCo", "pH nominal"):
            series(f"sim/q/{p}/{KEY[who]}", who, t, np.degrees(flat(traj[who]["theta"])[:, j]))
            series(f"sim/tau/{p}/{KEY[who]}", who, t, flat(traj[who]["tau"])[:, j])
    kinds = [k for k in ("joints", "tilt", "gyro", "speed", "torque") if k in res["rows"]["MuJoCo"]]
    for k in kinds:
        for who in ("nothing moves", "constant velocity", "MuJoCo", "pH nominal"):
            series(f"sim/error/{k}/{KEY[who]}", who, t, flat(m[who][k]), 2.0 if who in ("MuJoCo", "pH nominal") else 1.0)
    text = "\n".join(line for line in report_md.splitlines() if not line.startswith("!["))
    rr.log("sim/summary", rr.TextDocument(text, media_type=rr.MediaType.MARKDOWN), static=True)

    # ---------------------------------------------------------------- per window: one point per shown window
    wm = res.get("window_means", {})
    ts_pick = t_start[pick]
    for k in ("joints", "tilt"):
        for who in ("nothing moves", "MuJoCo", "pH nominal"):
            if k in wm.get(who, {}):
                series(f"sim/window/{k}/{KEY[who]}", who, ts_pick, np.asarray(wm[who][k], dtype=float)[pick], points=True)
        if k in res.get("window_diff", {}):
            rr.log(f"sim/window/{k}_difference/zero", rr.SeriesLines(colors=[(120, 120, 120)], names=["0"], widths=[0.8]),
                   static=True)
            view._scalars(f"sim/window/{k}_difference/zero", ts_pick[[0, -1]], np.zeros(2))
            series(f"sim/window/{k}_difference/ph_minus_mujoco", "pH nominal", ts_pick,
                   np.asarray(res["window_diff"][k], dtype=float)[pick], points=True)

    # ---------------------------------------------------------------- energy: hips and knees
    mot = load_motors()
    rk2 = mot.R / mot.k ** 2

    def power(d: dict) -> tuple[np.ndarray, np.ndarray]:
        tau, qd = flat(d["tau"])[:, HIP_KNEE], flat(d["qd"])[:, HIP_KNEE]
        return (tau * qd).sum(1), (tau ** 2 * rk2[HIP_KNEE]).sum(1)
    for who, d in (("robot", obs), ("MuJoCo", traj["MuJoCo"]), ("pH nominal", traj["pH nominal"])):
        if np.isfinite(d["tau"]).any():
            p_motor, p_cu = power(d)
            series(f"sim/energy/motor_power/{KEY[who]}", who, t, p_motor, 2.0 if who == "robot" else 1.2)
            series(f"sim/energy/copper/{KEY[who]}", who, t, p_cu, 2.0 if who == "robot" else 1.2)

    # ---------------------------------------------------------------- the start: soles over the run, the turns
    flat_feet = st.get("mode") == "flat-feet"
    if flat_feet:
        feet = calib.Feet()
        every = max(1, round(0.02 / log.dt))
        rows = np.arange(0, len(log.t), every)
        tt = log.t[rows] + offset
        fixed = keep.get("log", log)
        for name, L in (("logged", log), ("parallel", fixed)):
            tilt = np.array([calib.tilt_deg(feet.normals(L.quat[i], L.q[i])) for i in rows])     # (n, 2 feet, 2)
            for f, side in enumerate(calib.SIDES):
                for c, ang in enumerate(("roll", "pitch")):
                    path = f"sim/start/sole_{ang}/{side}_{name}"
                    colour = (60, 140, 60) if side == "left" else (150, 60, 150)
                    rr.log(path, rr.SeriesLines(colors=[colour], names=[f"{side} sole, {name}"],
                                                widths=[1.0 if name == "logged" else 2.0]), static=True)
                    view._scalars(path, tt, tilt[:, f, c])
        still = calib.still_mask(log)[rows].astype(float)
        rr.log("sim/start/still", rr.SeriesLines(colors=[(120, 120, 120)], names=["standing still (1)"], widths=[1.0]),
               static=True)
        view._scalars("sim/start/still", tt, still)
        turns = keep.get("turns") or []
        if turns:
            applied = np.array([i["turn_deg"] if i["applied"] else np.nan for i in turns])
            rr.log("sim/start/turn/applied", rr.SeriesPoints(colors=[(42, 120, 214)], names=["turn applied [deg]"],
                                                             marker_sizes=[3.0]), static=True)
            ok = np.isfinite(applied)
            if ok.any():
                view._scalars("sim/start/turn/applied", t_start[ok], applied[ok])
            skipped = np.array([np.nan if i["applied"] else min(i["turn_deg"], 5.0) for i in turns])
            rr.log("sim/start/turn/not_applied", rr.SeriesPoints(colors=[(170, 170, 170)],
                                                                 names=["turn not applied (moving, edge or > 3 deg)"],
                                                                 marker_sizes=[2.0]), static=True)
            ok = np.isfinite(skipped)
            if ok.any():
                view._scalars("sim/start/turn/not_applied", t_start[ok], skipped[ok])

    # ---------------------------------------------------------------- 3D: the engines' own trunks and foot forces
    run = read_rec(rec)
    tr = run.t - run.t[0]
    q50 = run.cols("q", 22)
    rpy50 = np.stack([run.col("roll"), run.col("pitch"), run.col("yaw")], axis=1)
    pos50 = trunk_positions(q50, rpy50)
    pos50[:, 2] = rest_on_ground(q50, rpy50, pos50)
    yaw50 = np.unwrap(rpy50[:, 2])
    g_idx = np.arange(0, len(t), GHOST_EVERY)
    tg = t[g_idx]
    ww = g_idx // H                                                      # which shown window each ghost pose is in
    a_t = t_start[pick]
    a_pos = np.stack([np.interp(a_t, tr, pos50[:, k]) for k in range(2)], 1)[ww]       # anchor: the real trunk at
    a_yaw = np.interp(a_t, tr, yaw50)[ww]                                              # the window start
    robot = Robot()
    mj = robot.model
    sole_geoms = [g for g in robot.feet if "ankle_roll" in mj.body(mj.geom_bodyid[g]).name]
    sole_geoms.sort(key=lambda g: 0 if "left" in mj.body(mj.geom_bodyid[g]).name else 1)
    for who in ("MuJoCo", "pH nominal"):
        root = f"world/{KEY[who]}"
        for g in robot.visual:
            body = mj.body(mj.geom_bodyid[g]).name
            verts, faces, normals, _ = robot.mesh(g)
            rr.log(f"{root}/{body}/mesh{g}", rr.Mesh3D(vertex_positions=verts, triangle_indices=faces,
                                                       vertex_normals=normals, albedo_factor=RGB[who] + (150,)),
                   static=True)
        base = flat(traj[who]["base"])[g_idx]                            # (n, 7): position, quaternion w x y z
        theta = flat(traj[who]["theta"])[g_idx]
        ok = np.isfinite(theta).all(1) & np.isfinite(base).all(1)
        c, s = np.cos(a_yaw), np.sin(a_yaw)
        pos = np.stack((a_pos[:, 0] + c * base[:, 0] - s * base[:, 1], a_pos[:, 1] + s * base[:, 0] + c * base[:, 1],
                        base[:, 2]), 1)
        quat = calib.qmul(yaw_quat(a_yaw), base[:, 3:7])
        poses = robot.poses(pos[ok], quat[ok], theta[ok])
        # between the shown windows the engines have nothing to show: at each window's end they shrink to a point
        # (scale 1e-6: a zero scale is an invalid transform to Rerun, which warns) and come back at the next window's
        # first pose; else they would stand frozen where it ended
        n_ok = int(ok.sum())
        t_ok = tg[ok]
        t_gone = t_start[pick] + H * log.dt                              # every shown window's end
        order = np.argsort(np.r_[t_ok, t_gone], kind="stable")
        t_all = np.r_[t_ok, t_gone][order]
        scale = np.r_[np.ones(n_ok), np.full(len(t_gone), 1e-6)][order][:, None].repeat(3, 1)
        for body, (trans, q_xyzw) in poses.items():
            tr_all = np.r_[trans, np.zeros((len(t_gone), 3))][order]
            q_all = np.r_[q_xyzw, np.tile([0.0, 0.0, 0.0, 1.0], (len(t_gone), 1))][order]
            rr.send_columns(f"{root}/{body}", indexes=[rr.TimeColumn("time", duration=t_all)],
                            columns=rr.Transform3D.columns(translation=tr_all, quaternion=q_all, scale=scale))
        # the floor force on each foot, from the sole box's centre, straight up
        force = flat(traj[who]["foot_force"])[g_idx][ok]                # (n, 2) left, right [N]
        origins = np.zeros((ok.sum(), 2, 3))
        for f, g in enumerate(sole_geoms):
            trans, q_xyzw = poses[mj.body(mj.geom_bodyid[g]).name]
            w_, x_, y_, z_ = q_xyzw[:, 3], q_xyzw[:, 0], q_xyzw[:, 1], q_xyzw[:, 2]
            R = np.stack([np.stack([1 - 2 * (y_ * y_ + z_ * z_), 2 * (x_ * y_ - w_ * z_), 2 * (x_ * z_ + w_ * y_)], -1),
                          np.stack([2 * (x_ * y_ + w_ * z_), 1 - 2 * (x_ * x_ + z_ * z_), 2 * (y_ * z_ - w_ * x_)], -1),
                          np.stack([2 * (x_ * z_ - w_ * y_), 2 * (y_ * z_ + w_ * x_), 1 - 2 * (x_ * x_ + y_ * y_)], -1)], 1)
            origins[:, f] = trans + R @ mj.geom_pos[g]
        vectors = np.zeros_like(origins)
        vectors[..., 2] = np.clip(np.nan_to_num(force), 0, None) * FORCE_SCALE
        rr.log(f"{root}/forces", rr.Arrows3D.from_fields(colors=[RGB[who]], radii=[0.004]), static=True)
        o_all = np.r_[origins, np.zeros((len(t_gone), 2, 3))][order]
        v_all = np.r_[vectors, np.zeros((len(t_gone), 2, 3))][order]          # zero arrows between the windows
        rr.send_columns(f"{root}/forces", indexes=[rr.TimeColumn("time", duration=t_all)],
                        columns=rr.Arrows3D.columns(origins=o_all.reshape(-1, 3),
                                                    vectors=v_all.reshape(-1, 3)).partition([2] * len(t_all)))

    # ---------------------------------------------------------------- the window starts as events
    groups = res.get("window_group", ["all"] * len(starts))
    for i in pick:
        rr.set_time("time", duration=float(t_start[i]))
        line = f"window start ({groups[i]}): both engines put into the robot's state"
        if wm:
            line += (f"; mean joint error over the window MuJoCo {wm['MuJoCo']['joints'][i]:.2f}, "
                     f"pH {wm['pH nominal']['joints'][i]:.2f} deg")
        turns = keep.get("turns")
        if turns:
            line += f"; start turned {turns[i]['turn_deg']:.1f} deg" if turns[i]["applied"] else "; no start turn"
        rr.log("events/windows", rr.TextLog(line))

    # ---------------------------------------------------------------- the blueprint
    scroll = dict(time_ranges=[rrb.VisibleTimeRange("time", start=rrb.TimeRangeBoundary.cursor_relative(seconds=-SPAN_S),
                                                    end=rrb.TimeRangeBoundary.cursor_relative(seconds=SPAN_S))],
                  axis_x=rrb.TimeAxis(link=rrbc.LinkAxis.LinkToGlobal))

    def grid(kind: str, joints, name: str, cols: int = 2):
        return rrb.Grid(*[rrb.TimeSeriesView(origin=f"sim/{kind}/{jp[j]}", name=jp[j].split("/")[1].replace("_", " "),
                                             **scroll) for j in joints], grid_columns=cols, name=name)
    units = {"joints": "joint angles [deg]", "tilt": "tilt [deg]", "gyro": "gyro [rad/s]",
             "speed": "leg joint speeds [rad/s]", "torque": "hip and knee torque [N m]"}
    errors = rrb.Vertical(*[rrb.TimeSeriesView(origin=f"sim/error/{k}", name=f"error: {units[k]}", **scroll)
                            for k in kinds], name="prediction error")
    per_window = rrb.Vertical(
        rrb.TimeSeriesView(origin="sim/window/joints", name="per window: mean joint error [deg] (click: go there)"),
        rrb.TimeSeriesView(origin="sim/window/joints_difference", name="per window: pH − MuJoCo, joints [deg] (below 0: pH closer)"),
        rrb.TimeSeriesView(origin="sim/window/tilt", name="per window: tilt error [deg]"), name="per window")
    tabs = [rrb.TextDocumentView(origin="sim/summary", name="sim vs robot: summary"), per_window, errors]
    if flat_feet:
        tabs.append(rrb.Vertical(
            rrb.TimeSeriesView(origin="sim/start/sole_roll", name="soles' roll [deg]: logged (thin), soles parallel (thick)"),
            rrb.TimeSeriesView(origin="sim/start/sole_pitch", name="soles' pitch [deg]"),
            rrb.TimeSeriesView(origin="sim/start/turn", name="start turn per window [deg]"),
            rrb.TimeSeriesView(origin="sim/start/still", name="the robot stands still"), name="start: flat feet"))
    if res["has_torque"]:
        tabs.append(rrb.Vertical(
            rrb.TimeSeriesView(origin="sim/energy/motor_power", name="hips and knees: motor power Σ τ·θ̇ [W]", **scroll),
            rrb.TimeSeriesView(origin="sim/energy/copper", name="hips and knees: copper loss Σ τ²R/k² [W] (robot: tau_est)",
                               **scroll), name="energy"))
    tabs += [grid("q", range(10, 16), "angle [deg]: left leg"), grid("q", range(16, 22), "angle [deg]: right leg")]
    if res["has_torque"]:
        tabs += [grid("tau", range(10, 16), "torque [N m]: left leg"), grid("tau", range(16, 22), "torque [N m]: right leg")]
    low_file = low_beside(rec)
    tabs += view._low_tabs(read_k1low(low_file)) if low_file else []
    cam = camera_beside(rec) if camera else None
    # the real robot see-through (else it hides the engines wherever they agree with it), the camera on its trunk
    see_through = {f"world/robot/{mj.body(mj.geom_bodyid[g]).name}/mesh{g}": rr.Mesh3D.from_fields(
        albedo_factor=[(140, 140, 140, 70)]) for g in robot.visual}
    robot_view = rrb.Spatial3DView(origin="world", name="robot (grey, see-through), MuJoCo (orange), pH (blue); arrows: "
                                   "floor force", overrides=see_through,
                                   eye_controls=rrb.EyeControls3D(tracking_entity="world/robot/trunk"))
    left = rrb.Vertical(robot_view, rrb.Horizontal(rrb.Spatial2DView(origin="world/camera/left", name="left camera"),
                                                   rrb.Spatial2DView(origin="world/camera/right", name="right camera")),
                        row_shares=[3, 2]) if cam else robot_view
    rr.send_blueprint(rrb.Blueprint(rrb.Horizontal(
        left, rrb.Vertical(rrb.Tabs(*tabs), rrb.TextLogView(origin="events", name="events"), row_shares=[5, 1]),
        column_shares=[2, 3]), collapse_panels=True))
    print(f"[sim2real] Rerun: {len(pick)} windows of {res['horizon_s']:.1f} s in a row, {len(tg)} poses per engine"
          f"{', saved ' + save if save else ''}", flush=True)

r"""The sim-to-real comparison (mbrl/sim2real.py) in Rerun, on k1_debug's view of the run.

    python -m mbrl.sim2real <rec.csv> --view            # opens the Rerun viewer
    python -m mbrl.sim2real <rec.csv> --save run.rrd    # writes a recording (open it with `rerun run.rrd`)

Added to k1_debug's recording (the robot posed from the log, camera, full-rate plots), all on its "time" line:
    3D      two more robots at the real one's place: MuJoCo (orange) and the pH plant (blue).  The run is cut into
            windows that follow each other; at the start of each the two are put into the robot's state, then they
            run free on the logged commands.  So they snap back onto the robot at every window start and drift away
            in between.  They show the simulated joints and trunk tilt; the trunk's place and heading are the robot's.
    tabs    summary (the report's tables); prediction error over time (joints, tilt, gyro, speeds, torque: both
            engines and the two floors); per leg joint the angle and the torque of robot, MuJoCo and pH.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import paths

RGB = {"robot": (40, 40, 40), "MuJoCo": (235, 104, 52), "pH nominal": (42, 120, 214), "nothing moves": (150, 150, 150),
       "constant velocity": (200, 170, 60)}
KEY = {"MuJoCo": "mujoco", "pH nominal": "ph", "nothing moves": "still", "constant velocity": "constant"}
GHOST_EVERY = 10          # one ghost pose per 10 ms


def chain(starts: np.ndarray, length: int) -> list[int]:
    """Indices of windows that follow each other without overlap."""
    out, end = [], -1
    for i, s in enumerate(starts):
        if s >= end:
            out.append(i)
            end = s + length
    return out


def show(rec: Path, log, res: dict, keep: dict, report_md: str, save: str | None, spawn: bool) -> None:
    import rerun as rr
    import rerun.blueprint as rrb

    paths.k1_debug_root()
    from k1_debug import view
    from k1_debug.logs import camera_beside, low_beside, read_k1low, read_rec
    from k1_debug.low import joint_paths
    from k1_debug.robot import Robot, rest_on_ground, rpy_to_quat_wxyz, trunk_positions

    view.build(rec, save=save, spawn=spawn)      # rr.init + viewer or file inside
    rp, traj, obs, m = keep["rp"], keep["traj"], keep["obs"], keep["measures"]
    H = rp.H
    starts = np.array([w.start for w in rp.windows])
    pick = chain(starts, H)
    offset = log.extra.get("t_offset", 0.0)
    idx = (starts[pick][:, None] + np.arange(H)[None] + 1).reshape(-1)        # grid index of every kept step
    t = log.t[idx] + offset
    flat = lambda a: a[pick].reshape(len(pick) * H, *a.shape[2:])  # noqa: E731
    jp = joint_paths()

    def series(path: str, who: str, values: np.ndarray, width: float = 1.2) -> None:
        rr.log(path, rr.SeriesLines(colors=[RGB[who]], names=[who], widths=[width]), static=True)
        ok = np.isfinite(values)
        if ok.any():
            view._scalars(path, t[ok], values[ok])

    for j, p in enumerate(jp):
        series(f"sim/q/{p}/robot", "robot", np.degrees(flat(obs["theta"])[:, j]), 2.0)
        series(f"sim/tau/{p}/robot", "robot", flat(obs["tau"])[:, j], 2.0)
        for who in ("MuJoCo", "pH nominal"):
            series(f"sim/q/{p}/{KEY[who]}", who, np.degrees(flat(traj[who]["theta"])[:, j]))
            series(f"sim/tau/{p}/{KEY[who]}", who, flat(traj[who]["tau"])[:, j])
    kinds = [k for k in ("joints", "tilt", "gyro", "speed", "torque") if k in res["rows"]["MuJoCo"]]
    for k in kinds:
        for who in ("nothing moves", "constant velocity", "MuJoCo", "pH nominal"):
            series(f"sim/error/{k}/{KEY[who]}", who, flat(m[who][k]), 2.0 if who in ("MuJoCo", "pH nominal") else 1.0)
    text = "\n".join(line for line in report_md.splitlines() if not line.startswith("!["))
    rr.log("sim/summary", rr.TextDocument(text, media_type=rr.MediaType.MARKDOWN), static=True)

    # the two simulated robots at the real one's place and heading
    run = read_rec(rec)
    tr = run.t - run.t[0]
    q50 = run.cols("q", 22)
    rpy50 = np.stack([run.col("roll"), run.col("pitch"), run.col("yaw")], axis=1)
    pos50 = trunk_positions(q50, rpy50)
    pos50[:, 2] = rest_on_ground(q50, rpy50, pos50)
    g_idx = np.arange(0, len(t), GHOST_EVERY)
    tg = t[g_idx]
    pos = np.stack([np.interp(tg, tr, pos50[:, k]) for k in range(3)], 1)
    yaw = np.interp(tg, tr, np.unwrap(rpy50[:, 2]))
    robot = Robot()
    mj = robot.model
    for who in ("MuJoCo", "pH nominal"):
        root = f"world/{KEY[who]}"
        for g in robot.visual:
            body = mj.body(mj.geom_bodyid[g]).name
            verts, faces, normals, _ = robot.mesh(g)
            rr.log(f"{root}/{body}/mesh{g}", rr.Mesh3D(vertex_positions=verts, triangle_indices=faces,
                                                       vertex_normals=normals, albedo_factor=RGB[who] + (150,)),
                   static=True)
        grav = flat(traj[who]["grav"])[g_idx]
        theta = flat(traj[who]["theta"])[g_idx]
        ok = np.isfinite(theta).all(1) & np.isfinite(grav).all(1)
        rpy = np.stack((np.arctan2(-grav[:, 1], -grav[:, 2]), np.arcsin(np.clip(grav[:, 0], -1, 1)), yaw), 1)
        poses = robot.poses(pos[ok], rpy_to_quat_wxyz(rpy[ok]), theta[ok])
        for body, (trans, quat) in poses.items():
            rr.send_columns(f"{root}/{body}", indexes=[rr.TimeColumn("time", duration=tg[ok])],
                            columns=rr.Transform3D.columns(translation=trans, quaternion=quat))
    for i in pick:
        rr.set_time("time", duration=float(log.t[starts[i]] + offset))
        rr.log("sim/windows", rr.TextLog("window start: both engines put into the robot's state"))

    def grid(kind: str, joints, name: str, cols: int = 2):
        return rrb.Grid(*[rrb.TimeSeriesView(origin=f"sim/{kind}/{jp[j]}", name=jp[j].split("/")[1].replace("_", " "))
                          for j in joints], grid_columns=cols, name=name)
    units = {"joints": "joint angles [deg]", "tilt": "tilt [deg]", "gyro": "gyro [rad/s]",
             "speed": "leg joint speeds [rad/s]", "torque": "hip and knee torque [N m]"}
    errors = rrb.Vertical(*[rrb.TimeSeriesView(origin=f"sim/error/{k}", name=f"error: {units[k]}") for k in kinds],
                          name="prediction error")
    tabs = [rrb.TextDocumentView(origin="sim/summary", name="sim vs robot: summary"), errors,
            grid("q", range(10, 16), "angle [deg]: left leg"), grid("q", range(16, 22), "angle [deg]: right leg")]
    if res["has_torque"]:
        tabs += [grid("tau", range(10, 16), "torque [N m]: left leg"), grid("tau", range(16, 22), "torque [N m]: right leg")]
    low_file = low_beside(rec)
    tabs += view._low_tabs(read_k1low(low_file)) if low_file else []
    cam = camera_beside(rec)
    robot_view = rrb.Spatial3DView(origin="world", name="robot (grey), MuJoCo (orange), pH (blue)")
    left = rrb.Vertical(robot_view, rrb.Horizontal(rrb.Spatial2DView(origin="world/camera/left", name="left camera"),
                                                   rrb.Spatial2DView(origin="world/camera/right", name="right camera")),
                        row_shares=[3, 2]) if cam else robot_view
    rr.send_blueprint(rrb.Blueprint(rrb.Horizontal(
        left, rrb.Vertical(rrb.Tabs(*tabs), rrb.TextLogView(origin="events", name="events"), row_shares=[5, 1]),
        column_shares=[2, 3]), collapse_panels=True))
    print(f"[sim2real] Rerun: {len(pick)} windows of {res['horizon_s']:.1f} s in a row, {len(tg)} poses per engine"
          f"{', saved ' + save if save else ''}", flush=True)

r"""Replay logged commands through the plant from logged states: the model's short-horizon prediction error.

A window is H policy steps (default 20 = 0.4 s) that start at a logged policy step.  Every robot of one batched
K1PHPlantNative is one (candidate parameter set, window) pair:
  1. state from the log at the window start (mbrl/state.py), the candidate's parameter row, driver gains x its
     factors, its delay; the delay line is filled with the commands of the last 100 ms, so the driver holds what
     the robot's driver held;
  2. H policy steps with the logged commands through the robot's command path (policy_step, 1 ms plant steps);
  3. the joint angles, the gyro and the trunk tilt after every step against the log.
Short horizons on purpose (Lecture 11: errors compound over long roll-outs, and the state estimate at the start
is not perfect).

    rp = Replay(logs, make_windows(logs), space, n_candidates=64)
    out = rp.run(theta)          # theta (N, D) -> out["loss"] (N,), per-group RMSE, ...
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from . import paths  # noqa: F401
from .data import G, RobotLog
from .params import GROUP_OF_JOINT, GROUPS, ParamSpace
from .state import init_currents, set_state
from ph_k1 import firmware as fw
from ph_k1.firmware import FirmwareConfig
from ph_k1.model import NB, NJ, NV, load_k1
from ph_k1.native import K1PHPlantNative

POLICY_DT = 0.02
JOINT_WEIGHT = {"head": 0.25, "arm": 0.5, "hip": 1.0, "knee": 1.0, "ankle": 1.0}


@dataclass(frozen=True)
class Window:
    log: int     # index into the log list
    start: int   # grid index of the first policy step


def make_windows(logs: list[RobotLog], horizon: int = 20, stride_s: float = 0.1, history_s: float = 0.1,
                 max_acc_dev: float = 2.0, max_count: int | None = None, seed: int = 0) -> list[Window]:
    """Windows that start at a policy step, have commands over the whole window and the 100 ms before it, and
    start where the accelerometer is not in an impact (| |acc| - g | < max_acc_dev: the tilt estimate is good)."""
    out = []
    for li, L in enumerate(logs):
        sub = round(POLICY_DT / L.dt)
        hist = int(np.ceil(history_s / L.dt))
        last = None
        for s in L.cmd_change:
            end = s + horizon * sub + 1
            if s - hist < 0 or end >= len(L):
                continue
            if last is not None and L.t[s] - L.t[last] < stride_s - 1e-6:
                continue
            if not L.cmd_ok[s - hist:end].all():
                continue
            if abs(np.linalg.norm(L.acc[s]) - G) > max_acc_dev:
                continue
            out.append(Window(li, int(s)))
            last = s
    if max_count is not None and len(out) > max_count:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(out), max_count, replace=False))
        out = [out[i] for i in keep]
    return out


class Replay:
    def __init__(self, logs: list[RobotLog], windows: list[Window], space: ParamSpace | None = None,
                 n_candidates: int = 32, horizon: int = 20, device: str = "cpu", dtype=torch.float64,
                 plant_dt: float = 1e-3, max_delay_ms: float = 100.0, w_gyro: float = 0.02, w_grav: float = 1.0):
        if not windows:
            raise ValueError("no windows")
        slews = {logs[w.log].meta.slew for w in windows}
        if len(slews) != 1:
            raise ValueError("all logs of one Replay must share the slew switch (it is a batch-wide setting)")
        self.logs, self.windows = logs, windows
        self.space = space or ParamSpace()
        self.W, self.H = len(windows), horizon
        self.n_per_batch = max(1, n_candidates)
        self.B = self.n_per_batch * self.W
        self.dtype, self.device = dtype, torch.device(device)
        self.h = plant_dt
        self.substeps = round(POLICY_DT / plant_dt)
        self.w_gyro, self.w_grav = w_gyro, w_grav
        model = load_k1(dtype=dtype, device=self.device)
        self.cfg = FirmwareConfig(gains="standup", slew=slews.pop(), delay_ms=(20.0, 20.0), max_delay_ms=max_delay_ms,
                                  policy_dt=POLICY_DT)
        self.plant = K1PHPlantNative(model=model, batch=self.B, timestep=plant_dt, firmware=self.cfg)
        self.D = fw.delay_line_length(self.cfg, plant_dt)
        self._tensors(model)

    # ------------------------------------------------------------------ the windows as tensors (W, ...)
    def _tensors(self, model):
        T = lambda a: torch.as_tensor(np.asarray(a), dtype=self.dtype, device=self.device)  # noqa: E731
        H, D = self.H, self.D
        th0, qd0, quat0, gy0, prev, hist, tgt, th, gy, gr, kp, kd = ([] for _ in range(12))
        for w in self.windows:
            L = self.logs[w.log]
            sub = round(POLICY_DT / L.dt)
            s = w.start
            th0.append(L.q[s]); qd0.append(L.qd[s]); quat0.append(L.quat[s]); gy0.append(L.gyro[s])
            prev.append(L.cmd[s - 1])
            # delay line: the command held (d+1) plant steps before the window, d = 0 .. D-1
            back = s - np.ceil((np.arange(D) + 1) * self.h / L.dt - 1e-9).astype(int)
            hist.append(L.cmd[np.clip(back, 0, None)])
            idx_t = np.minimum(s + np.arange(H) * sub + 1, len(L) - 1)      # the new command of each step
            idx_o = s + (np.arange(H) + 1) * sub                            # the state after each step
            tgt.append(L.cmd[idx_t]); th.append(L.q[idx_o]); gy.append(L.gyro[idx_o]); gr.append(L.grav[idx_o])
            k_p, k_d = fw.gains(model, L.meta.gains)
            kp.append(k_p.cpu().numpy()); kd.append(k_d.cpu().numpy())
        self.th0, self.qd0, self.quat0, self.gyro0 = T(th0), T(qd0), T(quat0), T(gy0)
        self.prev_cmd, self.hist = T(prev), T(hist)                   # (W,22), (W,D,22)
        self.target, self.obs_theta, self.obs_gyro, self.obs_grav = T(tgt), T(th), T(gy), T(gr)  # (W,H,...)
        self.base_kp, self.base_kd = T(kp), T(kd)
        jw = torch.tensor([JOINT_WEIGHT[GROUP_OF_JOINT[k]] for k in range(NJ)], dtype=self.dtype, device=self.device)
        self.joint_w = jw / jw.sum()

    def _rep(self, x: torch.Tensor, n: int) -> torch.Tensor:
        """(W, ...) -> (n*W, ...): window w of candidate c is robot c*W + w."""
        return x.repeat((n,) + (1,) * (x.dim() - 1))

    # ------------------------------------------------------------------ one batch of candidates
    def _run_batch(self, theta: torch.Tensor, residual=None, keep_traj: bool = False) -> dict:
        n, W, H = theta.shape[0], self.W, self.H
        pl, sp = self.plant, self.space
        rep = lambda x: self._rep(x, n)  # noqa: E731
        cand = lambda x: x.repeat_interleave(W, 0)  # noqa: E731
        theta = theta.to(self.dtype).to(self.device)
        pl.robot_params = cand(sp.robot_params(theta)).contiguous()
        kps, kds = sp.gain_scales(theta)
        kp, kd = (rep(self.base_kp) * cand(kps)).contiguous(), (rep(self.base_kd) * cand(kds)).contiguous()
        set_state(pl, rep(self.th0), rep(self.qd0), rep(self.quat0), rep(self.gyro0))
        lag = torch.round(cand(sp.delay_ms(theta)) * 1e-3 / self.h).clamp(0, self.D - 1).long()
        pl.lag = lag.to(torch.int32)
        pl.ring = rep(self.hist).flip(1).contiguous()          # ring[D-1-d] = command d+1 steps ago, write index 0
        pl.ring_idx = torch.zeros(pl.B, dtype=torch.int32, device=self.device)
        pl.last_target = rep(self.prev_cmd).contiguous()
        held = torch.where((lag > 0)[:, None], rep(self.hist)[torch.arange(pl.B), (lag - 1).clamp_min(0)],
                           rep(self.target[:, 0]))
        init_currents(pl, held, kp, kd, rep(self.qd0))
        gyro = rep(self.gyro0)
        th_pred = torch.empty(pl.B, H, NJ, dtype=self.dtype, device=self.device)
        gy_pred = torch.empty(pl.B, H, 3, dtype=self.dtype, device=self.device)
        gr_pred = torch.empty(pl.B, H, 3, dtype=self.dtype, device=self.device)
        qd_pred = torch.empty(pl.B, H, NJ, dtype=self.dtype, device=self.device) if keep_traj else None
        targets = rep(self.target)
        for k in range(H):
            gen = None
            if residual is not None:
                qd = pl.velocity()[:, NB:]
                tau = residual(pl.theta, qd, projected_gravity(pl.q[:, 3:7]), gyro, targets[:, k])
                gen = torch.zeros(pl.B, NV, dtype=self.dtype, device=self.device)
                gen[:, NB:] = tau.to(self.dtype)
            info = pl.policy_step(targets[:, k].contiguous(), self.substeps, kp=kp, kd=kd, gen_force=gen)
            gyro = info.extra["gyro"].clone()
            th_pred[:, k], gy_pred[:, k] = pl.theta, gyro
            gr_pred[:, k] = projected_gravity(pl.q[:, 3:7])
            if keep_traj:
                qd_pred[:, k] = info.extra["nu"][:, NB:]
        e_th = (th_pred - rep(self.obs_theta)) ** 2            # (B,H,22)
        e_gy = ((gy_pred - rep(self.obs_gyro)) ** 2).sum(-1)    # (B,H)
        e_gr = ((gr_pred - rep(self.obs_grav)) ** 2).sum(-1)
        per_joint = e_th.mean(1)                                # (B,22)
        loss = (per_joint * self.joint_w).sum(-1) + self.w_gyro * e_gy.mean(1) + self.w_grav * e_gr.mean(1)
        bad = ~torch.isfinite(loss)
        loss = torch.where(bad, torch.full_like(loss, 1e3), loss)
        out = {
            "loss": loss.view(n, W).mean(1),
            "loss_windows": loss.view(n, W),
            "joint_mse": torch.nan_to_num(per_joint, nan=1e3).view(n, W, NJ).mean(1),   # (n,22) rad^2
            "gyro_mse": torch.nan_to_num(e_gy.mean(1), nan=1e3).view(n, W).mean(1),
            "tilt_err": torch.nan_to_num(e_gr.mean(1), nan=1e3).view(n, W).mean(1),
            "diverged": bad.view(n, W).float().mean(1),
        }
        if keep_traj:
            out["theta"] = th_pred.view(n, W, H, NJ)
            out["gyro"] = gy_pred.view(n, W, H, 3)
            out["qd"] = qd_pred.view(n, W, H, NJ)
        return out

    def run(self, theta: torch.Tensor, residual=None, keep_traj: bool = False) -> dict:
        """theta (N, D): every candidate on every window, in batches of n_candidates."""
        theta = theta.reshape(-1, self.space.D)
        N, n = theta.shape[0], self.n_per_batch
        parts = []
        for i in range(0, N, n):
            chunk = theta[i:i + n]
            pad = n - chunk.shape[0]
            if pad:
                chunk = torch.cat((chunk, chunk[-1:].expand(pad, -1)))
            o = self._run_batch(chunk, residual, keep_traj)
            parts.append({k: v[: n - pad] for k, v in o.items()})
        out = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
        out["group_rmse_deg"] = group_rmse_deg(out["joint_mse"])
        return out


def projected_gravity(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(-1)
    return torch.stack((-2 * (x * z - w * y), -2 * (y * z + w * x), -(1 - 2 * (x * x + y * y))), -1)


def group_rmse_deg(joint_mse: torch.Tensor) -> dict[str, torch.Tensor]:
    """(N,22) rad^2 -> {group: (N,) RMSE in degrees}."""
    return {g: torch.rad2deg(joint_mse[:, list(js)].mean(-1).sqrt()) for g, js in GROUPS.items()}

r"""The kick plan: a feedback controller built around the walk policy the robot already runs.

Why not an open-loop joint motion (as the get-up's first plan): played open loop on a standing robot it tips over
(getup-mbrl-v2: all robots fell within 2 s after the plan ended), and a network cannot imitate a clock it does not
see.  Here the balance feedback stays on during the whole plan, because the plan acts THROUGH the walk policy:

    walking (no request)   a = walk(frame, command)                    the env's command: the deployed walk as it is
    request, approach      a = walk(frame, c*)                         c* = P-controller on the walk's command that
                                                                        brings the trunk to a stand-off pose behind
                                                                        the ball, facing the kick direction
    request, kick          a = walk(frame, 0) + offset(t - t_kick)     a keyframe motion of the 12 leg joints, in
                                                                        action units, added to the walk's action
    request, afterwards    a = walk(frame, 0)                          the walk policy stands

The kick starts when the trunk has been at the stand-off pose for hold_s (or latest_s after the request).  The
planner (plan.py) searches the parameters: where the ball lies relative to the trunk at the kick (back, side), the
keyframe offsets and their durations.  They are defined for a right-foot kick; a left-foot kick is their mirror image.
The foot is the one on the ball's side at the request (the env's foot_request).

The controller reads the simulation's true state (ball, trunk pose): it is a teacher.  The policy that learns from
it (train.py) sees only the robot's inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .. import paths  # noqa: F401
from rl.kick import contract as C

NJ = 22
LEG = 6
# sign of a right-leg joint offset when the kick is mirrored to the left leg: hip pitch, roll, yaw, knee, ankle
# pitch, ankle roll (rl_core k1_skill MIRRORED_SIGNS: roll and yaw joints flip)
MIRROR = (1.0, -1.0, -1.0, 1.0, 1.0, -1.0)
LEFT, RIGHT = slice(10, 16), slice(16, 22)


@dataclass
class PlanSpace:
    """The plan's parameters z (unbounded, D numbers) and what they mean."""

    keyframes: int = 4
    # the largest offset per leg joint in action units: hip pitch, roll, yaw, knee, ankle pitch, ankle roll (one unit:
    # hip pitch 0.21 rad, hip roll 0.24, hip yaw 0.12, knee 0.35, ankles 0.19).  k1_kick's kick-018 turns the kicking
    # leg's hip yaw by up to 14 units and its hip roll by 5 (an inside-foot kick); the firmware clamps the targets.
    offset_scale: tuple = (8.0, 6.0, 14.0, 6.0, 6.0, 4.0)
    d_min: float = 0.06            # s per segment
    d_max: float = 0.30
    back: tuple = (0.22, 0.45)     # m: the ball this far ahead of the trunk at the kick (toes: 0.07, ball radius 0.075)
    side: tuple = (0.05, 0.14)     # m: and this far to the kicking foot's side
    return_s: float = 0.25         # the offsets go back to zero in this time after the last keyframe

    @property
    def D(self) -> int:
        return 2 + self.keyframes * 2 * LEG + self.keyframes

    def decode(self, z: torch.Tensor) -> dict:
        """z (N, D) -> back (N,), side (N,), offsets (N, K, 12) [kicking leg 6 | standing leg 6], durations (N, K)."""
        K = self.keyframes
        s = torch.sigmoid(z)
        n_off = K * 2 * LEG
        scale = torch.tensor(self.offset_scale * 2, dtype=z.dtype, device=z.device)
        return {
            "back": self.back[0] + (self.back[1] - self.back[0]) * s[:, 0],
            "side": self.side[0] + (self.side[1] - self.side[0]) * s[:, 1],
            "offsets": scale * torch.tanh(z[:, 2:2 + n_off]).reshape(-1, K, 2 * LEG),
            "durations": self.d_min + (self.d_max - self.d_min) * s[:, 2 + n_off:],
        }

    def encode(self, back, side, offsets, durations) -> torch.Tensor:
        """The inverse of decode (values outside the ranges go to their edge): -> z (N, D)."""
        unit = lambda x, r: ((x - r[0]) / (r[1] - r[0])).clamp(0.02, 0.98)  # noqa: E731
        logit = lambda u: torch.log(u / (1 - u))  # noqa: E731
        scale = torch.tensor(self.offset_scale * 2, dtype=offsets.dtype, device=offsets.device)
        off = torch.atanh((offsets / scale).clamp(-0.98, 0.98)).reshape(offsets.shape[0], -1)
        return torch.cat((logit(unit(back, self.back))[:, None], logit(unit(side, self.side))[:, None], off,
                          logit(unit(durations, (self.d_min, self.d_max)))), -1)

    def init_mean(self) -> torch.Tensor:
        return torch.zeros(self.D)   # the middle of every range, no offsets: the robot walks up and stands

    def init_std(self, offset_std: float, other_std: float) -> torch.Tensor:
        std = torch.full((self.D,), other_std)
        std[2:2 + self.keyframes * 2 * LEG] = offset_std
        return std


def interpolate(params: dict, speed: torch.Tensor, speeds: torch.Tensor) -> dict:
    """A library of S plans (decoded, one per requested ball speed in `speeds`, ascending) -> the plan of every robot
    for its own requested speed (B,): linear between the neighbours, the first / last plan outside."""
    S = len(speeds)
    if S == 1:
        return {k: v[:1].expand(len(speed), *v.shape[1:]) for k, v in params.items()}
    hi = torch.searchsorted(speeds, speed.contiguous()).clamp(1, S - 1)
    lo = hi - 1
    w = ((speed - speeds[lo]) / (speeds[hi] - speeds[lo])).clamp(0.0, 1.0)
    out = {}
    for k, v in params.items():
        wk = w.reshape(-1, *([1] * (v.dim() - 1)))
        out[k] = (1 - wk) * v[lo] + wk * v[hi]
    return out


class KickController:
    """The plan for every robot of a KickEnv, as a function of the env's state: act(obs) -> actions (B, 22).

    State per robot: when the kick was triggered and since when the trunk is at the stand-off pose, both in request
    time, so calling act() twice at the same step gives the same answer."""

    def __init__(self, env, walk, space: PlanSpace | None = None, gains=(2.5, 2.5, 2.5),
                 limits=((-0.4, 0.8), (-0.4, 0.4), (-1.2, 1.2)), tol_pos: float = 0.03, tol_yaw: float = 0.08,
                 hold_s: float = 0.10, latest_s: float = 2.4):
        self.env, self.walk, self.space = env, walk, space or PlanSpace()
        self.gains, self.limits = gains, limits
        self.tol_pos, self.tol_yaw, self.hold_s, self.latest_s = tol_pos, tol_yaw, hold_s, latest_s
        B, dev, dt = env.num_envs, env.device, env.dtype
        self.t_kick = torch.full((B,), -1.0, dtype=dt, device=dev)      # request time of the trigger (-1: not yet)
        self.near_since = torch.full((B,), -1.0, dtype=dt, device=dev)
        self.mirror = torch.tensor(MIRROR, dtype=dt, device=dev)
        self.params = None
        self.error = torch.zeros(B, 3, dtype=dt, device=dev)            # stand-off error (x, y, yaw), heading frame
        self.phase = torch.zeros(B, dtype=torch.long, device=dev)       # 0 walk | 1 approach | 2 kick | 3 afterwards
        self.command = torch.zeros(B, 3, dtype=dt, device=dev)

    def set_plans(self, params: dict) -> None:
        """One decoded plan per robot (PlanSpace.decode, or interpolate() of a library)."""
        self.params = {k: v.to(self.env.device, self.env.dtype) for k, v in params.items()}

    def reset(self, ids: torch.Tensor | None = None) -> None:
        if ids is None:
            self.t_kick[:], self.near_since[:] = -1.0, -1.0
        else:
            self.t_kick[ids], self.near_since[ids] = -1.0, -1.0

    def kick_length(self) -> torch.Tensor:
        return self.params["durations"].sum(-1) + self.space.return_s

    def _offsets(self, t: torch.Tensor) -> torch.Tensor:
        """The 12 leg offsets at kick time t (B,): linear from zero through the keyframes and back to zero."""
        off, dur = self.params["offsets"], self.params["durations"]
        B, K = dur.shape
        ends = torch.cumsum(dur, -1)                                             # (B, K)
        zeros = torch.zeros(B, 1, 2 * LEG, dtype=off.dtype, device=off.device)
        knots = torch.cat((zeros, off, zeros), 1)                                # (B, K + 2, 12)
        times = torch.cat((torch.zeros_like(ends[:, :1]), ends, ends[:, -1:] + self.space.return_s), -1)   # (B, K + 2)
        tt = t.clamp(min=0.0)[:, None]
        seg = (tt >= times[:, 1:]).sum(-1).clamp(max=K)                          # segment index 0..K
        rows = torch.arange(B, device=off.device)
        t0, t1 = times[rows, seg], times[rows, seg + 1]
        w = ((t - t0) / (t1 - t0).clamp(min=1e-6)).clamp(0.0, 1.0)[:, None]
        return (1 - w) * knots[rows, seg] + w * knots[rows, seg + 1]

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        env, dt = self.env, self.env.dtype
        active = env.active
        t_req = env.time_in_request()
        new = active & (t_req <= 0.0)
        self.reset(new.nonzero(as_tuple=False).squeeze(-1))
        self.reset((~active).nonzero(as_tuple=False).squeeze(-1))

        right = env.foot_request == 1
        side = torch.where(right, -self.params["side"], self.params["side"])     # the ball's y in the trunk frame
        back = self.params["back"]
        # the stand-off pose: the trunk where the ball lies at (back, side) in a frame facing the kick direction
        psi = torch.atan2(env.target_w[:, 1], env.target_w[:, 0])
        c, s = torch.cos(psi), torch.sin(psi)
        goal = torch.stack((env.ball_pos[:, 0] - (c * back - s * side), env.ball_pos[:, 1] - (s * back + c * side)), -1)
        e_xy = C.to_heading(env.yaw, goal - env.base_pos[:, :2])
        e_yaw = C.wrap(psi - env.yaw)
        near = (torch.linalg.vector_norm(e_xy, dim=1) < self.tol_pos) & (e_yaw.abs() < self.tol_yaw)

        waiting = active & (self.t_kick < 0)
        self.near_since = torch.where(waiting & near & (self.near_since < 0), t_req, self.near_since)
        self.near_since = torch.where(waiting & ~near, torch.full_like(t_req, -1.0), self.near_since)
        held = (self.near_since >= 0) & (t_req - self.near_since >= self.hold_s)
        trigger = waiting & (held | (t_req >= self.latest_s))
        self.t_kick = torch.where(trigger, t_req, self.t_kick)

        kicking = active & (self.t_kick >= 0)
        tk = t_req - self.t_kick
        in_kick = kicking & (tk < self.kick_length())
        approach = active & ~kicking
        g, lim = self.gains, self.limits
        cmd = torch.stack(((g[0] * e_xy[:, 0]).clamp(*lim[0]), (g[1] * e_xy[:, 1]).clamp(*lim[1]),
                           (g[2] * e_yaw).clamp(*lim[2])), -1)
        cmd = cmd * (approach & ~near)[:, None].to(dt)
        cmd = torch.where(active[:, None], cmd, env.commands)                    # walking: the env's own command
        with torch.no_grad():
            a = self.walk(torch.cat((obs[:, :C.FRAME], cmd), -1).float()).to(dt)

        off = self._offsets(tk) * in_kick[:, None].to(dt)                        # (B, 12): kicking leg | standing leg
        kick_leg, stand_leg = off[:, :LEG], off[:, LEG:]
        r = right[:, None]
        a = a.clone()
        a[:, RIGHT] += torch.where(r, kick_leg, stand_leg * self.mirror)
        a[:, LEFT] += torch.where(r, stand_leg, kick_leg * self.mirror)

        # only before the kick is triggered does the stand-off error mean anything (the ball moves afterwards)
        self.error = torch.cat((e_xy, e_yaw[:, None]), -1) * approach[:, None].to(dt)
        self.phase = torch.where(~active, 0, torch.where(approach, 1, torch.where(in_kick, 2, 3)))
        self.command = cmd
        return a

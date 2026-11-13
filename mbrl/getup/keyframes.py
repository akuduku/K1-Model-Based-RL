r"""The get-up plan as numbers: K keyframes of joint targets plus K+1 durations, mirrored left/right.

One parameter vector z (D = 10 K + K + 1) describes a whole get-up:

    keyframe k  the LEFT arm (4) and leg (6) joint targets, each squashed into its joint range,
                target = low + (high - low) * sigmoid(z); the right side is the mirror image (roll and yaw joints
                change sign, as RL_Core's mirror_joints), the head stays at HOME
    durations   d_1 .. d_{K+1} in [d_min, d_max] s: from the start pose to keyframe 1, ..., from keyframe K to HOME

The targets move from the robot's own start angles theta0 through the K keyframes to HOME and stay there; between two
points they follow a smoothstep (zero speed at the points: no jerks at the keyframes).  The robot's command path
(firmware filter, delay, motor-driver PD with the StandupAgent's fixed gains) turns the targets into motion.
"""

from __future__ import annotations

import torch

# MJCF = firmware joint order (rl/getup/contract.py)
HEAD = (0, 1)
LEFT = (2, 3, 4, 5, 10, 11, 12, 13, 14, 15)     # shoulder pitch/roll, elbow pitch/yaw | hip p/r/y, knee, ankle p/r
RIGHT = (6, 7, 8, 9, 16, 17, 18, 19, 20, 21)
# the right joint = SIGN x the left joint (RL_Core k1_skill.MIRRORED_SIGNS: rolls and yaws flip)
SIGN = (1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0)
N_SIDE = len(LEFT)


def logit(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(1e-3, 1 - 1e-3)
    return torch.log(p) - torch.log1p(-p)


class Keyframes:
    def __init__(self, low: torch.Tensor, high: torch.Tensor, home: torch.Tensor, n_keys: int = 5,
                 d_min: float = 0.15, d_max: float = 0.7):
        """low, high, home: (22,) joint range and HOME [rad] (on the device the plans run on)."""
        self.low, self.high, self.home = low, high, home
        self.K, self.d_min, self.d_max = int(n_keys), float(d_min), float(d_max)
        dev, dt = home.device, home.dtype
        self.left = torch.as_tensor(LEFT, device=dev)
        self.right = torch.as_tensor(RIGHT, device=dev)
        self.sign = torch.as_tensor(SIGN, dtype=dt, device=dev)
        self.D = self.K * N_SIDE + self.K + 1

    # ------------------------------------------------------------------ parameters
    def split(self, z: torch.Tensor):
        """z (N, D) -> joint logits (N, K, 10), duration logits (N, K + 1)."""
        n = z.shape[0]
        return z[:, :self.K * N_SIDE].reshape(n, self.K, N_SIDE), z[:, self.K * N_SIDE:]

    def decode(self, z: torch.Tensor):
        """z (N, D) -> keyframes (N, K, 22) [rad], durations (N, K + 1) [s]."""
        zj, zd = self.split(z.to(self.home.dtype))
        lo, hi = self.low[self.left], self.high[self.left]
        left = lo + (hi - lo) * torch.sigmoid(zj)
        right = torch.minimum(torch.maximum(left * self.sign, self.low[self.right]), self.high[self.right])
        kf = self.home.expand(z.shape[0], self.K, -1).clone()
        kf[..., self.left] = left
        kf[..., self.right] = right
        dur = self.d_min + (self.d_max - self.d_min) * torch.sigmoid(zd)
        return kf, dur

    def init_mean(self, duration: float = 0.4) -> torch.Tensor:
        """Every keyframe at HOME and every duration `duration` s: the 'from scratch' start of the search."""
        lo, hi = self.low[self.left], self.high[self.left]
        zj = logit((self.home[self.left] - lo) / (hi - lo)).expand(self.K, -1).reshape(-1)
        zd = logit(torch.full((self.K + 1,), (duration - self.d_min) / (self.d_max - self.d_min),
                              dtype=self.home.dtype, device=self.home.device))
        return torch.cat((zj, zd))

    def init_std(self, joint: float = 1.5, duration: float = 1.0) -> torch.Tensor:
        dev, dt = self.home.device, self.home.dtype
        return torch.cat((torch.full((self.K * N_SIDE,), joint, dtype=dt, device=dev),
                          torch.full((self.K + 1,), duration, dtype=dt, device=dev)))

    # ------------------------------------------------------------------ the motion
    def targets(self, kf: torch.Tensor, dur: torch.Tensor, theta0: torch.Tensor, t) -> torch.Tensor:
        """Joint targets (N, 22) at time t [s] (a number or (N,)) of the plans (kf, dur) from start angles theta0."""
        return targets_at(kf, dur, theta0, self.home, t)

    def length(self, dur: torch.Tensor) -> torch.Tensor:
        """(N,) time until HOME is reached [s]."""
        return dur.sum(-1)


def targets_at(kf: torch.Tensor, dur: torch.Tensor, theta0: torch.Tensor, home: torch.Tensor, t) -> torch.Tensor:
    """Smoothstep interpolation theta0 -> kf[0] -> ... -> kf[K-1] -> HOME (then HOME), at time t (scalar or (N,))."""
    n, K, J = kf.shape
    pts = torch.cat((theta0[:, None, :], kf, home.expand(n, 1, J)), 1)                  # (N, K + 2, 22)
    cum = torch.cat((torch.zeros_like(dur[:, :1]), torch.cumsum(dur, -1)), -1)          # (N, K + 2): arrival times
    t = torch.as_tensor(t, dtype=kf.dtype, device=kf.device)
    t = t.expand(n) if t.dim() == 0 else t
    seg = ((t[:, None] >= cum).sum(-1) - 1).clamp(0, K)                                # segment start point
    t0 = cum.gather(1, seg[:, None]).squeeze(1)
    d = dur.gather(1, seg[:, None]).squeeze(1)
    w = ((t - t0) / d).clamp(0.0, 1.0)
    w = w * w * (3.0 - 2.0 * w)
    a = pts.gather(1, seg[:, None, None].expand(n, 1, J)).squeeze(1)
    b = pts.gather(1, (seg + 1)[:, None, None].expand(n, 1, J)).squeeze(1)
    return a + w[:, None] * (b - a)

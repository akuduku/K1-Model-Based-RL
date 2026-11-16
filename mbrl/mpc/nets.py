r"""The policy and value networks of the model-based learner.

Same shape and checkpoint layout as rsl-rl's MLPModel (keys mlp.N.weight / mlp.N.bias, obs_normalizer._mean / _std),
so Full Body's rl/common/export.py writes the firmware's .bin from a checkpoint and rl/kick/policy.py loads it.
The firmware computes (x - mean) / (std + 0.01) in front of the 512-256-128 ELU network.
"""

from __future__ import annotations

import torch
from torch import nn

EPS = 0.01  # rl/common/export.NORMALIZER_EPS (the firmware's kNormalizerEps)


class Normalizer(nn.Module):
    """Running mean and standard deviation of the inputs (rsl-rl's EmpiricalNormalization)."""

    def __init__(self, n: int):
        super().__init__()
        self.register_buffer("_mean", torch.zeros(1, n))
        self.register_buffer("_var", torch.ones(1, n))
        self.register_buffer("_std", torch.ones(1, n))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.detach().to(self._mean.dtype)
        n = x.shape[0]
        if n == 0:
            return
        self.count += n
        rate = n / float(self.count)
        delta = x.mean(0, keepdim=True) - self._mean
        self._mean += rate * delta
        self._var += rate * (x.var(0, unbiased=False, keepdim=True) - self._var + delta * (x.mean(0, keepdim=True) - self._mean))
        self._std = self._var.clamp_min(0.0).sqrt()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._mean) / (self._std + EPS)


class MLP(nn.Module):
    def __init__(self, n_in: int, n_out: int, hidden=(512, 256, 128), last_scale: float = 1.0):
        super().__init__()
        self.obs_normalizer = Normalizer(n_in)
        layers, at = [], n_in
        for h in hidden:
            layers += [nn.Linear(at, h), nn.ELU()]
            at = h
        layers.append(nn.Linear(at, n_out))
        self.mlp = nn.Sequential(*layers)
        with torch.no_grad():  # a small last layer: an untrained policy commands (nearly) the start pose
            self.mlp[-1].weight.mul_(last_scale)
            self.mlp[-1].bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.obs_normalizer(x.to(self.mlp[0].weight.dtype)))


def checkpoint(policy: MLP, value: MLP, iteration: int, infos: dict | None = None) -> dict:
    """rsl-rl's layout: what rl/common/export.actor_tensors reads."""
    return {"actor_state_dict": policy.state_dict(), "critic_state_dict": value.state_dict(), "iter": iteration,
            "infos": infos or {}}

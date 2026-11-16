r"""The planner: MPPI around the policy, rolled out in the model.

World = the task env with R robots (noise, pushes, delays).  Model = a second instance of the same env class with
R x N robots and nothing random.  One call of plan():

  1. copy every world robot's state into its N model copies (env.snapshot / env.restore);
  2. roll the model H steps: a = policy(model observation) + sigma * eps, eps time-correlated Gaussian noise;
     copy 0 of every robot gets no noise (the bare policy);
  3. score every copy: G = sum_h gamma^h r_h + gamma^H V(c_H), nothing after a termination;
  4. the action to execute is the MPPI-weighted mean of the copies' first actions,
     w ~ exp((G - max G) / (temperature * std G)).

The model is the pH plant itself (Full Body, C++): it is the same code as the world, with the world robot's own
gains and command delay in the snapshot, so the plan is made with a perfect model of a noisy world.
"""

from __future__ import annotations

import math

import torch


def index_snapshot(snap, idx: torch.Tensor):
    """Rows idx of every tensor of a (nested) snapshot."""
    if isinstance(snap, dict):
        return {k: index_snapshot(v, idx) for k, v in snap.items()}
    return snap[idx]


class Planner:
    def __init__(self, world, model, policy, value, candidates: int, horizon: int, sigma: float = 0.3,
                 beta: float = 0.7, gamma: float = 0.99, temperature: float = 0.5, after_restore=None):
        R, N = world.num_envs, candidates
        if model.num_envs != R * N:
            raise ValueError(f"the model needs {R} x {N} = {R * N} robots, it has {model.num_envs}")
        self.world, self.model, self.policy, self.value = world, model, policy, value
        self.R, self.N, self.H = R, N, horizon
        self.sigma, self.beta, self.gamma, self.temperature = sigma, beta, gamma, temperature
        self.after_restore = after_restore
        self.rows = torch.arange(R, device=world.device).repeat_interleave(N)       # model robot -> world robot
        self.bare = (torch.arange(R * N, device=world.device) % N) == 0             # copy 0: the policy without noise
        self.model_steps = 0
        self.value_weight = 1.0   # the learner raises it from 0 while the value network is still untrained

    @torch.no_grad()
    def plan(self) -> tuple[torch.Tensor, dict]:
        """-> (actions (R, A) for the world, {return_policy, return_planner, weight_entropy, ...} each (R,))."""
        world, model, R, N = self.world, self.model, self.R, self.N
        dev = self.rows.device
        model.restore(index_snapshot(world.snapshot(), self.rows))
        if self.after_restore is not None:
            self.after_restore(model)
        obs = model.get_observations()
        G = torch.zeros(R * N, device=dev)
        alive = torch.ones(R * N, dtype=torch.bool, device=dev)
        eps, first, disc = None, None, 1.0
        keep = math.sqrt(1.0 - self.beta**2)
        for h in range(self.H):
            mu = self.policy(obs["policy"]).float()
            xi = torch.randn_like(mu)
            eps = xi if eps is None else self.beta * eps + keep * xi
            a = mu + self.sigma * eps * (~self.bare)[:, None]
            if h == 0:
                first = a
            obs, r, done, _ = model.step(a.to(model.dtype))
            G += disc * r.float() * alive
            alive &= ~done.bool()
            disc *= self.gamma
        if self.value_weight > 0:
            G += self.value_weight * disc * self.value(obs["critic"]).float().squeeze(-1) * alive
        self.model_steps += R * N * self.H
        G = G.reshape(R, N)
        spread = G.std(-1, keepdim=True).clamp_min(1e-6)
        w = torch.softmax((G - G.amax(-1, keepdim=True)) / (self.temperature * spread), -1)
        action = (w[..., None] * first.reshape(R, N, -1)).sum(1)
        info = {"return_policy": G[:, 0], "return_planner": (w * G).sum(-1), "return_best": G.amax(-1), "return_mean": G.mean(-1),
                "weight_entropy": -(w * (w + 1e-12).log()).sum(-1) / math.log(N),
                "policy_action": first.reshape(R, N, -1)[:, 0]}
        return action.to(world.dtype), info

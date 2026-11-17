r"""Term 2 learned: a value network V(x) at the end of every window (roadmap W3).

The professor's formula charges the state at the end of a window, term 2 = ||x(T) - x_T||^2.  Standing has a goal
state x_T (upright, home pose, at rest); walking has none.  So term 2 becomes  -V(x(T)):  minus the reward the robot
can still expect from x(T) on, learned from what the robots really get.  The learner then maximises

    sum over the window of gamma^t r_t  +  gamma^K V(x(T))

and its derivative goes through the pH model into V as before into the fixed term 2 (SHAC, Xu et al. 2022, uses
the same split: gradients through the model inside the window, a learned value beyond it).

What V learns from (train.py --value-reward): `smooth` (default) = the objective the learner differentiates: the
reward terms with a derivative and the fall charge.  `full` = the full reward, including the terms without a
derivative (rl/stand: alive, feet_grounded, undesired_contacts, termination; rl/walk: air_time, terminated).
`full` failed on the laptop (17 Nov, docs/EXPERIMENTS.md W3): while the robots stand, nobody falls, so the counting
terms are the same for every state and V is a constant plus fitting noise; the learner follows the slope of that
noise.  The terms with a derivative differ from state to state, so their V has a slope worth following.

Targets: TD(lambda) returns from each window, bootstrapped with a slowly following copy of V (the target network);
the copy is also what the learner differentiates at the window's end.

    lambda_returns(rewards, values, ends, gamma, lam)     the targets of one window
    WeightLearner(names, init, ...)                       --term2 weights: V = b - sum_k s_k c_k(x), learned weights
                                                          s_k >= 0 of the costs the end state causes.  Works (W3,
                                                          attempt 3): its slope is a mix of physical slopes
    ValueLearner(critic_dim, ...)                         --term2 value: a free network V.  Failed twice on standing
                                                          (W3, attempts 1 and 2); kept for the comparison
"""

from __future__ import annotations

import copy

import torch

from mbrl.mpc.nets import MLP


def lambda_returns(rewards: torch.Tensor, values: torch.Tensor, ends: torch.Tensor, gamma: float, lam: float) -> torch.Tensor:
    """TD(lambda) returns of one window.

    rewards (K,B): the reward of step t; values (K+1,B): V of the state before step t, values[K] of the state at the
    window's end; ends (K,B) bool: the episode ended in step t (a fall: nothing follows, no bootstrap).
    -> (K,B):  G_t = r_t + gamma * (1 - end_t) * ((1 - lam) V_{t+1} + lam G_{t+1}),  G_K = V_K.
    Entries after a robot's end are meaningless (the caller masks them); the step of the end itself is valid."""
    K = rewards.shape[0]
    out = torch.empty_like(rewards)
    G = values[K]
    for t in reversed(range(K)):
        cont = (~ends[t]).to(rewards.dtype)
        G = rewards[t] + gamma * cont * ((1.0 - lam) * values[t + 1] + lam * G)
        out[t] = G
    return out


class WeightLearner:
    """Term 2 with learned weights (--term2 weights): term 2 = sum_k s_k c_k(x(T)), V(x) = b - sum_k s_k c_k(x).

    c_k are the costs the end state causes (env.state_costs(): the running cost's terms that read only the state,
    per second, and for standing the kinetic energy), s_k >= 0 how long each of them will last ("seconds"), b the
    level.  Standing's fixed term 2 is this sum with s = (1, 1, 1, 1 per J, 0, 0, 0), so the learner starts exactly
    there (env.fixed_term2_weights) and learns the weights from the same TD(lambda) returns as the network would.
    The slope of term 2 is then always a positive mix of physical slopes: attempts 1 and 2 (a free network) had a
    slope unrelated to the physics (docs/EXPERIMENTS.md, W3)."""

    def __init__(self, names, init, device="cpu", dtype=torch.float64, lr: float = 1e-2, epochs: int = 5,
                 minibatches: int = 4, buffer: int = 4, tau: float = 0.2):
        self.names = tuple(names)
        self.s = torch.tensor(init, dtype=dtype, device=device).requires_grad_(True)
        self.b = torch.zeros((), dtype=dtype, device=device, requires_grad=True)
        self.s_target, self.b_target = self.s.detach().clone(), self.b.detach().clone()
        self.opt = torch.optim.Adam([self.s, self.b], lr=lr)
        self.epochs, self.minibatches, self.buffer, self.tau = epochs, minibatches, buffer, tau
        self.replay: list[tuple[torch.Tensor, torch.Tensor]] = []

    @staticmethod
    def inputs(env) -> torch.Tensor:
        return env.state_costs()

    def __call__(self, costs: torch.Tensor) -> torch.Tensor:
        """V (B,) with the target copy of the weights: what the learner differentiates and bootstraps with."""
        return self.b_target - costs @ self.s_target

    def fit(self, costs: torch.Tensor, returns: torch.Tensor) -> dict:
        costs, returns = costs.detach(), returns.detach()
        with torch.no_grad():
            before = self.b - costs @ self.s
            explained = 1.0 - float((returns - before).var() / returns.var().clamp_min(1e-12))
        self.replay.append((costs, returns))
        self.replay = self.replay[-self.buffer:]
        x = torch.cat([r[0] for r in self.replay])
        y = torch.cat([r[1] for r in self.replay])
        loss = torch.zeros(())
        for _ in range(self.epochs):
            perm = torch.randperm(x.shape[0], device=x.device)
            for mb in perm.split(max(1, x.shape[0] // self.minibatches)):
                loss = (self.b - x[mb] @ self.s - y[mb]).square().mean()
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                with torch.no_grad():
                    self.s.clamp_(min=0.0)            # a cost cannot turn into a reward
        with torch.no_grad():
            self.s_target.mul_(1.0 - self.tau).add_(self.tau * self.s)
            self.b_target.mul_(1.0 - self.tau).add_(self.tau * self.b)
        out = {"loss": float(loss.detach()), "return_mean": float(returns.mean()), "value_mean": float(before.mean()),
               "explained_variance": explained, "samples": int(x.shape[0]), "level": float(self.b_target)}
        out.update({f"s_{n}": float(v) for n, v in zip(self.names, self.s_target)})
        return out

    def state_dict(self) -> dict:
        return {"names": self.names, "s": self.s.detach(), "b": self.b.detach(), "s_target": self.s_target,
                "b_target": self.b_target, "opt": self.opt.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        with torch.no_grad():
            for k in ("s", "b", "s_target", "b_target"):
                getattr(self, k).copy_(state[k])
        if "opt" in state:
            self.opt.load_state_dict(state["opt"])


class ValueLearner:
    """V and its target copy, fitted on the TD(lambda) returns of the last `buffer` windows."""

    @staticmethod
    def inputs(env) -> torch.Tensor:
        return env.critic_obs()

    def __init__(self, critic_dim: int, device="cpu", dtype=torch.float64, lr: float = 5e-4, epochs: int = 5,
                 minibatches: int = 4, buffer: int = 4, tau: float = 0.2, hidden=(128, 128)):
        # small, as SHAC's critics (64-64 to 128-128): the learner follows V's slope, and a large network fitted on a
        # few thousand samples has a noisy one
        self.net = MLP(critic_dim, 1, hidden=tuple(hidden)).to(device=device, dtype=dtype)
        self.target = copy.deepcopy(self.net).requires_grad_(False)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.epochs, self.minibatches, self.buffer, self.tau = epochs, minibatches, buffer, tau
        self.replay: list[tuple[torch.Tensor, torch.Tensor]] = []

    def __call__(self, critic_obs: torch.Tensor) -> torch.Tensor:
        """The target copy's V (B,): what the learner differentiates at the window's end and bootstraps with."""
        return self.target(critic_obs).squeeze(-1)

    def fit(self, critic_obs: torch.Tensor, returns: torch.Tensor) -> dict:
        """One update from a window's valid samples (N, critic_dim), (N,): the replay of the last windows, a few
        epochs of minibatches, then the target copy follows (Polyak, tau).  -> numbers to log."""
        critic_obs, returns = critic_obs.detach(), returns.detach()
        with torch.no_grad():
            before = self.net(critic_obs).squeeze(-1)
            explained = 1.0 - float((returns - before).var() / returns.var().clamp_min(1e-12))
        self.net.obs_normalizer.update(critic_obs)
        self.replay.append((critic_obs, returns))
        self.replay = self.replay[-self.buffer:]
        x = torch.cat([r[0] for r in self.replay])
        y = torch.cat([r[1] for r in self.replay])
        loss = torch.zeros(())
        for _ in range(self.epochs):
            perm = torch.randperm(x.shape[0], device=x.device)
            for mb in perm.split(max(1, x.shape[0] // self.minibatches)):
                loss = (self.net(x[mb]).squeeze(-1) - y[mb]).square().mean()
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.opt.step()
        with torch.no_grad():
            for pt, p in zip(self.target.parameters(), self.net.parameters()):
                pt.mul_(1.0 - self.tau).add_(self.tau * p)
            for bt, b in zip(self.target.buffers(), self.net.buffers()):
                bt.copy_(b)
        return {"loss": float(loss.detach()), "return_mean": float(returns.mean()), "value_mean": float(before.mean()),
                "explained_variance": explained, "samples": int(x.shape[0])}

    def state_dict(self) -> dict:
        return {"net": self.net.state_dict(), "target": self.target.state_dict(), "opt": self.opt.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.net.load_state_dict(state["net"])
        self.target.load_state_dict(state["target"])
        if "opt" in state:
            self.opt.load_state_dict(state["opt"])

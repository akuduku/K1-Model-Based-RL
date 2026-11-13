r"""The tracking task: get up like the plan, on the robot's StandupAgent contract (72 inputs -> 22 joint targets).

What changes against Full Body's GetUpEnv (which this is, as a subclass):
  * the policy:  22 targets with the StandupAgent's fixed gains (kKp, kKd), no tank, no variable gains; it observes
                 what the firmware's StandupAgent observes (gyro 3 | projected gravity 3 | q - HOME 22 | qdot 22 |
                 last action 22 = 72), so a trained policy runs on the robot without a firmware change;
  * the reward:  the get-up reward + term 1 of the professor's formula, tracking the plan's motion
                     track_q = exp(-|theta - theta_plan(t)|^2 / sigma_q^2),  track_h = exp(-(h - h_plan(t))^2 / sigma_h^2)
                 (t = time since the start; standing starts track HOME); the tracking weight fades from 1 to
                 track_final once the success EMA reaches track_fade_ema, so the policy may leave the plan where
                 that saves energy (the energy terms ramp in with the success, as in GetUpEnv);
  * the critic:  sees the plan (phase, theta_plan - theta, h_plan - h) and the start kind: asymmetric actor-critic,
                 the actor does not get the phase (the robot has none);
  * no assist (the plan replaces it).
teacher_actions() gives the plan's own targets for every robot (from its own start angles): the expert of the DAgger
warm start in mbrl/getup/train.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from tensordict import TensorDict

from .. import paths  # noqa: F401
from .keyframes import targets_at
from rl.getup.env import REWARDS, GetUpConfig, GetUpEnv
from rl.getup.lying import KIND_NAMES, KINDS, STANDING

TRACK_REWARDS = {"track_q": 3.0, "track_h": 3.0}
NJ = 22
OBS_DIM = 72
NUM_ACTIONS = 22


def load_plan(path) -> dict:
    p = torch.load(path, map_location="cpu", weights_only=False)
    ref = p["reference"]
    return {"kind": p["kind"], "keyframes": torch.as_tensor(p["keyframes"]), "durations": torch.as_tensor(p["durations"]),
            "theta": ref["theta"].float(), "com_z": ref["com_z"].float(), "success": p.get("success"),
            "path": str(path)}


@dataclass
class TrackConfig(GetUpConfig):
    plans: tuple = ()                 # plan .pt files of mbrl.getup.plan, one per lying kind
    track_sigma_q: float = 0.5        # rad (norm over the 22 joints)
    track_sigma_h: float = 0.05       # m
    track_fade_ema: float = 0.8
    track_final: float = 0.2
    track_fade_iters: int = 500
    # GetUpEnv settings that differ for this task
    tank: bool = False
    assist_force: float = 0.0
    gain_free_ema: float | None = 1e9   # the gains never unlock
    rewards: dict = field(default_factory=lambda: {**REWARDS, **TRACK_REWARDS})


class TrackEnv(GetUpEnv):
    EXTRA_REWARDS = tuple(TRACK_REWARDS)

    def __init__(self, cfg: TrackConfig):
        if not cfg.plans:
            raise ValueError("TrackConfig.plans: give the plan files of mbrl.getup.plan (one per lying kind)")
        dev = torch.device(cfg.device)
        dt = getattr(torch, cfg.dtype)
        self.plans = {}
        for path in cfg.plans:
            p = load_plan(path)
            self.plans[KINDS[p["kind"]]] = {k: (v.to(dev, dt) if isinstance(v, torch.Tensor) else v) for k, v in p.items()}
        missing = [k for k in cfg.lying_kinds if KINDS[k] not in self.plans]
        if missing:
            raise ValueError(f"no plan for the lying kinds {missing}")
        self.theta0 = None
        self.track_scale, self.track_start_iter = 1.0, None
        super().__init__(cfg)
        self.num_actions = NUM_ACTIONS
        self.gain_scale = 0.0
        self._pad = torch.zeros(self.num_envs, 2 * NJ, dtype=self.dtype, device=self.device)

    # ------------------------------------------------------------------ reset / curriculum
    def _reset_idx(self, ids: torch.Tensor) -> None:
        super()._reset_idx(ids)
        if self.theta0 is None:
            self.theta0 = self.plant.theta.clone()
        self.theta0[ids] = self.plant.theta[ids]

    def _update_curriculum(self, ids):
        super()._update_curriculum(ids)
        if not self.kinds:
            return
        c = self.cfg
        ema = min(self.success_ema.values())
        it = self.iteration
        if self.track_start_iter is None and ema >= c.track_fade_ema:
            self.track_start_iter = it
            print(f"[track] tracking weight fades from iteration {it:.0f}", flush=True)
        if self.track_start_iter is not None:
            ramp = min(1.0, (it - self.track_start_iter) / max(1, c.track_fade_iters))
            self.track_scale = 1.0 - (1.0 - c.track_final) * ramp
        self.extra_log["Curriculum/track_scale"] = self.track_scale

    def curriculum_state(self) -> dict:
        return {**super().curriculum_state(), "track_scale": self.track_scale, "track_start_iter": self.track_start_iter}

    def load_curriculum_state(self, s: dict) -> None:
        super().load_curriculum_state(s)
        self.gain_scale = 0.0
        self.track_scale = float(s.get("track_scale", 1.0))
        self.track_start_iter = s.get("track_start_iter")

    def locked_actions(self) -> torch.Tensor:
        return torch.zeros(NUM_ACTIONS, dtype=torch.bool)

    # ------------------------------------------------------------------ the plan
    def _time(self) -> torch.Tensor:
        """(B,) time since the start of the episode [s] at the state now (after the last step)."""
        return self.episode_length_buf.to(self.dtype) * self.cfg.policy_dt

    def reference(self):
        """theta_plan (B, 22), h_plan (B,), phase (B,) of every robot at the state now."""
        B = self.num_envs
        q_ref = self.contract.home.expand(B, -1).clone()
        h_ref = torch.full((B,), self.com_stand, dtype=self.dtype, device=self.device)
        phase = torch.ones(B, dtype=self.dtype, device=self.device)
        idx = (self.episode_length_buf - 1).clamp(min=0)
        for k, p in self.plans.items():
            rows = self.start_type == k
            if not bool(rows.any()):
                continue
            T = p["theta"].shape[0]
            i = idx[rows].clamp(max=T - 1)
            q_ref[rows] = p["theta"][i]
            h_ref[rows] = p["com_z"][i]
            phase[rows] = ((i + 1).to(self.dtype) / T).clamp(max=1.0)
        return q_ref, h_ref, phase

    def teacher_actions(self) -> torch.Tensor:
        """(B, 22): the plan's own targets for the coming step, from every robot's own start angles (HOME when
        standing), as actions of the contract."""
        B = self.num_envs
        home = self.contract.home
        tgt = home.expand(B, -1).clone()
        t = self._time()
        for k, p in self.plans.items():
            rows = (self.start_type == k).nonzero(as_tuple=False).squeeze(-1)
            if len(rows) == 0:
                continue
            n = len(rows)
            kf = p["keyframes"].expand(n, -1, -1)
            dur = p["durations"].expand(n, -1)
            tgt[rows] = targets_at(kf, dur, self.theta0[rows], home, t[rows])
        return (tgt - home) / self.contract.action_scale

    # ------------------------------------------------------------------ reward
    def _reward_scale(self, name: str) -> float:
        if name in TRACK_REWARDS:
            return self.track_scale
        return super()._reward_scale(name)

    def _terms(self, info, target_raw, P, d_supplied, d_contact, d_stops) -> dict:
        terms = super()._terms(info, target_raw, P, d_supplied, d_contact, d_stops)
        q_ref, h_ref, _ = self.reference()
        theta = self.plant.theta
        terms["track_q"] = torch.exp(-(theta - q_ref).square().sum(-1) / self.cfg.track_sigma_q ** 2)
        terms["track_h"] = torch.exp(-(self.com_z - h_ref).square() / self.cfg.track_sigma_h ** 2)
        return terms

    # ------------------------------------------------------------------ observations / step
    def get_observations(self) -> TensorDict:
        q = self.plant.q
        theta = q[:, 7:]
        home = self.contract.home
        g = self.contract.projected_gravity(q[:, 3:7])
        last = self.actions[:, :NJ]
        clean = torch.cat((self.gyro, g, theta - home, self.qd, last), -1)
        actor = torch.cat((self._noise(self.gyro, "gyro"), self._noise(g, "gravity"),
                           self._noise(theta, "joint_pos") - home, self._noise(self.qd, "joint_vel"), last), -1)
        active = self.contact.to(self.dtype)
        feet = torch.stack(((active @ self.left_sole) > 0, (active @ self.right_sole) > 0), -1).to(self.dtype)
        others = (self.contact & self.other_point).to(self.dtype).mean(-1, keepdim=True)
        q_ref, h_ref, phase = self.reference()
        kind = torch.stack([(self.start_type == k) for k in (STANDING, KINDS["supine"], KINDS["prone"])], -1)
        critic = torch.cat((clean, self.v_body, q[:, 2:3] - self.stand_z, (self.com_z - self.com_stand)[:, None], feet,
                            others, self.standing.to(self.dtype)[:, None], self.battery_w[:, None] / 100.0,
                            self.kinetic[:, None] / 10.0, phase[:, None], q_ref - theta, (h_ref - self.com_z)[:, None],
                            kind.to(self.dtype)), -1)
        return TensorDict({"policy": torch.nan_to_num(actor), "critic": torch.nan_to_num(critic)},
                          batch_size=[self.num_envs], device=self.device)

    def step(self, actions: torch.Tensor):
        a = actions.to(self.device, self.dtype)
        if a.shape[-1] == NUM_ACTIONS:
            a = torch.cat((a, self._pad[: a.shape[0]]), -1)
        obs, rew, dones, extras = super().step(a)
        extras["log"]["Metrics/track_q"] = torch.exp(-(self.plant.theta - self.reference()[0]).square().sum(-1)
                                                     / self.cfg.track_sigma_q ** 2).mean()
        return obs, rew, dones, extras

    def config_dict(self) -> dict:
        d = super().config_dict()
        d["contract"] = {"agent": "standup (StandupAgent, sprint.cpp)", "obs_dim": OBS_DIM, "num_actions": NUM_ACTIONS,
                         "hidden_dims": list(d["contract"]["hidden_dims"]), "gains": "standup (fixed)"}
        d["plans"] = {KIND_NAMES[k]: {"path": p["path"], "success": p["success"]} for k, p in self.plans.items()}
        return d

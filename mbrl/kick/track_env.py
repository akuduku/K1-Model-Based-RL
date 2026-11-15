r"""The tracking task: walk and kick like the plan, on the robot's kick_walk contract (165 inputs + a clock -> 22).

What changes against Full Body's KickEnv (which this is, as a subclass):
  * the teacher: the plan of mbrl/kick/controller.py with the library of mbrl/kick/plan.py (one plan per requested
    ball speed, interpolated in between); teacher_actions() is what it does in every robot's state: the expert of
    the DAgger warm start in mbrl/kick/train.py.  It reads the true ball and trunk pose; the policy does not;
  * the reward: the kick env's task terms + term 1 of the professor's formula, acting like the plan,
        track = exp(-mean (a - a_plan)^2 / sigma^2)
    in action units, while walking and kicking (it replaces the kick env's walk_imitation, the same term for the walk
    alone); its weight fades from 1 to track_final once the share of good kicks (EMA) reaches track_fade_ema, so the
    policy may leave the plan where the task pays more;
  * the actor: the 166th input, the seconds since the request (rl/kick/contract.py): the plan's kick is a function of
    time, and a network cannot follow a clock it does not see (the get-up tracker had none);
  * the critic: also sees the plan (its phase, the time in the kick, the stand-off error, the plan's action);
  * hold_heading: the heading at the touch (the plan turns the robot to face the kick direction before it kicks).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from tensordict import TensorDict

from .. import paths  # noqa: F401
from .controller import KickController, interpolate
from .plan import DEFAULT_WALK, load_library
from rl.kick import contract as C
from rl.kick.env import REWARDS, KickConfig, KickEnv

TRACK_REWARDS = {"track": 6.0}


def track_rewards() -> dict:
    return {**{k: v for k, v in REWARDS.items() if k != "walk_imitation"}, **TRACK_REWARDS}


@dataclass
class TrackConfig(KickConfig):
    plan: str = ""                      # the library of mbrl.kick.plan (kick-<tag>.pt)
    walk_teacher: str | None = str(DEFAULT_WALK)
    phase_input: bool = True
    imitate_std: float = math.sqrt(0.1)  # action units (k1_kick's teacher_std)
    track_fade_ema: float = 0.8
    track_final: float = 0.2
    track_fade_iters: int = 500
    success_alpha: float = 0.01         # EMA per finished request
    ok_direction_deg: float = 15.0      # a good kick, as the planner's gate
    ok_speed_error: float = 0.35
    steps_per_env: int = 24             # PPO rollout length: the fade counts iterations with it
    rewards: dict = field(default_factory=track_rewards)


class TrackEnv(KickEnv):
    EXTRA_REWARDS = tuple(TRACK_REWARDS)
    EXTRA_PHASE = {"track": "both"}

    def __init__(self, cfg: TrackConfig):
        if not cfg.plan:
            raise ValueError("TrackConfig.plan: give the library file of mbrl.kick.plan")
        if not cfg.walk_teacher:
            raise ValueError("TrackConfig.walk_teacher: the plan acts through the walk policy")
        self.ctrl = None
        self.teacher_action = None
        self.track_scale, self.track_start_iter, self.success_ema = 1.0, None, 0.0
        self.common_step_counter = 0
        self.track_error = None
        super().__init__(cfg)
        self.space, self.lib_speeds, self.lib, self.lib_success = load_library(cfg.plan, self.device, self.dtype)
        self.ctrl = KickController(self, self.teacher, self.space)
        self.teacher_action = torch.zeros(self.num_envs, C.NUM_ACTIONS, dtype=self.dtype, device=self.device)
        self.track_error = torch.zeros(self.num_envs, dtype=self.dtype, device=self.device)

    @property
    def iteration(self) -> float:
        return self.common_step_counter / self.cfg.steps_per_env

    # ------------------------------------------------------------------ the plan
    def teacher_actions(self) -> torch.Tensor:
        """(B, 22): what the plan does in the state the last observation shows."""
        return self.teacher_action

    def get_observations(self) -> TensorDict:
        td = super().get_observations()
        self.ctrl.set_plans(interpolate(self.lib, self.target_speed, self.lib_speeds))
        self.teacher_action = self.ctrl.act(self.actor_obs)
        ctrl, f = self.ctrl, (lambda x: x.to(self.dtype))
        t_kick = (self.time_in_request() - ctrl.t_kick).clamp(min=0.0) * f(ctrl.t_kick >= 0) * f(self.active)
        phase = torch.nn.functional.one_hot(ctrl.phase, 4).to(self.dtype)
        critic = torch.cat((td["critic"], phase, t_kick[:, None], ctrl.error, self.teacher_action), -1)
        return TensorDict({"policy": td["policy"], "critic": critic}, batch_size=[self.num_envs], device=self.device)

    # ------------------------------------------------------------------ reward
    def _reward_scale(self, name: str) -> float:
        return self.track_scale if name in TRACK_REWARDS else super()._reward_scale(name)

    def _terms(self, supplied_w: torch.Tensor) -> dict:
        terms = super()._terms(supplied_w)
        err = (self.actions - self.teacher_action).square().mean(-1)   # the plan's action for what the policy saw
        self.track_error = err.sqrt()
        terms["track"] = torch.exp(-err / self.cfg.imitate_std**2)
        return terms

    def _update_kick(self) -> None:
        super()._update_kick()
        self.request_yaw = torch.where(self.touched, self.yaw, self.request_yaw)

    # ------------------------------------------------------------------ curriculum
    def _episode_log(self, ids: torch.Tensor, log: dict) -> None:
        super()._episode_log(ids, log)
        c = self.cfg
        shot = self.last_shot[ids]
        col = {k: shot[:, i] for i, k in enumerate(("requested", "kicked", "along", "direction_deg", "loft",
                                                    "speed_error", "want", "fell"))}
        req = col["requested"] > 0
        n = int(req.sum())
        if n > 0:
            good = ((col["kicked"] > 0) & (col["fell"] == 0) & (col["direction_deg"] < c.ok_direction_deg)
                    & (col["speed_error"] < c.ok_speed_error))[req].to(self.dtype)
            log["Track/good_kick"] = good
            a = 1.0 - (1.0 - c.success_alpha) ** n
            self.success_ema += a * (float(good.mean()) - self.success_ema)
        it = self.iteration
        if self.track_start_iter is None and self.success_ema >= c.track_fade_ema:
            self.track_start_iter = it
            print(f"[track] tracking weight fades from iteration {it:.0f}", flush=True)
        if self.track_start_iter is not None:
            ramp = min(1.0, (it - self.track_start_iter) / max(1, c.track_fade_iters))
            self.track_scale = 1.0 - (1.0 - c.track_final) * ramp

    def curriculum_state(self) -> dict:
        return {"track_scale": self.track_scale, "track_start_iter": self.track_start_iter,
                "success_ema": self.success_ema, "common_step_counter": self.common_step_counter}

    def load_curriculum_state(self, s: dict) -> None:
        self.track_scale = float(s.get("track_scale", 1.0))
        self.track_start_iter = s.get("track_start_iter")
        self.success_ema = float(s.get("success_ema", 0.0))
        self.common_step_counter = int(s.get("common_step_counter", 0))

    # ------------------------------------------------------------------ step
    def step(self, actions: torch.Tensor):
        self.common_step_counter += 1
        obs, rew, dones, extras = super().step(actions)
        log, ph = extras["log"], self.ctrl.phase
        log["Track/action_error"] = self.track_error.mean()
        log["Track/weight"] = self.track_scale
        log["Track/good_kick_ema"] = self.success_ema
        for k, name in enumerate(("walk", "approach", "kick", "after")):
            log[f"Track/plan_{name}"] = (ph == k).to(self.dtype).mean()
        return obs, rew, dones, extras

    def config_dict(self) -> dict:
        d = super().config_dict()
        d["plan"] = {"path": str(self.cfg.plan), "speeds": self.lib_speeds.tolist(), "success": self.lib_success}
        return d

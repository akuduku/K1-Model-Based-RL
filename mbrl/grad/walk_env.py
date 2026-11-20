r"""The walk task on the differentiable pH plant: Full Body's rl/walk task (rl/walk/task.py), with autograd from the
reward back to the actions.

    observation (rl/walk/contract.py, the firmware's sprint agent: 75) -> policy -> decode -> firmware chain
    (clamp, delay, motor driver) -> 20 plant steps of 1 ms -> the walk rewards of rl/walk/task.py

The task is not written here: rewards, commands and what they read from the plant come from rl/walk/task.py, the
file the C++ environment (rl/walk/env.py) uses too.  This class adds what the gradient learner needs: the state of a
control step by name (plant, firmware, buffers, the task's memory), so a step can be computed again in the
backward pass, and the split of the reward into the terms that have a derivative and the ones that only count.

step() returns the reward of the terms with a derivative (what is differentiated); the full reward, as PPO and the
planner get it, is in the info.  The counting terms (rl/walk/task.py COUNTS: air_time, terminated) have no derivative.
`terminated` reaches the learner as the fall charge (train.py).  `air_time` reached it in no way in W5 (the learned
term 2 neither learns it nor can express it), and the robots stood instead of stepping; `air_time_smooth=w` adds its
smooth stand-in (rl/walk/task.py STAND_INS: the weight a foot has taken off the ground and kept off for 3 control
steps) with weight w to what is differentiated (W5b).  The logged full reward stays Eric's terms.
"""

from __future__ import annotations

import math

import torch

from mbrl import paths  # noqa: F401  (puts Full Body on sys.path)
from mbrl.grad.env import POLICY_DT, SUBSTEPS, GradEnv
from ph_k1 import firmware as fwm
from ph_k1.plant import Ledger
from rl.kick.contract import yaw_of
from rl.walk import contract as C
from rl.walk.task import COUNTS, FLOAT_MEMORY, REWARDS, SHAPING, WalkParams, WalkTask

NA = C.NJ
OBS_NOISE = {"gyro": 0.2, "gravity": 0.05, "joint_pos": 0.01, "joint_vel": 1.5}     # rl/walk/env.py WalkConfig
# what the end of a window is charged for (terminal_cost): the terms that read only where the robot is and how it
# moves, not what it did before
STATE_TERMS = ("track_linear_velocity", "track_angular_velocity", "upright", "body_ang_vel", "angular_momentum",
               "dof_pos_limits", "upper_body_posture", "standing_pose", "leg_spacing", "feet_parallel", "arm_swing",
               "trunk_bounce", "foot_spacing", "knees_out")


class GradWalk(GradEnv):
    """rl/walk's task on the differentiable Python plant.

    Quiet by default (no sensor noise, no pushes, nominal gains, no encoder bias, 20 ms delay, facing +x).
    `obs_noise`, `push_dv` with `push_interval_s`, `delay_ms` as a range, `gain_scale`, `encoder_bias`, `joint_noise`
    and `random_yaw` switch on the randomisation of rl/walk/env.py.  `command` (vx, vy, wz) fixes the request;
    without it the task draws its own (x `command_scale`: 0 = stand, 1 = Eric's ranges).
    """

    FLOAT_STATE = (("q", "p", "xi", "lam", "ring", "last_target", "actions", "prev_actions", "qd", "gyro", "critic_extra")
                   + tuple(f"mem:{k}" for k in FLOAT_MEMORY))
    ENV_KEYS = ("actions", "prev_actions", "qd", "gyro", "contact", "contact_force", "critic_extra", "fresh")

    def __init__(self, robots: int, seed: int = 1, joint_noise: float = 0.0, device="cpu", dtype=torch.float64,
                 obs_noise: dict | None = None, push_dv: float = 0.0, push_interval_s=(1.5, 4.0),
                 delay_ms=(20.0, 20.0), gain_scale=(1.0, 1.0), encoder_bias: float = 0.0, random_yaw: bool = False,
                 command=None, command_scale: float = 1.0, top_speed: float | None = None,
                 rewards: dict | None = None, params: WalkParams | None = None, observe_target: bool = False,
                 action_clip: float = 20.0, air_time_smooth: float = 0.0, gait_clock: float = 0.0):
        self.joint_noise, self.random_yaw, self.observe_target = joint_noise, random_yaw, observe_target
        self.action_clip = action_clip          # rl/walk/env.py WalkConfig.action_clip: stops a runaway policy
        self.obs_noise, self.push_dv, self.push_interval_s = obs_noise or {}, push_dv, push_interval_s
        self.gain_scale, self.bias_range, self.command = gain_scale, encoder_bias, command
        self._build(robots, seed, device, dtype, C.GAINS, C.SLEW, delay_ms)
        if air_time_smooth > 0:                 # air_time's smooth stand-in, differentiated beside Eric's terms (W5b)
            rewards = {**(REWARDS if rewards is None else rewards), "air_time_smooth": air_time_smooth}
        self.gait_clock = gait_clock > 0        # the gait clock (W5c): two more inputs and a reward for stepping in time
        if self.gait_clock:
            rewards = {**(REWARDS if rewards is None else rewards), "gait_clock": gait_clock}
        self.task = WalkTask(self.model, POLICY_DT, params, rewards)
        self.task.command_scale = command_scale
        if top_speed is not None:
            self.task.top_speed = top_speed
        self.contract = self.task.contract
        self.smooth = tuple(k for k in self.task.rewards if k not in COUNTS)       # the terms with a derivative
        self.full = tuple(k for k in self.task.rewards if k not in SHAPING)        # the full reward: Eric's terms
        self.plant.reset(pose=self.contract.home)
        self.fw = fwm.K1Firmware(self.plant, self.fw_cfg)
        self.base_kp, self.base_kd = (g.clone() for g in fwm.gains(self.model, C.GAINS))
        self.push_timer = self._uniform(*push_interval_s, (robots,))
        self.mem = self.task.new_memory(robots, self._task_rand)
        self.reset()

    def _task_rand(self, *shape) -> torch.Tensor:
        return self._rand(shape)

    def _rest_critic(self, q: torch.Tensor, air_time: torch.Tensor) -> torch.Tensor:
        """The critic's extras of robots that just started: at rest, both feet down, no force yet (rl/walk/env.py)."""
        n, m = q.shape[0], self.model
        zero = torch.zeros(n, m.ncontact, dtype=self.dtype, device=self.device)
        rest = self.task.signals(q, torch.zeros(n, 28, dtype=self.dtype, device=self.device),
                                 torch.zeros(n, NA, dtype=self.dtype, device=self.device), zero + self.task.sole.sum(0),
                                 zero[..., None].expand(-1, -1, 3))
        return self.task.critic_extra(rest, {"air_time": air_time})

    # ------------------------------------------------------------------ state
    def _start(self, n: int):
        pose = self.contract.home + self.joint_noise * (2 * self._rand((n, NA)) - 1)
        pose = torch.minimum(torch.maximum(pose, self.model.jnt_low), self.model.jnt_high)
        yaw = self._uniform(-math.pi, math.pi, (n,)) if self.random_yaw else torch.zeros(n, dtype=self.dtype, device=self.device)
        quat = torch.stack((torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)), -1)
        kp = self._uniform(*self.gain_scale, (n, NA))
        kd = self._uniform(*self.gain_scale, (n, NA))
        bias = self.bias_range * (2 * self._rand((n, NA)) - 1)
        return pose, quat, kp, kd, bias

    def reset(self, ids: torch.Tensor | None = None) -> None:
        """Start again standing in the sprint's home pose: all robots, or the rows `ids`."""
        m = self.model
        if ids is None:
            pose, quat, kp, kd, bias = self._start(self.B)
            self.plant.reset(pose=pose, base_quat=quat)
            self.fw.reset()
            self.kp, self.kd, self.encoder_bias = self.base_kp * kp, self.base_kd * kd, bias
            self.actions, self.prev_actions = self._z(NA), self._z(NA)
            self.qd, self.gyro = self._z(NA), self._z(3)
            # both feet down, no force known yet (as rl/walk/env.py after a reset)
            self.contact = torch.zeros(self.B, m.ncontact, dtype=self.dtype, device=self.device) + self.task.sole.sum(0)
            self.contact_force = torch.zeros(self.B, m.ncontact, 3, dtype=self.dtype, device=self.device)
            self.fresh = torch.ones(self.B, dtype=torch.bool, device=self.device)
            self.mem = self.task.reset_memory(self.mem, torch.arange(self.B, device=self.device), self._task_rand)
            self.critic_extra = self._rest_critic(self.plant.q, self.mem["air_time"])
        else:
            ids = ids.reshape(-1)
            if ids.numel() == 0:
                return
            self.detach()                                           # the in-place row writes below need plain tensors
            pose, quat, kp, kd, bias = self._start(int(ids.numel()))
            self.plant.reset(pose=pose, base_quat=quat, env_ids=ids)
            self.fw.reset(env_ids=ids)
            for name, value in (("kp", self.base_kp * kp), ("kd", self.base_kd * kd), ("encoder_bias", bias),
                                ("actions", 0.0), ("prev_actions", 0.0), ("qd", 0.0), ("gyro", 0.0),
                                ("contact", self.task.sole.sum(0)), ("contact_force", 0.0), ("fresh", True)):
                t = getattr(self, name).clone()
                t[ids] = value
                setattr(self, name, t)
            self.mem = self.task.reset_memory(self.mem, ids, self._task_rand)
            extra = self.critic_extra.clone()
            extra[ids] = self._rest_critic(self.plant.q[ids], self.mem["air_time"][ids])
            self.critic_extra = extra
        if self.command is not None:
            self.mem = self.task.set_command(self.mem, self.command)

    def set_command(self, twist) -> None:
        self.command = tuple(float(v) for v in twist)
        self.mem = self.task.set_command(self.mem, self.command)

    def between_steps(self) -> None:
        """New transport delays, and the commands' clocks: expired requests are redrawn, the reference ramps on.
        Random and outside the derivative; the learner keeps the state after it."""
        self.redraw_lags()
        if self.command is None:
            dt = torch.where(self.fresh, 0.0, POLICY_DT).to(self.dtype)       # robots that just started: no time passed
            with torch.no_grad():
                new = self.task.update_commands(self.mem, yaw_of(self.plant.q[:, 3:7]), dt, self._task_rand)
            self.mem = {k: (self.mem[k] if k in FLOAT_MEMORY else v) for k, v in new.items()}
        self.fresh = torch.zeros_like(self.fresh)

    def draw_pushes(self) -> torch.Tensor | None:
        """rl/walk's pushes: now and then a horizontal change of the trunk velocity, each axis in +-push_dv."""
        if self.push_dv <= 0:
            return None
        self.push_timer = self.push_timer - POLICY_DT
        fire = self.push_timer <= 0
        dv = self._z(3)
        n = int(fire.sum())
        if n:
            dv[fire, :2] = self._uniform(-self.push_dv, self.push_dv, (n, 2))
            self.push_timer[fire] = self._uniform(*self.push_interval_s, (n,))
        return dv

    # ------------------------------------------------------------------ observations
    def _noisy(self, x, key):
        s = self.obs_noise.get(key, 0.0)
        return x if s <= 0 else x + s * (2 * self._rand(x.shape) - 1)

    def observe(self, noise: bool = True) -> torch.Tensor:
        q = self.plant.q
        theta, quat = q[:, 7:], q[:, 3:7]
        g = self.contract.projected_gravity(quat)
        command = self.mem["target"] if self.observe_target else self.mem["commands"]
        if not noise or not self.obs_noise:
            obs = self.contract.observe(self.gyro, g, theta + self.encoder_bias, self.qd, self.actions, command)
        else:
            obs = self.contract.observe(self._noisy(self.gyro, "gyro"), self._noisy(g, "gravity"),
                                        self._noisy(theta + self.encoder_bias, "joint_pos"),
                                        self._noisy(self.qd, "joint_vel"), self.actions, command)
        if self.gait_clock:                     # the clock's sin and cos after the 75 (0 while standing): 77 inputs
            obs = torch.cat((obs, self.task.clock(self.mem).to(obs.dtype)), -1)
        return obs

    def _signals(self, prev_qd):
        pl = self.plant
        kin = pl.kinematics()
        nu = pl.velocity(kin)
        return self.task.signals(pl.q, nu, prev_qd, self.contact, self.contact_force, kin=kin), nu

    def critic_obs(self) -> torch.Tensor:
        """rl/walk/env.py's critic input: the clean observation and the task's extras (trunk velocity, foot heights,
        air times, contacts, forces), the extras as the last step left them (part of the state, so a derivative of
        the critic's input reaches back into that step)."""
        return torch.cat((self.observe(noise=False), self.critic_extra), -1)

    # ------------------------------------------------------------------ one control step
    def step(self, a: torch.Tensor, push: torch.Tensor | None = None):
        """-> (reward (B,) of the terms with a derivative, dict of logged numbers).  Differentiable in a and in the
        state (FLOAT_STATE)."""
        pl, fw = self.plant, self.fw
        a = a.clamp(-self.action_clip, self.action_clip)     # as rl/walk/env.py; the derivative is 0 beyond it
        fw.ring = fw.ring.clone()                       # policy_step writes the ring in place
        pl.ledger = Ledger.zeros(self.B, pl.dtype, pl.device)
        info = fw.policy_step(self.contract.decode(a), SUBSTEPS, kp=self.kp, kd=self.kd,
                              push_dv=self._z(3) if push is None else push)
        self.contact, self.contact_force = info.contact_active.to(self.dtype), info.contact_force
        nu = info.extra["nu"]       # the last plant step's velocity: what the C++ plant reports and rl/walk/env.py reads
        sig = self.task.signals(pl.q, nu, self.qd, self.contact, self.contact_force)
        supplied_w = pl.ledger.supplied / POLICY_DT
        terminated = self.task.fallen(sig)
        terms, self.mem = self.task.terms(sig, self.mem, a, self.actions, self.prev_actions, supplied_w, terminated)
        reward = self.task.reward(terms, only=self.smooth)
        self.critic_extra = self.task.critic_extra(sig, self.mem)
        self.prev_actions, self.actions = self.actions, a
        self.qd, self.gyro = nu[:, 6:], info.extra["gyro"]
        self.sig = sig
        self.motor_target = info.extra["target"].detach()     # what the PD drives to (logs: the closed-loop H_d)
        with torch.no_grad():
            track = sig.heading_vel if self.task.p.heading_frame else sig.base_lin_vel
            led = pl.ledger                                    # this control step's energy flows [J] -> [W]
            log = {"copper_W": led.copper.detach() / POLICY_DT, "supplied_W": supplied_w.detach(),
                   "gearbox_W": led.gearbox.detach() / POLICY_DT, "contact_W": led.contact.detach() / POLICY_DT,
                   "stops_W": led.stops.detach() / POLICY_DT,
                   "reward_all": self.task.reward(terms, only=self.full).detach(), "terminated": terminated,
                   "speed_error": torch.linalg.vector_norm(self.mem["target"][:, :2] - track[:, :2], dim=1),
                   "requested": torch.linalg.vector_norm(self.mem["target"][:, :2], dim=1),
                   # does it step or shuffle: the share of feet in the air, and how fast the feet on the floor slide
                   "feet_air": (self.mem["air_time"] > 0).to(self.dtype).mean(1),
                   "slip_speed": terms["foot_slip"].detach().clamp_min(0.0).sqrt() if "foot_slip" in terms else None,
                   "terms": {k: (self.task.rewards[k] * POLICY_DT * v).detach() for k, v in terms.items()}}
        return reward, log

    # ------------------------------------------------------------------ term 2 and falls
    def terminal_cost(self, seconds: float = 1.0, kinetic: float = 0.0) -> torch.Tensor:
        """Term 2 of the professor's formula for a walking robot, in its simplest form: what the robot's state at
        the end of the window would cost if it stayed like this for `seconds` more.

        Charged: the STATE_TERMS of the running cost (speed and yaw-rate tracking, upright, posture, spacing), each
        as its distance from the best value it can have.  A place holder until the learned value of roadmap W3;
        `kinetic` is not used (a walking robot is not at rest).  -> (B,) >= 0."""
        return seconds * self.state_costs().sum(-1)

    STATE_COSTS = STATE_TERMS

    def state_costs(self) -> torch.Tensor:
        """(B, len(STATE_TERMS)) >= 0: what each of the STATE_TERMS costs per second at the current state, as its
        distance from its best value (a reward between 0 and 1: what is missing to 1, x its weight; a penalty:
        itself, x its weight).  Term 2 is a weighted sum of these."""
        sig, _ = self._signals(self.qd)
        zero = torch.zeros(self.B, dtype=self.dtype, device=self.device)
        terms, _ = self.task.terms(sig, self.mem, self.actions, self.actions, self.actions, zero, zero.bool())
        cols = []
        for k in STATE_TERMS:
            w = self.task.rewards.get(k, 0.0)
            cols.append(w * (1.0 - terms[k]) if w > 0 else -w * terms[k] if w < 0 else zero)
        return torch.stack(cols, -1)

    @staticmethod
    def fixed_term2_weights(seconds: float = 1.0, kinetic: float = 0.0) -> tuple:
        """The weights of STATE_COSTS that give terminal_cost(seconds)."""
        return (seconds,) * len(STATE_TERMS)

    def fallen(self) -> torch.Tensor:
        """The episode ends (rl/walk/task.py fallen): the firmware's tilt abort, a body other than a foot on the
        ground, or the trunk down."""
        q = self.plant.q.detach()
        ground = (self.contact.detach() @ self.task.not_sole) > 0
        return (self.contract.roll_pitch(q[:, 3:7]).abs() > C.TILT_ABORT).any(-1) | ground | (q[:, 2] < C.FALL_HEIGHT)


__all__ = ["GradWalk", "NA", "OBS_NOISE", "REWARDS", "STATE_TERMS"]

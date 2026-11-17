r"""The stand task on the differentiable pH plant, and what every task on it shares (GradEnv): what the professor's
formula is minimised on.

    L = sum over the window of (running cost, term 1) + (cost at the end of the window, term 2),
    x following the model  x' = (J - R) grad H + G u,  u from the controller (network + PD driver).

Full Body's rl/stand task rebuilt on the Python plant (ph_k1/plant.py + ph_k1/firmware.py are PyTorch), so autograd
reaches from the cost back to the network's weights:

    observation (rl/stand/contract.py) -> policy -> decode -> firmware chain (clamp, slew, delay, motor driver)
    -> 20 plant steps of 1 ms -> the smooth reward terms of rl/stand/env.py

Left out of the reward: the terms without a derivative (alive, feet_grounded, undesired_contacts, termination).

Memory: with `checkpoint=True` every control step is its own torch.utils.checkpoint block: the forward pass keeps
only the states between control steps, the backward pass recomputes one control step at a time.  Memory then
depends on the number of robots, not on the length of the window.
"""

from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from mbrl import paths  # noqa: F401  (puts Full Body on sys.path)
from ph_k1 import firmware as fwm
from ph_k1.model import load_k1
from ph_k1.plant import K1PHPlant, Ledger
from ph_k1 import solve as ph_solve
from rl.stand import contract as C
from rl.stand.env import REWARDS

SMOOTH = ("orientation", "base_height", "base_lin_vel", "base_ang_vel", "joint_vel", "joint_acc", "action_rate",
          "similar_to_default", "joint_limits", "action_excess", "copper_loss")
POLICY_DT, SUBSTEPS = 0.02, 20
MIN_HEIGHT = 0.28
NA = C.NUM_ACTIONS


class GradEnv:
    """What every task on the differentiable plant shares: the plant and the firmware chain, the random draws, the
    transport delays, and a control step's state by name (kept and put back by the learner's backward pass).

    A task (GradStand here, GradWalk in walk_env.py) adds: reset, observe, step, terminal_cost, fallen, and
        FLOAT_STATE   names of the state a derivative flows through
        ENV_KEYS      this env's own attributes that belong to the state
    A task with a memory keeps it in `self.mem` (a dict of tensors); its entries are state under "mem:<name>".
    """

    PLANT_KEYS = ("q", "p", "xi", "lam")
    FW_KEYS = ("ring", "last_target", "ring_idx", "lag")
    ENV_KEYS: tuple = ()
    FLOAT_STATE: tuple = ()
    mem: dict | None = None

    def _build(self, robots: int, seed: int, device, dtype, gains: str, slew: bool, delay_ms) -> None:
        """Generator, model, plant and the firmware's settings (the firmware itself after the task's first reset)."""
        self.B = robots
        self.device, self.dtype = torch.device(device), dtype
        self.gen = torch.Generator().manual_seed(seed)          # on the CPU: the same draws on every device
        ph_solve.probe(self.device, robots, dtype=dtype)       # a GPU without batched solves: solve on the CPU
        self.model = load_k1(dtype=dtype, device=self.device)
        self.plant = K1PHPlant(model=self.model, batch=robots, timestep=POLICY_DT / SUBSTEPS)
        # the lag is drawn here (redraw_lags), not inside policy_step: a recomputed control step must repeat exactly
        self.fw_cfg = fwm.FirmwareConfig(gains=gains, slew=slew, delay_ms=tuple(delay_ms), policy_dt=POLICY_DT,
                                         delay_hold_prob=None)

    # ------------------------------------------------------------------ helpers
    def _rand(self, shape) -> torch.Tensor:
        return torch.rand(shape, generator=self.gen, dtype=torch.float64).to(self.device, self.dtype)

    def _uniform(self, lo, hi, shape) -> torch.Tensor:
        return lo + (hi - lo) * self._rand(shape)

    def _z(self, n: int) -> torch.Tensor:
        return torch.zeros(self.B, n, dtype=self.dtype, device=self.device)

    # ------------------------------------------------------------------ state
    def detach(self) -> None:
        """Cut the graph: the next window starts from these numbers."""
        pl, fw = self.plant, self.fw
        for k in ("q", "p", "xi", "lam", "ball"):
            setattr(pl, k, getattr(pl, k).detach())
        pl.ledger = Ledger.zeros(self.B, pl.dtype, pl.device)
        fw.ring, fw.last_target = fw.ring.detach().clone(), fw.last_target.detach()
        for k in self.ENV_KEYS:
            setattr(self, k, getattr(self, k).detach())
        if self.mem is not None:
            self.mem = {k: v.detach() for k, v in self.mem.items()}

    def get_state(self) -> dict:
        """Everything a control step reads, by name."""
        pl, fw = self.plant, self.fw
        state = {k: getattr(pl, k) for k in self.PLANT_KEYS}
        state.update({k: getattr(fw, k) for k in self.FW_KEYS})
        state.update({k: getattr(self, k) for k in self.ENV_KEYS})
        if self.mem is not None:
            state.update({f"mem:{k}": v for k, v in self.mem.items()})
        return state

    def set_state(self, state: dict) -> None:
        pl, fw = self.plant, self.fw
        for k in self.PLANT_KEYS:
            setattr(pl, k, state[k])
        for k in self.FW_KEYS:
            setattr(fw, k, state[k])
        for k in self.ENV_KEYS:
            setattr(self, k, state[k])
        if self.mem is not None:
            self.mem = {k: state[f"mem:{k}"] for k in self.mem}

    def redraw_lags(self, hold_prob: float = 0.3) -> None:
        """Between control steps: some robots get a new transport delay (as rl/stand's delay_hold_prob)."""
        lo, hi = self.fw_cfg.delay_ms
        if hi <= lo:
            return
        redraw = self._rand((self.B,)) >= hold_prob
        if bool(redraw.any()):
            lag = self.fw.lag.clone()       # a new tensor: the old one belongs to control steps already taken
            lag[redraw] = fwm.draw_lags(self.fw_cfg, int(redraw.sum()), self.plant.h, self.device).long()
            self.fw.lag = lag

    def between_steps(self) -> None:
        """Everything random that happens between two control steps, outside the derivative (the learner calls it
        before every step and keeps the state after it)."""
        self.redraw_lags()


class GradStand(GradEnv):
    """rl/stand's task on the differentiable Python plant.

    Quiet by default (no sensor noise, no pushes, 20 ms delay).  `obs_noise` (rl/stand/env.py's dict), `push_dv`
    with `push_interval_s`, and `delay_ms` as a range switch the randomisation of the other rows on.
    """

    # everything a control step reads, by name; FLOAT_STATE are the parts a derivative flows through
    FLOAT_STATE = ("q", "p", "xi", "lam", "ring", "last_target", "actions", "qd", "gyro")
    ENV_KEYS = ("actions", "qd", "gyro", "contact", "prev_actions", "prev_qd", "v_body")

    def __init__(self, robots: int, seed: int = 1, joint_noise: float = 0.02, device="cpu", dtype=torch.float64,
                 checkpoint: bool = False, obs_noise: dict | None = None, push_dv: float = 0.0,
                 push_interval_s=(4.0, 8.0), delay_ms=(20.0, 20.0)):
        self.joint_noise, self.use_checkpoint = joint_noise, checkpoint
        self.obs_noise, self.push_dv, self.push_interval_s = obs_noise or {}, push_dv, push_interval_s
        self._build(robots, seed, device, dtype, C.GAINS, C.SLEW, delay_ms)
        self.contract = C.Contract(dtype, self.device)
        self.plant.reset(pose=self.contract.default_qpos)
        self.stand_height = float(self.plant.q[:, 2].mean())
        self.fw = fwm.K1Firmware(self.plant, self.fw_cfg)
        m = self.model   # which contact points are the left and the right sole (as rl/stand/env.py)
        sole, cb = m.contact_is_sole, m.contact_body
        left, right = m.body_names.index("left_foot_link"), m.body_names.index("right_foot_link")
        self.left_sole = (sole & (cb == left)).to(dtype)
        self.right_sole = (sole & (cb == right)).to(dtype)
        self.contact = torch.ones(robots, m.ncontact, dtype=dtype, device=self.device)
        self.push_timer = self._uniform(*push_interval_s, (robots,))
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self, ids: torch.Tensor | None = None) -> None:
        """Start again at the home pose (with a little joint noise): all robots, or the rows `ids`."""
        if ids is None:
            noise = self.joint_noise * (2 * self._rand((self.B, NA)) - 1)
            self.plant.reset(pose=self.contract.default_qpos + noise)
            self.fw.reset()
            self.actions, self.prev_actions = self._z(NA), self._z(NA)
            self.qd, self.prev_qd, self.gyro, self.v_body = self._z(NA), self._z(NA), self._z(3), self._z(3)
            return
        ids = ids.reshape(-1)
        if ids.numel() == 0:
            return
        self.detach()                                           # the in-place row writes below need plain tensors
        noise = self.joint_noise * (2 * self._rand((ids.numel(), NA)) - 1)
        self.plant.reset(pose=self.contract.default_qpos + noise, env_ids=ids)
        self.fw.reset(env_ids=ids)
        for k in ("actions", "prev_actions", "qd", "prev_qd", "gyro", "v_body"):
            t = getattr(self, k).clone()
            t[ids] = 0.0
            setattr(self, k, t)

    def draw_pushes(self) -> torch.Tensor | None:
        """rl/stand's pushes: now and then a horizontal change of the trunk velocity.  -> (B,3) or None."""
        if self.push_dv <= 0:
            return None
        self.push_timer = self.push_timer - POLICY_DT
        fire = self.push_timer <= 0
        dv = self._z(3)
        n = int(fire.sum())
        if n:
            angle = self._uniform(0.0, 2 * torch.pi, (n,))
            speed = self._uniform(0.0, self.push_dv, (n,))
            dv[fire] = torch.stack((speed * angle.cos(), speed * angle.sin(), torch.zeros_like(speed)), -1)
            self.push_timer[fire] = self._uniform(*self.push_interval_s, (n,))
        return dv

    # ------------------------------------------------------------------ observations
    def _noisy(self, x, key):
        s = self.obs_noise.get(key, 0.0)
        return x if s <= 0 else x + s * (2 * self._rand(x.shape) - 1)

    def observe(self, noise: bool = True) -> torch.Tensor:
        q = self.plant.q
        theta, quat = q[:, 7:], q[:, 3:7]
        if not noise or not self.obs_noise:
            return self.contract.observe(theta, self.qd, self.gyro, quat, self.actions)
        g = self.contract.projected_gravity(quat)
        return self.contract.observe(self._noisy(theta, "joint_pos"), self._noisy(self.qd, "joint_vel"),
                                     self._noisy(self.gyro, "gyro"), quat, self.actions, gravity=self._noisy(g, "gravity"))

    def critic_obs(self) -> torch.Tensor:
        """rl/stand/env.py's critic input: clean observation | trunk velocity | height error | feet on the ground."""
        q = self.plant.q
        feet = torch.stack(((self.contact @ self.left_sole) > 0, (self.contact @ self.right_sole) > 0), -1).to(self.dtype)
        return torch.cat((self.observe(noise=False), self.v_body, q[:, 2:3] - self.stand_height, feet), -1)

    # ------------------------------------------------------------------ one control step, as a function
    def _advance(self, a, push, lag, q, p, xi, lam, ring, ring_idx, last_target, actions, qd):
        """(action, push, state before) -> (reward, copper watts, reward terms, contacts, state after).

        Everything it reads comes in through the arguments, so torch.utils.checkpoint can run it again in the
        backward pass and get the same numbers."""
        pl, fw = self.plant, self.fw
        pl.q, pl.p, pl.xi, pl.lam = q, p, xi, lam
        fw.ring, fw.ring_idx, fw.last_target = ring.clone(), ring_idx, last_target   # policy_step writes the ring in place
        fw.lag = lag            # this step's transport delays (redraw_lags changes fw.lag between steps)
        pl.ledger = Ledger.zeros(self.B, pl.dtype, pl.device)
        info = fw.policy_step(self.contract.decode(a), SUBSTEPS, push_dv=push)
        nu = pl.velocity()
        qd1, v_body, gyro = nu[:, 6:], nu[:, 3:6], info.extra["gyro"]
        copper_w = pl.ledger.copper / POLICY_DT
        q1, m = pl.q, self.model
        theta = q1[:, 7:]
        g = self.contract.projected_gravity(q1[:, 3:7])
        terms = {
            "orientation": g[:, :2].square().sum(-1),
            "base_height": (q1[:, 2] - self.stand_height).square(),
            "base_lin_vel": v_body.square().sum(-1),
            "base_ang_vel": gyro.square().sum(-1),
            "joint_vel": qd1.square().sum(-1),
            "joint_acc": ((qd1 - qd) / POLICY_DT).square().sum(-1),
            "action_rate": (a - actions).square().sum(-1),
            "similar_to_default": (theta - self.contract.default_qpos).abs().sum(-1),
            "joint_limits": ((m.jnt_low - theta).clamp_min(0) + (theta - m.jnt_high).clamp_min(0)).sum(-1),
            "action_excess": ((a.abs() - 1.0).clamp_min(0).sum(-1)).clamp(max=10.0),
            "copper_loss": copper_w,
        }
        weighted = torch.stack([REWARDS[k] * terms[k] * POLICY_DT for k in SMOOTH])        # (terms, B)
        return (weighted.sum(0), copper_w, weighted, info.contact_active.to(self.dtype),
                pl.q, pl.p, pl.xi, pl.lam, fw.ring, fw.ring_idx, fw.last_target, qd1, v_body, gyro)

    def step(self, a: torch.Tensor, push: torch.Tensor | None = None):
        """-> (reward (B,) of the smooth terms, dict of logged numbers).  Differentiable in a and in the state."""
        pl, fw = self.plant, self.fw
        if push is None:
            push = self._z(3)
        lag = fw.lag.clone()    # a copy: a recomputed step must see the delays it had
        state = (pl.q, pl.p, pl.xi, pl.lam, fw.ring, fw.ring_idx, fw.last_target, self.actions, self.qd)
        if self.use_checkpoint and torch.is_grad_enabled():
            out = torch_checkpoint(self._advance, a, push, lag, *state, use_reentrant=False)
        else:
            out = self._advance(a, push, lag, *state)
        reward, copper_w, weighted, self.contact = out[:4]
        pl.q, pl.p, pl.xi, pl.lam, fw.ring, fw.ring_idx, fw.last_target = out[4:11]
        self.prev_qd, self.prev_actions = self.qd, self.actions
        self.qd, self.v_body, self.gyro = out[11:14]
        self.actions = a
        return reward, {"copper_W": copper_w.detach(), "terms": {k: weighted[i].detach() for i, k in enumerate(SMOOTH)}}

    # ------------------------------------------------------------------ term 2 and falls
    def terminal_cost(self, seconds: float = 1.0, kinetic: float = 1.0) -> torch.Tensor:
        """Term 2 of the professor's formula, ||x(T) - x_T||^2 with weights: x_T = the home pose, upright, at rest.

        "At rest" in pH terms: the kinetic energy T = 1/2 p^T M^-1 p [J] at the end of the window (weight `kinetic`
        per joule).  The pose, tilt and height errors are charged as if they stayed for `seconds` more (the running
        cost's own weights), so term 2 stands for the part of term 1 that the window does not see.  -> (B,) >= 0."""
        pl = self.plant
        q = pl.q
        nu = pl.velocity()
        T = 0.5 * (pl.p * nu).sum(-1)
        g = self.contract.projected_gravity(q[:, 3:7])
        stay = (-REWARDS["orientation"] * g[:, :2].square().sum(-1)
                - REWARDS["base_height"] * (q[:, 2] - self.stand_height).square()
                - REWARDS["similar_to_default"] * (q[:, 7:] - self.contract.default_qpos).abs().sum(-1))
        return kinetic * T + seconds * stay

    def fallen(self) -> torch.Tensor:
        rp = self.contract.roll_pitch(self.plant.q[:, 3:7].detach())
        return (rp.abs() > C.TILT_ABORT).any(-1) | (self.plant.q[:, 2].detach() < MIN_HEIGHT)

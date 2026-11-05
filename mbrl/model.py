r"""Turn a training plant into the fitted model: a parameter ensemble (+ an optional learned residual).

attach(plant, fit, residual) wraps the plant's reset and policy_step, so it works under any env built on
K1PHPlantNative (rl.stand, rl.getup) without changing the env:
  * at every reset each robot draws its own parameter set from the fit: one of the CEM elites (mode "elites", the
    ensemble of Lecture 11 / PDDM: a policy cannot exploit one wrong model) or a sample of the fitted Gaussian
    (mode "gaussian"); it sets the robot's row of plant.robot_params and its factors on the driver gains;
  * the transport delay is drawn from the range the elites span;
  * with a residual (mbrl/residual.py) every policy step adds its joint torque through the plant's gen_force port
    (its work is booked in ledger.mech_input, so the energy balance stays closed); each robot uses one ensemble
    member, drawn at reset (as PETS' trajectory sampling).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from . import paths  # noqa: F401
from .params import ParamSpace
from .replay import projected_gravity
from ph_k1.model import NB, NJ, NV
from ph_k1.robot_params import neutral_rows


class FittedModel:
    def __init__(self, space: ParamSpace, mean: torch.Tensor, std: torch.Tensor, elites: torch.Tensor,
                 elite_loss: torch.Tensor | None = None, source: str = ""):
        self.space, self.mean, self.std, self.elites = space, mean, std, elites
        self.elite_loss = elite_loss
        self.source = source

    @classmethod
    def load(cls, path: str | Path) -> FittedModel:
        d = json.loads(Path(path).read_text())
        space = ParamSpace()
        if d["space"] != space.names:
            raise ValueError(f"{path}: parameter names differ from mbrl/params.py (fit made with another version)")
        T = lambda x: torch.tensor(x, dtype=torch.float64)  # noqa: E731
        return cls(space, T(d["mean"]), T(d["std"]), T(d["elites"]), T(d.get("elite_loss", [])), str(path))

    def sample(self, n: int, generator: torch.Generator | None = None, mode: str = "elites") -> torch.Tensor:
        if mode == "elites":
            idx = torch.randint(len(self.elites), (n,), generator=generator)
            return self.elites[idx]
        if mode == "gaussian":
            return self.space.clamp(self.mean + self.std * torch.randn(n, self.space.D, generator=generator,
                                                                       dtype=torch.float64))
        if mode == "mean":
            return self.mean.expand(n, -1).clone()
        raise ValueError(f"mode {mode!r}: elites | gaussian | mean")

    def delay_range(self, min_width_ms: float = 5.0) -> tuple[float, float]:
        d = self.space.delay_ms(self.elites)
        lo, hi = float(d.min()), float(d.max())
        if hi - lo < min_width_ms:
            c = 0.5 * (lo + hi)
            lo, hi = c - min_width_ms / 2, c + min_width_ms / 2
        return max(lo, 0.0), hi


def attach(plant, fit: FittedModel | None = None, residual=None, mode: str = "elites", seed: int = 0,
           fit_delay: bool = True):
    """Wrap plant.reset / plant.policy_step (in place); returns the plant.  plant.mbrl holds the per-robot state."""
    B, dev, dt = plant.B, plant.device, plant.dtype
    gen = torch.Generator().manual_seed(seed)
    st = SimpleNamespace(kp_scale=torch.ones(B, NJ, dtype=dt, device=dev), kd_scale=torch.ones(B, NJ, dtype=dt, device=dev),
                         member=torch.zeros(B, dtype=torch.long, device=dev), gyro=torch.zeros(B, 3, dtype=dt, device=dev),
                         theta=None, fit=fit, residual=residual, mode=mode)
    if fit is not None:
        plant.robot_params = neutral_rows(B, dt, dev)
        st.theta = fit.mean.expand(B, -1).clone()

    def draw(env_ids=None):
        ids = torch.arange(B, device=dev) if env_ids is None else torch.as_tensor(env_ids, device=dev).long().reshape(-1)
        if ids.dtype == torch.bool:
            ids = ids.nonzero().squeeze(-1)
        n = int(ids.numel())
        if n == 0:
            return
        if fit is not None:
            theta = fit.sample(n, gen, mode)
            sp = fit.space
            plant.robot_params[ids] = sp.robot_params(theta).to(dev, dt)
            kps, kds = sp.gain_scales(theta)
            st.kp_scale[ids], st.kd_scale[ids] = kps.to(dev, dt), kds.to(dev, dt)
            st.theta[ids] = theta
        if residual is not None:
            st.member[ids] = torch.randint(residual.K, (n,), generator=gen).to(dev)
        st.gyro[ids] = 0.0

    orig_reset, orig_step = plant.reset, plant.policy_step

    def reset(*args, **kwargs):
        orig_reset(*args, **kwargs)
        draw(kwargs.get("env_ids"))

    def policy_step(targets, substeps=None, kp=None, kd=None, gain_scale=None, push_dv=None, trunk_force=None,
                    gen_force=None):
        if kp is None or kd is None:
            gkp, gkd = plant.gains(gain_scale=gain_scale)
            kp = gkp if kp is None else kp
            kd = gkd if kd is None else kd
        kp, kd = kp * st.kp_scale, kd * st.kd_scale
        if residual is not None:
            qd = plant.velocity()[:, NB:]
            tau = residual(plant.theta, qd, projected_gravity(plant.q[:, 3:7]), st.gyro, targets, member=st.member)
            g = torch.zeros(B, NV, dtype=dt, device=dev)
            g[:, NB:] = tau.to(dt)
            gen_force = g if gen_force is None else torch.as_tensor(gen_force, dtype=dt, device=dev) + g
        info = orig_step(targets, substeps, kp=kp, kd=kd, gain_scale=gain_scale, push_dv=push_dv,
                         trunk_force=trunk_force, gen_force=gen_force)
        st.gyro = info.extra["gyro"].clone()
        return info

    plant.reset, plant.policy_step = reset, policy_step
    if fit is not None and fit_delay:
        lo, hi = fit.delay_range()
        plant.firmware.delay_ms = (lo, hi)
        plant.set_delay(lo, hi)
    draw(None)
    plant.mbrl = st
    return plant

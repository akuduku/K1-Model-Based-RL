r"""The parameter vector theta that system identification fits, and how it becomes plant settings.

theta (D = 24), grouped so that two get-up logs can identify it (left/right share their values):

    contact    log k_n, log c_n, log mu          factors on ContactParams (k_t follows k_n, c_t follows c_n)
    per group  damp, log tau, log kp, log kd      g in head, arm, hip, knee, ankle:
               damp  added passive damping [N m s/rad] (>= 0)
               tau   factor on the motor torque ceiling
               kp/kd factors on the driver gains (the robot's effective PD vs the commanded one)
    delay      transport delay [ms]

theta = neutral() is today's plant (factors 1, no added damping) with a 20 ms delay (the best fit of the
25 Oct front get-up replay).  Rows for the plant: robot_params(theta) (B, NRP) and gain_scales(theta) (B, 22) x2.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from . import paths  # noqa: F401
from ph_k1.model import NJ
from ph_k1.robot_params import NRP, RP_CN, RP_CT, RP_DAMP, RP_KN, RP_KT, RP_MU, RP_TAU, neutral_rows

# joint groups in firmware / MJCF order (ph_k1 theta)
GROUPS = {
    "head": (0, 1),
    "arm": (2, 3, 4, 5, 6, 7, 8, 9),
    "hip": (10, 11, 12, 16, 17, 18),
    "knee": (13, 19),
    "ankle": (14, 15, 20, 21),
}
GROUP_OF_JOINT = [next(g for g, js in GROUPS.items() if k in js) for k in range(NJ)]


@dataclass(frozen=True)
class Spec:
    name: str
    init: float     # neutral value
    std: float      # CEM start spread
    lo: float
    hi: float


def _specs() -> list[Spec]:
    s = [Spec("log_k_n", 0.0, 0.5, -2.3, 2.3), Spec("log_c_n", 0.0, 0.5, -2.3, 2.3), Spec("log_mu", 0.0, 0.25, -1.0, 0.5)]
    for g in GROUPS:
        s += [Spec(f"{g}.damp", 0.0, 0.3, 0.0, 3.0), Spec(f"{g}.log_tau", 0.0, 0.2, -1.2, 0.4),
              Spec(f"{g}.log_kp", 0.0, 0.2, -1.0, 1.0), Spec(f"{g}.log_kd", 0.0, 0.3, -1.5, 1.5)]
    s.append(Spec("delay_ms", 20.0, 10.0, 0.0, 80.0))
    return s


class ParamSpace:
    def __init__(self):
        self.specs = _specs()
        self.names = [p.name for p in self.specs]
        self.D = len(self.specs)
        self.index = {n: i for i, n in enumerate(self.names)}
        f = lambda attr: torch.tensor([getattr(p, attr) for p in self.specs], dtype=torch.float64)  # noqa: E731
        self.init, self.std, self.lo, self.hi = f("init"), f("std"), f("lo"), f("hi")

    def neutral(self, n: int = 1) -> torch.Tensor:
        return self.init.expand(n, self.D).clone()

    def clamp(self, theta: torch.Tensor) -> torch.Tensor:
        lo, hi = self.lo.to(theta), self.hi.to(theta)
        return torch.maximum(torch.minimum(theta, hi), lo)

    def _per_joint(self, theta: torch.Tensor, key: str) -> torch.Tensor:
        """(N, 22) from the group entries `<group>.<key>`."""
        cols = [self.index[f"{GROUP_OF_JOINT[k]}.{key}"] for k in range(NJ)]
        return theta[:, cols]

    def robot_params(self, theta: torch.Tensor) -> torch.Tensor:
        """(N, NRP) plant rows (ph_k1/robot_params.py)."""
        N = theta.shape[0]
        r = neutral_rows(N, theta.dtype, theta.device)
        kn, cn, mu = (torch.exp(theta[:, self.index[n]]) for n in ("log_k_n", "log_c_n", "log_mu"))
        r[:, RP_KN], r[:, RP_KT] = kn, kn
        r[:, RP_CN], r[:, RP_CT] = cn, cn
        r[:, RP_MU] = mu
        r[:, RP_DAMP:RP_DAMP + NJ] = self._per_joint(theta, "damp").clamp_min(0.0)
        r[:, RP_TAU:RP_TAU + NJ] = torch.exp(self._per_joint(theta, "log_tau"))
        assert r.shape[1] == NRP
        return r

    def gain_scales(self, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Factors on the commanded kp, kd: (N, 22) each."""
        return torch.exp(self._per_joint(theta, "log_kp")), torch.exp(self._per_joint(theta, "log_kd"))

    def delay_ms(self, theta: torch.Tensor) -> torch.Tensor:
        return theta[:, self.index["delay_ms"]]

    def describe(self, theta: torch.Tensor) -> dict[str, float]:
        """Readable values of one theta (factors instead of logs)."""
        out = {}
        for n, v in zip(self.names, theta.reshape(-1).tolist()):
            if ".log_" in n or n.startswith("log_"):
                out[n.replace("log_", "") + "_factor"] = float(torch.exp(torch.tensor(v)))
            else:
                out[n] = v
        return out

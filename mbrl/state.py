r"""Put plant robots into a measured robot state (one log sample per robot).

The log has the joint angles and speeds, the gyro and (through the tilt filter) the trunk orientation without yaw.
It has neither the trunk position nor its linear velocity.  They come from the legs:
  * height: the lowest of the plant's 128 contact points is put on the floor (K1PHPlant.reset with base z=None);
  * linear velocity: that lowest point is taken as a stance point that does not move, so
        J_point(q) nu = 0,  nu = (omega_B = gyro, v_B, theta_dot)   ->  solved for v_B (3 equations, 3 unknowns).
Yaw and the horizontal position are free (flat floor): 0.
With preload=True the robot is then lowered until the plant's contact springs carry its weight (weight_sink): the
plant's floor is a spring, and a robot that starts just touching it first drops into it and bounces for ~20 ms.
The coil currents are set to what the motor driver would hold (init_currents), not to 0, so the motors do not
start with a 50 ms current ramp.
"""

from __future__ import annotations

import torch

from . import paths  # noqa: F401
from ph_k1.spatial import skew
from ph_k1.model import NB


def set_state(plant, theta: torch.Tensor, qd: torch.Tensor, quat: torch.Tensor, gyro: torch.Tensor,
              clearance: float = 0.0, stance_velocity: bool = True, preload: bool = False) -> torch.Tensor:
    """Every robot of `plant` (B rows each) to the measured state.  Returns the stance-point index per robot.
    stance_velocity=False: the trunk's linear velocity is 0 instead (how much does the stance-point guess matter).
    preload=True: the contact springs start with the robot's weight on them (the floor force is not measured; its
    mean is the weight)."""
    B = plant.B
    dt, dev = plant.dtype, plant.device
    T = lambda x: torch.as_tensor(x, dtype=dt, device=dev).reshape(B, -1)  # noqa: E731
    theta, qd, quat, gyro = T(theta), T(qd), T(quat), T(gyro)
    plant.reset(pose=theta, base_quat=quat, base_pos=(0.0, 0.0, None), clearance=clearance)
    if preload:
        q = plant.q.clone()
        q[:, 2] -= weight_sink(plant)
        plant.q = q.contiguous()
    kin = plant.kinematics()
    m = plant.model
    pc = plant._sole_points(kin)  # (B, nc, 3): all contact points
    low = pc[..., 2].argmin(-1)
    rows = torch.arange(B, device=dev)
    Jb = kin.J[rows, m.contact_body[low]]                    # (B, 6, 28) Jacobian of the stance point's body
    x = pc[rows, low]                                         # (B, 3) the point, world
    J = Jb[:, 3:] - skew(x) @ Jb[:, :3]                      # (B, 3, 28) its linear velocity
    rhs = -(J[:, :, :3] @ gyro[..., None] + J[:, :, NB:] @ qd[..., None]).squeeze(-1)
    v_body = torch.linalg.solve(J[:, :, 3:NB], rhs) if stance_velocity else torch.zeros_like(rhs)
    nu = torch.cat((gyro, v_body, qd), -1)
    plant.p = (kin.M @ nu[..., None]).squeeze(-1).contiguous()
    return low


def weight_sink(plant) -> torch.Tensor:
    """(B,) how far the robot has to sink so that the normal springs of its contact points carry its weight:
    sum_i k_n max(0, d - (z_i - z_min)) = m g, solved exactly on the sorted point heights."""
    z = plant._sole_points(plant.kinematics())[..., 2]
    gap = (z - z.min(-1, keepdim=True).values).sort(-1).values          # (B, nc) ascending, gap[:, 0] = 0
    k_n = plant.params().k_n
    k_n = k_n.reshape(-1) if isinstance(k_n, torch.Tensor) else torch.full((plant.B,), float(k_n), dtype=plant.dtype,
                                                                           device=plant.device)
    need = plant.model.total_mass * 9.81 / k_n                           # (B,) sum of the penetrations [m]
    n = torch.arange(1, gap.shape[1] + 1, dtype=plant.dtype, device=plant.device)
    d = (need[:, None] + gap.cumsum(-1)) / n                             # the sink if exactly the n lowest points touch
    nxt = torch.cat((gap[:, 1:], torch.full_like(gap[:, :1], float("inf"))), -1)
    ok = (d >= gap) & (d <= nxt)
    first = ok.float().argmax(-1)
    return d[torch.arange(plant.B, device=plant.device), first]


def init_currents(plant, target: torch.Tensor, kp: torch.Tensor, kd: torch.Tensor, qd: torch.Tensor) -> None:
    """Coil currents = the driver's PD torque at this state (clipped at the robot's torque ceiling) / k."""
    mot = plant.motors
    cap = plant.params().tau_max
    tau = kp * (target - plant.theta) - kd * qd
    tau = torch.maximum(torch.minimum(tau, cap), -cap)
    plant.lam = (mot.L * tau / mot.k).contiguous()

r"""Put plant robots into a measured robot state (one log sample per robot).

The log has the joint angles and speeds, the gyro and (through the tilt filter) the trunk orientation without yaw.
It has neither the trunk position nor its linear velocity.  They come from the legs:
  * height: the lowest of the plant's 128 contact points is put on the floor (K1PHPlant.reset with base z=None);
  * linear velocity: that lowest point is taken as a stance point that does not move, so
        J_point(q) nu = 0,  nu = (omega_B = gyro, v_B, theta_dot)   ->  solved for v_B (3 equations, 3 unknowns).
Yaw and the horizontal position are free (flat floor): 0.
The coil currents are set to what the motor driver would hold (init_currents), not to 0, so the motors do not
start with a 50 ms current ramp.
"""

from __future__ import annotations

import torch

from . import paths  # noqa: F401
from ph_k1.spatial import skew
from ph_k1.model import NB


def set_state(plant, theta: torch.Tensor, qd: torch.Tensor, quat: torch.Tensor, gyro: torch.Tensor,
              clearance: float = 0.0) -> torch.Tensor:
    """Every robot of `plant` (B rows each) to the measured state.  Returns the stance-point index per robot."""
    B = plant.B
    dt, dev = plant.dtype, plant.device
    T = lambda x: torch.as_tensor(x, dtype=dt, device=dev).reshape(B, -1)  # noqa: E731
    theta, qd, quat, gyro = T(theta), T(qd), T(quat), T(gyro)
    plant.reset(pose=theta, base_quat=quat, base_pos=(0.0, 0.0, None), clearance=clearance)
    kin = plant.kinematics()
    m = plant.model
    pc = plant._sole_points(kin)  # (B, nc, 3): all contact points
    low = pc[..., 2].argmin(-1)
    rows = torch.arange(B, device=dev)
    Jb = kin.J[rows, m.contact_body[low]]                    # (B, 6, 28) Jacobian of the stance point's body
    x = pc[rows, low]                                         # (B, 3) the point, world
    J = Jb[:, 3:] - skew(x) @ Jb[:, :3]                      # (B, 3, 28) its linear velocity
    rhs = -(J[:, :, :3] @ gyro[..., None] + J[:, :, NB:] @ qd[..., None]).squeeze(-1)
    v_body = torch.linalg.solve(J[:, :, 3:NB], rhs)
    nu = torch.cat((gyro, v_body, qd), -1)
    plant.p = (kin.M @ nu[..., None]).squeeze(-1).contiguous()
    return low


def init_currents(plant, target: torch.Tensor, kp: torch.Tensor, kd: torch.Tensor, qd: torch.Tensor) -> None:
    """Coil currents = the driver's PD torque at this state (clipped at the robot's torque ceiling) / k."""
    mot = plant.motors
    cap = plant.params().tau_max
    tau = kp * (target - plant.theta) - kd * qd
    tau = torch.maximum(torch.minimum(tau, cap), -cap)
    plant.lam = (mot.L * tau / mot.k).contiguous()

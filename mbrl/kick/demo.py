r"""A kick policy's leg motion as a plan: where the planner's searches can start (--init-demo).

The policy (e.g. Eric Plass's k1_kick kick-018, trained in MuJoCo) kicks a ball that lies at the plan's stand-off
position, from the stance, on the pH plant.  What it does with the 12 leg joints BEYOND what the walk policy would do
in the same state is the kick motion in the plan's own terms (mbrl/kick/controller.py: offsets on the walk policy's
action).  That motion is read at four times around the touch and becomes the keyframes; left-foot kicks are mirrored
onto the right foot and averaged with the right-foot ones.

This is a starting point, not the plan: replayed open loop through the walk policy it need not kick as well as the
policy does with its own feedback.  The planner then searches around it in the pH model.
"""

from __future__ import annotations

import torch

from .. import paths  # noqa: F401
from .controller import LEG, MIRROR, PlanSpace
from rl.kick import contract as C
from rl.kick import policy as P
from rl.kick.env import KickConfig, KickEnv
from rl.kick.evaluate import quiet

SPAN = 0.15   # the keyframes cover the touch -+ SPAN seconds, evenly spaced


class DemoEnv(KickEnv):
    scenario: dict | None = None

    def _draw_request(self, ids):
        s = self.scenario
        if s is None:
            return super()._draw_request(ids)
        return s["local"][ids].clone(), s["angle"][ids].clone(), s["speed"][ids].clone()


def from_policy(policy_path, walk, speeds, space: PlanSpace | None = None, back: float = 0.32, side: float = 0.09,
                robots: int = 8, seconds: float = 1.6, log=print) -> torch.Tensor:
    """-> z (len(speeds), D): one plan per requested ball speed, from the policy's kicks at that speed."""
    space = space or PlanSpace()
    policy, _ = P.load(policy_path, "cpu", torch.float32)
    cfg = quiet(KickConfig(num_envs=robots, device="cpu", dtype="float32", seed=1, stand_prob=1.0, joint_noise=0.0,
                           yaw_noise=0.0, delay_hold_prob=None, energy_log_envs=0))
    mirror = torch.tensor(MIRROR)
    out = []
    for speed in speeds:
        env = DemoEnv(cfg)
        sign = torch.where(torch.arange(robots) % 2 == 0, -1.0, 1.0)          # even robots: ball right, right foot
        env.scenario = {"local": torch.stack((torch.full((robots,), back), sign * side), -1),
                        "angle": torch.zeros(robots), "speed": torch.full((robots,), float(speed))}
        env._reset_idx(env.index)
        env._refresh()
        env._start_requests()
        steps = int(round(seconds / env.cfg.policy_dt))
        offs, touch = [], torch.full((robots,), -1, dtype=torch.long)
        with torch.no_grad():
            obs = env.get_observations()["policy"]
            for k in range(steps):
                a = policy(obs)
                a_walk = walk(torch.cat((obs[:, :C.FRAME], torch.zeros(robots, 3)), -1).float())
                d = a - a_walk
                right = (env.foot_request == 1)[:, None]
                kick = torch.where(right, d[:, 16:22], d[:, 10:16] * mirror)   # right-foot convention
                stand = torch.where(right, d[:, 10:16], d[:, 16:22] * mirror)
                offs.append(torch.cat((kick, stand), -1))
                obs = env.step(a)[0]["policy"]
                touch = torch.where((touch < 0) & env.kicked, torch.full_like(touch, k), touch)
        offs = torch.stack(offs, 1)                                            # (robots, steps, 12)
        good = (touch > 0) & ~env.terminated
        if not bool(good.any()):
            raise RuntimeError(f"{policy_path}: no kick at {speed} m/s from the stand-off position")
        along = (env.launch_vel * env.target_w).sum(-1)
        t_touch = float(touch[good].float().mean()) * env.cfg.policy_dt
        K = space.keyframes
        first = max(t_touch - SPAN, space.d_min)
        times = torch.linspace(first, max(t_touch + SPAN, first + K * space.d_min), K).tolist()
        keys = []
        for t in times:
            i = min(int(round(t / env.cfg.policy_dt)), steps - 1)
            lo, hi = max(i - 1, 0), min(i + 2, steps)
            keys.append(offs[good][:, lo:hi].mean((0, 1)))
        durations = torch.tensor([times[0]] + [times[i] - times[i - 1] for i in range(1, len(times))])
        z = space.encode(torch.tensor([back]), torch.tensor([side]), torch.stack(keys)[None], durations[None])
        out.append(z[0])
        log(f"[demo] {speed:g} m/s: {int(good.sum())}/{robots} kicks, touch after {t_touch:.2f} s, ball "
            f"{float(along[good].mean()):.2f} m/s; keyframes at {[round(t, 2) for t in times]} s, largest offset "
            f"{float(torch.stack(keys).abs().max()):.1f} action units")
    return torch.stack(out)

"""The model-based kick: the plan's parameters and controller, the planner's code path, the tracking env.

    python -m pytest tests/test_kick_mbrl.py -q        (needs the built ph_k1_native, rsl-rl-lib and tensordict)
"""

from __future__ import annotations

import pytest
import torch

import mbrl  # noqa: F401
from mbrl.kick.controller import LEG, MIRROR, KickController, PlanSpace, interpolate
from mbrl.kick.plan import DEFAULT_WALK, PlanConfig, Planner, load_library, save

pytest.importorskip("tensordict")
pytestmark = pytest.mark.skipif(not DEFAULT_WALK.exists(), reason="no walk policy (policies/walk_teacher.pt)")

from rl.kick import contract as C  # noqa: E402
from rl.kick.env import KickConfig, KickEnv  # noqa: E402
from rl.kick.evaluate import quiet  # noqa: E402


def test_plan_space():
    sp = PlanSpace()
    assert sp.D == 2 + 4 * 12 + 4 == 54
    d = sp.decode(torch.zeros(3, sp.D))
    assert d["offsets"].shape == (3, 4, 12) and (d["offsets"] == 0).all()        # z = 0: no kick motion
    assert torch.allclose(d["back"], torch.full((3,), 0.335)) and torch.allclose(d["durations"], torch.full((3, 4), 0.18))
    wild = sp.decode(50 * torch.randn(8, sp.D))
    assert (wild["offsets"].abs() <= torch.tensor(sp.offset_scale * 2) + 1e-6).all()
    inside = sp.decode(0.5 * torch.randn(5, sp.D))                # encode is decode's inverse inside the ranges
    again = sp.decode(sp.encode(**inside))
    assert all(torch.allclose(inside[k], again[k], atol=1e-4) for k in inside)
    assert (wild["durations"] >= sp.d_min).all() and (wild["durations"] <= sp.d_max).all()
    assert (wild["back"] >= sp.back[0]).all() and (wild["side"] <= sp.side[1]).all()


def test_interpolate_between_speeds():
    sp = PlanSpace()
    lib = sp.decode(torch.randn(3, sp.D))
    speeds = torch.tensor([2.0, 4.0, 6.0])
    got = interpolate(lib, torch.tensor([1.0, 2.0, 3.0, 6.0, 9.0]), speeds)
    assert torch.allclose(got["offsets"][0], lib["offsets"][0]) and torch.allclose(got["offsets"][1], lib["offsets"][0])
    assert torch.allclose(got["offsets"][2], 0.5 * (lib["offsets"][0] + lib["offsets"][1]))
    assert torch.allclose(got["back"][3], lib["back"][2]) and torch.allclose(got["back"][4], lib["back"][2])


@pytest.fixture(scope="module")
def walk():
    return torch.jit.load(str(DEFAULT_WALK), map_location="cpu").eval()


def make_env(n=8, **kw):
    return KickEnv(quiet(KickConfig(num_envs=n, device="cpu", seed=5, stand_prob=1.0, energy_log_envs=0, **kw)))


def test_offsets_follow_the_keyframes_and_mirror(walk):
    env = make_env(4)
    sp = PlanSpace()
    ctrl = KickController(env, walk, sp)
    plan = sp.decode(torch.randn(4, sp.D))
    ctrl.set_plans(plan)
    ends = torch.cumsum(plan["durations"], -1)
    assert (ctrl._offsets(torch.zeros(4)) == 0).all()
    for k in range(sp.keyframes):
        assert torch.allclose(ctrl._offsets(ends[:, k]), plan["offsets"][:, k], atol=1e-5)
    assert torch.allclose(ctrl._offsets(ends[:, -1] + sp.return_s), torch.zeros(4, 12), atol=1e-6)
    # the same plan for a right- and a left-foot kick: mirror images in the action
    env.foot_request[:] = torch.tensor([1, 0, 1, 0])
    same = {k: v[:1].expand(4, *v.shape[1:]) for k, v in plan.items()}
    ctrl.set_plans(same)
    ctrl.t_kick[:] = 0.0                                              # the kick has just been triggered
    env.episode_length_buf[:] = env.request_step + 5                  # 0.1 s into it
    obs = torch.zeros(4, C.OBS_DIM)
    a = ctrl.act(obs)
    base = walk(torch.zeros(1, 75))[0]
    off = a - base
    m = torch.tensor(MIRROR)
    assert torch.allclose(off[0, 16:22], off[1, 10:16] * m, atol=1e-5)     # kicking leg
    assert torch.allclose(off[0, 10:16], off[1, 16:22] * m, atol=1e-5)     # standing leg
    assert off[:, :10].abs().max() < 1e-5                                  # head and arms: the walk policy's own
    assert off[0, 16:22].abs().max() > 0


def test_approach_reaches_the_stand_off_pose(walk):
    """Without a kick motion the plan walks the robot behind the ball, facing the kick direction, and stands."""
    env = make_env(8)
    sp = PlanSpace()
    ctrl = KickController(env, walk, sp, latest_s=10.0)
    ctrl.set_plans(sp.decode(torch.zeros(8, sp.D)))
    obs = env.get_observations()["policy"]
    for _ in range(110):
        obs = env.step(ctrl.act(obs))[0]["policy"]
    psi = torch.atan2(env.target_w[:, 1], env.target_w[:, 0])
    ball = C.to_heading(env.yaw, env.ball_pos - env.base_pos)[:, :2]
    assert not env.terminated.any() and not env.kicked.any()
    assert (C.wrap(psi - env.yaw).abs() < 0.2).all()
    assert (ball[:, 0] - 0.335).abs().max() < 0.08                         # the ball `back` ahead of the trunk
    assert (ball[:, 1].abs() - 0.095).abs().max() < 0.08                   # and `side` towards the kicking foot
    assert ((ball[:, 1] < 0) == (env.foot_request == 1)).all()


def test_planner_and_tracker_code_path(tmp_path, walk):
    cfg = PlanConfig(speeds=(3.5,), candidates=6, variants=2, iterations=1, horizon_s=2.0, tail_s=0.5, device="cpu")
    planner = Planner(cfg, log=lambda *a: None)
    state = planner.search()
    result = planner.final(state)
    path = save(result, state, cfg, planner.space, tmp_path, "t")
    space, speeds, lib, success = load_library(path)
    assert speeds.tolist() == [3.5] and lib["offsets"].shape == (1, 4, 12) and len(success) == 1
    assert len(state["history"]) == 1 and len(result["per_speed"][0]["ok"]) == 2

    from mbrl.kick.track_env import TrackConfig, TrackEnv

    env = TrackEnv(TrackConfig(num_envs=6, device="cpu", seed=2, plan=str(path), stand_prob=0.5, energy_log_envs=0))
    obs = env.get_observations()
    assert obs["policy"].shape == (6, 166)
    walking = ~env.active
    assert walking.any() and env.active.any()
    want = walk(torch.cat((obs["policy"][:, :72], env.commands), -1).float())
    assert torch.allclose(env.teacher_actions()[walking], want[walking], atol=1e-5)   # walking: the walk policy itself
    for _ in range(30):
        obs, rew, dones, extras = env.step(env.teacher_actions())
        assert torch.isfinite(rew).all()
    assert extras["log"]["Track/action_error"] < 1e-5          # the teacher follows itself
    assert "Track/plan_approach" in extras["log"] and obs["critic"].shape[1] == env.get_observations()["critic"].shape[1]

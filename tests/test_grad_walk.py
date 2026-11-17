"""The walk task on the differentiable plant (mbrl/grad/walk_env.py): the same task as on the C++ plant, and the
learner's window works on it."""

from types import SimpleNamespace

import pytest
import torch

from mbrl.grad.walk_env import NA, OBS_NOISE, GradWalk
from rl.walk import contract as C
from rl.walk.backends import CHECKPOINTS
from rl.walk.task import COUNTS, REWARDS

torch.set_num_threads(1)
COMMAND = (0.5, 0.0, 0.0)


def test_both_plants_pay_the_same_for_the_same_motion():
    """Eric's walk drives the C++ environment; the Python environment gets the same actions.  Every reward term of
    every step must agree (the two plants are two programs for the same equations)."""
    if not (CHECKPOINTS / "walk_v9.pt").exists():
        pytest.skip("k1_sprint is not next to this repository")
    from rl.walk.env import WalkConfig, WalkEnv, quiet

    cpp = WalkEnv(quiet(WalkConfig(num_envs=1, device="cpu", dtype="float64", seed=3), fixed_command=COMMAND))
    py = GradWalk(1, seed=3, command=COMMAND)
    net = torch.jit.load(str(CHECKPOINTS / "walk_v9.pt"), map_location="cpu").eval()
    obs = cpp.get_observations()
    worst, total, stepped = {}, [0.0, 0.0], False
    with torch.no_grad():
        assert torch.allclose(obs["policy"], py.observe(), atol=1e-9)
        for _ in range(40):                                  # 0.8 s: the robot starts to walk and takes steps
            a = net(obs["policy"].float()).double()
            obs, reward, _, extras = cpp.step(a)
            py.between_steps()
            _, log = py.step(a)
            for k, w in REWARDS.items():
                d = float((w * C.POLICY_DT * extras["terms"][k] - log["terms"][k]).abs().max())
                worst[k] = max(worst.get(k, 0.0), d)
            total[0] += float(reward)
            total[1] += float(log["reward_all"])
            stepped |= not bool(cpp.sig.contact.all())
            assert torch.equal(cpp.sig.contact, py.sig.contact)
            assert torch.allclose(obs["policy"], py.observe(), atol=1e-5)
    assert stepped                                           # a foot was in the air: the test saw a step
    assert max(worst.values()) < 1e-5, {k: v for k, v in worst.items() if v > 1e-6}
    assert total[0] == pytest.approx(total[1], abs=1e-5)


def test_step_returns_the_terms_with_a_derivative():
    env = GradWalk(2, seed=1, command=COMMAND)
    a = (0.1 * torch.randn(2, NA, dtype=torch.float64)).requires_grad_(True)
    reward, log = env.step(a)
    counted = sum(log["terms"][k] for k in COUNTS)
    assert torch.allclose(reward.detach() + counted, log["reward_all"], atol=1e-12)
    (g,) = torch.autograd.grad(reward.sum(), a)
    assert bool(torch.isfinite(g).all()) and float(g.abs().max()) > 0.0
    assert env.observe().shape == (2, C.OBS_DIM) and env.critic_obs().shape == (2, C.OBS_DIM + 15)
    assert float(env.terminal_cost().min()) >= 0.0 and not bool(env.fallen().any())


def test_restarts_draw_new_requests_and_a_fixed_command_stays():
    env = GradWalk(6, seed=2, joint_noise=0.1, random_yaw=True)
    env.between_steps()                                      # the first step of all: no time has passed yet
    env.step(torch.zeros(6, NA, dtype=torch.float64))
    env.reset(torch.tensor([1, 4]))
    env.between_steps()
    age = env.mem["command_age"]
    assert torch.allclose(age[[0, 2, 3, 5]], torch.full((4,), C.POLICY_DT, dtype=torch.float64))   # their clocks run on
    assert bool((age[[1, 4]] == 0).all())                    # the two that started again: a fresh request, no time passed
    assert float(env.mem["target"][[1, 4], :2].abs().max()) <= env.task.p.accel * C.POLICY_DT + 1e-12   # the speed ramps from 0
    assert float(env.plant.p[[1, 4]].abs().max()) == 0.0 and float(env.actions[[1, 4]].abs().max()) == 0.0
    fixed = GradWalk(3, seed=2, command=COMMAND)
    for _ in range(3):
        fixed.between_steps()
        fixed.step(torch.zeros(3, NA, dtype=torch.float64))
    fixed.reset(torch.tensor([0]))
    assert torch.equal(fixed.mem["commands"], torch.tensor([COMMAND] * 3, dtype=torch.float64))


def test_window_step_by_step_equals_one_graph_on_the_walk():
    """The learner's window on the walk task, with noise, pushes, changing delays, gain errors and changing
    requests: backward one control step at a time gives the same action gradients as one graph."""
    from mbrl.grad.train import window, window_one_graph
    from mbrl.mpc.nets import MLP

    args = SimpleNamespace(steps=4, clip=3.0, action_noise=0.05, gamma=0.99, terminal_seconds=1.0,
                           terminal_kinetic=0.0, fall_weight=1.0)
    world = {"obs_noise": OBS_NOISE, "push_dv": 0.5, "push_interval_s": (0.02, 0.06), "delay_ms": (10.0, 70.0),
             "gain_scale": (0.6, 1.3), "encoder_bias": 0.015, "joint_noise": 0.1, "random_yaw": True}
    torch.manual_seed(1)
    policy = MLP(C.OBS_DIM, NA, last_scale=0.5).double()
    out = []
    for fn in (lambda e: window(e, policy, args)["g"], lambda e: window_one_graph(e, policy, args)):
        env = GradWalk(3, seed=4, **world)
        env.mem["command_time_left"][0] = 0.03               # robot 0 gets a new request inside the window
        torch.manual_seed(7)
        out.append(fn(env))
    assert out[0].shape == (4, 3, NA) and float(out[0].abs().max()) > 0.0
    assert torch.allclose(out[0], out[1], rtol=1e-8, atol=1e-12), float((out[0] - out[1]).abs().max())

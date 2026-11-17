"""The learned term 2 (mbrl/grad/value.py, roadmap W3): the targets, the fit, and the window's derivative through V."""

from types import SimpleNamespace

import pytest
import torch

from mbrl.grad.env import GradStand
from mbrl.grad.train import window, window_one_graph
from mbrl.grad.value import ValueLearner, WeightLearner, lambda_returns
from mbrl.mpc.nets import MLP

torch.set_num_threads(1)


def test_lambda_returns_by_hand():
    g, lam = 0.9, 0.5
    r = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    v = torch.tensor([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0]])
    ends = torch.tensor([[False, False], [False, True], [False, False]])
    out = lambda_returns(r, v, ends, g, lam)
    g2 = 3.0 + g * 40.0                                           # the last step: one-step return
    g1 = 2.0 + g * ((1 - lam) * 30.0 + lam * g2)
    g0 = 1.0 + g * ((1 - lam) * 20.0 + lam * g1)
    assert out[:, 0].tolist() == pytest.approx([g0, g1, g2])
    assert float(out[1, 1]) == 2.0                                 # a fall in step 1: nothing follows it
    assert float(out[0, 1]) == pytest.approx(1.0 + g * ((1 - lam) * 20.0 + lam * 2.0))
    assert lambda_returns(r, v, ends, g, 0.0)[0, 0] == pytest.approx(1.0 + g * 20.0)     # lambda 0: TD(0)


def test_the_value_learns_a_function_of_its_input():
    torch.manual_seed(0)
    learner = ValueLearner(3, dtype=torch.float64, lr=3e-3, buffer=8)
    w = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    explained = []
    for _ in range(30):
        x = torch.randn(256, 3, dtype=torch.float64)
        explained.append(learner.fit(x, x @ w + 3.0)["explained_variance"])
    assert explained[0] < 0.5 < 0.95 < explained[-1]
    x = torch.randn(64, 3, dtype=torch.float64)
    assert float((learner(x) - (x @ w + 3.0)).abs().mean()) < 0.2       # the target copy has followed


def test_the_full_reward_adds_the_counting_terms():
    env = GradStand(2, seed=1, joint_noise=0.0)          # the home pose exactly: both soles on the floor
    r, info = env.step(torch.zeros(2, 22, dtype=torch.float64))
    # standing on both feet, nothing else on the ground, not fallen: alive 1.5 + feet_grounded 0.5, per second
    assert torch.allclose(info["reward_all"] - r, torch.full((2,), 2.0 * 0.02, dtype=torch.float64))
    q = env.plant.q.clone()
    q[0, 3:7] = torch.tensor([0.8776, 0.0, 0.4794, 0.0], dtype=torch.float64)      # robot 0 pitched by 1 rad
    env.plant.q = q
    assert bool(env.fallen()[0]) and not bool(env.fallen()[1])


def args_(**kw):
    return SimpleNamespace(**{"steps": 4, "clip": 3.0, "action_noise": 0.05, "gamma": 0.99, "lam": 0.95,
                              "terminal_seconds": 1.0, "terminal_kinetic": 1.0, "fall_weight": 1.0, **kw})


def a_value(critic_dim: int) -> ValueLearner:
    """A value network that is not flat: its derivative must reach the actions."""
    torch.manual_seed(3)
    v = ValueLearner(critic_dim, dtype=torch.float64)
    with torch.no_grad():
        for p in v.target.parameters():
            p.add_(0.3 * torch.randn_like(p))
    return v


@pytest.mark.parametrize("weight", (1.0, 0.5))
def test_window_with_the_learned_term2_equals_one_graph_on_standing(weight):
    from rl.stand import contract as C

    world = {"obs_noise": {"gyro": 0.2, "joint_pos": 0.01, "joint_vel": 1.0, "gravity": 0.05}, "push_dv": 0.3,
             "push_interval_s": (0.02, 0.06), "delay_ms": (10.0, 70.0)}
    torch.manual_seed(1)
    policy = MLP(C.OBS_DIM, 22, last_scale=0.5).double()
    value = a_value(C.OBS_DIM + 6)
    out = []
    for fn in (lambda e: window(e, policy, args_(), value=value, value_weight=weight),
               lambda e: window_one_graph(e, policy, args_(), value=value, value_weight=weight)):
        env = GradStand(3, seed=4, **world)
        torch.manual_seed(7)
        out.append(fn(env))
    w = out[0]
    assert torch.allclose(w["g"], out[1], rtol=1e-8, atol=1e-12), float((w["g"] - out[1]).abs().max())
    assert w["critic"].shape == (4, 3, C.OBS_DIM + 6) and w["value_target"].shape == (4, 3)
    assert bool(w["valid"].all()) and bool(torch.isfinite(w["value_target"]).all())
    # V's derivative is in the action gradient: the same window with the fixed term 2 gives another one
    env = GradStand(3, seed=4, **world)
    torch.manual_seed(7)
    fixed = window(env, policy, args_())["g"]
    assert float((fixed - w["g"]).abs().max()) > 1e-6


@pytest.mark.parametrize("kind", ("weights", "network"))
def test_window_with_the_learned_term2_equals_one_graph_on_walking(kind):
    from mbrl.grad.walk_env import NA, OBS_NOISE, GradWalk
    from rl.walk import contract as C

    world = {"obs_noise": OBS_NOISE, "push_dv": 0.5, "push_interval_s": (0.02, 0.06), "delay_ms": (10.0, 70.0),
             "gain_scale": (0.6, 1.3), "encoder_bias": 0.015, "joint_noise": 0.1, "random_yaw": True}
    torch.manual_seed(1)
    policy = MLP(C.OBS_DIM, NA, last_scale=0.5).double()
    if kind == "network":
        value = a_value(C.OBS_DIM + 15)
    else:                                                    # weights away from the start, one of them zero
        n = len(GradWalk.STATE_COSTS)
        value = WeightLearner(GradWalk.STATE_COSTS, [0.5 + 0.1 * i if i != 3 else 0.0 for i in range(n)])
    out = []
    for fn in (lambda e: window(e, policy, args_(terminal_kinetic=0.0), value=value)["g"],
               lambda e: window_one_graph(e, policy, args_(terminal_kinetic=0.0), value=value)):
        env = GradWalk(3, seed=4, **world)
        torch.manual_seed(7)
        out.append(fn(env))
    assert float(out[0].abs().max()) > 0.0
    assert torch.allclose(out[0], out[1], rtol=1e-8, atol=1e-12), float((out[0] - out[1]).abs().max())


# ------------------------------------------------------------------ term 2 with learned weights (--term2 weights)
def test_the_fixed_weights_give_the_fixed_term2():
    from mbrl.grad.walk_env import GradWalk

    torch.manual_seed(2)
    stand = GradStand(3, seed=1)
    for _ in range(3):
        stand.step(0.3 * torch.randn(3, 22, dtype=torch.float64))
    w = torch.tensor(stand.fixed_term2_weights(1.3, 0.7), dtype=torch.float64)
    assert torch.allclose(stand.state_costs() @ w, stand.terminal_cost(1.3, 0.7), rtol=1e-12, atol=1e-14)
    walk = GradWalk(3, seed=1, command=(0.5, 0.0, 0.0))
    for _ in range(3):
        walk.step(0.3 * torch.randn(3, 22, dtype=torch.float64))
    w = torch.tensor(walk.fixed_term2_weights(1.3), dtype=torch.float64)
    assert torch.allclose(walk.state_costs() @ w, walk.terminal_cost(1.3), rtol=1e-12, atol=1e-14)
    assert bool((stand.state_costs() >= 0).all()) and bool((walk.state_costs() >= 0).all())


def test_learned_weights_start_as_the_fixed_term2_and_their_derivative_is_exact():
    from rl.stand import contract as C

    world = {"obs_noise": {"gyro": 0.2, "joint_pos": 0.01, "joint_vel": 1.0, "gravity": 0.05}, "push_dv": 0.3,
             "push_interval_s": (0.02, 0.06), "delay_ms": (10.0, 70.0)}
    torch.manual_seed(1)
    policy = MLP(C.OBS_DIM, 22, last_scale=0.5).double()
    learner = WeightLearner(GradStand.STATE_COSTS, GradStand.fixed_term2_weights(1.0, 1.0))
    runs = []
    for value in (None, learner):
        env = GradStand(3, seed=4, **world)
        torch.manual_seed(7)
        runs.append(window(env, policy, args_(), value=value))
    assert torch.allclose(runs[0]["g"], runs[1]["g"], rtol=1e-10, atol=1e-14)   # at the start: the fixed term 2
    assert runs[1]["critic"].shape == (4, 3, len(GradStand.STATE_COSTS))
    with torch.no_grad():                                                        # other weights: still exact
        learner.s_target.copy_(torch.tensor([0.5, 2.0, 0.1, 3.0, 0.4, 0.2, 0.05], dtype=torch.float64))
    out = []
    for fn in (lambda e: window(e, policy, args_(), value=learner)["g"],
               lambda e: window_one_graph(e, policy, args_(), value=learner)):
        env = GradStand(3, seed=4, **world)
        torch.manual_seed(7)
        out.append(fn(env))
    assert torch.allclose(out[0], out[1], rtol=1e-8, atol=1e-12), float((out[0] - out[1]).abs().max())
    assert float((out[0] - runs[0]["g"]).abs().max()) > 1e-6


def test_the_weights_are_learned_from_returns_and_stay_positive():
    torch.manual_seed(0)
    learner = WeightLearner(("a", "b", "c"), (1.0, 1.0, 1.0), lr=3e-2, buffer=8)
    true = torch.tensor([2.0, 0.0, 0.5], dtype=torch.float64)
    for _ in range(200):
        x = torch.rand(256, 3, dtype=torch.float64)
        fit = learner.fit(x, -0.3 - x @ true)
    assert learner.s_target.tolist() == pytest.approx([2.0, 0.0, 0.5], abs=0.05)
    assert fit["level"] == pytest.approx(-0.3, abs=0.05) and bool((learner.s >= 0).all())

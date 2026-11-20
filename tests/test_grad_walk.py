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


def test_both_plants_observe_the_same_clock():
    """With the gait clock (W5c) both environments give the network the same 77 numbers while Eric's walk drives
    them (it reads the first 75): the clock's sin and cos follow the same rule on the C++ and the Python plant."""
    if not (CHECKPOINTS / "walk_v9.pt").exists():
        pytest.skip("k1_sprint is not next to this repository")
    from rl.walk.env import WalkConfig, WalkEnv, quiet

    cpp = WalkEnv(quiet(WalkConfig(num_envs=1, device="cpu", dtype="float64", seed=3, gait_clock=True), fixed_command=COMMAND))
    py = GradWalk(1, seed=3, command=COMMAND, gait_clock=1.0)
    net = torch.jit.load(str(CHECKPOINTS / "walk_v9.pt"), map_location="cpu").eval()
    obs = cpp.get_observations()
    with torch.no_grad():
        for _ in range(30):
            assert obs["policy"].shape == (1, 77)
            assert torch.allclose(obs["policy"], py.observe(), atol=1e-5)
            a = net(obs["policy"][:, :75].float()).double()
            obs, *_ = cpp.step(a)
            py.between_steps()
            py.step(a)
    assert float(obs["policy"][0, 76]) != 1.0                     # the clock has moved on


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


def test_air_time_stand_in_pays_for_the_weight_taken_off_a_foot():
    """air_time's smooth stand-in (W5b) on chosen ground forces, unload histories, foot clocks and requests: it pays
    the unload a foot has kept for 3 control steps (Eric's 0.05 s minimum), for at most 0.5 s of unload."""
    import dataclasses

    n = 10
    env = GradWalk(n, seed=1, command=COMMAND, air_time_smooth=0.1)
    env.step(torch.zeros(n, NA, dtype=torch.float64))
    half, f64 = env.task.half_weight, torch.float64
    assert 90.0 < half < 105.0                                           # 19.7 kg on two feet
    assert env.task.hold_steps == 3                                      # Eric's air clock passes 0.05 s at 0.06 s
    force = torch.zeros(n, 2, 3, dtype=f64)
    force[0, :, 2] = half                                                # both feet carry their half
    force[1, 0, 2] = 2 * half                                            # all on the left, the right foot is up
    force[2, :, 2] = 0.5 * half                                          # each foot carries a quarter (pushing off)
    force[3, 0, 2] = 2 * half                                            # the right foot has been up for 0.6 s
    force[4, 0, 2] = 2 * half                                            # as robot 1, but asked for 0.1 m/s only
    force[5, 0, 2], force[5, 1, 2] = 1.5 * half, 0.5 * half              # three quarters of the weight on the left
    force[6, 0, 2] = 2 * half                                            # the right foot rests on the floor without
    #                                                                      weight for 0.6 s: standing on one leg
    force[7, 0, 2], force[7, 1, 2] = 1.5 * half, 0.5 * half              # rocking: the right foot was loaded a step ago
    force[8, 0, 2], force[8, 1, 2] = 1.5 * half, 0.5 * half              # leaning on the left foot for over 1 s
    force[9, 0, 2], force[9, 1, 2] = 1.05 * half, 0.95 * half            # a quiet stance, a little to the left
    contact = torch.ones(n, 2, dtype=torch.bool)
    contact[[1, 3, 4], 1] = False
    sig = dataclasses.replace(env.sig, contact_force=force, contact=contact)
    mem = dict(env.mem)
    mem["air_time"] = torch.zeros(n, 2, dtype=f64)
    mem["air_time"][[1, 3, 4], 1] = torch.tensor([0.1, 0.6, 0.1], dtype=f64)
    mem["unloaded_time"] = torch.zeros(n, 2, dtype=f64)
    mem["unloaded_time"][[1, 3, 4, 6, 8, 9], 1] = torch.tensor([0.1, 0.6, 0.1, 0.6, 0.5, 0.6], dtype=f64)
    history = torch.zeros(n, 2, 3, dtype=f64)                            # the unloads of the steps before, oldest first
    history[[1, 3, 4, 6], 1] = 1.0
    history[2], history[[5, 8], 1] = 0.5, 0.5
    history[7, 1] = torch.tensor([0.5, 0.0, 0.5], dtype=f64)
    history[9, 1] = 0.05
    mem["unload_history"] = history
    mem["target"] = torch.tensor([COMMAND] * 4 + [(0.1, 0.0, 0.0)] + [COMMAND] * 5, dtype=f64)
    z = torch.zeros(n, NA, dtype=f64)
    terms, new = env.task.terms(sig, mem, z, z, z, torch.zeros(n, dtype=f64), torch.zeros(n, dtype=torch.bool))
    paid = torch.tensor([0.0, 1.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.05], dtype=f64)
    assert torch.allclose(terms["air_time_smooth"], paid), terms["air_time_smooth"]
    assert terms["air_time"][1] == 1.0 and terms["air_time"][3] == 0.0   # where Eric's count pays and where it stopped
    assert float(new["unloaded_time"][0].abs().max()) == 0.0             # a loaded foot starts the clock again
    assert float(new["unloaded_time"][6, 1]) == pytest.approx(0.62)      # the resting foot's clock runs on
    assert float(new["unloaded_time"][8, 1]) == pytest.approx(0.51)      # half unloaded: half a second per second
    assert float(new["unloaded_time"][9, 1]) == 0.0                      # 95 % of its half: a new unload starts
    assert torch.allclose(new["unload_history"][7, 1], torch.tensor([0.0, 0.5, 0.5], dtype=f64))


def test_air_time_stand_in_gives_a_slope_through_the_plant():
    """With the stand-in, the differentiated reward of a standing robot asked to walk gets a new slope, and autograd
    through the pH plant agrees with finite differences.  6 steps: an action reaches the feet a step later (the
    firmware's delay), and the stand-in pays the smallest unload of 3 steps."""
    start, K = None, 6
    grads = []
    for weight in (0.0, 0.1):
        env = GradWalk(2, seed=1, command=COMMAND, air_time_smooth=weight)
        with torch.no_grad():
            for _ in range(20):         # 0.4 s: settled (one foot keeps a little weight off), the request past 0.2
                env.between_steps()
                env.step(torch.zeros(2, NA, dtype=torch.float64))
        start = {k: v.clone() for k, v in env.get_state().items()}

        def rollout(actions):
            env.set_state({k: v.clone() for k, v in start.items()})
            total = 0.0
            for a in actions:
                reward, _ = env.step(a)
                total = total + reward.sum()
            return total

        actions = (0.05 * torch.randn(K, 2, NA, dtype=torch.float64, generator=torch.Generator().manual_seed(5)))
        actions.requires_grad_(True)
        (g,) = torch.autograd.grad(rollout(actions), actions)
        grads.append(g)
    assert float((grads[1] - grads[0]).abs().max()) > 1e-6               # the stand-in adds a slope
    flat, eps = grads[1].flatten(), 1e-6
    with torch.no_grad():
        for j in flat.abs().topk(4).indices.tolist():                    # autograd against central differences
            d = torch.zeros_like(flat)
            d[j] = eps
            d = d.reshape(actions.shape)
            fd = float((rollout(actions.detach() + d) - rollout(actions.detach() - d)) / (2 * eps))
            assert fd == pytest.approx(float(flat[j]), rel=1e-4, abs=1e-9)


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


@pytest.mark.parametrize("stand_in", [0.0, 1.0])
def test_window_step_by_step_equals_one_graph_on_the_walk(stand_in):
    """The learner's window on the walk task, with noise, pushes, changing delays, gain errors and changing
    requests: backward one control step at a time gives the same action gradients as one graph.  With air_time's
    stand-in, the derivative also runs through the memory of the last unloads."""
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
        env = GradWalk(3, seed=4, air_time_smooth=stand_in, **world)
        env.mem["command_time_left"][0] = 0.03               # robot 0 gets a new request inside the window
        torch.manual_seed(7)
        out.append(fn(env))
    assert out[0].shape == (4, 3, NA) and float(out[0].abs().max()) > 0.0
    assert torch.allclose(out[0], out[1], rtol=1e-8, atol=1e-12), float((out[0] - out[1]).abs().max())

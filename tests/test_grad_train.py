"""The trainer of the thesis method (mbrl/grad/train.py) for walking (roadmap W5): it runs, writes the file the robot
loads, its curriculum follows its rule, and standing is unchanged."""

import json
import sys
from argparse import Namespace

import numpy as np
import pytest
import torch

from mbrl.grad import train
from mbrl.mpc.nets import MLP

torch.set_num_threads(1)


def run_main(monkeypatch, tmp_path, *args):
    monkeypatch.setattr(sys, "argv", ["train", "--no-wandb", "--threads", "1", "--save-every-min", "0",
                                      "--log-root", str(tmp_path), *args])
    train.main()


def test_walking_writes_the_policy_file_the_robot_loads(monkeypatch, tmp_path):
    from rl.walk import contract as C
    from rl.walk.export import bin_policy, read_bin

    run_main(monkeypatch, tmp_path, "--task", "walk", "--robots", "3", "--steps", "3", "--iterations", "2",
             "--eval-robots", "0", "--run-name", "w")
    out = tmp_path / "w"
    config = json.loads((out / "config.json").read_text())
    assert config["task"] == "walk" and config["term2"] == "weights" and config["episode_s"] == 20.0
    rows = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and all(r["Curriculum/command_scale"] == 0.25 for r in rows)
    assert all(np.isfinite(r[f"Walk/{k}"]) for r in rows for k in ("speed_error", "requested_speed", "feet_air", "slip_speed"))
    # what W&B shows besides: the derivative through the pH model, the Hamiltonian, every reward term, the network
    r = rows[-1]
    for k in ("Gradient/action_first_step", "Gradient/costate_q_start", "Gradient/costate_p_end",
              "Energy/supplied_W", "Energy/copper_W", "Energy/H_total_J", "Energy/Hd_total_J", "Energy/unbooked_W",
              "Reward/track_linear_velocity", "Reward/(differentiated)", "Learn/action_change", "Train/episodes"):
        assert np.isfinite(r[k]), k
    assert r["Reward/(differentiated)"] == pytest.approx(sum(r[f"Reward/{k}"] for k in train_terms()), rel=1e-6)
    ck = torch.load(out / "model_2.pt", map_location="cpu", weights_only=False)
    assert ck["infos"]["task"] == "walk" and ck["infos"]["command_scale"] == 0.25
    assert len(ck["infos"]["term2_weights"]) == 14                     # the 14 state terms of walking
    policy = MLP(C.OBS_DIM, C.NJ).double()
    policy.load_state_dict({k: v.double() for k, v in ck["actor_state_dict"].items()})
    obs = torch.randn(32, C.OBS_DIM, dtype=torch.float64)
    with torch.no_grad():
        want = policy(obs).numpy()
    got = bin_policy(read_bin(out / "deploy_2.bin"))(obs.numpy())     # the firmware's forward pass on the file
    assert np.abs(got - want).max() < 1e-5


def train_terms():
    """The walk task's terms with a derivative: what the learner differentiates."""
    from rl.walk.task import COUNTS, REWARDS

    return [k for k in REWARDS if k not in COUNTS]


def test_the_gradient_figure_of_a_window(tmp_path):
    """The figure W&B gets with every evaluation, from a real window: the action gradient per step and the co-state."""
    from types import SimpleNamespace

    from mbrl.grad.figures import gradient_figure
    from mbrl.grad.walk_env import GradWalk
    from rl.walk import contract as C

    torch.manual_seed(1)
    env = GradWalk(3, seed=4, command=(0.5, 0.0, 0.0))
    args = SimpleNamespace(steps=3, clip=3.0, action_noise=0.05, gamma=0.99, terminal_seconds=1.0, terminal_kinetic=0.0,
                           fall_weight=1.0)
    w = train.window(env, MLP(C.OBS_DIM, C.NJ, last_scale=0.5).double(), args)
    trace = w["trace"]
    assert len(trace["g"]) == 3 and all(len(trace[k]) == 4 for k in ("q", "p", "flux"))
    assert all(lo <= mid <= hi for lo, mid, hi in trace["g"] + trace["q"] + trace["p"])
    assert trace["flux"][-1][1] == 0.0 and trace["flux"][0][1] > 0.0      # term 2 does not read the coils; step 1 does
    assert gradient_figure(trace, tmp_path / "g.png", 0.02, title="test").stat().st_size > 10_000


def args_(**kw):
    return Namespace(**{"command_scale_max": 1.0, "command_scale_step": 0.05, "curriculum_updates": 3,
                        "curriculum_falls": 0.002, "curriculum_error": 0.3, **kw})


def test_the_curriculum_rises_stops_and_never_falls():
    good, falls, slow = (0.0, 0.05, 0.25), (0.01, 0.05, 0.25), (0.0, 0.2, 0.25)   # (fell, speed error, requested)
    a = args_()
    assert train.next_command_scale(0.25, [good] * 2, a) == 0.25        # not enough updates yet
    assert train.next_command_scale(0.25, [good] * 3, a) == 0.30        # rarely falls, follows: one step up
    assert train.next_command_scale(0.25, [good, good, falls], a) == 0.25        # 0.33 % of robot-windows fell
    assert train.next_command_scale(0.25, [good, good, slow], a) == 0.25         # mean error 0.1 > 0.3 x 0.25
    assert train.next_command_scale(0.98, [good] * 3, a) == 1.0         # stops at its maximum
    assert train.next_command_scale(1.0, [good] * 3, a) == 1.0
    worse = [(0.5, 1.0, 0.25)] * 3
    assert train.next_command_scale(0.6, worse, a) == 0.6               # never goes down


def test_the_curriculum_survives_a_resume(monkeypatch, tmp_path):
    common = ("--task", "walk", "--robots", "3", "--steps", "2", "--eval-robots", "0", "--curriculum-updates", "1000")
    run_main(monkeypatch, tmp_path, *common, "--iterations", "1", "--command-scale", "0.6", "--run-name", "a")
    run_main(monkeypatch, tmp_path, *common, "--iterations", "2", "--resume", str(tmp_path / "a" / "model_1.pt"),
             "--run-name", "b")                                         # --command-scale left at its default 0.25
    rows = [json.loads(line) for line in (tmp_path / "b" / "metrics.jsonl").read_text().splitlines()]
    assert [r["iteration"] for r in rows] == [2] and rows[0]["Curriculum/command_scale"] == 0.6
    assert torch.load(tmp_path / "b" / "model_2.pt", weights_only=False)["infos"]["command_scale"] == 0.6


def test_standing_is_unchanged_by_walking(monkeypatch, tmp_path):
    """The numbers of 2 updates of standing (4 robots, window 4, seed 3, learned term-2 weights), recorded with the
    trainer as it was before walking was added (19 Nov 2026, commit e91df64)."""
    run_main(monkeypatch, tmp_path, "--robots", "4", "--steps", "4", "--iterations", "2", "--eval-robots", "0",
             "--seed", "3", "--term2", "weights", "--run-name", "s")
    rows = [json.loads(line) for line in (tmp_path / "s" / "metrics.jsonl").read_text().splitlines()]
    then = [{"Grad/objective": -0.07744921025261876, "Grad/reward_step": -0.0035191038031258826,
             "Grad/grad_median": 0.001892035895827536, "Grad/term2": 0.06616440536428819,
             "Learn/fit_loss": 0.0015360866467552632, "Value/s_orientation": 0.9807661254838794},
            {"Grad/objective": -0.06823180291006276, "Grad/reward_step": -0.0021606219137352113,
             "Grad/grad_median": 0.004696941877725274, "Grad/term2": 0.06216431976644638,
             "Learn/fit_loss": 0.001634123622766003, "Value/s_orientation": 0.9688295663430034}]
    for row, want in zip(rows, then):
        for k, v in want.items():
            assert row[k] == pytest.approx(v, rel=1e-9, abs=1e-15), k
    assert "Curriculum/command_scale" not in rows[0] and "Grad/speed_error" not in rows[0]

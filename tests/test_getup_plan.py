"""The model-based get-up: keyframes, planner, tracking env, export (all on the CPU, tiny)."""

import pytest
import torch

pytest.importorskip("tensordict")

import mbrl  # noqa: E402,F401
from mbrl.getup.keyframes import LEFT, RIGHT, SIGN, Keyframes, targets_at  # noqa: E402
from ph_k1.model import load_k1  # noqa: E402
from rl.getup import contract as C  # noqa: E402


def _kf(K=3):
    m = load_k1(dtype=torch.float32)
    return Keyframes(m.jnt_low, m.jnt_high, torch.tensor(C.HOME), K), m


def test_keyframes_mirror_and_range():
    kf, m = _kf()
    z = 3.0 * torch.randn(5, kf.D, generator=torch.Generator().manual_seed(0))
    frames, dur = kf.decode(z)
    assert frames.shape == (5, 3, 22) and dur.shape == (5, 4)
    assert ((frames >= m.jnt_low - 1e-6) & (frames <= m.jnt_high + 1e-6)).all()
    left, right = frames[..., list(LEFT)], frames[..., list(RIGHT)]
    mirrored = torch.minimum(torch.maximum(left * torch.tensor(SIGN), m.jnt_low[list(RIGHT)]), m.jnt_high[list(RIGHT)])
    assert torch.allclose(right, mirrored)
    assert torch.equal(frames[..., :2], torch.zeros(5, 3, 2))          # head at HOME
    assert ((dur >= kf.d_min) & (dur <= kf.d_max)).all()


def test_init_mean_is_home():
    kf, _ = _kf()
    frames, dur = kf.decode(kf.init_mean()[None])
    assert torch.allclose(frames[0], torch.tensor(C.HOME).expand(3, -1), atol=2e-3)
    assert torch.allclose(dur, torch.full_like(dur, 0.4), atol=1e-4)


def test_targets_pass_the_keyframes():
    kf, m = _kf()
    frames, dur = kf.decode(torch.randn(2, kf.D, generator=torch.Generator().manual_seed(1)))
    theta0 = m.jnt_low + 0.3 * (m.jnt_high - m.jnt_low)
    theta0 = theta0.expand(2, -1)
    assert torch.allclose(targets_at(frames, dur, theta0, kf.home, 0.0), theta0)
    t1 = dur[:, 0]
    assert torch.allclose(targets_at(frames, dur, theta0, kf.home, t1), frames[:, 0], atol=1e-5)
    end = dur.sum(-1) + 1.0
    assert torch.allclose(targets_at(frames, dur, theta0, kf.home, end), kf.home.expand(2, -1))


def _planner(fit=None):
    from mbrl.getup.plan import PlanConfig, Planner
    cfg = PlanConfig(kind="prone", restarts=2, candidates=2, variants=2, keyframes=2, horizon_s=0.2, iterations=1,
                     library_size=8, device="cpu", seed=3)
    return Planner(cfg, fit, log=lambda *a: None)


def test_planner_rollout_is_deterministic_and_finite():
    pl = _planner()
    z = pl.kf.init_mean().expand(pl.B, -1).clone()
    z += 0.5 * torch.randn(z.shape, generator=torch.Generator().manual_seed(2))
    a = pl.rollout(z)
    b = pl.rollout(z)
    assert torch.isfinite(a["cost"]).all()
    assert torch.allclose(a["cost"], b["cost"])


def test_planner_search_and_plan_file(tmp_path):
    from mbrl.getup.plan import save_plan
    from mbrl.getup.track_env import load_plan
    from test_model import _fit
    pl = _planner(_fit())
    state = pl.search()
    assert state["it"] == 1 and len(state["history"]) == 1
    res = pl.final(state)
    ref = res["reference"]
    assert ref["theta"].shape == (pl.steps, 22) and ref["com_z"].shape == (pl.steps,)
    assert len(res["success_variants"]) == 2 and res["success_fitted"] is not None
    path = save_plan(res, state, pl.cfg, tmp_path, "t", "fit.json")
    p = load_plan(path)
    assert p["kind"] == "prone" and p["keyframes"].shape == (2, 22) and p["durations"].shape == (3,)


def test_assist_schedule_and_init_plan(tmp_path):
    from mbrl.getup.plan import PlanConfig, Planner, save_plan
    cfg = PlanConfig(kind="prone", restarts=2, candidates=2, variants=1, keyframes=2, horizon_s=0.1, iterations=4,
                     library_size=8, device="cpu", seed=6, assist=100.0, assist_until=0.5)
    pl = Planner(cfg, log=lambda *a: None)
    st = pl.search()
    assists = [row["assist_N"] for row in st["history"]]
    assert assists[0] == 100.0 and assists[1] == 50.0 and assists[2] == 0.0 and assists[3] == 0.0
    path = save_plan(pl.final(st), st, cfg, tmp_path, "a", None)
    cfg2 = PlanConfig(**{**cfg.__dict__, "iterations": 1, "init_plan": str(path)})
    pl2 = Planner(cfg2, log=lambda *a: None)
    st2 = pl2.search()
    prev = torch.load(path, weights_only=False)
    assert st2["history"][0]["it"] == 0
    assert torch.allclose(pl2.kf.decode(torch.tensor(prev["z"])[None].float())[0],
                          pl.kf.decode(torch.tensor(prev["z"])[None].float())[0])


def _plan_files(tmp_path):
    """Two tiny plans (supine, prone) made by the planner."""
    from mbrl.getup.plan import PlanConfig, Planner, save_plan
    files = []
    for kind in ("supine", "prone"):
        cfg = PlanConfig(kind=kind, restarts=1, candidates=2, variants=1, keyframes=2, horizon_s=0.3, iterations=1,
                         library_size=8, device="cpu", seed=4)
        pl = Planner(cfg, log=lambda *a: None)
        st = pl.search()
        files.append(str(save_plan(pl.final(st), st, cfg, tmp_path, "t", None)))
    return tuple(files)


def test_track_env_contract_and_teacher(tmp_path):
    from mbrl.getup.track_env import NUM_ACTIONS, OBS_DIM, TrackConfig, TrackEnv
    env = TrackEnv(TrackConfig(num_envs=6, device="cpu", seed=5, library_size=8, plans=_plan_files(tmp_path)))
    obs = env.get_observations()
    assert obs["policy"].shape == (6, OBS_DIM)
    crit = obs["critic"].shape[1]
    teacher = env.teacher_actions()
    assert teacher.shape == (6, NUM_ACTIONS)
    # at t = 0 the plan's target is a lying robot's own start angles; a standing robot's target is HOME
    from rl.getup.lying import STANDING
    expect = (env.plant.theta - env.contract.home) / env.contract.action_scale
    lying = env.start_type != STANDING
    assert torch.allclose(teacher[lying], expect[lying], atol=1e-5)
    assert torch.equal(teacher[~lying], torch.zeros_like(teacher[~lying]))
    for _ in range(3):
        obs, rew, dones, extras = env.step(env.teacher_actions())
    assert obs["policy"].shape == (6, OBS_DIM) and obs["critic"].shape[1] == crit
    assert torch.isfinite(rew).all()
    assert "Episode_Reward/track_q" not in extras["log"] or torch.isfinite(torch.as_tensor(extras["log"]["Episode_Reward/track_q"]))
    s = env.curriculum_state()
    assert s["track_scale"] == 1.0
    env.load_curriculum_state({**s, "track_scale": 0.5})
    assert env.track_scale == 0.5 and env.gain_scale == 0.0


def test_export_is_the_standup_agent(tmp_path):
    from mbrl.getup.train import export_bin
    from rl.common import export as E
    dims = (72, 512, 256, 128, 22)
    state = {}
    for n, i in enumerate((0, 2, 4, 6)):
        state[f"mlp.{i}.weight"] = torch.randn(dims[n + 1], dims[n])
        state[f"mlp.{i}.bias"] = torch.randn(dims[n + 1])
    state["obs_normalizer._mean"] = torch.zeros(1, 72)
    state["obs_normalizer._std"] = torch.ones(1, 72)
    out = export_bin({"actor_state_dict": state}, tmp_path / "deploy.bin")
    mats = E.read_bin(out)
    assert mats["actor.0.weight"].shape == (512, 72) and mats["actor.6.weight"].shape == (22, 128)

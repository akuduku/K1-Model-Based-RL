import torch

import mbrl  # noqa: F401
from mbrl.params import ParamSpace
from mbrl.replay import Replay, make_windows

from helpers import synthetic_log


def test_true_parameters_explain_their_own_log_best():
    """A log made by the plant with known parameters: replaying with those parameters predicts it much better
    than with wrong ones (delay, ankle torque, knee damping)."""
    sp = ParamSpace()
    true = sp.neutral(1)[0]
    true[sp.index["delay_ms"]] = 15.0
    true[sp.index["knee.damp"]] = 0.8
    true[sp.index["hip.log_kp"]] = -0.3
    log = synthetic_log(true)
    ws = make_windows([log], horizon=10, stride_s=0.1)
    assert len(ws) >= 4
    wrong = true.clone()
    wrong[sp.index["delay_ms"]] = 60.0
    wrong[sp.index["knee.damp"]] = 0.0
    wrong[sp.index["hip.log_kp"]] = 0.5
    rp = Replay([log], ws, sp, n_candidates=3, horizon=10)
    out = rp.run(torch.stack((true, wrong, sp.neutral(1)[0])))
    l_true, l_wrong, l_neutral = out["loss"].tolist()
    assert l_true < 0.5 * l_wrong and l_true < l_neutral, out["loss"]
    assert float(out["diverged"].max()) == 0.0
    # what is left with the true parameters is the state estimate (smoothed theta_dot, trunk speed and tilt): ~1 deg
    assert out["group_rmse_deg"]["knee"][0] < 1.5     # degrees over 0.2 s


def test_batches_give_the_same_numbers():
    sp = ParamSpace()
    log = synthetic_log(sp.neutral(1)[0], seconds=0.8)
    ws = make_windows([log], horizon=5, stride_s=0.1)
    th = sp.clamp(sp.neutral(5) + 0.2 * sp.std * torch.randn(5, sp.D, dtype=torch.float64))
    a = Replay([log], ws, sp, n_candidates=5, horizon=5).run(th)["loss"]
    b = Replay([log], ws, sp, n_candidates=2, horizon=5).run(th)["loss"]
    assert torch.allclose(a, b, rtol=1e-9, atol=1e-12)

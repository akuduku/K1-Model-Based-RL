import torch

import mbrl  # noqa: F401
from mbrl.model import FittedModel, attach
from mbrl.params import ParamSpace
from mbrl.residual import ResidualEnsemble
from ph_k1.model import load_k1
from ph_k1.native import K1PHPlantNative


def _fit():
    sp = ParamSpace()
    el = sp.neutral(3)
    el[0, sp.index["log_mu"]] = -0.5
    el[1, sp.index["knee.damp"]] = 1.0
    el[2, sp.index["ankle.log_kp"]] = 0.3
    el[:, sp.index["delay_ms"]] = torch.tensor([20.0, 25.0, 30.0], dtype=torch.float64)
    return FittedModel(sp, el.mean(0), el.std(0), el)


def test_each_robot_is_one_ensemble_member_and_resets_redraw_only_their_rows():
    fit = _fit()
    pl = K1PHPlantNative(model=load_k1(dtype=torch.float64), batch=6)
    attach(pl, fit, seed=1)
    member_rows = fit.space.robot_params(fit.elites)
    for r in pl.robot_params:
        assert any(torch.allclose(r, m) for m in member_rows)
    assert pl.firmware.delay_ms == fit.delay_range() == (20.0, 30.0)
    before = pl.robot_params.clone()
    for _ in range(5):
        pl.reset(env_ids=torch.tensor([2]))
    keep = [0, 1, 3, 4, 5]
    assert torch.equal(pl.robot_params[keep], before[keep])
    info = pl.policy_step(pl.theta.clone())
    assert torch.isfinite(info.torque).all()


def test_residual_torque_enters_the_ledger_as_a_port():
    pl = K1PHPlantNative(model=load_k1(dtype=torch.float64), batch=2)
    res = ResidualEnsemble(K=2, hidden=8).double()
    with torch.no_grad():
        for net in res.nets:
            net[-1].bias[22:] = 1.0                       # a constant torque u = 0.1 effort tanh(1)
    attach(pl, None, res)
    for _ in range(5):
        pl.policy_step(pl.theta.clone())
    assert pl.ledger.mech_input.abs().max() > 1e-6
    assert pl.mbrl.member.max() < 2

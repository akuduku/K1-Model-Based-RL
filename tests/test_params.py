import math

import torch

import mbrl  # noqa: F401
from mbrl.params import GROUPS, ParamSpace
from ph_k1.robot_params import NRP, RP_DAMP, RP_KN, RP_KT, RP_MU, RP_TAU, neutral_rows


def test_neutral_theta_is_todays_plant():
    sp = ParamSpace()
    th = sp.neutral(3)
    assert torch.equal(sp.robot_params(th), neutral_rows(3, torch.float64))
    kps, kds = sp.gain_scales(th)
    assert torch.equal(kps, torch.ones(3, 22, dtype=torch.float64)) and torch.equal(kds, kps)
    assert float(sp.delay_ms(th)[0]) == 20.0


def test_groups_map_to_joints_and_contact_ties():
    sp = ParamSpace()
    th = sp.neutral(1)
    th[0, sp.index["knee.damp"]] = 0.7
    th[0, sp.index["ankle.log_tau"]] = math.log(0.5)
    th[0, sp.index["log_k_n"]] = math.log(2.0)
    th[0, sp.index["log_mu"]] = math.log(0.8)
    r = sp.robot_params(th)[0]
    assert r.shape == (NRP,)
    damp, tau = r[RP_DAMP:RP_DAMP + 22], r[RP_TAU:RP_TAU + 22]
    for k in range(22):
        assert abs(damp[k] - (0.7 if k in GROUPS["knee"] else 0.0)) < 1e-12
        assert abs(tau[k] - (0.5 if k in GROUPS["ankle"] else 1.0)) < 1e-12
    assert abs(r[RP_KN] - 2.0) < 1e-12 and abs(r[RP_KT] - 2.0) < 1e-12 and abs(r[RP_MU] - 0.8) < 1e-12


def test_clamp_keeps_bounds():
    sp = ParamSpace()
    th = sp.clamp(100 * torch.randn(50, sp.D, dtype=torch.float64))
    assert (th >= sp.lo).all() and (th <= sp.hi).all()
    assert (sp.robot_params(th)[:, RP_DAMP:RP_DAMP + 22] >= 0).all()

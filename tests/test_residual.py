import torch

import mbrl  # noqa: F401
from mbrl.residual import NFEAT, ResidualEnsemble, features, train


def _toy(n, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, NFEAT, generator=g)
    qd = x[:, 22:44]
    y = -0.3 * qd + 0.5 * torch.tanh(x[:, :22])
    return {"x": x, "qd": qd, "y": y}


def test_shapes_and_passive_part():
    ens = ResidualEnsemble(K=3, hidden=16)
    B = 7
    th, qd, gr, gy, tg = torch.randn(B, 22), torch.randn(B, 22), torch.randn(B, 3), torch.randn(B, 3), torch.randn(B, 22)
    assert features(th, qd, gr, gy, tg).shape == (B, NFEAT)
    d, u = ens.components(features(th, qd, gr, gy, tg))
    assert d.shape == (3, B, 22) and (d >= 0).all() and (d <= ens.damp_max).all()
    assert (u.abs() <= ens.tau_bound + 1e-6).all()
    assert ens(th, qd, gr, gy, tg, member=torch.tensor([0, 1, 2, 0, 1, 2, 0])).shape == (B, 22)


def test_training_learns_a_damping_law_and_round_trips(tmp_path):
    tr, va = _toy(2000, 0), _toy(500, 1)
    ens0 = ResidualEnsemble(K=2, hidden=32)
    ens0.x_mean.copy_(tr["x"].mean(0)); ens0.x_std.copy_(tr["x"].std(0))
    with torch.no_grad():
        base = float(((ens0.member_torque(va["x"], va["qd"]).mean(0) - va["y"]) ** 2).mean())
    ens, hist = train(tr, va, K=2, hidden=32, epochs=60, lr=3e-3, log=lambda *_: None)
    after = float(((ens.member_torque(va["x"], va["qd"]).mean(0) - va["y"]) ** 2).mean())
    assert after < 0.3 * base
    p = tmp_path / "r.pt"
    ens.save(p)
    back = ResidualEnsemble.load(p)
    assert torch.allclose(back.member_torque(va["x"], va["qd"]), ens.member_torque(va["x"], va["qd"]))

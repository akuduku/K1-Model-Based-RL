r"""A learned residual in port-Hamiltonian form: what the fitted physics still misses, as a joint torque.

    tau_res(x) = -d(x) * theta_dot  +  u(x)
        d(x) >= 0  (softplus, at most damp_max)   extra joint damping: a dissipative element, it can only remove energy
        u(x)       bounded (tanh, at most tau_frac x effort)   an unmodelled torque, entering as an input port

Features x (72): theta, theta_dot, gravity in the trunk frame, gyro, target - theta.  The torque is held for one
policy step (20 ms) and enters the plant through policy_step(gen_force=...), so its work is booked in
ledger.mech_input: the energy balance stays closed and the plant stays a pH system with one more port.

Training data (build_dataset): for every logged policy step, start the FITTED plant in the logged state, run that
one step with the logged command, and compare the joint speeds after 20 ms:
    tau_target = M_jj(theta) (theta_dot_real - theta_dot_sim) / 0.02
(M_jj = joint block of the mass matrix; trunk coupling neglected).  This is ordinary supervised regression, the
"model learning" step of Lecture 11.  An ensemble of K networks on bootstrapped data; the members' disagreement
is the model's uncertainty.

    python -m mbrl.residual --fit fits/getup-v1.json --out residual/getup-v1.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import paths
from .data import load_known
from .params import ParamSpace
from .replay import Replay, make_windows
from ph_k1.dynamics import forward_kinematics
from ph_k1.model import NB, NJ, NQ, load_k1
from ph_k1 import servo

NFEAT = 3 * NJ + 6  # theta, qd, target - theta (3 x 22) + gravity, gyro (6) = 72


def features(theta, qd, grav, gyro, target) -> torch.Tensor:
    return torch.cat((theta, qd, grav, gyro, target - theta), -1)


class ResidualEnsemble(nn.Module):
    def __init__(self, K: int = 5, hidden: int = 64, damp_max: float = 0.5, tau_frac: float = 0.1):
        super().__init__()
        self.K, self.hidden = K, hidden
        self.nets = nn.ModuleList(
            nn.Sequential(nn.Linear(NFEAT, hidden), nn.ELU(), nn.Linear(hidden, hidden), nn.ELU(),
                          nn.Linear(hidden, 2 * NJ)) for _ in range(K))
        for net in self.nets:  # start at "no residual"
            nn.init.zeros_(net[-1].weight)
            nn.init.constant_(net[-1].bias[:NJ], -6.0)
            nn.init.zeros_(net[-1].bias[NJ:])
        self.register_buffer("x_mean", torch.zeros(NFEAT))
        self.register_buffer("x_std", torch.ones(NFEAT))
        self.register_buffer("damp_max", torch.tensor(float(damp_max)))
        self.register_buffer("tau_bound", tau_frac * torch.tensor(servo.EFFORT, dtype=torch.float32))
        self.config = {"K": K, "hidden": hidden, "damp_max": damp_max, "tau_frac": tau_frac}

    def components(self, x: torch.Tensor):
        """x (B, 72) -> d, u: (K, B, 22) each."""
        z = (x.to(self.x_mean.dtype) - self.x_mean) / self.x_std
        o = torch.stack([net(z) for net in self.nets])
        d = self.damp_max * torch.sigmoid(o[..., :NJ])
        u = self.tau_bound * torch.tanh(o[..., NJ:])
        return d, u

    def member_torque(self, x: torch.Tensor, qd: torch.Tensor) -> torch.Tensor:
        """(K, B, 22): every member's torque."""
        d, u = self.components(x)
        return -d * qd.to(d.dtype)[None] + u

    def forward(self, theta, qd, grav, gyro, target, member: torch.Tensor | None = None) -> torch.Tensor:
        """(B, 22): the torque of each robot's member, or the ensemble mean (member=None)."""
        tau = self.member_torque(features(theta, qd, grav, gyro, target), qd)
        if member is None:
            return tau.mean(0)
        return tau[member, torch.arange(tau.shape[1], device=tau.device)]

    def save(self, path: str | Path, meta: dict | None = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": self.config, "state": self.state_dict(), "meta": meta or {}}, path)

    @classmethod
    def load(cls, path: str | Path, device="cpu") -> ResidualEnsemble:
        d = torch.load(path, map_location=device, weights_only=False)
        m = cls(**d["config"])
        m.load_state_dict(d["state"])
        m.meta = d.get("meta", {})
        return m.to(device)


# ------------------------------------------------------------------ data
def build_dataset(logs, theta: torch.Tensor, device="cpu", dtype=torch.float64, batch: int = 256) -> dict:
    """One sample per logged policy step: features at the start, theta_dot (for the damping term), tau_target."""
    space = ParamSpace()
    wins = make_windows(logs, horizon=1, stride_s=0.0, max_acc_dev=1e9)
    model = load_k1(dtype=dtype, device=device)
    out = {k: [] for k in ("x", "qd", "y", "log")}
    for i in range(0, len(wins), batch):
        ws = wins[i:i + batch]
        rp = Replay(logs, ws, space, n_candidates=1, horizon=1, device=device, dtype=dtype)
        r = rp.run(theta.reshape(1, -1), keep_traj=True)
        qd_sim = r["qd"][0, :, 0]                                   # (W, 22) after 20 ms
        L_end = torch.stack([torch.as_tensor(logs[w.log].qd[w.start + 10], dtype=dtype) for w in ws])
        q = torch.zeros(len(ws), NQ, dtype=dtype, device=device)
        q[:, 3] = 1.0
        q[:, 7:] = rp.th0
        Mjj = forward_kinematics(model, q).M[:, NB:, NB:]
        y = (Mjj @ (L_end.to(device) - qd_sim)[..., None]).squeeze(-1) / 0.02
        out["x"].append(features(rp.th0, rp.qd0, torch.as_tensor(np.stack([logs[w.log].grav[w.start] for w in ws]),
                                                                  dtype=dtype, device=device),
                                 rp.gyro0, rp.target[:, 0]))
        out["qd"].append(rp.qd0)
        out["y"].append(y)
        out["log"] += [logs[w.log].name for w in ws]
    return {"x": torch.cat(out["x"]).float().cpu(), "qd": torch.cat(out["qd"]).float().cpu(),
            "y": torch.cat(out["y"]).float().cpu(), "log": out["log"]}


def train(data: dict, val: dict | None = None, K: int = 5, hidden: int = 64, epochs: int = 400, lr: float = 1e-3,
          weight_decay: float = 1e-3, batch: int = 128, seed: int = 0, log=print) -> tuple[ResidualEnsemble, list]:
    torch.manual_seed(seed)
    ens = ResidualEnsemble(K, hidden)
    x, qd, y = data["x"], data["qd"], data["y"]
    ens.x_mean.copy_(x.mean(0))
    ens.x_std.copy_(x.std(0).clamp_min(1e-3))
    n = len(x)
    boots = [torch.randint(n, (n,)) for _ in range(K)]
    opt = torch.optim.AdamW(ens.parameters(), lr=lr, weight_decay=weight_decay)
    hist, best = [], (float("inf"), None)

    def val_loss():
        if val is None:
            return float("nan")
        with torch.no_grad():
            return float(((ens.member_torque(val["x"], val["qd"]).mean(0) - val["y"]) ** 2).mean())

    for ep in range(epochs):
        perm = [b[torch.randperm(n)] for b in boots]
        tot = 0.0
        for s in range(0, n, batch):
            loss = 0.0
            for k in range(K):
                idx = perm[k][s:s + batch]
                d, u = ens.components(x[idx])
                pred = -d[k] * qd[idx] + u[k]
                loss = loss + ((pred - y[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach())
        if ep % 20 == 0 or ep == epochs - 1:
            vl = val_loss()
            hist.append({"epoch": ep, "train": tot / max(1, (n + batch - 1) // batch) / K, "val": vl})
            log(f"[residual] epoch {ep:4d}  train {hist[-1]['train']:.3f}  val {vl:.3f}")
            if val is not None and vl < best[0]:
                best = (vl, {k: v.clone() for k, v in ens.state_dict().items()})
    if best[1] is not None:  # early stopping on the validation logs
        ens.load_state_dict(best[1])
    return ens, hist


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", required=True, help="fit JSON of mbrl.fit (the calibrated physics underneath)")
    ap.add_argument("--train", default="getup_2236")
    ap.add_argument("--val", default="getup_front_2250")
    ap.add_argument("--out", default=str(paths.REPO / "residual" / "residual.pt"))
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--horizon", type=int, default=20, help="policy steps of the k-step check")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    fit = json.loads(Path(a.fit).read_text())
    theta = torch.tensor(fit["mean"], dtype=torch.float64)
    names = lambda s: [x for x in s.split(",") if x]  # noqa: E731
    tl, vl = load_known(names(a.train)), load_known(names(a.val))
    t0 = time.time()
    dtr = build_dataset(tl, theta, a.device)
    dva = build_dataset(vl, theta, a.device) if vl else None
    print(f"[residual] {len(dtr['y'])} training / {0 if dva is None else len(dva['y'])} validation samples "
          f"({time.time() - t0:.0f} s); |tau_target| rms train {dtr['y'].pow(2).mean().sqrt():.2f} N m")
    ens, hist = train(dtr, dva, a.members, a.hidden, a.epochs)
    # the k-step check: physics fit alone vs physics fit + residual, on both log sets
    space = ParamSpace()
    report = {}
    for tag, logs in (("train", tl), ("val", vl)):
        if not logs:
            continue
        ws = make_windows(logs, a.horizon)
        rp = Replay(logs, ws, space, 1, a.horizon, a.device)
        ens_d = ens.to(a.device).double()
        base, res = rp.run(theta[None]), rp.run(theta[None], residual=ens_d)
        report[tag] = {"fit": float(base["loss"][0]), "fit+residual": float(res["loss"][0]),
                       "group_rmse_deg": {g: [round(float(base["group_rmse_deg"][g][0]), 3),
                                              round(float(res["group_rmse_deg"][g][0]), 3)] for g in base["group_rmse_deg"]}}
        print(f"[residual] {tag}: k-step loss fit {report[tag]['fit']:.4f} -> fit+residual {report[tag]['fit+residual']:.4f}")
    ens.float().save(a.out, {"fit": a.fit, "train": names(a.train), "val": names(a.val), "history": hist, "kstep": report})
    print(f"[residual] wrote {a.out}")


if __name__ == "__main__":
    main()

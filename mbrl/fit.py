r"""System identification of the pH plant on robot logs with the cross-entropy method (CEM).

Every CEM candidate is one parameter vector theta (mbrl/params.py); one batched plant runs all candidates on all
windows at once (mbrl/replay.py).  CEM, as in Lecture 11: sample N candidates from a Gaussian, rank them by the
replay loss, refit the Gaussian to the best M (the elites), repeat.  The mean of the current Gaussian is always
candidate 0, so the reported mean is an evaluated point.

    python -m mbrl.fit --smoke                                          # laptop, ~1 min
    python -m mbrl.fit --out fits/getup-v1.json                         # CPU: minutes to an hour
    python -m mbrl.fit --device cuda --candidates 256 --out fits/getup-v1.json   # Colab/Kaggle GPU

Train and validation are separate logs (default: back get-up for training, front get-up for validation).
The JSON keeps the final Gaussian, the elites (the parameter ensemble for training: mbrl/model.py), the
loss and per-joint-group RMSE of today's plant ("neutral") and of the fit on both log sets, and the history.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import torch

from . import paths
from .data import load_known
from .params import ParamSpace
from .replay import Replay, make_windows


def metrics(out: dict, i: int = 0) -> dict:
    return {"loss": float(out["loss"][i]), "tilt_err": float(out["tilt_err"][i]), "gyro_mse": float(out["gyro_mse"][i]),
            "diverged": float(out["diverged"][i]),
            "group_rmse_deg": {g: round(float(v[i]), 3) for g, v in out["group_rmse_deg"].items()}}


def cem(rp: Replay, space: ParamSpace, n: int, n_elite: int, iterations: int, alpha: float = 0.7,
        min_std_frac: float = 0.05, seed: int = 0, log=print):
    g = torch.Generator().manual_seed(seed)
    mean, std = space.init.clone(), space.std.clone()
    min_std = min_std_frac * space.std
    history, best = [], (float("inf"), mean.clone())
    for it in range(iterations):
        t0 = time.time()
        theta = mean + std * torch.randn(n, space.D, generator=g, dtype=torch.float64)
        theta[0] = mean
        theta = space.clamp(theta)
        loss = rp.run(theta)["loss"].cpu()
        order = torch.argsort(loss)
        elite = theta[order[:n_elite]]
        if float(loss[order[0]]) < best[0]:
            best = (float(loss[order[0]]), theta[order[0]].clone())
        mean = alpha * elite.mean(0) + (1 - alpha) * mean
        std = torch.maximum(alpha * elite.std(0) + (1 - alpha) * std, min_std)
        history.append({"it": it, "mean_loss": float(loss[0]), "best": float(loss[order[0]]),
                        "elite_loss": float(loss[order[:n_elite]].mean()), "seconds": round(time.time() - t0, 1)})
        log(f"[cem] it {it:3d}  loss(mean) {loss[0]:.4f}  best {loss[order[0]]:.4f}  "
            f"elites {loss[order[:n_elite]].mean():.4f}  ({time.time() - t0:.1f} s)")
    # the elites of the final distribution, evaluated (the ensemble)
    theta = space.clamp(mean + std * torch.randn(n, space.D, generator=g, dtype=torch.float64))
    theta[0] = mean
    loss = rp.run(theta)["loss"].cpu()
    order = torch.argsort(loss)
    return mean, std, theta[order[:n_elite]], loss[order[:n_elite]], best, history


def git_rev(path: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default="getup_2236", help="comma list of data.KNOWN log names")
    ap.add_argument("--val", default="getup_front_2250")
    ap.add_argument("--out", default=str(paths.REPO / "fits" / "fit.json"))
    ap.add_argument("--candidates", type=int, default=64)
    ap.add_argument("--elites", type=int, default=8)
    ap.add_argument("--iterations", type=int, default=15)
    ap.add_argument("--horizon", type=int, default=20, help="policy steps per window (20 = 0.4 s)")
    ap.add_argument("--stride", type=float, default=0.1, help="s between window starts")
    ap.add_argument("--max-windows", type=int, default=0, help="0: all")
    ap.add_argument("--batch-candidates", type=int, default=16, help="candidates per plant batch")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="tiny run for a quick check")
    a = ap.parse_args(argv)
    if a.smoke:
        a.candidates, a.elites, a.iterations, a.horizon, a.max_windows, a.batch_candidates = 12, 3, 3, 10, 12, 12
    dtype = getattr(torch, a.dtype)
    space = ParamSpace()
    names = lambda s: [x for x in s.split(",") if x]  # noqa: E731
    train_logs, val_logs = load_known(names(a.train)), load_known(names(a.val)) if a.val else []
    mw = a.max_windows or None
    tw = make_windows(train_logs, a.horizon, a.stride, max_count=mw, seed=a.seed)
    rp = Replay(train_logs, tw, space, a.batch_candidates, a.horizon, a.device, dtype)
    print(f"[fit] train {a.train}: {len(tw)} windows of {a.horizon * 0.02:.2f} s; {rp.B} robots per batch on {a.device}")
    base = rp.run(space.neutral(1))
    print(f"[fit] today's plant (neutral): loss {base['loss'][0]:.4f}  "
          + "  ".join(f"{g} {v[0]:.2f}°" for g, v in base["group_rmse_deg"].items()))
    t0 = time.time()
    mean, std, elites, elite_loss, best, hist = cem(rp, space, a.candidates, a.elites, a.iterations, seed=a.seed)
    fit = rp.run(mean[None])
    print(f"[fit] fitted mean: loss {fit['loss'][0]:.4f}  "
          + "  ".join(f"{g} {v[0]:.2f}°" for g, v in fit["group_rmse_deg"].items()) + f"  ({time.time() - t0:.0f} s)")
    result = {
        "space": space.names, "neutral": space.init.tolist(), "mean": mean.tolist(), "std": std.tolist(),
        "elites": elites.tolist(), "elite_loss": elite_loss.tolist(), "best": best[1].tolist(), "best_loss": best[0],
        "readable_mean": space.describe(mean),
        "train": {"logs": names(a.train), "windows": len(tw), "neutral": metrics(base), "fit": metrics(fit)},
        "history": hist, "config": vars(a), "full_body_commit": git_rev(paths.FULL_BODY),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if val_logs:
        vw = make_windows(val_logs, a.horizon, a.stride, max_count=mw, seed=a.seed)
        rv = Replay(val_logs, vw, space, 2, a.horizon, a.device, dtype)
        v = rv.run(torch.stack((space.init, mean)))
        result["val"] = {"logs": names(a.val), "windows": len(vw), "neutral": metrics(v, 0), "fit": metrics(v, 1)}
        print(f"[fit] validation {a.val}: neutral {v['loss'][0]:.4f} -> fit {v['loss'][1]:.4f}  "
              + "  ".join(f"{g} {v['group_rmse_deg'][g][0]:.2f}->{v['group_rmse_deg'][g][1]:.2f}°"
                          for g in v["group_rmse_deg"]))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(f"[fit] wrote {out}")
    return result


if __name__ == "__main__":
    main()

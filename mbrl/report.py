r"""How well does the model predict the robot?  Today's plant vs the fit (vs the fit + residual), per log.

    python -m mbrl.report fits/getup-v1.json [--residual residual/getup-v1.pt] [--out outputs/getup-v1]

Writes report.md (k-step loss and RMSE per joint group, in degrees, over 0.4 s windows), kstep.png (RMSE per
group) and traces.png (left hip pitch / knee / ankle pitch over a few windows: robot, today's plant, fit).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from . import paths  # noqa: E402
from .data import KNOWN, load_known  # noqa: E402
from .params import GROUPS, ParamSpace  # noqa: E402
from .replay import POLICY_DT, Replay, make_windows  # noqa: E402

TRACE_JOINTS = {"L hip pitch": 10, "L knee": 13, "L ankle pitch": 14}


def evaluate(logs, theta: torch.Tensor, residual=None, horizon: int = 20, device="cpu"):
    ws = make_windows(logs, horizon)
    rp = Replay(logs, ws, ParamSpace(), theta.shape[0], horizon, device)
    out = rp.run(theta, keep_traj=True)
    res = rp.run(theta[-1:], residual=residual, keep_traj=True) if residual is not None else None
    return rp, out, res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fit")
    ap.add_argument("--residual", default=None)
    ap.add_argument("--logs", default=",".join(KNOWN))
    ap.add_argument("--out", default=None)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    fit = json.loads(Path(a.fit).read_text())
    out_dir = Path(a.out or paths.REPO / "outputs" / Path(a.fit).stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    space = ParamSpace()
    theta = torch.stack((space.init, torch.tensor(fit["mean"], dtype=torch.float64)))
    residual = None
    if a.residual:
        from .residual import ResidualEnsemble
        residual = ResidualEnsemble.load(a.residual, a.device).double()
    train_logs = set(fit.get("train", {}).get("logs", []))
    lines = [f"# Model check: {Path(a.fit).name}", "",
             f"k-step prediction over {a.horizon * POLICY_DT:.2f} s windows (logged commands from logged states). "
             "RMSE of the joint angles per group in degrees; loss = weighted joint MSE + gyro + tilt.", "",
             "| log | set | model | loss | " + " | ".join(GROUPS) + " | tilt | diverged |",
             "|---|---|---|---|" + "---|" * len(GROUPS) + "---|---|"]
    fig_k, axk = plt.subplots(1, len(a.logs.split(",")), figsize=(6 * len(a.logs.split(",")), 3.5), squeeze=False)
    fig_t, axt = plt.subplots(len(TRACE_JOINTS), 3, figsize=(13, 8), squeeze=False)
    for li, name in enumerate(a.logs.split(",")):
        logs = load_known([name])
        rp, out, res = evaluate(logs, theta, residual, a.horizon, a.device)
        rows = [("today's plant", out, 0), ("fit", out, 1)] + ([("fit + residual", res, 0)] if res is not None else [])
        tag = "train" if name in train_logs else "held out"
        for label, o, i in rows:
            g = " | ".join(f"{float(o['group_rmse_deg'][k][i]):.2f}" for k in GROUPS)
            lines.append(f"| {name} | {tag} | {label} | {float(o['loss'][i]):.4f} | {g} | "
                         f"{float(o['tilt_err'][i]):.4f} | {float(o['diverged'][i]):.2f} |")
        x = np.arange(len(GROUPS))
        for j, (label, o, i) in enumerate(rows):
            axk[0, li].bar(x + 0.27 * (j - 1), [float(o["group_rmse_deg"][k][i]) for k in GROUPS], 0.27, label=label)
        axk[0, li].set_xticks(x, list(GROUPS))
        axk[0, li].set_ylabel("RMSE over 0.4 s [deg]")
        axk[0, li].set_title(f"{name} ({tag})")
        axk[0, li].legend(fontsize=8)
        if li == 0:  # traces of three windows of the first log
            t = (np.arange(a.horizon) + 1) * POLICY_DT
            for c, w in enumerate(np.linspace(0, rp.W - 1, 3).astype(int)):
                for r, (jn, j) in enumerate(TRACE_JOINTS.items()):
                    ax = axt[r, c]
                    ax.plot(t, np.rad2deg(rp.obs_theta[w, :, j].cpu().numpy()), "k", lw=2, label="robot")
                    ax.plot(t, np.rad2deg(out["theta"][0, w, :, j].cpu().numpy()), "--", label="today's plant")
                    ax.plot(t, np.rad2deg(out["theta"][1, w, :, j].cpu().numpy()), label="fit")
                    if res is not None:
                        ax.plot(t, np.rad2deg(res["theta"][0, w, :, j].cpu().numpy()), ":", label="fit + residual")
                    ax.set_title(f"{jn}, window {w}", fontsize=9)
                    if r == len(TRACE_JOINTS) - 1:
                        ax.set_xlabel("s after the window start")
            axt[0, 0].legend(fontsize=8)
    lines += ["", "Fitted parameters (factors and added damping in N m s/rad):", "", "```",
              json.dumps(fit.get("readable_mean", {}), indent=1), "```"]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    fig_k.tight_layout()
    fig_k.savefig(out_dir / "kstep.png", dpi=120)
    fig_t.tight_layout()
    fig_t.savefig(out_dir / "traces.png", dpi=120)
    print("\n".join(lines[:lines.index("Fitted parameters (factors and added damping in N m s/rad):") - 1]))
    print(f"[report] wrote {out_dir}/report.md, kstep.png, traces.png")


if __name__ == "__main__":
    main()

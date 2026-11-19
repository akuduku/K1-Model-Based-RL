r"""The derivative through the pH model in one window, as a figure (logged to W&B as Gradient/through_window).

The learner's backward pass (train.py window) carries the derivative of the cost back through the pH model one
control step at a time: the co-state lambda_t = d(cost from step t to the end, term 2 included)/d(state before step t),
one number per state entry, the adjoint of the port-Hamiltonian dynamics.  Its parts for the positions q, the
momenta p and the motors' coil fluxes show how far back in time, and through which part of the state, the cost
reaches; the action gradient d(cost)/d(a_t) is what each control step's action is pushed by.

    gradient_figure(window_output["trace"], "gradient_00100.png", dt=0.02, title="update 100")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

PARTS = (("q", "positions q", "#7570b3"), ("p", "momenta p", "#d95f02"), ("flux", "coil fluxes (motors)", "#1b9e77"))


def gradient_figure(trace: dict, path: str | Path, dt: float = 0.02, title: str | None = None) -> Path:
    """trace (train.py window): "g" (K, 3) and "q", "p", "flux" (K+1, 3): 10th percentile, median and 90th percentile
    over the robots of |d cost/d a_t| per control step, and of the co-state's norm before every step and at the end."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = np.asarray(trace["g"])
    K = g.shape[0]
    t_act = np.arange(K) * dt
    t_state = np.arange(K + 1) * dt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    a = ax[0]
    a.fill_between(t_act, g[:, 0], g[:, 2], color="#1f78b4", alpha=0.25, label="10-90 % of the robots")
    a.plot(t_act, g[:, 1], color="#1f78b4", lw=2, marker="o", ms=3, label="median")
    a.set(yscale="log", xlabel="time in the window [s] (0 = its first control step)",
          ylabel="|d cost / d action| (clipped per robot)",
          title="(a) How strongly each control step's action is pushed")
    a.legend(fontsize=8)
    b = ax[1]
    for key, label, color in PARTS:
        if key not in trace:
            continue
        c = np.asarray(trace[key], dtype=np.float64)
        c = np.where(c > 1e-12 * max(float(np.nanmax(c)), 1e-30), c, np.nan)   # exact zeros (term 2 does not read
        b.fill_between(t_state, c[:, 0], c[:, 2], color=color, alpha=0.15)   # the coil fluxes) are left out
        b.plot(t_state, c[:, 1], color=color, lw=2, marker="o", ms=3, label=label)
    b.set(yscale="log", xlabel="time in the window [s] (the derivative goes from the end back to 0)",
          ylabel=r"|co-state $\lambda$| (median, band: 10-90 %)",
          title="(b) The co-state: the cost's derivative carried back through the pH model")
    b.axvline(K * dt, color="black", lw=0.8)
    b.text(K * dt, b.get_ylim()[1], " term 2", va="top", fontsize=8)
    b.legend(fontsize=8)
    for axis in ax:
        axis.grid(alpha=0.3, which="both")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path

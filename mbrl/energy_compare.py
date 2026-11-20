r"""Energy of one walking policy on the pH plant and on MuJoCo: booked every 1 ms, and sampled every 20 ms as before.

    python -m mbrl.energy_compare [--policy sprint-repro-001] [--delay-ms 20] [--seconds 10] [--keep-from 4]

Per engine and requested speed (0, 0.5, ..., 2.5 m/s) the policy walks for `seconds` on the standard robot model;
the stretch from keep_from on is averaged.  Writes outputs/energy/<policy>/report.md and results.json.

Booked (Full Body rl/walk/backends.py energy_book, at every 1 ms plant step):
    P_bus    the battery: sum_j v_j i_j, joints on one bus (a braking joint feeds the others)
    P_cu     copper loss sum_j R_j i_j^2
    P_motor  the motors' net mechanical work sum_j tau_j theta_dot_j
pH from its own ledger; MuJoCo with an ideal DC motor behind each torque (i = tau / k, R / k^2 = w_max / tau_stall).
Sampled (the method of physics_comparison.pdf, 8 Nov 2026, scripts in its physics_comparison_data/): the same
formulas on the torque and speed at the end of each 20 ms policy step, and P_mech = sum_j |tau_j theta_dot_j|.
pH only: the ledger's residual over the stretch, dH - (supplied + mech_input - dissipated), as % of supplied: energy
the implicit integrator loses at the foot impacts that the ledger does not see.

Not measurements on the robot: both columns are what two simulators predict for the same policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import paths

SPEEDS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)
DT = 0.02


def run(backend, policy, speed: float, seconds: float, keep_from: float, r_over_k2: np.ndarray) -> dict:
    from rl.walk.backends import PHBackend, Walker

    w = Walker(backend, policy)
    w.reset()
    n, keep0 = int(round(seconds / DT)), int(round(keep_from / DT))
    ph = isinstance(backend, PHBackend)
    tau, qd, x = [], [], []
    book0 = led0 = None
    for i in range(n):
        if i == keep0:
            book0 = backend.energy_book()
            if ph:
                L = backend.plant.ledger
                led0 = (float(backend.plant.energy()["total"][0]), float(L.supplied[0]), float(L.mech_input[0]),
                        float(L.dissipated[0]))
        r = w.tick(np.array([speed, 0.0, 0.0]))
        if i >= keep0:
            tau.append(r.torque.copy())
            qd.append(r.qd.copy())
            x.append(r.pos.copy())
    dur = (n - keep0) * DT
    book1 = backend.energy_book()
    tau, qd, x = np.array(tau), np.array(qd), np.array(x)
    p = tau * qd
    v = float(np.linalg.norm(x[-1, :2] - x[0, :2]) / ((len(x) - 1) * DT))
    out = {"engine": backend.name, "request": speed, "speed": v, "fell": bool(w.fell),
           "booked": {k: (book1[k] - book0[k]) / dur for k in book1},
           "sampled": {"mech": float(np.abs(p).sum(1).mean()), "copper": float((tau ** 2 * r_over_k2).sum(1).mean()),
                       "bus": float((p + tau ** 2 * r_over_k2).sum(1).mean()), "motor": float(p.sum(1).mean())}}
    if led0 is not None:
        L = backend.plant.ledger
        H1 = float(backend.plant.energy()["total"][0])
        sup = float(L.supplied[0]) - led0[1]
        resid = (H1 - led0[0]) - (sup + float(L.mech_input[0]) - led0[2] - (float(L.dissipated[0]) - led0[3]))
        out["ledger_residual_pct"] = 100.0 * resid / sup if sup > 0 else float("nan")
    return out


def report(res: dict) -> str:
    rows = {(r["engine"], r["request"]): r for r in res["runs"]}
    L = [f"# Energy of {res['policy']} on both engines: booked every 1 ms, sampled every 20 ms", "",
         f"Standard robot model, {res['delay_ms']:g} ms transport delay, {res['seconds']:g} s per request, the stretch "
         f"from {res['keep_from']:g} s on averaged. Each cell: pH / MuJoCo [W]. Both columns are simulator "
         "predictions, not measurements on the robot.", "",
         "| request [m/s] | speed [m/s] | P_bus booked | P_cu booked | P_motor booked | P_bus sampled | P_cu sampled | "
         "P_bus pH vs MuJoCo, booked | sampled | pH ledger residual |", "|---|---|---|---|---|---|---|---|---|---|"]
    for s in res["speeds"]:
        a, b = rows[("pH", s)], rows[("MuJoCo", s)]
        f = lambda k, w: f"{a[w][k]:.0f} / {b[w][k]:.0f}"  # noqa: E731
        rel = lambda w: f"{100 * (a[w]['bus'] / b[w]['bus'] - 1):+.0f} %"  # noqa: E731
        fell = " (fell)" if a["fell"] or b["fell"] else ""
        L.append(f"| {s:g}{fell} | {a['speed']:.2f} / {b['speed']:.2f} | {f('bus', 'booked')} | {f('copper', 'booked')} | "
                 f"{f('motor', 'booked')} | {f('bus', 'sampled')} | {f('copper', 'sampled')} | {rel('booked')} | "
                 f"{rel('sampled')} | {a.get('ledger_residual_pct', float('nan')):+.1f} % |")
    L += ["", "## Cost of transport (booked bus power)", "", "| request [m/s] | pH | MuJoCo |", "|---|---|---|"]
    for s in res["speeds"]:
        a, b = rows[("pH", s)], rows[("MuJoCo", s)]
        if a["speed"] > 0.1 and b["speed"] > 0.1:
            cot = lambda r: r["booked"]["bus"] / (res["mass"] * 9.81 * r["speed"])  # noqa: E731
            L.append(f"| {s:g} | {cot(a):.2f} | {cot(b):.2f} |")
    L += ["", "## Reading", "",
          "- Booked against sampled: the 20 ms samples miss the current's ups and downs between two policy steps. "
          "The copper loss grows with the square of the torque, so it is the column that moves most.",
          "- The pH ledger residual is energy its integrator loses at foot impacts without booking it. Differences "
          "between the engines smaller than it are not shown by this comparison.",
          "- MuJoCo's electrical side is an ideal DC motor in steady state (no inductance); the pH plant has coils. "
          "Both use the same R/k² from the motors' torque-speed line.", ""]
    return "\n".join(L)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="sprint-repro-001")
    ap.add_argument("--delay-ms", type=float, default=20.0)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--keep-from", type=float, default=4.0)
    ap.add_argument("--out", default=str(paths.REPO / "outputs" / "energy"))
    a = ap.parse_args(argv)
    from ph_k1.model import load_k1
    from ph_k1.motors import build_motors
    from rl.walk.backends import MuJoCoBackend, PHBackend, load_walk_policy

    model = load_k1()
    mot = build_motors(model)
    r_over_k2 = (mot.R / mot.k ** 2).numpy()
    policy = load_walk_policy(a.policy)
    res = {"policy": a.policy, "delay_ms": a.delay_ms, "seconds": a.seconds, "keep_from": a.keep_from,
           "speeds": list(SPEEDS), "mass": float(model.total_mass), "model": str(model.xml_path), "runs": []}
    for make in (PHBackend, MuJoCoBackend):
        b = make(model=model, delay_ms=a.delay_ms)
        for s in SPEEDS:
            r = run(b, policy, s, a.seconds, a.keep_from, r_over_k2)
            res["runs"].append(r)
            print(f"[energy] {r['engine']:6s} {s:.1f} m/s: speed {r['speed']:.2f}, bus booked {r['booked']['bus']:.0f} W "
                  f"(sampled {r['sampled']['bus']:.0f}), copper booked {r['booked']['copper']:.0f} W "
                  f"(sampled {r['sampled']['copper']:.0f})"
                  + (f", ledger residual {r['ledger_residual_pct']:+.1f} %" if "ledger_residual_pct" in r else ""),
                  flush=True)
    out = Path(a.out) / a.policy
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1))
    (out / "report.md").write_text(report(res))
    print(f"[energy] wrote {out}")
    return res


if __name__ == "__main__":
    main()

r"""How fast is a gradient through the pH model?  CPU against GPU, float32 against float64, by number of robots.

One measurement = one training window (mbrl.grad.train.window, the code the training runs) of K control steps with
an untrained policy: forward without a graph, backward one control step at a time (memory of one step).

    python -m mbrl.grad.bench                                   # what this machine has
    python -m mbrl.grad.bench --robots 64 256 1024 --devices cpu cuda --dtypes float32 float64

Prints, per row: seconds per control step (forward, backward), robot-control-steps per second with gradient, and
the peak memory.  The numbers set the sizes of the training run (mbrl.grad.train).
"""

from __future__ import annotations

import argparse
import os
import resource
import sys
import time
import traceback

if "--trace" in sys.argv:   # before CUDA starts: errors are then reported at the operation that caused them
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

import torch

from mbrl.grad.env import GradStand, NA
from mbrl.mpc.nets import MLP
from rl.stand import contract as C


def measure(device: str, dtype, robots: int, steps: int) -> dict:
    """One training window (mbrl.grad.train.window: the same code the training runs) with an untrained policy."""
    from types import SimpleNamespace

    from mbrl.grad.train import OBS_NOISE, window

    env = GradStand(robots, device=device, dtype=dtype, obs_noise=OBS_NOISE, push_dv=0.3, delay_ms=(10.0, 70.0))
    policy = MLP(C.OBS_DIM, NA, last_scale=0.01).to(device=device, dtype=dtype)
    args = SimpleNamespace(steps=steps, clip=3.0, action_noise=0.05, gamma=0.99, terminal_seconds=1.0,
                           terminal_kinetic=1.0, fall_weight=1.0)
    cuda = device.startswith("cuda")
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    w = window(env, policy, args)
    if cuda:
        torch.cuda.synchronize()
    total = time.time() - t0
    mem = torch.cuda.max_memory_allocated() / 2**30 if cuda else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    return {"forward": w["stats"]["seconds_forward"] / steps, "backward": w["stats"]["seconds_backward"] / steps,
            "rate": robots * steps / total, "mem_GiB": mem, "finite": bool(torch.isfinite(w["g"]).all())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robots", type=int, nargs="+", default=[8, 32])
    ap.add_argument("--devices", nargs="+", default=["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
    ap.add_argument("--dtypes", nargs="+", default=["float32", "float64"])
    ap.add_argument("--steps", type=int, default=20, help="control steps per window, as in the training")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--trace", action="store_true", help="print the traceback of a failing row (and run CUDA blocking)")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    print(f"torch {torch.__version__}, {torch.get_num_threads()} CPU threads, "
          f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}", flush=True)
    print(f"{'device':6s} {'dtype':8s} {'robots':>6s}  {'fwd s/step':>10s} {'bwd s/step':>10s} {'robot-steps/s':>13s} "
          f"{'memory GiB':>10s}  ok", flush=True)
    for device in args.devices:
        for name in args.dtypes:
            for robots in args.robots:
                try:
                    m = measure(device, getattr(torch, name), robots, args.steps)
                    print(f"{device:6s} {name:8s} {robots:6d}  {m['forward']:10.2f} {m['backward']:10.2f} {m['rate']:13.1f} "
                          f"{m['mem_GiB']:10.2f}  {'yes' if m['finite'] else 'NOT FINITE'}", flush=True)
                except Exception as exc:  # noqa: BLE001  (out of memory on the larger rows is an answer too)
                    print(f"{device:6s} {name:8s} {robots:6d}  failed: {type(exc).__name__}: {str(exc)[:90]}", flush=True)
                    if args.trace:
                        print("".join(traceback.format_exc().splitlines(keepends=True)[-14:]), flush=True)
                    if device.startswith("cuda"):
                        torch.cuda.empty_cache()
    print("CPU memory is the process's peak so far (it never goes down); GPU memory is per row.", flush=True)


if __name__ == "__main__":
    main()

r"""How fast is a gradient through the pH model?  CPU against GPU, float32 against float64, by number of robots.

One measurement = a window of K control steps with an untrained policy (the robot stays near the home pose),
forward and backward, with one checkpoint block per control step (memory independent of K).

    python -m mbrl.grad.bench                                   # what this machine has
    python -m mbrl.grad.bench --robots 64 256 1024 --devices cpu cuda --dtypes float32 float64

Prints, per row: seconds per control step (forward, backward), robot-control-steps per second with gradient, and
the peak memory.  The numbers set the sizes of the training run (mbrl.grad.train).
"""

from __future__ import annotations

import argparse
import resource
import time

import torch

from mbrl.grad.env import GradStand, NA
from mbrl.mpc.nets import MLP
from rl.stand import contract as C


def measure(device: str, dtype, robots: int, steps: int) -> dict:
    env = GradStand(robots, device=device, dtype=dtype, checkpoint=True)
    policy = MLP(C.OBS_DIM, NA, last_scale=0.01).to(device=device, dtype=dtype)
    cuda = device.startswith("cuda")
    env.detach()
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    total = 0.0
    for _ in range(steps):
        r, _ = env.step(policy(env.observe()))
        total = total + r.mean()
    total = total - env.terminal_cost().mean()
    if cuda:
        torch.cuda.synchronize()
    t1 = time.time()
    (-total).backward()
    if cuda:
        torch.cuda.synchronize()
    t2 = time.time()
    grad = torch.cat([p.grad.flatten() for p in policy.parameters()])
    mem = torch.cuda.max_memory_allocated() / 2**30 if cuda else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    return {"forward": (t1 - t0) / steps, "backward": (t2 - t1) / steps, "rate": robots * steps / (t2 - t0),
            "mem_GiB": mem, "finite": bool(torch.isfinite(grad).all())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robots", type=int, nargs="+", default=[8, 32])
    ap.add_argument("--devices", nargs="+", default=["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
    ap.add_argument("--dtypes", nargs="+", default=["float32", "float64"])
    ap.add_argument("--steps", type=int, default=3, help="control steps per measurement")
    ap.add_argument("--threads", type=int, default=0)
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
                    if device.startswith("cuda"):
                        torch.cuda.empty_cache()
    print("CPU memory is the process's peak so far (it never goes down); GPU memory is per row.", flush=True)


if __name__ == "__main__":
    main()

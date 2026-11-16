r"""Learn to stand from scratch with the professor's formula: derivatives of the cost through the pH model.

    minimise over the network's weights w:   L = sum of the running cost (term 1) + cost at the window's end (term 2)
    where the state follows the pH model and the input comes from the controller (network + PD driver).

Every update (mbrl/grad/env.py is the model):
  1. roll all robots K control steps from where the last window ended, with a little action noise;
  2. backward pass through the model: g = d(-L)/d(action) for every robot and step, each robot's g limited to
     `clip` x the batch median;
  3. controlled step: action targets  y = a + delta * g / RMS(g);  the network is fitted to (observation, y);
  4. robots that fell are charged term 2 at that moment, stop counting in the window and start again.

The world is the one of the other rows of the thesis table: sensor noise, pushes, a random transport delay
(`--quiet` switches them off).  The policy is scored on the fast C++ plant with Full Body's rl/stand/evaluate.py,
the same evaluation as PPO and the planner.  Nothing is taken from another policy: the network starts random.

    python -m mbrl.grad.train --robots 8 --steps 20 --minutes 30 --no-wandb                       # laptop
    python -m mbrl.grad.train --robots 256 --steps 20 --device cuda --dtype float32 --hours 6     # a T4

Writes <log-root>/<run-name>/model_<it>.pt (rsl-rl's layout), deploy_<it>.bin (the firmware's file), metrics.jsonl.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import signal
import time
from pathlib import Path

import torch

from mbrl import paths
from mbrl.grad.env import GradStand, NA, POLICY_DT
from mbrl.mpc.nets import MLP, checkpoint
from rl.stand import contract as C

OBS_NOISE = {"gyro": 0.2, "gravity": 0.05, "joint_pos": 0.01, "joint_vel": 1.0}   # rl/stand/env.py StandConfig


def window(env: GradStand, policy: MLP, args, alive_age: torch.Tensor | None = None) -> dict:
    """One look-ahead window.  -> observations, actions, d(objective)/d(action) (clipped per robot), numbers to log.

    Forward without a graph, remembering the state before every control step and every random draw.  Backward one
    control step at a time: the step is computed again with a graph, and the derivative of everything later (the
    co-state, one number per state entry) is carried back through it.  Memory is one control step, whatever the
    length of the window; `window_one_graph` is the same calculation in a single graph (the test compares them)."""
    B, K, dev = env.B, args.steps, env.device
    t_start = time.time()
    env.detach()
    states, noises, epss, pushes, lives, falls, discs = [], [], [], [], [], [], []
    obs_list, act_list, grads = [], [], []
    alive = torch.ones(B, dtype=torch.bool, device=dev)
    fell_now = torch.zeros(B, dtype=torch.bool, device=dev)
    reward_sum, watts, objective, disc = 0.0, [], 0.0, 1.0
    with torch.no_grad():
        for _ in range(K):
            env.redraw_lags()
            states.append(env.get_state())
            clean = env.observe(noise=False)
            o = env.observe()
            a = policy(o)
            eps = args.action_noise * torch.randn(a.shape, device=a.device, dtype=a.dtype)
            push = env.draw_pushes()
            r, info = env.step(a + eps, push)
            live = alive.to(r.dtype)
            objective += disc * float((r * live).mean())
            reward_sum += float((r * live).sum())
            watts.append(float((info["copper_W"] * live).sum() / live.sum().clamp_min(1.0)))
            fallen = env.fallen() & alive
            if bool(fallen.any()):      # the end state of a fall is far from x_T: term 2, charged now
                objective -= disc * float((env.terminal_cost(args.terminal_seconds, args.terminal_kinetic)
                                           * fallen.to(r.dtype)).mean()) * args.fall_weight
            noises.append(o - clean); epss.append(eps); pushes.append(push); lives.append(live)
            falls.append(fallen); discs.append(disc)
            obs_list.append(o); act_list.append(a)
            fell_now |= fallen
            alive &= ~fallen
            disc *= args.gamma
        end = env.get_state()
    t_forward = time.time() - t_start

    def clip_and_keep(g):   # per robot: at most clip x the batch median; what flows on is the clipped one
        n = g.norm(dim=-1, keepdim=True)
        g = g * (args.clip * n.median() / n.clamp_min(1e-30)).clamp(max=1.0)
        grads.append(g.detach())
        return g

    def leaves_of(state):
        return {k: state[k].detach().requires_grad_(True) for k in env.FLOAT_STATE}

    def co_state(objective_part, leaves):
        g = torch.autograd.grad(objective_part, list(leaves.values()), allow_unused=True)
        return {k: (torch.zeros_like(leaves[k]) if gi is None else gi) for k, gi in zip(leaves, g)}

    # the end of the window: term 2
    leaves = leaves_of(end)
    env.set_state({**end, **leaves})
    term2 = env.terminal_cost(args.terminal_seconds, args.terminal_kinetic)
    lam = co_state(-disc * (term2 * alive.to(term2.dtype)).mean(), leaves)
    objective -= disc * float((term2.detach() * alive.to(term2.dtype)).mean())
    for t in reversed(range(K)):
        leaves = leaves_of(states[t])
        env.set_state({**states[t], **leaves})
        a = policy(env.observe(noise=False) + noises[t])
        a.register_hook(clip_and_keep)
        r, _ = env.step(a + epss[t], pushes[t])
        part = discs[t] * (r * lives[t]).mean()
        if bool(falls[t].any()):
            part = part - discs[t] * (env.terminal_cost(args.terminal_seconds, args.terminal_kinetic)
                                      * falls[t].to(r.dtype)).mean() * args.fall_weight
        after = env.get_state()
        part = part + sum((lam[k] * after[k]).sum() for k in env.FLOAT_STATE)     # everything later, through this step
        lam = co_state(part, leaves)
    env.set_state(end)               # the next window goes on from the end of this one
    g = torch.stack(grads[::-1])     # the hooks ran last step first; they saw d(objective)/d(action)
    norms = g.norm(dim=-1)
    return {"obs": torch.stack(obs_list), "act": torch.stack(act_list), "g": g, "fell": fell_now,
            "stats": {"reward_step": reward_sum / (B * K), "copper_W": sum(watts) / len(watts),
                      "term2": float(term2.detach().mean()), "grad_median": float(norms.median()),
                      "grad_max_over_median": float(norms.max() / norms.median().clamp_min(1e-30)),
                      "clipped": float((norms >= 0.999 * args.clip * norms.median(dim=1, keepdim=True).values).float().mean()),
                      "fell": float(fell_now.float().mean()), "objective": objective,
                      "seconds_forward": t_forward, "seconds_backward": time.time() - t_start - t_forward}}


def window_one_graph(env: GradStand, policy: MLP, args) -> torch.Tensor:
    """The same d(objective)/d(action) as `window`, from one graph over the whole window (memory grows with the
    window; for the test only).  -> (K, B, 22)."""
    B = env.B
    env.detach()
    grads = []
    alive = torch.ones(B, dtype=torch.bool, device=env.device)
    objective, disc = 0.0, 1.0

    def clip_and_keep(g):
        n = g.norm(dim=-1, keepdim=True)
        g = g * (args.clip * n.median() / n.clamp_min(1e-30)).clamp(max=1.0)
        grads.append(g.detach())
        return g

    for _ in range(args.steps):
        env.redraw_lags()
        a = policy(env.observe())
        a.register_hook(clip_and_keep)
        a = a + args.action_noise * torch.randn(a.shape, device=a.device, dtype=a.dtype)
        r, _ = env.step(a, env.draw_pushes())
        objective = objective + disc * (r * alive.to(r.dtype)).mean()
        fallen = env.fallen() & alive
        if bool(fallen.any()):
            objective = objective - disc * (env.terminal_cost(args.terminal_seconds, args.terminal_kinetic)
                                            * fallen.to(r.dtype)).mean() * args.fall_weight
            alive &= ~fallen
        disc *= args.gamma
    term2 = env.terminal_cost(args.terminal_seconds, args.terminal_kinetic)
    objective = objective - disc * (term2 * alive.to(term2.dtype)).mean()
    objective.backward()
    return torch.stack(grads[::-1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", default="grad-stand")
    ap.add_argument("--robots", type=int, default=8, help="0 = as many as fit: measured on the GPU at the start")
    ap.add_argument("--max-robots", type=int, default=1024, help="upper bound for --robots 0")
    ap.add_argument("--steps", type=int, default=20, help="control steps per window (K)")
    ap.add_argument("--delta", type=float, default=0.05, help="RMS change of the action targets per update (action units)")
    ap.add_argument("--delta-final", type=float, default=None,
                    help="shrink delta (and the learning rate with it) linearly to this over the time budget")
    ap.add_argument("--clip", type=float, default=3.0, help="one robot's action gradient at most this x the batch median")
    ap.add_argument("--regress", type=int, default=8, help="fitting steps of the network onto the action targets")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--action-noise", type=float, default=0.05)
    ap.add_argument("--terminal-seconds", type=float, default=1.0, help="term 2: pose, tilt, height error charged this long")
    ap.add_argument("--terminal-kinetic", type=float, default=1.0, help="term 2: cost per joule of kinetic energy")
    ap.add_argument("--fall-weight", type=float, default=1.0, help="term 2 at the moment of a fall, times this")
    ap.add_argument("--episode-s", type=float, default=10.0, help="robots start again after this long")
    ap.add_argument("--quiet", action="store_true", help="no sensor noise, no pushes, 20 ms delay")
    ap.add_argument("--iterations", type=int, default=1_000_000)
    ap.add_argument("--minutes", type=float, default=0.0)
    ap.add_argument("--hours", type=float, default=0.0)
    ap.add_argument("--save-every-min", type=float, default=10.0)
    ap.add_argument("--eval-every-min", type=float, default=20.0)
    ap.add_argument("--eval-robots", type=int, default=64, help="robots of the evaluation on the C++ plant; 0 = none")
    ap.add_argument("--eval-device", default="cpu")
    ap.add_argument("--resume", default=None, help="model_<it>.pt to continue from")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--log-root", default=str(paths.REPO / "logs"))
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-project", default="Akos pH MB")
    ap.add_argument("--fake-oom", action="store_true", help=argparse.SUPPRESS)   # test of the out-of-memory path
    ap.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", "karl-jahnel-universit-t-leipzig"))
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)
    run_dir = Path(args.log_root) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=1))

    world = {} if args.quiet else {"obs_noise": OBS_NOISE, "push_dv": 0.3, "delay_ms": (10.0, 70.0)}

    def make_env(robots: int) -> GradStand:
        return GradStand(robots, seed=args.seed, device=args.device, dtype=dtype, **world)

    if args.robots <= 0:      # measure one short window with 64 robots, then take what fits in 60 % of the GPU
        args.robots = 16
        if str(args.device).startswith("cuda"):
            probe = make_env(64)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            window(probe, MLP(C.OBS_DIM, NA, last_scale=0.01).to(device=args.device, dtype=dtype),
                   argparse.Namespace(**{**vars(args), "steps": 2}))
            torch.cuda.synchronize()
            per_robot = torch.cuda.max_memory_allocated() / 64
            total = torch.cuda.get_device_properties(0).total_memory
            probe = None
            gc.collect()
            torch.cuda.empty_cache()
            args.robots = max(32, min(args.max_robots, int(0.6 * total / per_robot) // 32 * 32))
            print(f"[grad] {per_robot / 2**20:.1f} MiB per robot measured, GPU {total / 2**30:.1f} GiB: "
                  f"{args.robots} robots", flush=True)
    env = make_env(args.robots)
    policy = MLP(C.OBS_DIM, NA, last_scale=0.01).to(device=args.device, dtype=dtype)
    value = MLP(C.OBS_DIM + 6, 1)       # not used by this learner; kept in the file so the layout is rsl-rl's
    opt = torch.optim.Adam(policy.mlp.parameters(), lr=args.lr)
    iteration, env_steps = 0, 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        policy.load_state_dict(ck["actor_state_dict"])
        policy.to(device=args.device, dtype=dtype)
        iteration, env_steps = int(ck.get("iter", 0)), int(ck.get("infos", {}).get("executed_steps", 0))
        print(f"[grad] continuing at iteration {iteration}", flush=True)

    evaluate = None
    if args.eval_robots > 0:
        from rl.stand.env import StandConfig
        from rl.stand.evaluate import StandEval
        evaluate = StandEval(StandConfig(num_envs=args.eval_robots, device=args.eval_device, dtype="float32", seed=args.seed),
                             args.eval_robots, device=args.eval_device)
    run = None
    if not args.no_wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.run_name,
                         config={"task": "stand", "method": "gradients through the pH model", **vars(args)})

    def act(obs):   # the bare policy on the evaluation's device and precision
        with torch.no_grad():
            x = obs["policy"]
            return policy(x.to(device=args.device, dtype=dtype)).to(device=x.device, dtype=x.dtype)

    def save() -> Path:
        from rl.stand.export import export_bin
        state = {k: (v.detach().cpu().float() if v.is_floating_point() else v.detach().cpu())
                 for k, v in policy.state_dict().items()}
        ck = checkpoint(policy, value, iteration, {"executed_steps": env_steps, "method": "grad"})
        ck["actor_state_dict"] = state
        path = run_dir / f"model_{iteration}.pt"
        torch.save(ck, path)
        try:
            export_bin(path, path.with_name(f"deploy_{iteration}.bin"))
        except Exception as exc:  # noqa: BLE001
            print(f"[export] failed: {type(exc).__name__}: {exc}", flush=True)
        return path

    print(f"[grad] stand from scratch: {args.robots} robots x {args.steps} control steps per window on {args.device} "
          f"({args.dtype}), world {'quiet' if args.quiet else 'with noise, pushes, delay'}, logs -> {run_dir}", flush=True)
    stop = {"why": None}
    old = signal.signal(signal.SIGINT, lambda *_: stop.update(why="SIGINT"))
    budget = max(args.minutes, 60 * args.hours)
    t0 = time.time()
    last = {"save": t0, "eval": t0}
    max_age = round(args.episode_s / POLICY_DT)
    # staggered ages: every window contains robots that have just started, not only robots in the middle of a stand
    age = torch.rand(args.robots, device=env.device) * max_age
    falls = 0
    with open(run_dir / "metrics.jsonl", "a") as metrics:
        try:
            while iteration < args.iterations and stop["why"] is None:
                t_it = time.time()
                # Memory grows with robots x window length (measured 16 Nov: about 11 MB per robot for the control
                # step being differentiated plus about 1.5 MB per robot and window step).  If the GPU runs out,
                # go on with half the robots instead of losing the run.
                out_of_memory = False
                try:
                    if args.fake_oom:
                        args.fake_oom = False
                        raise torch.OutOfMemoryError("test")
                    w = window(env, policy, args, age)
                except torch.OutOfMemoryError:
                    out_of_memory = True
                if out_of_memory:        # cleaned up outside the except block: the traceback holds the window's tensors
                    if args.robots <= 16:
                        raise SystemExit("out of memory even with 16 robots")
                    for p in policy.parameters():
                        p.grad = None
                    env = None
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    args.robots //= 2
                    print(f"[grad] out of memory: continuing with {args.robots} robots", flush=True)
                    env = make_env(args.robots)
                    age = torch.rand(args.robots, device=env.device) * max_age
                    continue
                g = w["g"]
                delta = args.delta
                if args.delta_final is not None and budget > 0:    # smaller steps towards the end of the run
                    done = min(1.0, (time.time() - t0) / (60 * budget))
                    delta = args.delta + done * (args.delta_final - args.delta)
                    for group in opt.param_groups:
                        group["lr"] = args.lr * delta / args.delta
                target = w["act"] + delta * g / g.square().mean().sqrt().clamp_min(1e-30)
                obs_b, target_b = w["obs"].flatten(0, 1), target.flatten(0, 1)
                policy.obs_normalizer.update(obs_b)
                loss = torch.zeros(())
                for _ in range(args.regress):                      # the network follows the improved actions
                    opt.zero_grad()
                    loss = (policy(obs_b) - target_b).square().mean()
                    loss.backward()
                    opt.step()
                # who starts again: the fallen and those whose episode is over
                age += args.steps
                again = w["fell"] | (age >= max_age)
                falls += int(w["fell"].sum())
                if bool(again.any()):
                    ids = again.nonzero().squeeze(-1)
                    env.reset(ids)
                    age[ids] = 0
                iteration += 1
                env_steps += args.robots * args.steps
                now = time.time()
                row = {"iteration": iteration, "Cost/executed_steps": env_steps, "Cost/seconds": now - t_it,
                       "Learn/fit_loss": float(loss), "Learn/delta": delta, **{f"Grad/{k}": v for k, v in w["stats"].items()}}
                if evaluate is not None and args.eval_every_min > 0 and (now - last["eval"]) / 60 >= args.eval_every_min:
                    try:
                        row.update({k: float(v) for k, v in evaluate(act, iteration).items()})
                    except Exception as exc:  # noqa: BLE001  (an evaluation must never end a training)
                        print(f"[eval] failed: {type(exc).__name__}: {exc}", flush=True)
                    last["eval"] = time.time()
                if args.save_every_min > 0 and (now - last["save"]) / 60 >= args.save_every_min:
                    save()
                    last["save"] = time.time()
                metrics.write(json.dumps(row) + "\n")
                metrics.flush()
                if run is not None:
                    run.log(row, step=iteration)
                s = w["stats"]
                extra = "".join(f"  {k.split('/')[-1]} {v:.3g}" for k, v in row.items() if k.startswith("Eval/"))
                print(f"[grad] it {iteration:4d}  steps {env_steps:8d}  reward/step {s['reward_step']:+.4f}  copper "
                      f"{s['copper_W']:6.1f} W  term2 {s['term2']:7.3f}  fell {s['fell']:.0%}  |g| {s['grad_median']:.3g} "
                      f"(max {s['grad_max_over_median']:.0f}x, clipped {s['clipped']:.0%})  ({row['Cost/seconds']:.0f} s){extra}",
                      flush=True)
                if budget > 0 and (time.time() - t0) / 60 > budget:
                    stop["why"] = f"time budget of {budget:g} min used up"
        finally:
            signal.signal(signal.SIGINT, old)
    final = {}
    if evaluate is not None:
        try:
            final = {k: float(v) for k, v in evaluate(act, iteration).items()}
            if run is not None:
                run.log(final, step=iteration)
        except Exception as exc:  # noqa: BLE001
            print(f"[eval] failed: {type(exc).__name__}: {exc}", flush=True)
    path = save()
    print(f"[grad] {'stopped: ' + stop['why'] if stop['why'] else 'done'} after iteration {iteration} "
          f"({(time.time() - t0) / 60:.1f} min, {env_steps} steps, {falls} falls); {path.name}"
          + "".join(f"  {k.split('/')[-1]} {v:.3g}" for k, v in final.items()), flush=True)
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()

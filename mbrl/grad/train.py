r"""Learn to stand or to walk from scratch with the professor's formula: derivatives of the cost through the pH model.

    minimise over the network's weights w:   L = sum of the running cost (term 1) + cost at the window's end (term 2)
    where the state follows the pH model and the input comes from the controller (network + PD driver).

Every update (mbrl/grad/env.py is the model):
  1. roll all robots K control steps from where the last window ended, with a little action noise;
  2. backward pass through the model: g = d(-L)/d(action) for every robot and step, each robot's g limited to
     `clip` x the batch median;
  3. controlled step: action targets  y = a + delta * g / RMS(g);  the network is fitted to (observation, y);
  4. robots that fell are charged term 2 at that moment, stop counting in the window and start again.

Term 2 at the window's end (--term2): `fixed` = ||x(T) - x_T||^2 with x_T upright, home pose, at rest
(GradStand.terminal_cost); `value` = -V(x(T)), a value network learned from the robots' rewards (TD(lambda),
mbrl/grad/value.py; roadmap W3).  With `value`, V learns from the start, the window's end keeps the fixed term 2 for
--value-start updates and then goes over to -V within --value-ramp updates; falls inside the window are charged the
fixed term 2 in both.

The world is the one of the other rows of the thesis table: sensor noise, pushes, a random transport delay
(`--quiet` switches them off).  The policy is scored on the fast C++ plant with Full Body's rl/stand/evaluate.py,
the same evaluation as PPO and the planner.  Nothing is taken from another policy: the network starts random.

--task walk (roadmap W5): Full Body's rl/walk task on mbrl/grad/walk_env.py (Eric's rewards and velocity requests,
the firmware's sprint agent 75 -> 22, the world of rl/walk/env.py), term 2 with learned weights by default.  A
curriculum multiplies every request by --command-scale (0.25 at the start: forward up to 0.5 m/s) and takes one step
up when the robots rarely fall and follow the requests (next_command_scale), up to Eric's ranges.  In-run
evaluation: rl/walk/evaluate.py WalkEval, six fixed commands on the C++ plant.

    python -m mbrl.grad.train --robots 8 --steps 20 --minutes 30 --no-wandb                       # laptop
    python -m mbrl.grad.train --robots 256 --steps 20 --device cuda --dtype float32 --hours 6     # a T4
    python -m mbrl.grad.train --robots 16 --minutes 40 --term2 weights --no-wandb                 # W3, laptop
    python -m mbrl.grad.train --task walk --robots 16 --minutes 30 --no-wandb                     # W5, laptop

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


def end_cost(env, args, value=None, value_weight: float = 1.0) -> torch.Tensor:
    """Term 2 at the end of a window, (B,): the fixed one (env.terminal_cost: x_T = upright, home pose, at rest), or
    minus the learned value V(x(T)) (value.py), or during the hand-over (value_weight < 1) a mix of both."""
    if value is None or value_weight <= 0.0:
        return env.terminal_cost(args.terminal_seconds, args.terminal_kinetic)
    learned = -value(value.inputs(env))
    if value_weight >= 1.0:
        return learned
    return (1.0 - value_weight) * env.terminal_cost(args.terminal_seconds, args.terminal_kinetic) + value_weight * learned


def window(env: GradStand, policy: MLP, args, alive_age: torch.Tensor | None = None, value=None,
           value_weight: float = 1.0) -> dict:
    """One look-ahead window.  -> observations, actions, d(objective)/d(action) (clipped per robot), numbers to log.

    Forward without a graph, remembering the state before every control step and every random draw.  Backward one
    control step at a time: the step is computed again with a graph, and the derivative of everything later (the
    co-state, one number per state entry) is carried back through it.  Memory is one control step, whatever the
    length of the window; `window_one_graph` is the same calculation in a single graph (the test compares them).

    value (value.ValueLearner or None): term 2 at the window's end is -V(x(T)) (end_cost); the window then also
    returns the critic's inputs and the TD(lambda) targets of V (full reward, falls charged as in the objective)."""
    B, K, dev = env.B, args.steps, env.device
    crits, full_rewards = [], []
    walk_stats = {"speed_error": [], "requested_speed": [], "feet_air": [], "slip_speed": []}   # walking only
    # the Hamiltonian's flows (what the battery supplies, what turns into heat where) and every reward term: sums
    # over the live robots and control steps, as tensors (no wait for the GPU in every step)
    flows, terms_sum, live_steps = {}, {}, 0.0
    t_start = time.time()
    env.detach()
    # the energy balance per robot over the window: supplied - heat - change of H.  Not zero: the plant's implicit
    # Euler step damps fast current changes (1/2 L di^2 per 1 ms step, Full Body ph_k1/plant.py) and the ledger does
    # not book it; pushes add energy outside the ledger, so pushed robots are left out
    with torch.no_grad():
        H_start = env.plant.energy()["total"]
    supplied_r = torch.zeros(B, dtype=H_start.dtype, device=dev)
    heat_r = torch.zeros(B, dtype=H_start.dtype, device=dev)
    pushed = torch.zeros(B, dtype=torch.bool, device=dev)
    states, noises, epss, pushes, lives, falls, discs = [], [], [], [], [], [], []
    obs_list, act_list, grads = [], [], []
    alive = torch.ones(B, dtype=torch.bool, device=dev)
    fell_now = torch.zeros(B, dtype=torch.bool, device=dev)
    reward_sum, watts, objective, disc = 0.0, [], 0.0, 1.0
    with torch.no_grad():
        for _ in range(K):
            env.between_steps()
            states.append(env.get_state())
            if value is not None:
                crits.append(value.inputs(env))
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
            if "speed_error" in info:   # walking: how well it follows the requests, and whether it steps
                for k, key in (("speed_error", "speed_error"), ("requested_speed", "requested"),
                               ("feet_air", "feet_air"), ("slip_speed", "slip_speed")):
                    if info.get(key) is not None:
                        walk_stats[k].append(float((info[key] * live).sum() / live.sum().clamp_min(1.0)))
            for key in ("supplied_W", "copper_W", "gearbox_W", "contact_W", "stops_W"):
                if info.get(key) is not None:
                    flows[key] = flows.get(key, 0.0) + (info[key] * live).sum()
                    if key == "supplied_W":
                        supplied_r = supplied_r + info[key] * POLICY_DT
                    else:
                        heat_r = heat_r + info[key] * POLICY_DT
            if push is not None:
                pushed |= push.abs().sum(-1) > 0
            for k, v in info.get("terms", {}).items():
                terms_sum[k] = terms_sum.get(k, 0.0) + (v * live).sum()
            if "reward_all" in info:
                terms_sum["(full reward)"] = terms_sum.get("(full reward)", 0.0) + (info["reward_all"] * live).sum()
                terms_sum["(differentiated)"] = terms_sum.get("(differentiated)", 0.0) + (r * live).sum()
            live_steps = live_steps + live.sum()
            fallen = env.fallen() & alive
            fall_cost = torch.zeros_like(r)
            if bool(fallen.any()):      # the end state of a fall is far from x_T: term 2, charged now
                tc = env.terminal_cost(args.terminal_seconds, args.terminal_kinetic) * fallen.to(r.dtype)
                objective -= disc * float(tc.mean()) * args.fall_weight
                fall_cost = tc * args.fall_weight
            if value is not None:       # what V learns from (--value-reward), the fall charged as in the objective
                learn = info["reward_all"] if getattr(args, "value_reward", "smooth") == "full" else r
                full_rewards.append(learn - fall_cost)
            noises.append(o - clean); epss.append(eps); pushes.append(push); lives.append(live)
            falls.append(fallen); discs.append(disc)
            obs_list.append(o); act_list.append(a)
            fell_now |= fallen
            alive &= ~fallen
            disc *= args.gamma
        end = env.get_state()
        # the Hamiltonian's tanks at the window's end (robots still up), and the energy the PD law adds to it: the
        # closed-loop H_d = H + sum 1/2 kp (theta - target)^2 (the motor law is a virtual spring to the target)
        energy_end = {}
        if bool(alive.any()):
            tanks = env.plant.energy()
            for k in ("kinetic", "gravity", "contact", "stops", "coil", "total"):
                energy_end[f"H_{k}_J"] = float(tanks[k][alive].mean())
            if getattr(env, "motor_target", None) is not None and getattr(env, "kp", None) is not None:
                shaping = 0.5 * (env.kp * (env.plant.q[:, 7:] - env.motor_target) ** 2).sum(-1)
                energy_end["Hd_shaping_J"] = float(shaping[alive].mean())
                energy_end["Hd_total_J"] = energy_end["H_total_J"] + energy_end["Hd_shaping_J"]
            clean = alive & ~pushed
            if "supplied_W" in flows and bool(clean.any()):
                unbooked = supplied_r - heat_r - (tanks["total"] - H_start)
                energy_end["unbooked_W"] = float(unbooked[clean].mean()) / (K * POLICY_DT)
                energy_end["unbooked_share"] = float(unbooked[clean].sum() / supplied_r[clean].sum().clamp_min(1e-9))
        value_out = {}
        if value is not None:       # the targets of V: TD(lambda) returns, bootstrapped with V's target copy
            from mbrl.grad.value import lambda_returns
            crit = torch.stack(crits + [value.inputs(env)])                        # (K+1, B, n)
            v = value(crit.flatten(0, 1)).reshape(K + 1, B)
            ret = lambda_returns(torch.stack(full_rewards), v, torch.stack(falls), args.gamma, args.lam)
            value_out = {"critic": crit[:-1], "value_target": ret, "valid": torch.stack(lives) > 0,
                         "value_end": float(v[-1][alive].mean()) if bool(alive.any()) else float("nan")}
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

    quantiles = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)

    def costate_sizes(lam) -> dict:   # per robot |lambda| of the positions, momenta, coil fluxes: 10 %, median, 90 %
        return {name: torch.quantile(lam[key].detach().flatten(1).norm(dim=-1).double().cpu(), quantiles).tolist()
                for key, name in (("q", "q"), ("p", "p"), ("lam", "flux")) if key in lam}

    # the end of the window: term 2
    leaves = leaves_of(end)
    env.set_state({**end, **leaves})
    term2 = end_cost(env, args, value, value_weight)
    lam = co_state(-disc * (term2 * alive.to(term2.dtype)).mean(), leaves)
    objective -= disc * float((term2.detach() * alive.to(term2.dtype)).mean())
    sizes = [costate_sizes(lam)]      # the co-state at the end, then before every step going back
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
        sizes.append(costate_sizes(lam))
    env.set_state(end)               # the next window goes on from the end of this one
    g = torch.stack(grads[::-1])     # the hooks ran last step first; they saw d(objective)/d(action)
    norms = g.norm(dim=-1)
    # the derivative through the pH model, step by step: the action gradient's size per control step and the
    # co-state's before every step (index 0) up to the window's end (index K); 10 %, median, 90 % over the robots
    sizes = sizes[::-1]
    trace = {"g": torch.quantile(norms.detach().double().cpu(), quantiles, dim=1).T.tolist(),
             **{name: [s[name] for s in sizes] for name in sizes[0]}}
    gradient = {"action_first_step": trace["g"][0][1], "action_middle_step": trace["g"][K // 2][1],
                "action_last_step": trace["g"][-1][1]}
    for name in sizes[0]:
        gradient[f"costate_{name}_end"], gradient[f"costate_{name}_start"] = trace[name][-1][1], trace[name][0][1]
    n_live = max(float(live_steps), 1.0)
    return {**value_out, "obs": torch.stack(obs_list), "act": torch.stack(act_list), "g": g, "fell": fell_now,
            "stats": {"reward_step": reward_sum / (B * K), "copper_W": sum(watts) / len(watts),
                      "term2": float(term2.detach().mean()), "grad_median": float(norms.median()),
                      "grad_max_over_median": float(norms.max() / norms.median().clamp_min(1e-30)),
                      "clipped": float((norms >= 0.999 * args.clip * norms.median(dim=1, keepdim=True).values).float().mean()),
                      "fell": float(fell_now.float().mean()), "objective": objective,
                      "seconds_forward": t_forward, "seconds_backward": time.time() - t_start - t_forward},
            "walk": {k: sum(v) / len(v) for k, v in walk_stats.items() if v},
            "energy": {**{k: float(v) / n_live for k, v in flows.items()}, **energy_end},
            "reward": {k: float(v) / n_live for k, v in terms_sum.items()},
            "gradient": gradient, "trace": trace}


def window_one_graph(env: GradStand, policy: MLP, args, value=None, value_weight: float = 1.0) -> torch.Tensor:
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
        env.between_steps()
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
    term2 = end_cost(env, args, value, value_weight)
    objective = objective - disc * (term2 * alive.to(term2.dtype)).mean()
    objective.backward()
    return torch.stack(grads[::-1])


def next_command_scale(scale: float, recent: list, args) -> float:
    """The walking curriculum (W5): every request is multiplied by `scale` (WalkTask.command_scale).  One step up
    (args.command_scale_step, at most args.command_scale_max) when, over the last args.curriculum_updates updates,
    at most args.curriculum_falls of the robot-windows ended in a fall and the speed error was at most
    args.curriculum_error x the requested speed.  recent: (share of robots fallen, speed error, requested speed) per
    update.  It never goes down."""
    if scale >= args.command_scale_max or len(recent) < args.curriculum_updates:
        return scale
    n = len(recent)
    falls, error, requested = (sum(r[i] for r in recent) / n for i in range(3))
    if falls <= args.curriculum_falls and error <= args.curriculum_error * requested:
        return min(args.command_scale_max, round(scale + args.command_scale_step, 6))
    return scale


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=("stand", "walk"), default="stand",
                    help="stand (W1, W3) or walk (W5: rl/walk's task, the firmware's sprint agent, a curriculum)")
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
    ap.add_argument("--term2", choices=("fixed", "weights", "value"), default=None,
                    help="term 2 at the window's end: the fixed goal state; its weights learned (W3); a learned "
                         "value network V (W3, failed on the laptop).  Default: fixed for stand, weights for walk")
    ap.add_argument("--weights-lr", type=float, default=1e-2, help="--term2 weights: Adam's step on the weights")
    ap.add_argument("--lam", type=float, default=0.95, help="TD(lambda) of the value's targets")
    ap.add_argument("--value-lr", type=float, default=5e-4)
    ap.add_argument("--value-epochs", type=int, default=5)
    ap.add_argument("--value-minibatches", type=int, default=4)
    ap.add_argument("--value-buffer", type=int, default=4, help="windows the value learns from (the newest)")
    ap.add_argument("--value-tau", type=float, default=0.2, help="how fast V's target copy follows V")
    ap.add_argument("--value-start", type=int, default=None,
                    help="updates V only learns before it is used (default: 20 for value, 0 for weights)")
    ap.add_argument("--value-ramp", type=int, default=None,
                    help="then: updates over which term 2 goes over to -V (default: 10 for value, 1 for weights)")
    ap.add_argument("--value-hidden", default="128,128", help="V's hidden layers")
    ap.add_argument("--value-reward", choices=("smooth", "full"), default="smooth",
                    help="what V learns: the differentiated objective (smooth terms + fall charge), or the full reward")
    ap.add_argument("--episode-s", type=float, default=None,
                    help="robots start again after this long (default: 10 s stand, 20 s walk as rl/walk/env.py)")
    ap.add_argument("--command-scale", type=float, default=0.25,
                    help="walk: every request x this at the start (0.25: forward up to 0.5 m/s; 1 = Eric's ranges)")
    ap.add_argument("--command-scale-max", type=float, default=1.0, help="walk: where the curriculum stops")
    ap.add_argument("--command-scale-step", type=float, default=0.05, help="walk: one step of the curriculum")
    ap.add_argument("--curriculum-updates", type=int, default=20, help="walk: updates the curriculum looks back over")
    ap.add_argument("--curriculum-falls", type=float, default=0.002,
                    help="walk: one step up only if at most this share of robot-windows ended in a fall")
    ap.add_argument("--curriculum-error", type=float, default=0.3,
                    help="walk: ... and the speed error was at most this x the requested speed")
    ap.add_argument("--gait-clock", type=float, default=0.0,
                    help="walk: weight of the gait clock's reward (W5c; Full Body rl/walk/task.py gait_clock), and the clock's "
                         "two inputs after the 75 (77 inputs, the firmware's walk_clock agent); 0 = off, as in W5")
    ap.add_argument("--air-time-smooth", type=float, default=0.0,
                    help="walk: weight of air_time's smooth stand-in (Full Body rl/walk/task.py STAND_INS: the weight a "
                         "foot has taken off the ground and kept off for 3 control steps), differentiated beside Eric's "
                         "terms (W5b); 0 = off, as in W5")
    ap.add_argument("--quiet", action="store_true", help="no sensor noise, no pushes, 20 ms delay")
    ap.add_argument("--iterations", type=int, default=1_000_000)
    ap.add_argument("--minutes", type=float, default=0.0)
    ap.add_argument("--hours", type=float, default=0.0)
    ap.add_argument("--save-every-min", type=float, default=10.0)
    ap.add_argument("--eval-every-min", type=float, default=20.0)
    ap.add_argument("--eval-robots", type=int, default=64, help="robots of the evaluation on the C++ plant; 0 = none")
    ap.add_argument("--eval-device", default="cpu")
    ap.add_argument("--videos", action="store_true",
                    help="walk: with every evaluation a video of the policy (needs MuJoCo rendering: MUJOCO_GL=egl on "
                         "a GPU machine) and the Hamiltonian figure of that episode")
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
    walk = args.task == "walk"
    if args.term2 is None:
        args.term2 = "weights" if walk else "fixed"
    if args.episode_s is None:
        args.episode_s = 20.0 if walk else 10.0
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)
    run_dir = Path(args.log_root) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=1))

    curriculum = {"scale": args.command_scale, "recent": []}     # walk: what the requests are multiplied by
    if walk:   # the world of rl/walk/env.py WalkConfig, where the walk is scored
        from mbrl.grad.walk_env import OBS_NOISE as WALK_OBS_NOISE
        from mbrl.grad.walk_env import GradWalk
        from rl.walk import contract as WC
        world = {} if args.quiet else {"obs_noise": WALK_OBS_NOISE, "push_dv": 0.5, "delay_ms": (10.0, 70.0),
                                       "gain_scale": (0.6, 1.3), "encoder_bias": 0.015, "joint_noise": 0.1,
                                       "random_yaw": True}
        obs_dim = WC.OBS_DIM + (2 if args.gait_clock > 0 else 0)          # the clock: sin, cos after the 75
        critic_dim = obs_dim + 15
    else:
        world = {} if args.quiet else {"obs_noise": OBS_NOISE, "push_dv": 0.3, "delay_ms": (10.0, 70.0)}
        obs_dim, critic_dim = C.OBS_DIM, C.OBS_DIM + 6

    def make_env(robots: int):
        if walk:
            return GradWalk(robots, seed=args.seed, device=args.device, dtype=dtype, command_scale=curriculum["scale"],
                            air_time_smooth=args.air_time_smooth, gait_clock=args.gait_clock, **world)
        return GradStand(robots, seed=args.seed, device=args.device, dtype=dtype, **world)

    if args.robots <= 0:      # measure one short window with 64 robots, then take what fits in 60 % of the GPU
        args.robots = 16
        if str(args.device).startswith("cuda"):
            probe = make_env(64)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            window(probe, MLP(obs_dim, NA, last_scale=0.01).to(device=args.device, dtype=dtype),
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
    policy = MLP(obs_dim, NA, last_scale=0.01).to(device=args.device, dtype=dtype)
    value = MLP(critic_dim, 1)          # with --term2 fixed not used; kept in the file so the layout is rsl-rl's
    learner_v = None
    if args.value_start is None:
        args.value_start = 20 if args.term2 == "value" else 0
    if args.value_ramp is None:
        args.value_ramp = 10 if args.term2 == "value" else 1
    if args.term2 == "weights":     # starts as the fixed term 2 and learns its weights
        from mbrl.grad.value import WeightLearner
        learner_v = WeightLearner(env.STATE_COSTS, env.fixed_term2_weights(args.terminal_seconds, args.terminal_kinetic),
                                  device=args.device, dtype=dtype, lr=args.weights_lr, epochs=args.value_epochs,
                                  minibatches=args.value_minibatches, buffer=args.value_buffer, tau=args.value_tau)
    if args.term2 == "value":
        from mbrl.grad.value import ValueLearner
        learner_v = ValueLearner(env.critic_obs().shape[-1], device=args.device, dtype=dtype, lr=args.value_lr,
                                 epochs=args.value_epochs, minibatches=args.value_minibatches, buffer=args.value_buffer,
                                 tau=args.value_tau, hidden=[int(h) for h in args.value_hidden.split(",")])
        value = learner_v.net
    opt = torch.optim.Adam(policy.mlp.parameters(), lr=args.lr)
    iteration, env_steps = 0, 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        policy.load_state_dict(ck["actor_state_dict"])
        policy.to(device=args.device, dtype=dtype)
        iteration, env_steps = int(ck.get("iter", 0)), int(ck.get("infos", {}).get("executed_steps", 0))
        if learner_v is not None and "value_learner" in ck:
            learner_v.load_state_dict(ck["value_learner"])
        if walk:                        # the curriculum goes on where it was
            curriculum["scale"] = float(ck.get("infos", {}).get("command_scale", curriculum["scale"]))
            env.task.command_scale = curriculum["scale"]
        print(f"[grad] continuing at iteration {iteration}", flush=True)

    evaluate = None
    if args.eval_robots > 0 and walk:   # the six fixed commands, under the training's conditions
        from rl.walk.env import WalkConfig
        from rl.walk.evaluate import WalkEval
        evaluate = WalkEval(WalkConfig(num_envs=args.eval_robots, device=args.eval_device, dtype="float32", seed=args.seed,
                                       gait_clock=args.gait_clock > 0),
                            args.eval_robots, device=args.eval_device)
    elif args.eval_robots > 0:
        from rl.stand.env import StandConfig
        from rl.stand.evaluate import StandEval
        evaluate = StandEval(StandConfig(num_envs=args.eval_robots, device=args.eval_device, dtype="float32", seed=args.seed),
                             args.eval_robots, device=args.eval_device)
    run = None
    if not args.no_wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.run_name,
                         config={"method": "gradients through the pH model", **vars(args)})
    print(f"[grad] term 2 at the window's end: {dict(fixed='fixed goal state', weights='learned weights (W3)', value='learned value network V (W3)')[args.term2]}",
          flush=True)
    if walk and args.gait_clock > 0:
        print(f"[grad] gait clock: {obs_dim} inputs (the 75 and sin, cos of the clock), its reward with weight "
              f"{args.gait_clock:g} (W5c)", flush=True)
    if walk and args.air_time_smooth > 0:
        print(f"[grad] air_time's smooth stand-in (weight taken off a foot) differentiated with weight "
              f"{args.air_time_smooth:g} (W5b)", flush=True)

    def act(obs):   # the bare policy on the evaluation's device and precision
        with torch.no_grad():
            x = obs["policy"]
            return policy(x.to(device=args.device, dtype=dtype)).to(device=x.device, dtype=x.dtype)

    def save() -> Path:
        if walk:
            from rl.walk.export import export_bin       # the firmware's sprint agent, 75 -> 22
        else:
            from rl.stand.export import export_bin      # the firmware's stand agent, 72 -> 22
        state = {k: (v.detach().cpu().float() if v.is_floating_point() else v.detach().cpu())
                 for k, v in policy.state_dict().items()}
        ck = checkpoint(policy, value, iteration, {"executed_steps": env_steps, "method": "grad", "term2": args.term2})
        if walk:
            ck["infos"].update(task="walk", command_scale=curriculum["scale"])
        if args.term2 == "weights":
            ck["infos"]["term2_weights"] = dict(zip(learner_v.names, learner_v.s_target.tolist()))
        ck["actor_state_dict"] = state
        if learner_v is not None:
            ck["value_learner"] = learner_v.state_dict()
        path = run_dir / f"model_{iteration}.pt"
        torch.save(ck, path)
        try:
            if walk:                                    # the sprint agent's 75, or walk_clock's 77 (W5c)
                export_bin(path, path.with_name(f"deploy_{iteration}.bin"), obs_dim)
            else:
                export_bin(path, path.with_name(f"deploy_{iteration}.bin"))
        except Exception as exc:  # noqa: BLE001
            print(f"[export] failed: {type(exc).__name__}: {exc}", flush=True)
        return path

    video = None
    if walk and args.videos:   # one quiet episode on the C++ plant, the requests changing, and its Hamiltonian figure
        from rl.walk.env import WalkConfig
        from rl.walk.video import WalkVideo
        video = WalkVideo(WalkConfig(num_envs=1, device="cpu", dtype="float32", seed=args.seed, gait_clock=args.gait_clock > 0),
                          run_dir / "videos")

    def make_media(it: int, w: dict) -> dict:
        """With every evaluation: the derivative through the pH model in this update's window (figures.py), and with
        --videos a video of the policy and the Hamiltonian figure of that episode.  -> {W&B key: file}."""
        from mbrl.grad.figures import gradient_figure

        out = {}
        try:
            out["Gradient/through_window"] = gradient_figure(
                w["trace"], run_dir / "figures" / f"gradient_{it:05d}.png", POLICY_DT,
                title=f"update {it}: the derivative through the pH model in one window of {args.steps} control steps, "
                      f"{args.robots} robots")
        except Exception as exc:  # noqa: BLE001  (a figure must never end a training)
            print(f"[figure] failed: {type(exc).__name__}: {exc}", flush=True)
        if video is not None:
            result = video(act, it)
            path, extra = result if isinstance(result, tuple) else (result, {})
            out.update({k: v for k, v in extra.items() if v})
            if path:
                out["Video/walk"] = path
        return out

    def to_wandb(media: dict) -> dict:
        if not media:
            return {}
        import wandb

        return {k: (wandb.Video(str(v), format="mp4") if str(v).endswith(".mp4") else wandb.Image(str(v)))
                for k, v in media.items()}

    print(f"[grad] {args.task} from scratch: {args.robots} robots x {args.steps} control steps per window on {args.device} "
          f"({args.dtype}), world {'quiet' if args.quiet else 'with noise, pushes, delay'}, logs -> {run_dir}"
          + (f"; requests x {curriculum['scale']:g} (curriculum up to {args.command_scale_max:g})" if walk else ""),
          flush=True)
    stop = {"why": None}
    old = signal.signal(signal.SIGINT, lambda *_: stop.update(why="SIGINT"))
    budget = max(args.minutes, 60 * args.hours)
    t0 = time.time()
    last = {"save": t0, "eval": t0}
    max_age = round(args.episode_s / POLICY_DT)
    # staggered ages: every window contains robots that have just started, not only robots in the middle of a stand
    age = torch.rand(args.robots, device=env.device) * max_age
    falls = episodes = 0
    w = None
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
                    weight = min(1.0, max(0.0, (iteration - args.value_start + 1) / max(args.value_ramp, 1)))
                    w = window(env, policy, args, age, value=learner_v, value_weight=weight)
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
                with torch.no_grad():
                    before = policy(obs_b)
                loss = torch.zeros(())
                for _ in range(args.regress):                      # the network follows the improved actions
                    opt.zero_grad()
                    loss = (policy(obs_b) - target_b).square().mean()
                    loss.backward()
                    opt.step()
                with torch.no_grad():   # how far the network's actions moved, against the step the targets asked for
                    change = float((policy(obs_b) - before).square().mean().sqrt())
                learn_row = {"Learn/action_change": change, "Learn/step_taken": change / max(delta, 1e-12)}
                value_row = {}
                if learner_v is not None:                          # V learns from this window's returns
                    fit = learner_v.fit(w["critic"][w["valid"]], w["value_target"][w["valid"]])
                    value_row = {f"Value/{k}": v for k, v in fit.items()}
                    value_row.update({"Value/weight": weight, "Value/end_mean": w["value_end"]})
                # who starts again: the fallen and those whose episode is over
                age += args.steps
                again = w["fell"] | (age >= max_age)
                falls += int(w["fell"].sum())
                episode_row = {}
                if bool(again.any()):
                    ids = again.nonzero().squeeze(-1)
                    episodes += int(ids.numel())                   # episodes that ended: how long they lasted
                    episode_row = {"Train/episode_s": float(age[ids].mean()) * POLICY_DT,
                                   "Train/fall_share": float(w["fell"][ids].float().mean())}
                    env.reset(ids)
                    age[ids] = 0
                episode_row.update({"Train/episodes": episodes, "Train/falls": falls})
                iteration += 1
                env_steps += args.robots * args.steps
                walk_row = {}
                if walk:                                           # the curriculum: one step up when it walks well
                    s = {**w["stats"], **w["walk"]}
                    curriculum["recent"] = (curriculum["recent"] + [(s["fell"], s["speed_error"], s["requested_speed"])]
                                            )[-args.curriculum_updates:]
                    new = next_command_scale(curriculum["scale"], curriculum["recent"], args)
                    if new != curriculum["scale"]:
                        print(f"[grad] curriculum: requests x {new:g} (was x {curriculum['scale']:g})", flush=True)
                        curriculum["scale"], curriculum["recent"] = new, []
                        env.task.command_scale = new
                    walk_row = {"Curriculum/command_scale": curriculum["scale"]}
                now = time.time()
                row = {"iteration": iteration, "Cost/executed_steps": env_steps, "Cost/seconds": now - t_it,
                       "Learn/fit_loss": float(loss.detach()), "Learn/delta": delta, **{f"Grad/{k}": v for k, v in w["stats"].items()},
                       **value_row, **walk_row, **learn_row, **episode_row,
                       **{f"Gradient/{k}": v for k, v in w["gradient"].items()},
                       **{f"Energy/{k}": v for k, v in w["energy"].items()},
                       **{f"Reward/{k}": v for k, v in w["reward"].items()},
                       **{f"Walk/{k}": v for k, v in w["walk"].items()}}
                evaluated = False
                if evaluate is not None and args.eval_every_min > 0 and (now - last["eval"]) / 60 >= args.eval_every_min:
                    try:
                        row.update({k: float(v) for k, v in evaluate(act, iteration).items()})
                        evaluated = True
                    except Exception as exc:  # noqa: BLE001  (an evaluation must never end a training)
                        print(f"[eval] failed: {type(exc).__name__}: {exc}", flush=True)
                    last["eval"] = time.time()
                # every evaluated update is saved too: the scoring picks among evaluated updates with a policy file
                if evaluated or (args.save_every_min > 0 and (now - last["save"]) / 60 >= args.save_every_min):
                    save()
                    last["save"] = time.time()
                media = make_media(iteration, w) if evaluated else {}     # with every evaluation: figures, video
                metrics.write(json.dumps({**row, **{f"{k} (file)": str(v) for k, v in media.items()}}) + "\n")
                metrics.flush()
                if run is not None:
                    run.log({**row, **to_wandb(media)}, step=iteration)
                s = {**w["stats"], **w["walk"]}
                extra = "".join(f"  {k.split('/')[-1]} {v:.3g}" for k, v in row.items() if k.startswith("Eval/"))
                print(f"[grad] it {iteration:4d}  steps {env_steps:8d}  reward/step {s['reward_step']:+.4f}  copper "
                      f"{s['copper_W']:6.1f} W  term2 {s['term2']:7.3f}  fell {s['fell']:.0%}  |g| {s['grad_median']:.3g} "
                      f"(max {s['grad_max_over_median']:.0f}x, clipped {s['clipped']:.0%})  ({row['Cost/seconds']:.0f} s)"
                      + (f"  V {value_row['Value/value_mean']:+.3f} return {value_row['Value/return_mean']:+.3f} "
                         f"expl.var {value_row['Value/explained_variance']:+.2f} weight {value_row['Value/weight']:.2f}"
                         if value_row else "")
                      + ("  s " + " ".join(f"{v:.2f}" for k, v in value_row.items() if k.startswith("Value/s_"))
                         if args.term2 == "weights" else "")
                      + (f"  speed err {s['speed_error']:.2f} of {s['requested_speed']:.2f} m/s  feet in air "
                         f"{s['feet_air']:.0%}  slip {s['slip_speed']:.2f} m/s  requests x{curriculum['scale']:g}"
                         if walk else "") + extra,
                      flush=True)
                if budget > 0 and (time.time() - t0) / 60 > budget:
                    stop["why"] = f"time budget of {budget:g} min used up"
        finally:
            signal.signal(signal.SIGINT, old)
    final = {}
    if evaluate is not None:
        try:
            final = {k: float(v) for k, v in evaluate(act, iteration).items()}
            media = make_media(iteration, w) if w is not None else {}       # the final policy's figures and video
            if run is not None:
                run.log({**final, **to_wandb(media)}, step=iteration)
        except Exception as exc:  # noqa: BLE001
            print(f"[eval] failed: {type(exc).__name__}: {exc}", flush=True)
    path = save()
    print(f"[grad] {'stopped: ' + stop['why'] if stop['why'] else 'done'} after iteration {iteration} "
          f"({(time.time() - t0) / 60:.1f} min, {env_steps} steps, {falls} falls); {path.name}"
          + "".join(f"  {k.split('/')[-1]} {v:.3g}" for k, v in final.items() if k.startswith("Eval/")), flush=True)
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()

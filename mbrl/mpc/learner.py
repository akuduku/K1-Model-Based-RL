r"""The model-based learner's loop: plan, act in the world, learn from what was executed.

One iteration:
  1. T control steps.  At each the planner (planner.py) gives every world robot an action; a share `policy_share` of
     the robots executes the bare policy's action instead (so the policy also sees the states its own mistakes lead
     to), but every robot's label is the planner's action.
  2. The value network fits the lambda-returns of the executed rollout.
  3. The policy regresses on the planner's actions, on a replay of the last `replay` rollouts.

What the harness gives PPO is here too: W&B, a time budget with a clean stop (also on Ctrl-C), checkpoints with the
firmware's .bin, resume, an evaluation of the bare policy, a video.  Costs are counted twice: executed steps (what a
world robot did) and model steps (the planner's rollouts).
"""

from __future__ import annotations

import json
import signal
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .nets import MLP, checkpoint
from .planner import Planner


@dataclass
class LearnConfig:
    candidates: int = 16          # N: model copies per world robot
    horizon: int = 10             # H: planner steps (x 20 ms)
    sigma: float = 0.3            # action noise of the planner's samples (action units)
    beta: float = 0.7             # its correlation from step to step
    temperature: float = 0.5      # MPPI: weights ~ exp((G - max) / (temperature x std G))
    gamma: float = 0.99
    lam: float = 0.95
    steps: int = 24               # T: control steps per iteration
    policy_share: float = 0.25    # share of the robots that execute the bare policy
    replay: int = 20              # rollouts the policy learns from
    epochs: int = 4
    minibatches: int = 4
    lr: float = 1e-3
    hidden: tuple = (512, 256, 128)
    seed: int = 1


class Learner:
    def __init__(self, world, model, cfg: LearnConfig, run_dir: Path, export=None, evaluate=None, video=None,
                 after_restore=None, wandb_run=None, log=None):
        self.world, self.model, self.cfg = world, model, cfg
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.export, self.evaluate, self.video, self.wandb = export, evaluate, video, wandb_run
        self.log = log or (lambda *a: print(*a, flush=True))
        torch.manual_seed(cfg.seed)
        dev = world.device
        obs = world.get_observations()
        self.policy = MLP(obs["policy"].shape[1], world.num_actions, cfg.hidden, last_scale=0.01).to(dev)
        self.value = MLP(obs["critic"].shape[1], 1, cfg.hidden).to(dev)
        self.opt_p = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr)
        self.opt_v = torch.optim.Adam(self.value.parameters(), lr=cfg.lr)
        self.planner = Planner(world, model, self.policy, self.value, cfg.candidates, cfg.horizon, cfg.sigma, cfg.beta,
                               cfg.gamma, cfg.temperature, after_restore)
        self.buffer: deque = deque(maxlen=cfg.replay)
        self.iteration, self.executed = 0, 0
        self.stop = None
        (self.run_dir / "config.json").write_text(json.dumps(
            {"learner": asdict(cfg), "world_robots": world.num_envs, "model_robots": model.num_envs,
             "env": world.config_dict() if hasattr(world, "config_dict") else {}}, indent=1, default=str))

    # ------------------------------------------------------------------ files
    def save(self) -> Path:
        path = self.run_dir / f"model_{self.iteration}.pt"
        ck = checkpoint(self.policy, self.value, self.iteration, {
            "executed_steps": self.executed, "model_steps": self.planner.model_steps,
            "opt_p": self.opt_p.state_dict(), "opt_v": self.opt_v.state_dict()})
        torch.save(ck, path)
        if self.export is not None:
            try:
                self.export(path, path.with_name(f"deploy_{self.iteration}.bin"))
            except Exception as exc:  # noqa: BLE001
                self.log(f"[export] failed: {type(exc).__name__}: {exc}")
        return path

    def load(self, path) -> None:
        ck = torch.load(path, map_location=self.world.device, weights_only=False)
        self.policy.load_state_dict(ck["actor_state_dict"])
        self.value.load_state_dict(ck["critic_state_dict"])
        info = ck.get("infos", {})
        self.iteration = int(ck.get("iter", 0))
        self.executed = int(info.get("executed_steps", 0))
        self.planner.model_steps = int(info.get("model_steps", 0))
        for opt, key in ((self.opt_p, "opt_p"), (self.opt_v, "opt_v")):
            if key in info:
                opt.load_state_dict(info[key])
        self.log(f"[mb] continuing at iteration {self.iteration} ({self.executed} executed steps)")

    def act(self, obs) -> torch.Tensor:
        """The bare policy (no planner): what the robot would run."""
        with torch.no_grad():
            return self.policy(obs["policy"].to(next(self.policy.parameters()).device)).to(obs["policy"].dtype)

    # ------------------------------------------------------------------ one iteration
    def collect(self) -> dict:
        world, cfg, R = self.world, self.cfg, self.world.num_envs
        dev = world.device
        O, C, A, Rw, D, TO, V = [], [], [], [], [], [], []
        stats = {k: [] for k in ("return_policy", "return_planner", "return_best", "weight_entropy")}
        env_log: dict = {}
        obs = world.get_observations()
        with torch.no_grad():
            for _ in range(cfg.steps):
                a_plan, info = self.planner.plan()
                bare = torch.rand(R, 1, device=dev) < cfg.policy_share
                a = torch.where(bare, info["policy_action"].to(a_plan.dtype), a_plan)
                O.append(obs["policy"].float())
                C.append(obs["critic"].float())
                A.append(a_plan.float())
                V.append(self.value(obs["critic"]).squeeze(-1))
                obs, r, done, extras = world.step(a)
                Rw.append(r.float())
                D.append(done.float())
                TO.append(extras.get("time_outs", torch.zeros_like(done)).float())
                for k in stats:
                    stats[k].append(info[k])
                for k, v in extras.get("log", {}).items():
                    v = torch.as_tensor(v).float().reshape(-1)
                    if v.numel():
                        env_log.setdefault(k, []).append(v.cpu())
            last = self.value(obs["critic"]).squeeze(-1)
            adv = torch.zeros_like(last)
            returns = []
            for t in reversed(range(cfg.steps)):
                nxt = last if t == cfg.steps - 1 else V[t + 1]
                # an episode that ended by its time limit is not a failure: its value goes on (as rsl-rl does)
                r = Rw[t] + cfg.gamma * V[t] * TO[t]
                nonterm = 1.0 - D[t]
                delta = r + cfg.gamma * nxt * nonterm - V[t]
                adv = delta + cfg.gamma * cfg.lam * nonterm * adv
                returns.insert(0, adv + V[t])
        self.executed += R * cfg.steps
        out = {"obs": torch.cat(O), "critic": torch.cat(C), "action": torch.cat(A), "ret": torch.cat(returns)}
        out["stats"] = {k: float(torch.cat(v).mean()) for k, v in stats.items()}
        out["stats"]["reward_step"] = float(torch.stack(Rw).mean())
        out["stats"]["done_rate"] = float(torch.stack(D).mean())
        out["env_log"] = {k: float(torch.cat(v).mean()) for k, v in env_log.items()}
        return out

    def update(self, batch: dict) -> dict:
        cfg = self.cfg
        self.policy.obs_normalizer.update(batch["obs"])
        self.value.obs_normalizer.update(batch["critic"])
        self.buffer.append((batch["obs"], batch["action"]))
        obs = torch.cat([b[0] for b in self.buffer])
        act = torch.cat([b[1] for b in self.buffer])
        loss_p = loss_v = torch.zeros(())
        for _ in range(cfg.epochs):
            perm = torch.randperm(obs.shape[0], device=obs.device)
            for mb in perm.split(max(1, obs.shape[0] // cfg.minibatches)):
                loss_p = (self.policy(obs[mb]) - act[mb]).square().mean()
                self.opt_p.zero_grad()
                loss_p.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.opt_p.step()
            n = batch["critic"].shape[0]
            perm = torch.randperm(n, device=obs.device)
            for mb in perm.split(max(1, n // cfg.minibatches)):
                loss_v = (self.value(batch["critic"][mb]).squeeze(-1) - batch["ret"][mb]).square().mean()
                self.opt_v.zero_grad()
                loss_v.backward()
                torch.nn.utils.clip_grad_norm_(self.value.parameters(), 1.0)
                self.opt_v.step()
        return {"policy_loss": float(loss_p), "value_loss": float(loss_v), "value_mean": float(batch["ret"].mean())}

    # ------------------------------------------------------------------ the run
    def train(self, iterations: int, minutes: float = 0.0, save_every_min: float = 10.0, eval_every_min: float = 0.0,
              video_every_min: float = 0.0, print_every: int = 1) -> None:
        t0 = time.time()
        last = {"save": t0, "eval": t0, "video": t0}
        old = signal.signal(signal.SIGINT, lambda *_: setattr(self, "stop", "SIGINT"))
        metrics = (self.run_dir / "metrics.jsonl").open("a")
        try:
            end = self.iteration + iterations
            while self.iteration < end and self.stop is None:
                t_it = time.time()
                batch = self.collect()
                learn = self.update(batch)
                self.iteration += 1
                now = time.time()
                row = {"iteration": self.iteration, "Cost/executed_steps": self.executed,
                       "Cost/model_steps": self.planner.model_steps, "Cost/seconds": now - t_it,
                       **{f"Plan/{k}": v for k, v in batch["stats"].items()},
                       **{f"Learn/{k}": v for k, v in learn.items()}, **batch["env_log"]}
                if self.evaluate is not None and eval_every_min > 0 and (now - last["eval"]) / 60 >= eval_every_min:
                    row.update(self.run_eval())
                    last["eval"] = time.time()
                if self.video is not None and video_every_min > 0 and (now - last["video"]) / 60 >= video_every_min:
                    self.run_video()
                    last["video"] = time.time()
                if save_every_min > 0 and (now - last["save"]) / 60 >= save_every_min:
                    self.save()
                    last["save"] = time.time()
                metrics.write(json.dumps(row) + "\n")
                metrics.flush()
                if self.wandb is not None:
                    self.wandb.log(row, step=self.iteration)
                if self.iteration % print_every == 0:
                    s = batch["stats"]
                    extra = "".join(f"  {k.split('/')[-1]} {v:.3g}" for k, v in row.items() if k.startswith("Eval/"))
                    self.log(f"[mb] it {self.iteration:4d}  executed {self.executed:8d}  planner {s['return_planner']:7.3f} "
                             f"policy {s['return_policy']:7.3f}  done/step {s['done_rate']:.4f}  "
                             f"loss p {learn['policy_loss']:.4f} v {learn['value_loss']:.4f}  ({row['Cost/seconds']:.1f} s){extra}")
                if minutes > 0 and (time.time() - t0) / 60 > minutes:
                    self.stop = f"time budget of {minutes:g} min used up"
        finally:
            signal.signal(signal.SIGINT, old)
            metrics.close()
        final = {}
        if self.evaluate is not None:
            final = self.run_eval()
            if self.wandb is not None:
                self.wandb.log(final, step=self.iteration)
        if self.video is not None and video_every_min > 0:
            self.run_video()
        path = self.save()
        self.log(f"[mb] {'stopped: ' + self.stop if self.stop else 'done'} after iteration {self.iteration} "
                 f"({(time.time() - t0) / 60:.1f} min, {self.executed} executed and {self.planner.model_steps} model "
                 f"steps); {path.name}" + "".join(f"  {k.split('/')[-1]} {v:.3g}" for k, v in final.items()))

    def run_eval(self) -> dict:
        try:
            return {k: float(v) for k, v in self.evaluate(self.act, self.iteration).items()}
        except Exception as exc:  # noqa: BLE001  (an evaluation must never end a training)
            self.log(f"[eval] failed: {type(exc).__name__}: {exc}")
            return {}

    def run_video(self) -> None:
        out = self.video(self.act, self.iteration)
        images = {}
        if isinstance(out, tuple):
            out, images = out
        if out:
            self.log(f"[video] {Path(out).name}")
            if self.wandb is not None:
                try:
                    import wandb
                    self.wandb.log({"video": wandb.Video(str(out), format="mp4"),
                                    **{k: wandb.Image(str(v)) for k, v in images.items()}}, step=self.iteration)
                except Exception as exc:  # noqa: BLE001
                    self.log(f"[video] not uploaded: {type(exc).__name__}: {exc}")

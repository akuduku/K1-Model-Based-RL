r"""Model-based kick planner: search the kick IN the pH model x-dot with the cross-entropy method (CEM).

Lecture 11, approach 1b (sampling-based planning), offline, as mbrl/getup/plan.py: the model (Full Body's batched pH
plant with its ball, behind the robot's command path) rolls out every candidate; physics is a hard constraint; the
cost is the task.

A candidate is a plan of mbrl/kick/controller.py (D = 54 numbers: where the ball lies relative to the trunk at the
kick, 4 keyframes of offsets on the 12 leg joints, their durations), run THROUGH the walk policy, so the robot keeps
its balance feedback during the whole rollout.  One search per requested ball speed (--speeds), side by side in one
batch; every candidate is judged on V variants (ball positions over the task's range, kick directions, delays 0-70
ms, driver gain errors; with --fit half of them on fitted parameters): cost = mean + risk x std over the variants.

    cost =  miss x (1 + closest the kicking foot came to the ball)        if the ball was not touched
            miss + early                                                   if it was bumped before the kick started
            direction x (direction error / 0.3 rad)^2                      else  (capped)
          + speed x (relative speed error)^2  + loft x vertical ball speed
          + fall                                                           if the robot fell
          + stand x (share of the last tail_s seconds NOT standing in the stance)
          + time x seconds until the touch  + energy x kJ drawn from the battery

"Standing" must hold for the whole last tail_s = 2 s of the rollout (the get-up's plan was only checked for 0.5 s
and tipped over right after).  From scratch: every search starts with zero offsets (the robot walks up to the ball
and stands there) and a wide distribution.  --init-demo POLICY starts them around the leg motion of a kick policy
instead (mbrl/kick/demo.py: what that policy does beyond the walk policy, as keyframes).

    python -m mbrl.kick.plan --smoke                                    # laptop CPU, about a minute
    python -m mbrl.kick.plan --walk ../k1_kick/checkpoints/walk_teacher.pt --minutes 60 --run-name kick-plan-v1

Output in --out (default plans/): kick-<tag>.pt (the library: speeds, z, decoded plans, success per variant, history)
and kick-<tag>.json (the same without the arrays), plus a video per speed with --video.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import torch  # noqa: E402

from .. import paths  # noqa: E402
from .controller import KickController, PlanSpace  # noqa: E402
from rl.kick import contract as C  # noqa: E402
from rl.kick.env import KickConfig, KickEnv  # noqa: E402

PLANS = paths.REPO / "plans"
# the walk the robot runs (k1_deploy's sprint.pt = k1_kick/checkpoints/walk_teacher.pt), TorchScript 75 -> 22
_WALKS = (paths.REPO / "policies" / "walk_teacher.pt", paths.REPO.parent / "k1_kick" / "checkpoints" / "walk_teacher.pt")
DEFAULT_WALK = next((w for w in _WALKS if w.exists()), _WALKS[0])


@dataclass
class PlanConfig:
    speeds: tuple = (2.0, 3.5, 5.0, 6.0)   # requested ball speeds, one search each (from 5: as hard as possible)
    candidates: int = 64                   # per search and iteration (candidate 0 = the search's mean)
    variants: int = 8
    horizon_s: float = 5.0                 # approach (<= 2.4 s) + kick (<= 1.5 s) + standing
    tail_s: float = 2.0                    # the robot must stand in its stance for all of the last tail_s
    iterations: int = 40
    minutes: float = 0.0                   # time budget of the search (0: iterations only)
    elite_frac: float = 0.125
    alpha: float = 0.7
    init_offset_std: float = 1.2           # logit units of the offsets (tanh x 5 action units)
    init_other_std: float = 1.0
    min_std: float = 0.05
    risk: float = 0.5
    # cost weights
    miss: float = 10.0
    early: float = 5.0                     # the ball moved before the kick was triggered (the approach bumped it)
    direction: float = 4.0
    direction_scale: float = 0.3           # rad
    direction_cap: float = 6.0
    speed: float = 4.0
    loft: float = 0.3
    fall: float = 20.0
    stand: float = 10.0
    time: float = 0.5
    energy: float = 1.0                    # per kJ
    # what counts as a good kick (the gate, per variant)
    ok_direction_deg: float = 15.0
    ok_speed_error: float = 0.35
    ok_standing: float = 0.9
    # the variants
    delay_ms: tuple = (0.0, 70.0)
    gain_dr: float = 0.10
    init_plan: str = ""                    # start the searches around this library (a previous run)
    init_demo: str = ""                    # ... or around the leg motion of this kick policy (mbrl/kick/demo.py)
    init_std_scale: float = 0.5
    walk: str = str(DEFAULT_WALK)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float32"
    seed: int = 1


class PlanEnv(KickEnv):
    """KickEnv whose requests are given (one scenario per robot) instead of drawn."""

    scenario: dict | None = None

    def _draw_request(self, ids):
        s = self.scenario
        if s is None:  # the constructor's own first requests, before the scenarios are set
            return super()._draw_request(ids)
        return s["local"][ids].clone(), s["angle"][ids].clone(), s["speed"][ids].clone()


def plan_env_config(cfg: PlanConfig, B: int) -> KickConfig:
    """Every robot gets its request at once, from the stance; nothing random but what the variants fix."""
    return KickConfig(num_envs=B, device=cfg.device, dtype=cfg.dtype, seed=cfg.seed, stand_prob=1.0,
                      request_s=cfg.horizon_s + 1.0, joint_noise=0.0, yaw_noise=0.0, gain_dr=0.0, push_dv=0.0,
                      obs_noise={}, delay_ms=tuple(cfg.delay_ms), delay_hold_prob=None, energy_log_envs=0)


class Planner:
    def __init__(self, cfg: PlanConfig, fit=None, log=None, space: PlanSpace | None = None):
        self.cfg = cfg
        self.log = log or (lambda *a: print(*a, flush=True))
        self.space = space or PlanSpace()
        self.speeds = [float(s) for s in cfg.speeds]
        R, N, V = len(self.speeds), cfg.candidates, cfg.variants
        self.R, self.B = R, R * N * V
        B = self.B
        self.env = env = PlanEnv(plan_env_config(cfg, B))
        self.dev, self.dt = env.device, env.dtype
        self.walk = torch.jit.load(str(cfg.walk), map_location=self.dev).eval()
        self.ctrl = KickController(env, self.walk, self.space)
        self.steps = int(round(cfg.horizon_s / env.cfg.policy_dt))
        self.tail = max(1, int(round(cfg.tail_s / env.cfg.policy_dt)))

        # the V variants, fixed for the whole search (common random numbers): slot b = (r * N + n) * V + v
        g = torch.Generator().manual_seed(cfg.seed)
        c = env.cfg
        perm = lambda: torch.randperm(V, generator=g)  # noqa: E731
        lin = lambda lo, hi: (torch.linspace(lo, hi, V) if V > 1 else torch.tensor([0.5 * (lo + hi)]))  # noqa: E731
        self.var_local = torch.stack((lin(*c.ball_x)[perm()], lin(*c.ball_y)[perm()]), -1)       # (V, 2)
        self.var_angle = lin(*c.kick_angle)[perm()]
        self.var_delay = lin(*cfg.delay_ms)[perm()].tolist()
        u = lambda: (1 - cfg.gain_dr) + 2 * cfg.gain_dr * torch.rand(V, 22, generator=g)  # noqa: E731
        self.gain_err = (u().to(self.dev, self.dt), u().to(self.dev, self.dt))
        self.variant = torch.arange(B, device=self.dev) % V
        self.search_of = torch.arange(B, device=self.dev) // (N * V)
        speed = torch.tensor(self.speeds, dtype=self.dt, device=self.dev)[self.search_of]
        env.scenario = {"local": self.var_local.to(self.dev, self.dt)[self.variant],
                        "angle": self.var_angle.to(self.dev, self.dt)[self.variant], "speed": speed}
        self.fit = fit
        self.fitted = torch.zeros(V, dtype=torch.bool)
        if fit is not None:  # odd variants: a fitted parameter set (mbrl.fit), as the get-up planner
            from mbrl.model import attach
            from ph_k1.robot_params import neutral_rows
            attach(env.plant, fit, None, "elites", seed=cfg.seed, fit_delay=False)
            self.fitted = torch.arange(V) % 2 == 1
            elites = fit.elites
            theta = torch.stack([elites[(v // 2) % len(elites)] for v in range(V)])
            rows = neutral_rows(V, self.dt, self.dev)
            rows[self.fitted.to(self.dev)] = fit.space.robot_params(theta[self.fitted]).to(self.dev, self.dt)
            self.var_rows = rows

    # ------------------------------------------------------------------ rollouts
    def _reset(self) -> None:
        env, cfg = self.env, self.cfg
        ids = torch.arange(self.B, device=self.dev)
        env._reset_idx(ids)
        for v in range(cfg.variants):
            rows = (self.variant == v).nonzero(as_tuple=False).squeeze(-1)
            env.plant.set_delay(self.var_delay[v], self.var_delay[v], env_ids=rows)
        env.kp = env.base_kp * self.gain_err[0][self.variant]
        env.kd = env.base_kd * self.gain_err[1][self.variant]
        if self.fit is not None:
            env.plant.robot_params = self.var_rows[self.variant].contiguous()
        env._refresh()
        env._start_requests()
        self.ctrl.reset()

    def rollout(self, z: torch.Tensor, video=None, record: int | None = None) -> dict:
        """z (B, D): one plan per robot -> per-robot results (cost and what it is made of)."""
        env, cfg = self.env, self.cfg
        f = lambda x: x.to(self.dt)  # noqa: E731
        self._reset()
        self.ctrl.set_plans(self.space.decode(z.to(self.dev, self.dt)))
        B, dt = self.B, env.cfg.policy_dt
        zb = lambda: torch.zeros(B, dtype=torch.bool, device=self.dev)  # noqa: E731
        alive = ~zb()
        fell, kicked = zb(), zb()
        launch = torch.zeros(B, 3, dtype=self.dt, device=self.dev)
        touch_time = torch.zeros(B, dtype=self.dt, device=self.dev)
        t_kick = torch.full((B,), -1.0, dtype=self.dt, device=self.dev)
        closest = torch.full((B,), 10.0, dtype=self.dt, device=self.dev)
        tail = torch.zeros(B, dtype=self.dt, device=self.dev)
        supplied0 = env.plant.ledger.supplied.clone()
        energy = torch.zeros(B, dtype=self.dt, device=self.dev)
        rec = {k: [] for k in ("theta", "target", "phase", "ball")} if record is not None else None
        with torch.no_grad():
            obs = env.get_observations()["policy"]
            for s in range(self.steps):
                a = self.ctrl.act(obs)
                o, _, dones, _ = env.step(a)
                obs = o["policy"]
                dones = dones.bool()
                fell |= alive & dones            # the request does not time out inside the rollout: an end is a fall
                # the last finished episode's shot is in env.last_shot; while alive the env's own state is this episode's
                foot = env.foot_pos[env.index, env.foot_request]
                dist = torch.linalg.vector_norm(env.ball_pos[:, :2] - foot[:, :2], dim=1)
                closest = torch.where(alive & ~dones & ~env.kicked, torch.minimum(closest, dist), closest)
                live = alive & ~dones
                kicked = torch.where(live, env.kicked, kicked)
                launch = torch.where((live & env.kicked)[:, None], env.launch_vel, launch)
                touch_time = torch.where(live & env.kicked, env.touch_time, touch_time)
                t_kick = torch.where(live, self.ctrl.t_kick, t_kick)
                energy = torch.where(live, env.plant.ledger.supplied - supplied0, energy)
                alive = live
                if s >= self.steps - self.tail:
                    tail += f(alive & self.standing())
                if rec is not None:
                    rec["theta"].append(env.theta[record].cpu().clone())
                    rec["target"].append(env.held_target[record].cpu().clone())
                    rec["phase"].append(int(self.ctrl.phase[record]))
                    rec["ball"].append(env.ball_pos[record].cpu().clone())
                if video is not None:
                    video.capture()
        target, want = env.target_w, env.target_speed
        along = (launch * target).sum(-1)
        speed = torch.linalg.vector_norm(launch[:, :2], dim=1)
        derr = C.direction_error(launch, target)
        full = want >= env.cfg.full_power
        serr = torch.where(full, (want - speed).clamp(min=0.0), (speed - want).abs()) / want
        stand = tail / self.tail
        early = kicked & ((t_kick < 0) | (touch_time < t_kick))    # touched before the kick was triggered: a bump
        proper = kicked & ~early
        k = f(proper)
        cost = (cfg.miss * (1 - f(kicked)) * (1 + closest) + f(early) * (cfg.miss + cfg.early)
                + k * (cfg.direction * (derr / cfg.direction_scale).square()).clamp(max=cfg.direction_cap)
                + k * cfg.speed * serr.square() + k * cfg.loft * launch[:, 2].abs()
                + cfg.fall * f(fell) + cfg.stand * (1 - stand) + cfg.time * touch_time * k
                + cfg.energy * energy.clamp(min=0.0) / 1000.0)
        ok = (proper & ~fell & (torch.rad2deg(derr) < cfg.ok_direction_deg) & (serr < cfg.ok_speed_error)
              & (stand >= cfg.ok_standing))
        out = {"cost": cost, "ok": ok, "kicked": kicked, "early": early, "fell": fell, "stand": stand, "along": along, "speed": speed,
               "direction_deg": torch.rad2deg(derr), "speed_error": serr, "loft": launch[:, 2].abs(),
               "touch_time": touch_time, "energy_J": energy, "closest": closest}
        out = {k_: v.cpu() for k_, v in out.items()}
        if rec is not None:
            out["record"] = {k_: (torch.stack(v) if isinstance(v[0], torch.Tensor) else torch.tensor(v))
                             for k_, v in rec.items()}
        return out

    def standing(self) -> torch.Tensor:
        """Upright, at standing height, both feet about where they stand, nearly at rest."""
        env = self.env
        tilt = torch.acos((-env.gravity[:, 2]).clamp(-1.0, 1.0))
        rel = C.to_heading(env.yaw, env.foot_pos - env.base_pos[:, None])[..., :2]
        offset = torch.linalg.vector_norm(rel - env.stance, dim=-1).sum(-1)
        return ((tilt < 0.25) & (env.base_pos[:, 2] > 0.9 * env.stand_height) & (offset < 0.25)
                & (env.qd.abs().amax(-1) < 3.0))

    # ------------------------------------------------------------------ CEM
    video_every_min = 0.0
    video_dir: Path | None = None

    def _recorder(self, slot: int):
        try:
            from ph_k1.video import VideoRecorder
            return VideoRecorder(self.env.plant, b=slot, fps=1.0 / self.env.cfg.policy_dt)
        except Exception as exc:  # noqa: BLE001
            self.log(f"[plan] no video: {type(exc).__name__}: {exc}")
            return None

    def _save_video(self, rec, path: Path, wandb_run=None, key: str = "video"):
        if rec is None:
            return None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            out = rec.save(path)
        except Exception as exc:  # noqa: BLE001  (no GL backend / MP4 writer: plan on without videos)
            self.log(f"[plan] video failed: {type(exc).__name__}: {exc}")
            self.video_every_min = 0.0
            return None
        if wandb_run is not None and out:
            try:
                import wandb
                wandb_run.log({key: wandb.Video(str(out), format="mp4")})
            except Exception as exc:  # noqa: BLE001
                self.log(f"[plan] video upload failed: {type(exc).__name__}: {exc}")
        self.log(f"[plan] video {Path(out).name}")
        return out

    def search(self, state: dict | None = None, wandb_run=None) -> dict:
        cfg = self.cfg
        R, N, V, D = self.R, cfg.candidates, cfg.variants, self.space.D
        n_el = max(2, int(round(cfg.elite_frac * N)))
        g = torch.Generator().manual_seed(cfg.seed + 7)
        if state is None:
            std = self.space.init_std(cfg.init_offset_std, cfg.init_other_std).double().expand(R, D).clone()
            mean = self.space.init_mean().double().expand(R, D).clone()
            if cfg.init_demo:
                from .demo import from_policy
                mean = from_policy(cfg.init_demo, self.walk, self.speeds, self.space, log=self.log).double()
                std = std * cfg.init_std_scale
            if cfg.init_plan:
                prev = torch.load(cfg.init_plan, map_location="cpu", weights_only=False)
                zp, sp = torch.as_tensor(prev["z"], dtype=torch.float64), list(prev["speeds"])
                if zp.shape[1] != D:
                    raise ValueError(f"{cfg.init_plan}: {zp.shape[1]} parameters, this planner has {D}")
                for r, s in enumerate(self.speeds):  # the plan of the nearest speed
                    mean[r] = zp[min(range(len(sp)), key=lambda i: abs(sp[i] - s))]
                std = std * cfg.init_std_scale
                self.log(f"[plan] searches start around {cfg.init_plan}")
            state = {"mean": mean, "std": std, "it": 0, "history": []}
        mean, std = state["mean"], state["std"]
        t_start, last_video = time.time(), time.time()
        while state["it"] < cfg.iterations:
            used = (time.time() - t_start) / 60
            if cfg.minutes > 0 and used > cfg.minutes:
                self.log(f"[plan] time budget of {cfg.minutes:g} min used up")
                break
            t0 = time.time()
            eps = torch.randn(R, N, D, generator=g, dtype=torch.float64)
            eps[:, 0] = 0.0                                                          # candidate 0: the mean plan
            z = mean[:, None] + std[:, None] * eps                                   # (R, N, D)
            film = (self.video_every_min > 0 and self.video_dir is not None
                    and ((time.time() - last_video) / 60 >= self.video_every_min or state["it"] == 0))
            rec = self._recorder((state["it"] % R) * N * V) if film else None        # a search's mean plan, variant 0
            out = self.rollout(z[:, :, None].expand(R, N, V, D).reshape(-1, D).float(), video=rec)
            if film:
                r_f = state["it"] % R
                self._save_video(rec, self.video_dir / f"kick_{self.speeds[r_f]:g}_it{state['it']:04d}.mp4", wandb_run,
                                 f"Plan/video_{self.speeds[r_f]:g}")
                last_video = time.time()
            sh = lambda k: out[k].double().reshape(R, N, V)  # noqa: E731
            c = sh("cost")
            cost = c.mean(-1) + cfg.risk * c.std(-1, unbiased=False)                 # (R, N)
            order = cost.argsort(-1)
            el = order[:, :n_el]
            elite = torch.gather(z, 1, el[..., None].expand(-1, -1, D))
            mean = cfg.alpha * elite.mean(1) + (1 - cfg.alpha) * mean
            std = torch.maximum(cfg.alpha * elite.std(1) + (1 - cfg.alpha) * std, torch.full_like(std, cfg.min_std))
            ok = sh("ok").mean(-1)                                                    # (R, N) share of good variants
            el_ok = torch.gather(ok, 1, el).mean(-1)
            row = {"it": state["it"], "cost_mean": cost[:, 0].tolist(),
                   "cost_best": cost.gather(1, order[:, :1]).squeeze(1).tolist(),
                   "elite_ok": el_ok.tolist(), "mean_ok": ok[:, 0].tolist(),
                   "mean_kicked": sh("kicked")[:, 0].mean(-1).tolist(), "mean_stand": sh("stand")[:, 0].mean(-1).tolist(),
                   "mean_speed": sh("speed")[:, 0].mean(-1).tolist(),
                   "mean_direction_deg": sh("direction_deg")[:, 0].mean(-1).tolist(),
                   "std_mean": std.mean(-1).tolist(), "seconds": round(time.time() - t0, 1)}
            state["history"].append(row)
            self.log(f"[plan] it {state['it']:3d}  " + "  ".join(
                f"{s:g} m/s: ok {row['mean_ok'][r]:.2f} (elites {row['elite_ok'][r]:.2f}) kicked "
                f"{row['mean_kicked'][r]:.2f} {row['mean_speed'][r]:.1f} m/s cost {row['cost_mean'][r]:.1f}"
                for r, s in enumerate(self.speeds)) + f"  ({row['seconds']} s)")
            if wandb_run is not None:
                data = {"Plan/iteration": state["it"], "Plan/seconds": row["seconds"]}
                for r, s in enumerate(self.speeds):
                    tag = f"{s:g}"
                    data.update({f"Plan/ok_{tag}": row["mean_ok"][r], f"Plan/elite_ok_{tag}": row["elite_ok"][r],
                                 f"Plan/cost_{tag}": row["cost_mean"][r], f"Plan/kicked_{tag}": row["mean_kicked"][r],
                                 f"Plan/speed_{tag}": row["mean_speed"][r],
                                 f"Plan/direction_deg_{tag}": row["mean_direction_deg"][r],
                                 f"Plan/standing_{tag}": row["mean_stand"][r], f"Plan/std_{tag}": row["std_mean"][r]})
                wandb_run.log(data)
            state.update(mean=mean, std=std, it=state["it"] + 1)
        return state

    def final(self, state: dict, video_dir: Path | None = None) -> dict:
        """Every search's mean plan on all variants; its reference motion; a video per speed."""
        cfg = self.cfg
        R, N, V, D = self.R, cfg.candidates, cfg.variants, self.space.D
        z = state["mean"][:, None, None].expand(R, N, V, D).reshape(-1, D).float()
        result = {"speeds": self.speeds, "z": state["mean"].clone(), "per_speed": []}
        for r, s in enumerate(self.speeds):
            slot = r * N * V                                           # this search's mean plan on variant 0
            rec = self._recorder(slot) if video_dir is not None else None
            out = self.rollout(z, video=rec, record=slot)
            if rec is not None:
                self._save_video(rec, video_dir / f"kick_{s:g}_final.mp4")
            rows = slice(slot, slot + V)                               # candidate 0 of search r: the mean plan
            item = {k: out[k][rows].float().tolist() for k in ("ok", "kicked", "early", "fell", "stand", "along", "speed",
                                                               "direction_deg", "speed_error", "loft", "touch_time",
                                                               "energy_J")}
            item.update(speed_request=s, success=float(out["ok"][rows].float().mean()),
                        record={k: v for k, v in out["record"].items()})
            result["per_speed"].append(item)
            self.log(f"[plan] final {s:g} m/s: good kick in {int(out['ok'][rows].sum())}/{V} variants; kicked "
                     f"{int(out['kicked'][rows].sum())}/{V} (bumped early {int(out['early'][rows].sum())}), fell {int(out['fell'][rows].sum())}/{V}, ball "
                     f"{float(out['speed'][rows].mean()):.2f} m/s, direction {float(out['direction_deg'][rows].mean()):.1f} deg, "
                     f"standing {float(out['stand'][rows].mean()):.2f}, {float(out['energy_J'][rows].mean()):.0f} J")
        result["decoded"] = {k: v.cpu() for k, v in self.space.decode(state["mean"].float()).items()}
        result["success"] = [p["success"] for p in result["per_speed"]]
        return result


def save(result: dict, state: dict, cfg: PlanConfig, space: PlanSpace, out_dir: Path, tag: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"kick-{tag}.pt"
    torch.save({"speeds": result["speeds"], "z": result["z"], "decoded": result["decoded"],
                "success": result["success"], "per_speed": result["per_speed"], "history": state["history"],
                "config": asdict(cfg), "space": asdict(space)}, path)
    light = {"speeds": result["speeds"], "success": result["success"], "config": asdict(cfg), "space": asdict(space),
             "per_speed": [{k: v for k, v in p.items() if k != "record"} for p in result["per_speed"]],
             "history": state["history"]}
    path.with_suffix(".json").write_text(json.dumps(light, indent=1))
    return path


def load_library(path, device="cpu", dtype=torch.float32):
    """-> (PlanSpace, speeds (S,), decoded plans (each (S, ...)), success per speed)."""
    p = torch.load(path, map_location="cpu", weights_only=False)
    space = PlanSpace(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in p["space"].items()})
    speeds = torch.as_tensor(p["speeds"], dtype=dtype, device=device)
    decoded = {k: v.to(device, dtype) for k, v in space.decode(torch.as_tensor(p["z"], dtype=torch.float32)).items()}
    return space, speeds, decoded, p.get("success")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speeds", default="2,3.5,5,6")
    ap.add_argument("--candidates", type=int, default=64)
    ap.add_argument("--variants", type=int, default=8)
    ap.add_argument("--iterations", type=int, default=40)
    ap.add_argument("--minutes", type=float, default=0.0)
    ap.add_argument("--horizon", type=float, default=5.0)
    ap.add_argument("--keyframes", type=int, default=4, help="keyframes of the kick motion (D = 2 + 13 per keyframe)")
    ap.add_argument("--d-min", type=float, default=0.06, help="shortest segment between keyframes [s]")
    ap.add_argument("--walk", default=str(DEFAULT_WALK), help="TorchScript walk policy (75 -> 22)")
    ap.add_argument("--fit", default=None, help="fit JSON of mbrl.fit: odd variants run on fitted parameters")
    ap.add_argument("--init-plan", default="")
    ap.add_argument("--init-demo", default="", help="kick policy file whose leg motion starts the searches")
    ap.add_argument("--init-std-scale", type=float, default=0.5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=str(PLANS))
    ap.add_argument("--run-name", default=time.strftime("kick-plan-%Y%m%d-%H%M"))
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--video", action="store_true", help="a video of every speed's final plan")
    ap.add_argument("--video-every-min", type=float, default=0.0)
    ap.add_argument("--wandb-project", default="")
    ap.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", "karl-jahnel-universit-t-leipzig"))
    ap.add_argument("--smoke", action="store_true", help="a tiny search: checks the code path, not the kick")
    args = ap.parse_args()

    cfg = PlanConfig(speeds=tuple(float(s) for s in args.speeds.split(",")), candidates=args.candidates,
                     variants=args.variants, iterations=args.iterations, minutes=args.minutes, horizon_s=args.horizon,
                     walk=args.walk, init_plan=args.init_plan, init_demo=args.init_demo,
                     init_std_scale=args.init_std_scale, device=args.device, seed=args.seed)
    if args.smoke:
        cfg = replace(cfg, speeds=(3.5,), candidates=8, variants=2, iterations=2, horizon_s=3.0, tail_s=1.0)
    fit = None
    if args.fit:
        from mbrl.model import FittedModel
        fit = FittedModel.load(args.fit)
    run = None
    if args.wandb_project:
        import wandb
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.run_name, config=asdict(cfg))
    planner = Planner(cfg, fit, space=PlanSpace(keyframes=args.keyframes, d_min=args.d_min))
    out_dir = Path(args.out)
    planner.video_every_min, planner.video_dir = args.video_every_min, out_dir / "videos" / args.run_name
    print(f"[plan] {planner.B} robots on {cfg.device}: {len(cfg.speeds)} searches x {cfg.candidates} candidates x "
          f"{cfg.variants} variants, {planner.space.D} parameters, horizon {cfg.horizon_s} s", flush=True)
    state = planner.search(wandb_run=run)
    result = planner.final(state, planner.video_dir if (args.video or args.video_every_min > 0) else None)
    path = save(result, state, cfg, planner.space, out_dir, args.tag)
    gate = all(s >= 0.75 for s in result["success"])
    print(f"[plan] wrote {path}; good kicks per speed {['%.2f' % s for s in result['success']]}; "
          f"gate (>= 0.75 of the variants for every speed): {'passed' if gate else 'NOT passed'}", flush=True)
    if run is not None:
        for s, ok in zip(result["speeds"], result["success"]):
            run.summary[f"Plan/final_ok_{s:g}"] = ok
        run.finish()


if __name__ == "__main__":
    main()

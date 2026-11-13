r"""Model-based get-up planner: search a get-up motion IN the pH model x-dot with the cross-entropy method (CEM).

Lecture 11, approach 1b (sampling-based planning), offline: the model (Full Body's batched pH plant behind the
robot's command path) rolls out every candidate motion; physics is a hard constraint (the professor's formula with
alpha -> infinity); the cost is the task:

    cost = - sum_t r_t  (the get-up env's own reward: height, upright, standing, head/neck, limits, torque, ...;
                         battery and impacts x the energy weight)
           - final_bonus x (share of the last 0.5 s spent standing)          <- term 2: finish standing
           + explode_cost if the plant blew up

A candidate is a keyframe motion (mbrl/getup/keyframes.py: 5 mirrored keyframes + 6 durations = 56 numbers), sent
as joint targets every 20 ms with the StandupAgent's fixed gains (no tank, no variable gains).  Every candidate is
judged on V variants (different settled lying poses, delays 10-40 ms, driver gain errors, and with --fit half of them
on the fitted parameters of mbrl.fit): cost = mean + risk x std over the variants, a plan that works open-loop on all
of them.  R independent CEM searches (restarts) run side by side in one batch: B = R x N x V robots.

From scratch: every search starts with all keyframes at HOME and a wide distribution.  Exploration help by
continuation: an upward assist force (as the get-up training's) starts at --assist N and drops by --assist-step each
time the best search's elites stand in >= --gate of the variants; once it is 0 the energy weight rises the same way.

    python -m mbrl.getup.plan --smoke                                   # laptop CPU, a few seconds
    python -m mbrl.getup.plan --kinds supine,prone --fit fits/getup-both-v1.json --minutes 70 --run-name getup-plan-v1

Output in --out (default plans/): getup-<kind>-<tag>.pt (keyframes, durations, the reference motion of the best plan
at 50 Hz: joint angles, centre-of-mass height, uprightness, standing, targets; success per variant; history) and
getup-<kind>-<tag>.json (the same without the arrays), plus a video of the best plan if --video.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import torch  # noqa: E402

from .. import paths  # noqa: E402
from .keyframes import Keyframes  # noqa: E402
from ph_k1.dynamics import body_frames, com_height  # noqa: E402
from rl.getup import contract as C  # noqa: E402
from rl.getup.env import GetUpConfig, GetUpEnv  # noqa: E402
from rl.getup.lying import KINDS  # noqa: E402

PLANS = paths.REPO / "plans"


@dataclass
class PlanConfig:
    kind: str = "supine"
    restarts: int = 4             # independent CEM searches in one batch
    candidates: int = 128         # per search and iteration (candidate 0 = the search's mean)
    variants: int = 8             # start poses / delays / gain errors / parameter sets every candidate is judged on
    keyframes: int = 5
    d_min: float = 0.15           # s, per segment
    d_max: float = 0.7
    horizon_s: float = 5.0        # the rollout; the motion reaches HOME after at most (K + 1) d_max = 4.2 s
    iterations: int = 40
    minutes: float = 0.0          # time budget of the search (0: iterations only)
    elite_frac: float = 0.125
    alpha: float = 0.7            # smoothing of the CEM update
    init_joint_std: float = 1.5   # logit units
    init_dur_std: float = 1.0
    min_std: float = 0.05
    risk: float = 0.5             # cost = mean + risk x std over the variants
    final_bonus: float = 10.0
    tail_s: float = 0.5           # "finish standing" = the share of the last tail_s seconds spent standing
    explode_cost: float = 100.0
    assist: float = 100.0         # N, upward force on the trunk while not standing (continuation)
    assist_step: float = 25.0
    gate: float = 0.75            # elites standing in this share of the variants -> less assist / more energy weight
    energy_step: float = 0.1
    delay_ms: tuple = (10.0, 40.0)
    gain_dr: float = 0.10
    self_collision: bool = True
    library_size: int = 64
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float32"
    seed: int = 1


class Planner:
    def __init__(self, cfg: PlanConfig, fit=None, log=print):
        self.cfg, self.log = cfg, log
        R, N, V = cfg.restarts, cfg.candidates, cfg.variants
        self.B = B = R * N * V
        self.kind = KINDS[cfg.kind]
        self.env = env = GetUpEnv(GetUpConfig(
            num_envs=B, device=cfg.device, dtype=cfg.dtype, seed=cfg.seed, episode_s=cfg.horizon_s + 1.0,
            delay_ms=tuple(cfg.delay_ms), delay_hold_prob=None, gain_dr=cfg.gain_dr, obs_noise={},
            self_collision=cfg.self_collision, tank=False, lying_kinds=(cfg.kind,), start=cfg.kind,
            library_size=cfg.library_size, assist_force=cfg.assist, push_dv=0.0,
            gain_free_ema=1e9))  # the gains never unlock: the StandupAgent's kKp / kKd
        env._update_curriculum = lambda ids: None  # the planner sets assist / energy / head weights itself
        env.gain_scale, env.head_scale, env.energy_scale = 0.0, 1.0, 0.0
        self.dev, self.dt = env.device, env.dtype
        m = env.model
        self.kf = Keyframes(m.jnt_low, m.jnt_high, env.contract.home, cfg.keyframes, cfg.d_min, cfg.d_max)
        self.scale = env.contract.action_scale
        self.steps = int(round(cfg.horizon_s / env.cfg.policy_dt))
        self.tail = max(1, int(round(cfg.tail_s / env.cfg.policy_dt)))

        # the V variants: fixed for the whole search (the same for every candidate)
        g = torch.Generator().manual_seed(cfg.seed)
        lib_idx = env.lib_index[self.kind].cpu()
        self.pose = lib_idx[torch.randperm(len(lib_idx), generator=g)[:V] % len(lib_idx)] if len(lib_idx) >= V \
            else lib_idx[torch.randint(len(lib_idx), (V,), generator=g)]
        lo, hi = cfg.delay_ms
        self.delay = torch.linspace(lo, hi, V).tolist() if V > 1 else [0.5 * (lo + hi)]
        u = lambda: (1 - cfg.gain_dr) + 2 * cfg.gain_dr * torch.rand(V, 22, generator=g)  # noqa: E731
        self.gain_err = (u().to(self.dev, self.dt), u().to(self.dev, self.dt))
        self.variant = torch.arange(B, device=self.dev) % V           # slot b = (r * N + n) * V + v
        self.fit = fit
        self.fitted = torch.zeros(V, dtype=torch.bool)
        if fit is not None:
            from mbrl.model import attach
            from ph_k1.robot_params import neutral_rows
            attach(env.plant, fit, None, "elites", seed=cfg.seed, fit_delay=False)
            self.fitted = torch.arange(V) % 2 == 1                   # odd variants: a fitted parameter set
            elites = fit.elites
            theta = torch.stack([elites[(v // 2) % len(elites)] for v in range(V)])
            rows = neutral_rows(V, self.dt, self.dev)
            rows[self.fitted.to(self.dev)] = fit.space.robot_params(theta[self.fitted]).to(self.dev, self.dt)
            kps, kds = fit.space.gain_scales(theta)
            one = torch.ones(V, 22, dtype=self.dt, device=self.dev)
            f = self.fitted.to(self.dev)[:, None]
            self.var_rows = rows
            self.var_kp = torch.where(f, kps.to(self.dev, self.dt), one)
            self.var_kd = torch.where(f, kds.to(self.dev, self.dt), one)
        self.assist, self.energy = float(cfg.assist), 0.0

    # ------------------------------------------------------------------ rollouts
    def _reset(self) -> torch.Tensor:
        """Every robot to its variant's lying pose (heading 0), delay, gain error and parameters; returns theta0."""
        env, cfg, m = self.env, self.cfg, self.env.model
        ids = torch.arange(self.B, device=self.dev)
        env._reset_idx(ids)
        lib = env.library.q
        for v in range(cfg.variants):
            rows = (self.variant == v).nonzero(as_tuple=False).squeeze(-1)
            ql = lib[int(self.pose[v])].expand(len(rows), -1)
            env.plant.reset(pose=ql[:, 7:], base_pos=(0.0, 0.0, ql[:, 2]), base_quat=ql[:, 3:7], env_ids=rows)
            env.plant.set_delay(self.delay[v], self.delay[v], env_ids=rows)
        env.gain_err_kp[:] = self.gain_err[0][self.variant]
        env.gain_err_kd[:] = self.gain_err[1][self.variant]
        if self.fit is not None:
            st = env.plant.mbrl
            env.plant.robot_params = self.var_rows[self.variant].contiguous()
            st.kp_scale[:] = self.var_kp[self.variant]
            st.kd_scale[:] = self.var_kd[self.variant]
        env.com_start[:] = com_height(m, body_frames(m, env.plant.q)[2])
        env.cfg.assist_force = self.assist
        env.assist_scale = {self.kind: 1.0}
        env.energy_scale = self.energy
        return env.plant.theta.clone()

    def rollout(self, z: torch.Tensor, record: int | None = None, video=None) -> dict:
        """z (B, D) one plan per robot -> per-robot cost, final standing share, ...; record: a robot whose motion is
        kept as the reference."""
        env, cfg = self.env, self.cfg
        kf, dur = self.kf.decode(z.to(self.dev))
        theta0 = self._reset()
        B, dt = self.B, env.cfg.policy_dt
        pad = torch.zeros(B, 2 * 22, dtype=self.dt, device=self.dev)
        total = torch.zeros(B, dtype=self.dt, device=self.dev)
        alive = torch.ones(B, dtype=torch.bool, device=self.dev)
        tail = torch.zeros(B, dtype=self.dt, device=self.dev)
        rec = {k: [] for k in ("theta", "com_z", "up", "standing", "target", "battery_J")} if record is not None else None
        with torch.no_grad():
            for s in range(self.steps):
                tgt = self.kf.targets(kf, dur, theta0, s * dt)
                a = (tgt - env.contract.home) / self.scale
                _, r, dones, _ = env.step(torch.cat((a, pad), -1))
                total += r * alive
                alive &= ~dones.bool()
                if s >= self.steps - self.tail:
                    tail += (env.standing & alive).to(self.dt)
                if rec is not None:
                    q = env.plant.q[record]
                    g = env.contract.projected_gravity(q[None, 3:7])[0]
                    rec["theta"].append(q[7:].cpu())
                    rec["com_z"].append(float(env.com_z[record]))
                    rec["up"].append(float(-g[2]))
                    rec["standing"].append(bool(env.standing[record]))
                    rec["target"].append(tgt[record].cpu())
                    rec["battery_J"].append(float(env.ep_battery[record]))
                if video is not None:
                    video.capture()
        final = tail / self.tail
        cost = -total - cfg.final_bonus * final + cfg.explode_cost * (~alive).to(self.dt)
        out = {"cost": cost.cpu(), "final": final.cpu(), "alive": alive.cpu(), "battery_J": env.ep_battery.cpu(),
               "time_to_stand": env.time_to_stand.cpu()}
        if rec is not None:
            out["record"] = {k: (torch.stack(v) if isinstance(v[0], torch.Tensor) else torch.tensor(v))
                             for k, v in rec.items()}
            out["record"]["theta0"] = theta0[record].cpu()
        return out

    # ------------------------------------------------------------------ CEM
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
            rec.close()
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

    video_every_min = 0.0   # set by main(): a video of the best search's mean plan every N minutes of search
    video_dir: Path | None = None

    def search(self, state: dict | None = None, wandb_run=None) -> dict:
        cfg = self.cfg
        R, N, V, D = cfg.restarts, cfg.candidates, cfg.variants, self.kf.D
        n_el = max(2, int(round(cfg.elite_frac * N)))
        g = torch.Generator().manual_seed(cfg.seed + 7)
        if state is None:
            base = self.kf.init_mean().cpu().double()
            mean = base.expand(R, D).clone()
            mean[1:] += 0.5 * torch.randn(R - 1, D, generator=g, dtype=torch.float64)  # different starts per search
            std = self.kf.init_std(cfg.init_joint_std, cfg.init_dur_std).cpu().double().expand(R, D).clone()
            state = {"mean": mean, "std": std, "it": 0, "history": [], "assist": self.assist, "energy": 0.0}
        mean, std = state["mean"], state["std"]
        self.assist, self.energy = state["assist"], state["energy"]
        t_start = time.time()
        last_video = t_start
        while state["it"] < cfg.iterations:
            if cfg.minutes > 0 and (time.time() - t_start) / 60 > cfg.minutes:
                self.log(f"[plan] {cfg.kind}: time budget of {cfg.minutes:g} min used up")
                break
            t0 = time.time()
            eps = torch.randn(R, N, D, generator=g, dtype=torch.float64)
            eps[:, 0] = 0.0
            z = mean[:, None] + std[:, None] * eps                                   # (R, N, D)
            # every video_every_min minutes: film the mean plan of the search that did best last time (variant 0)
            rec = None
            film = (self.video_every_min > 0 and self.video_dir is not None
                    and ((time.time() - last_video) / 60 >= self.video_every_min or state["it"] == 0))
            if film:
                prev = state["history"][-1]["elite_success"] if state["history"] else [0.0]
                top = int(torch.tensor(prev).argmax())
                rec = self._recorder((top * N) * V)
            out = self.rollout(z[:, :, None].expand(R, N, V, D).reshape(-1, D).float(), video=rec)
            if film:
                self._save_video(rec, self.video_dir / f"{cfg.kind}_it{state['it']:04d}.mp4", wandb_run,
                                 f"{cfg.kind}/video_search")
                last_video = time.time()
            c = out["cost"].double().reshape(R, N, V)
            fin = (out["final"] >= 0.8).double().reshape(R, N, V)
            cost = c.mean(-1) + cfg.risk * c.std(-1, unbiased=False)
            order = cost.argsort(-1)
            el = order[:, :n_el]
            elite = torch.gather(z, 1, el[..., None].expand(-1, -1, D))
            mean = cfg.alpha * elite.mean(1) + (1 - cfg.alpha) * mean
            std = torch.maximum(cfg.alpha * elite.std(1) + (1 - cfg.alpha) * std, torch.full_like(std, cfg.min_std))
            succ = fin.mean(-1)                                                      # (R, N): share of variants standing
            el_succ = torch.gather(succ, 1, el).mean(-1)                             # (R,)
            best = int(el_succ.argmax())
            row = {"it": state["it"], "assist_N": self.assist, "energy": self.energy,
                   "cost_mean": cost[:, 0].tolist(), "cost_best": cost.gather(1, order[:, :1]).squeeze(1).tolist(),
                   "elite_success": el_succ.tolist(), "mean_success": succ[:, 0].tolist(),
                   "battery_J_mean": float(out["battery_J"].mean()), "seconds": round(time.time() - t0, 1)}
            state["history"].append(row)
            self.log(f"[plan] {cfg.kind} it {state['it']:3d}  assist {self.assist:5.1f} N  energy {self.energy:.1f}  "
                     f"best search {best}: elites stand {el_succ[best]:.2f}, mean plan {succ[best, 0]:.2f}, "
                     f"cost {cost[best, 0]:.1f}  ({row['seconds']} s)")
            if wandb_run is not None:
                wandb_run.log({f"{cfg.kind}/elite_success": float(el_succ.max()),
                               f"{cfg.kind}/mean_success": float(succ[:, 0].max()),
                               f"{cfg.kind}/cost_mean": float(cost[:, 0].min()), f"{cfg.kind}/assist_N": self.assist,
                               f"{cfg.kind}/energy_weight": self.energy, f"{cfg.kind}/seconds": row["seconds"]})
            # continuation: less help, then more energy weight, while the best search's elites keep standing
            if float(el_succ.max()) >= cfg.gate:
                if self.assist > 0:
                    self.assist = max(0.0, self.assist - cfg.assist_step)
                elif self.energy < 1.0:
                    self.energy = min(1.0, self.energy + cfg.energy_step)
            state.update(mean=mean, std=std, it=state["it"] + 1, assist=self.assist, energy=self.energy)
        return state

    def final(self, state: dict, video_path: Path | None = None) -> dict:
        """Every search's mean plan without assist, on all variants; the best one with its reference motion."""
        cfg = self.cfg
        R, N, V, D = cfg.restarts, cfg.candidates, cfg.variants, self.kf.D
        self.assist = 0.0
        self.energy = max(self.energy, 0.0)
        z = state["mean"][:, None, None].expand(R, N, V, D).reshape(-1, D).float()
        out = self.rollout(z)
        c = out["cost"].double().reshape(R, N, V)[:, 0]
        fin = (out["final"] >= 0.8).reshape(R, N, V)[:, 0]
        cost = c.mean(-1) + cfg.risk * c.std(-1, unbiased=False)
        best = int(cost.argmin())
        # the reference: the best plan on variant 0 (nominal parameters), recorded
        slot = (best * N) * V + 0
        video = self._recorder(slot) if video_path is not None else None
        ref = self.rollout(z, record=slot, video=video)
        vid = self._save_video(video, video_path) if video is not None else None
        kf, dur = self.kf.decode(state["mean"][best:best + 1].float().to(self.dev))
        b = slice(best * N * V, best * N * V + V, 1)
        tts = ref["time_to_stand"][b]
        return {
            "kind": cfg.kind, "best_search": best, "z": state["mean"][best].tolist(),
            "keyframes": kf[0].cpu().tolist(), "durations": dur[0].cpu().tolist(), "home": self.kf.home.cpu().tolist(),
            "policy_dt": self.env.cfg.policy_dt, "horizon_s": cfg.horizon_s,
            "success_variants": fin[best].double().tolist(), "success": float(fin[best].double().mean()),
            "success_nominal": float(fin[best][~self.fitted].double().mean()),
            "success_fitted": float(fin[best][self.fitted].double().mean()) if bool(self.fitted.any()) else None,
            "cost_per_search": cost.tolist(), "success_per_search": fin.double().mean(-1).tolist(),
            "battery_J_variants": ref["battery_J"][b].tolist(),
            "time_to_stand_variants": [None if math.isnan(x) else x for x in tts.tolist()],
            "energy_weight": self.energy, "variants": {"pose_index": self.pose.tolist(), "delay_ms": self.delay,
                                                       "fitted": self.fitted.tolist()},
            "reference": ref["record"], "video": str(vid) if vid else None,
        }


def save_plan(result: dict, state: dict, cfg: PlanConfig, out_dir: Path, tag: str, fit_path: str | None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"getup-{cfg.kind}-{tag}"
    meta = {k: v for k, v in result.items() if k != "reference"}
    meta.update(config=asdict(cfg), fit=fit_path, history=state["history"], iterations=state["it"])
    torch.save({**meta, "reference": result["reference"]}, stem.with_suffix(".pt"))
    stem.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=str))
    return stem.with_suffix(".pt")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kinds", default="supine,prone")
    ap.add_argument("--fit", default=None, help="fit JSON of mbrl.fit: half of the variants use its elites")
    ap.add_argument("--out", default=str(PLANS))
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--run-name", default=time.strftime("getup-plan-%Y%m%d-%H%M"))
    ap.add_argument("--video", action="store_true", help="a video of every final plan (and of the search, see below)")
    ap.add_argument("--video-every-min", type=float, default=15.0,
                    help="with --video: also film the best search's current mean plan every N minutes (0: off)")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-project", default="Akos pH MBRL")
    ap.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", "karl-jahnel-universit-t-leipzig"))
    ap.add_argument("--smoke", action="store_true", help="tiny CPU run: 2 searches x 4 candidates x 2 variants, 1 s")
    for f, default in asdict(PlanConfig()).items():
        if f == "kind":
            continue
        name = "--" + f.replace("_", "-")
        if isinstance(default, bool):
            ap.add_argument(name, type=lambda s: s.lower() in ("1", "true", "yes"), default=default)
        elif isinstance(default, tuple):
            ap.add_argument(name, type=lambda s: tuple(float(x) for x in s.split(",")), default=default)
        else:
            ap.add_argument(name, type=type(default), default=default)
    a = ap.parse_args(argv)
    base = {f: getattr(a, f) for f in asdict(PlanConfig()) if f != "kind"}
    if a.smoke:
        base.update(device="cpu", restarts=2, candidates=4, variants=2, keyframes=2, horizon_s=1.0, iterations=2,
                    library_size=8)
        a.no_wandb = True
    fit = None
    if a.fit:
        from mbrl.model import FittedModel
        fit = FittedModel.load(a.fit)
    run = None
    if not a.no_wandb:
        import wandb
        run = wandb.init(entity=a.wandb_entity, project=a.wandb_project, name=a.run_name,
                         config={**base, "kinds": a.kinds, "fit": a.fit, "tag": a.tag})
    out_dir = Path(a.out)
    for kind in [k.strip() for k in a.kinds.split(",") if k.strip()]:
        cfg = PlanConfig(kind=kind, **base)  # --minutes: the time budget per kind
        t0 = time.time()
        pl = Planner(cfg, fit)
        if a.video:
            pl.video_every_min, pl.video_dir = a.video_every_min, out_dir / "videos"
        print(f"[plan] {kind}: {pl.B} robots ({cfg.restarts} searches x {cfg.candidates} candidates x "
              f"{cfg.variants} variants) on {cfg.device}, {pl.kf.D} parameters, {pl.steps} steps per rollout", flush=True)
        state_file = out_dir / f"getup-{kind}-{a.tag}-state.pt"
        state = torch.load(state_file, weights_only=False) if state_file.exists() else None
        if state is not None:
            print(f"[plan] {kind}: continuing at iteration {state['it']}", flush=True)
        state = pl.search(state, run)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(state, state_file)
        video = out_dir / f"getup-{kind}-{a.tag}.mp4" if a.video else None
        result = pl.final(state, video)
        path = save_plan(result, state, cfg, out_dir, a.tag, a.fit)
        print(f"[plan] {kind}: best search {result['best_search']} stands in {result['success']:.2f} of the variants "
              f"without assist (nominal {result['success_nominal']:.2f}, fitted {result['success_fitted']}), "
              f"battery {sum(result['battery_J_variants']) / len(result['battery_J_variants']):.0f} J; "
              f"{(time.time() - t0) / 60:.1f} min -> {path}", flush=True)
        if run is not None:
            run.summary[f"{kind}/success"] = result["success"]
            run.summary[f"{kind}/success_nominal"] = result["success_nominal"]
            if result["video"]:
                import wandb
                run.log({f"{kind}/video_final": wandb.Video(result["video"], format="mp4")})
            run.save(str(path), base_path=str(out_dir))
            run.save(str(path.with_suffix(".json")), base_path=str(out_dir))
        del pl
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()

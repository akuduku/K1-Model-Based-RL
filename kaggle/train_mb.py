"""Kaggle kernel: train a K1 skill (stand or getup) on the FITTED pH plant (mbrl.train) on a T4, with W&B.

The code comes from git: this repo (akuduku/K1-Model-Based-RL, private) is cloned at BRANCH together with its
submodule external/full-body (akuduku/K1-Full-Body-pH, private) using GITHUB_TOKEN; the token therefore needs read
access to BOTH repos.  GITHUB_TOKEN and WANDB_API_KEY come from the private dataset <user>/k1-secrets (attached by
kaggle/run.sh), or from Kaggle Secrets / the environment.

Steps: GPU + network check -> clone (recursive) -> colab/setup_gpu.sh (Full Body's setup: packages next to the
preinstalled CUDA torch, the CUDA plant, its tests ON THE GPU; then this repo's tests; no training if any fails)
-> python -m mbrl.train TASK --fit FIT (every robot draws one member of the fitted parameter ensemble at each reset)
-> runs/<RUN_NAME>/ in the kernel output, written there live (checkpoints, deploy_<it>.bin, videos).

The training gets the time left in the session (SESSION_HOURS minus a margin), or HOURS if that is smaller, and
stops cleanly (final checkpoint + deploy.bin + W&B artifact).  Continue in a new session with RESUME=<run name>.
kaggle/run.sh overrides the settings below:
    TASK=stand kaggle/run.sh                                  # stand-mb-v1, 3000 iterations (as stand-ph-v1)
    TASK=getup kaggle/run.sh                                  # getup-mb-v1, by time
    TASK=getup kaggle/run.sh RUN_NAME=getup-mb-v1b RESUME=getup-mb-v1
    TASK=getup-plan kaggle/run.sh HOURS=2.5                   # the model-based planner -> plans/ in the output
    TASK=getup-track kaggle/run.sh HOURS=6                    # DAgger + PPO on the committed plans (PLANS)
    TASK=getup-mbrl kaggle/run.sh HOURS=8                     # both in one session: plan (PLAN_HOURS), then track
    TASK=kick-plan kaggle/run.sh HOURS=2                      # the kick planner (mbrl.kick.plan) -> plans/ in the output
    TASK=kick-track kaggle/run.sh HOURS=6                     # DAgger + PPO on the committed kick library (KICK_PLAN)
    TASK=kick-mbrl kaggle/run.sh HOURS=8                      # both in one session
    TASK=mb-stand kaggle/run.sh HOURS=6 RUN_NAME=mb-stand-v1  # the model-based learner (mbrl.mpc) on standing
    TASK=grad-bench kaggle/run.sh HOURS=1                     # speed of gradients through the pH plant (mbrl.grad.bench)
    TASK=grad-stand kaggle/run.sh HOURS=6 RUN_NAME=grad-stand-v1            # the professor's formula, from scratch
    TASK=grad-walk kaggle/run.sh HOURS=6 RUN_NAME=grad-walk-v1              # the same, walking (W5: --task walk)

The kick tasks: kick-plan runs `python -m mbrl.kick.plan` (CEM over the kick plan of mbrl/kick/controller.py, one
search per ball speed in KICK_SPEEDS, through the walk policy WALK), kick-track runs `python -m mbrl.kick.train --plan
KICK_PLAN`; W&B project KICK_WANDB_PROJECT.  They use the standard robot model and no fitted parameters unless
KICK_FIT names a fit made on that model.

getup-plan runs `python -m mbrl.getup.plan` (CEM over keyframe motions in the pH plant; FIT gives half of the variants
fitted parameters), the time budget split over the PLAN_KINDS; its plans land in plans/ of the kernel output (commit
them to plans/ of this repo for getup-track).  getup-track runs `python -m mbrl.getup.train --plans PLANS`.
"""

import json
import os
import time

KERNEL_START = time.time()  # the session clock starts with the kernel
import shutil
import socket
import subprocess
import sys
from pathlib import Path

TASK = "stand"                            # stand | getup | getup-plan | getup-track
BRANCH = "main"                           # git branch of akuduku/K1-Model-Based-RL
RUN_NAME = ""                             # "" = <TASK>-mb-v1
FIT = "fits/getup-both-v1.json"           # the fitted model (mbrl.fit)
RESIDUAL = ""                             # optional residual/<name>.pt (mbrl.residual); "" = physics fit only
MODE = "elites"                           # elites | gaussian | mean
NUM_ENVS = 4096
ITERATIONS = 0                            # 0 = task default (stand 3000 as stand-ph-v1; getup: by time)
HOURS = 0.0                               # 0 = the whole session; else at most this much training
VIDEO_EVERY = -1                          # iterations; -1 = task default (stand 500, getup 0 = by time)
VIDEO_EVERY_MIN = -1.0                    # minutes; -1 = task default (stand 0, getup 30)
FIRST_VIDEO_MIN = 3.0                     # the first of the timed videos after 3 min: it proves the video path
SAVE_EVERY = 250
SAVE_EVERY_MIN = 10.0                     # a checkpoint every 10 min of training, so a kill loses little
WANDB_ENTITY = "karl-jahnel-universit-t-leipzig"
WANDB_PROJECT = "Akos pH MBRL"
TRAIN_ARGS = ""                           # extra arguments for Full Body's rl/<task>/train.py (or the getup-* scripts)
PLAN_KINDS = "supine,prone"               # getup-plan: one plan per lying kind
PLAN_HOURS = 2.5                          # getup-mbrl: the planning part; the tracking gets the rest of the session
PLAN_INIT = ""                            # getup-plan/-mbrl: start the searches around these plans, e.g.
                                          # "plans/getup-{kind}-v1.pt" (committed from a previous run)
FORCE_TRACK = False                       # getup-mbrl: track also plans that miss the gate (default: only those that
                                          # stand without assist in >= 75 % of the variants; none -> no tracking)
PLAN_TAG = "v1"                           # getup-plan: plans/getup-<kind>-<tag>.pt
PLAN_ITERATIONS = 200                     # getup-plan: upper bound; the time budget usually ends each search first
PLANS = "plans/getup-supine-v1.pt,plans/getup-prone-v1.pt"   # getup-track: the plans (committed in this repo)
DAGGER_ITERS = 300                        # getup-track: DAgger warm-start iterations before PPO
KICK_SPEEDS = "2,3.5,5,6"                 # kick-plan: requested ball speeds, one search each
KICK_CANDIDATES = 96                      # kick-plan: candidates per search and iteration
KICK_VARIANTS = 8                         # kick-plan: variants every candidate is judged on
KICK_PLAN = "plans/kick-v1.pt"            # kick-track: the library (committed in this repo); kick-plan/-mbrl with
                                          # KICK_PLAN_INIT=1 start their searches around it
KICK_PLAN_INIT = False
KICK_FIT = ""                             # a fit made on the standard robot model, or "" (fits/getup-both-v1.json
                                          # was made on the older hand model: not for the kick)
WALK = "policies/walk_teacher.pt"         # the walk policy the kick plan acts through (TorchScript, 75 -> 22)
KICK_WANDB_PROJECT = "Akos pH Kick MBRL"
MB_ROBOTS = 256                           # mb-*: world robots of the model-based learner (mbrl.mpc)
MB_CANDIDATES = 8                         # mb-*: model copies per world robot
MB_HORIZON = 5                            # mb-*: planner steps of 20 ms
GRAD_ROBOTS = 0                           # grad-stand/-walk: robots of the gradient learner (mbrl.grad.train); 0 = as
                                          # many as fit, measured on the GPU at the start
GRAD_STEPS = 20                           # grad-stand/-walk: control steps per window
GRAD_DEVICE = "cuda"                      # grad-stand/-walk: where the Python pH plant runs
GRAD_BENCH_ROBOTS = "64,256,1024"         # grad-bench: robots per row
GRAD_DTYPE = "float32"                    # grad-stand/-walk
MB_WANDB_PROJECT = "Akos pH MB"
RESUME = ""                               # W&B run (name or id) in WANDB_PROJECT to continue from; "" = fresh
SESSION_HOURS = 11.0                      # our own cap, 1 h under Kaggle's 12 h (kaggle/run.sh pushes with the same
                                          # timeout): the kernel never runs into Kaggle's limit
MARGIN_MINUTES = 30                       # kept free after the training: final checkpoint, deploy.bin, W&B artifact

TASK_DEFAULTS = {"stand": {"iterations": 3000, "video_every": 500, "video_every_min": 0.0},
                 "getup": {"iterations": 100000, "video_every": 0, "video_every_min": 30.0},
                 "getup-plan": {"iterations": 0, "video_every": 0, "video_every_min": 0.0},
                 "getup-track": {"iterations": 100000, "video_every": 0, "video_every_min": 30.0},
                 "getup-mbrl": {"iterations": 100000, "video_every": 0, "video_every_min": 30.0},
                 "kick-plan": {"iterations": 0, "video_every": 0, "video_every_min": 0.0},
                 "kick-track": {"iterations": 100000, "video_every": 0, "video_every_min": 30.0},
                 "kick-mbrl": {"iterations": 100000, "video_every": 0, "video_every_min": 30.0},
                 "mb-stand": {"iterations": 1000000, "video_every": 0, "video_every_min": 30.0},
                 "grad-bench": {"iterations": 0, "video_every": 0, "video_every_min": 0.0},
                 "grad-stand": {"iterations": 1000000, "video_every": 0, "video_every_min": 0.0},
                 "grad-walk": {"iterations": 1000000, "video_every": 0, "video_every_min": 20.0}}   # with every evaluation
REPO = "https://github.com/akuduku/K1-Model-Based-RL.git"
WORK = Path("/kaggle/working")
SRC = Path("/tmp/mbrl")
LOGS = WORK / "logs"


def log(msg):
    print(f"[kaggle] {msg}", flush=True)


def run(cmd, logfile, env=None, cwd=None, check=True):
    log("$ " + " ".join(map(str, cmd)))
    with open(LOGS / logfile, "a") as out:
        try:
            proc = subprocess.Popen(list(map(str, cmd)), env={**os.environ, **(env or {})}, cwd=cwd,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        except OSError as exc:
            out.write(f"{cmd[0]}: {exc}\n")
            print(f"{cmd[0]}: {exc}", flush=True)
            code = 127
        else:
            for line in proc.stdout:
                sys.stdout.write(line)
                out.write(line)
            code = proc.wait()
    if check and code:
        raise SystemExit(f"{cmd[0]} failed with exit code {code} (logs/{logfile})")
    return code


def secrets_file():
    import glob
    for path in sorted(glob.glob("/kaggle/input/**/k1_secrets.json", recursive=True)):
        try:
            return json.loads(Path(path).read_text())
        except (OSError, ValueError):
            pass
    return {}


def secret(name):
    if os.environ.get(name):
        return os.environ[name]
    if secrets_file().get(name):
        return secrets_file()[name]
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:
        return None


RENDER_PROBE = """
import torch
torch.zeros(1, device="cuda" if torch.cuda.is_available() else "cpu")   # CUDA first, as in the training
import mujoco
m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos="0 0 3"/><geom size=".1"/></worldbody></mujoco>')
r = mujoco.Renderer(m, height=480, width=640)
r.update_scene(mujoco.MjData(m))
img = r.render()
print("render ok", img.shape)
"""


def render_backend(py):
    """The MuJoCo GL backend for the video process (ph_k1/render_frames.py inherits it), tried in a child process:
    EGL, else OSMesa (software), else None (train without videos).  EGL renders in a fresh process on Kaggle but
    crashed with SIGSEGV inside the training process (getup-mb-v1, -v1b, -v1c), so the training never touches GL."""
    for gl in ("egl", "osmesa"):
        if gl == "osmesa":
            run(["bash", "-c", "apt-get install -y -qq libosmesa6 > /dev/null 2>&1 || true"], "install.log",
                check=False)
        code = run([py, "-c", RENDER_PROBE], "render.log", env={"MUJOCO_GL": gl, "PYOPENGL_PLATFORM": gl},
                   check=False)
        log(f"render probe {gl}: {'ok' if code == 0 else f'failed (exit {code})'}")
        if code == 0:
            return gl
    return None


def reachable(host):
    try:
        socket.create_connection((host, 443), timeout=5).close()
        return True
    except OSError:
        return False


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    if TASK not in TASK_DEFAULTS:
        raise SystemExit(f"TASK={TASK!r}: stand, getup, getup-plan, getup-track, getup-mbrl, kick-plan, kick-track, "
                         "kick-mbrl, mb-stand, grad-bench, grad-stand or grad-walk")
    d = TASK_DEFAULTS[TASK]
    run_name = RUN_NAME or (f"{TASK}-v1" if TASK.startswith(("getup-", "kick-", "mb-", "grad-")) else f"{TASK}-mb-v1")
    project = KICK_WANDB_PROJECT if TASK.startswith("kick-") else WANDB_PROJECT
    iterations = ITERATIONS or d["iterations"]
    video_every = d["video_every"] if VIDEO_EVERY < 0 else VIDEO_EVERY
    video_every_min = d["video_every_min"] if VIDEO_EVERY_MIN < 0 else VIDEO_EVERY_MIN
    settings = {k: globals()[k] for k in ("TASK", "BRANCH", "FIT", "RESIDUAL", "MODE", "NUM_ENVS", "HOURS", "SAVE_EVERY",
                                          "SAVE_EVERY_MIN", "WANDB_ENTITY", "WANDB_PROJECT", "TRAIN_ARGS", "RESUME",
                                          "SESSION_HOURS", "MARGIN_MINUTES")}
    settings.update(run_name=run_name, iterations=iterations, video_every=video_every, video_every_min=video_every_min)
    log(json.dumps(settings))
    log(f"network: github {'ok' if reachable('github.com') else 'NO'}, pypi {'ok' if reachable('pypi.org') else 'NO'}")
    github, wandb_key = secret("GITHUB_TOKEN"), secret("WANDB_API_KEY")
    missing = [n for n, v in (("GITHUB_TOKEN", github), ("WANDB_API_KEY", wandb_key)) if not v]
    if missing:
        raise SystemExit("secrets missing: " + ", ".join(missing) + " (private dataset <user>/k1-secrets: run "
                         "upload_secrets.sh of the k1-walk repo once; kaggle/run.sh attaches it)")
    if run(["nvidia-smi"], "install.log", check=False) and not os.environ.get("PH_ALLOW_CPU"):
        raise SystemExit("no GPU in this session (Settings: Accelerator GPU T4, Internet on, phone-verified account)")

    # the code, from git: this repo and the pH plant (submodule), with the token for both
    if SRC.exists():
        shutil.rmtree(SRC)
    helper = '!f() { echo username=x-access-token; echo "password=$GITHUB_TOKEN"; }; f'
    code = run(["git", "-c", "credential.helper=", "-c", f"credential.helper={helper}", "clone", "--depth", "1",
                "--recurse-submodules", "--shallow-submodules", "--branch", BRANCH, REPO, SRC], "install.log",
               env={"GITHUB_TOKEN": github}, check=False)
    if code:
        raise SystemExit("git clone failed: GITHUB_TOKEN needs read access (Contents) to akuduku/K1-Model-Based-RL "
                         "AND akuduku/K1-Full-Body-pH (the submodule)")
    run(["git", "-C", SRC, "log", "-1", "--format=commit %h %s (%cd)"], "install.log")
    run(["git", "-C", SRC, "submodule", "status"], "install.log")

    # packages + CUDA plant + all tests on the GPU (colab/setup_gpu.sh -> external/full-body/colab/setup_gpu.sh)
    py = sys.executable
    run(["bash", "-c", "apt-get install -y -qq libegl1 libgl1 libglvnd0 > /dev/null 2>&1 || true"], "install.log",
        check=False)
    code = run(["bash", "colab/setup_gpu.sh"], "setup.log", cwd=SRC, check=False,
               env={"PYTHON": py, "GITHUB_TOKEN": github, "PH_K1_CUDA": os.environ.get("PH_K1_CUDA", "1"),
                    "PH_TEST_DEVICE": os.environ.get("PH_TEST_DEVICE", "cuda")})
    if code:
        raise SystemExit("setup or GPU tests failed (logs/setup.log): not training on a plant that disagrees with "
                         "its reference")

    # videos only with a GL backend that rendered in a child process (else the training runs without videos)
    gl = render_backend(py)
    if gl is None:
        log("no working GL backend: training WITHOUT videos")
        video_every, video_every_min = 0, 0.0

    # training on the fitted model
    env = {"WANDB_API_KEY": wandb_key, "WANDB_ENTITY": WANDB_ENTITY, "WANDB_USERNAME": WANDB_ENTITY,
           "MUJOCO_GL": gl or "egl", "PYOPENGL_PLATFORM": gl or "egl", "PYTHONUNBUFFERED": "1",
           "PH_K1_ROOT": str(SRC / "external" / "full-body"),
           "PYTHONFAULTHANDLER": "1"}  # a crash in native code (SIGSEGV) prints the Python stack into train.log
    budget = SESSION_HOURS * 60 - (time.time() - KERNEL_START) / 60 - MARGIN_MINUTES
    if HOURS > 0:
        budget = min(budget, HOURS * 60)
    log(f"training time budget {budget:.0f} min (session {SESSION_HOURS:g} h, {MARGIN_MINUTES} min margin)")

    def harness(name, minutes):
        return ["--run-name", name, "--num-envs", NUM_ENVS, "--iterations", iterations,
                "--save-every", SAVE_EVERY, "--save-every-min", SAVE_EVERY_MIN,
                "--video-every", video_every, "--video-every-min", video_every_min,
                *(["--first-video-min", FIRST_VIDEO_MIN] if video_every_min > 0 else []),
                "--wandb-project", project, "--wandb-entity", WANDB_ENTITY, "--log-root", WORK / "runs",
                "--max-minutes", f"{max(minutes, 1):.1f}", *(["--resume-wandb", RESUME] if RESUME else [])]

    def kick_plan_cmd(name, minutes):
        search = max(minutes - 10.0, 1.0)  # the search, then ~10 min for the final rollouts and videos
        log(f"kick planner: {search:.0f} min for the speeds {KICK_SPEEDS}")
        return [py, "-m", "mbrl.kick.plan", "--speeds", KICK_SPEEDS, "--candidates", KICK_CANDIDATES,
                "--variants", KICK_VARIANTS, "--tag", PLAN_TAG, "--minutes", f"{search:.1f}",
                "--iterations", PLAN_ITERATIONS, "--walk", WALK, "--out", WORK / "plans", "--run-name", name, "--video",
                "--video-every-min", 15, "--wandb-project", project, "--wandb-entity", WANDB_ENTITY,
                *(["--fit", KICK_FIT] if KICK_FIT else []), *(["--init-plan", KICK_PLAN] if KICK_PLAN_INIT else [])]

    def kick_track_cmd(name, minutes, plan):
        return [py, "-m", "mbrl.kick.train", "--plan", plan, "--walk", WALK, "--dagger-iters", DAGGER_ITERS,
                *harness(name, minutes), *TRAIN_ARGS.split()]

    def plan_cmd(name, minutes):
        kinds = [k for k in PLAN_KINDS.split(",") if k.strip()]
        per_kind = max(minutes / len(kinds) - 6.0, 1.0)  # each kind: its search, then ~5 min for the final rollouts
        log(f"planner: {per_kind:.0f} min per kind for {kinds}")
        return [py, "-m", "mbrl.getup.plan", "--kinds", PLAN_KINDS, "--tag", PLAN_TAG, "--minutes", f"{per_kind:.1f}",
                "--iterations", PLAN_ITERATIONS, "--out", WORK / "plans", "--run-name", name, "--video",
                "--wandb-project", WANDB_PROJECT, "--wandb-entity", WANDB_ENTITY, *(["--fit", FIT] if FIT else []),
                *(["--init-plan", PLAN_INIT] if PLAN_INIT else []), *TRAIN_ARGS.split()]

    def track_cmd(name, minutes, plans, kinds=None):
        return [py, "-m", "mbrl.getup.train", "--plans", plans, "--dagger-iters", DAGGER_ITERS,
                *(["--lying-kinds", kinds] if kinds else []),
                *(["--fit", FIT] if FIT else []), *harness(name, minutes), *TRAIN_ARGS.split()]

    def go(cmd, what):
        start = time.time()
        code = run(cmd, "train.log", env=env, cwd=SRC, check=False)
        log(f"{what} exit code {code} after {(time.time() - start) / 60:.1f} min")
        if code:
            raise SystemExit(f"{what} failed (exit code {code}), see logs/train.log")

    if TASK == "grad-bench":
        # how fast is a gradient through the Python pH plant on the T4 (16 Nov on the kernel's CPU, 2 threads, float32:
        # 64 robots 24 and 256 robots 38 robot-control-steps per second)
        go([py, "-u", "-m", "mbrl.grad.bench", "--devices", "cuda", "--robots", *str(GRAD_BENCH_ROBOTS).split(","),
            "--dtypes", "float32", "--steps", 20, "--trace"], "benchmark (GPU)")
    elif TASK in ("grad-stand", "grad-walk"):
        # the professor's formula: standing (or walking, W5) from scratch with derivatives of the cost through the
        # pH model; walking is evaluated on six fixed commands, 64 robots each, with a video and the Hamiltonian
        # figure at every evaluation when the GL backend works
        walk = TASK == "grad-walk"
        go([py, "-u", "-m", "mbrl.grad.train", *(["--task", "walk"] if walk else []),
            *(["--videos"] if walk and video_every_min > 0 else []), "--run-name", run_name,
            "--robots", GRAD_ROBOTS, "--steps", GRAD_STEPS,
            "--device", GRAD_DEVICE, "--dtype", GRAD_DTYPE, "--minutes", f"{max(budget, 1):.1f}", "--delta-final", 0.01,
            "--save-every-min", SAVE_EVERY_MIN, "--eval-every-min", 20, "--eval-robots", 64 if walk else 128,
            "--log-root", WORK / "runs", "--wandb-project", MB_WANDB_PROJECT, "--wandb-entity", WANDB_ENTITY,
            *(["--resume", RESUME] if RESUME else []), *TRAIN_ARGS.split()], "training")
    elif TASK.startswith("mb-"):
        # the model-based learner: plan in the pH model, learn policy and value; resume = a checkpoint path in the repo
        go([py, "-m", "mbrl.mpc.train", TASK[3:], "--run-name", run_name, "--robots", MB_ROBOTS,
            "--candidates", MB_CANDIDATES, "--horizon", MB_HORIZON, "--sigma", 0.2, "--minutes", f"{max(budget, 1):.1f}",
            "--save-every-min", SAVE_EVERY_MIN, "--eval-every-min", 20, "--eval-robots", 128,
            "--video-every-min", video_every_min, "--log-root", WORK / "runs", "--wandb-project", MB_WANDB_PROJECT,
            "--wandb-entity", WANDB_ENTITY, *(["--resume", RESUME] if RESUME else []), *TRAIN_ARGS.split()], "training")
    elif TASK == "kick-plan":
        go(kick_plan_cmd(run_name, budget), "planning")
    elif TASK == "kick-track":
        go(kick_track_cmd(run_name, budget, KICK_PLAN), "training")
    elif TASK == "kick-mbrl":
        # 1. plan the kick with the model, 2. learn to track the fresh library with the time that is left
        go(kick_plan_cmd(f"{run_name}-plan", min(PLAN_HOURS * 60, budget - 60)), "planning")
        p = WORK / "plans" / f"kick-{PLAN_TAG}.pt"
        meta = json.loads(p.with_suffix(".json").read_text())
        ok = all(s >= 0.75 for s in meta["success"])
        log(f"kick library: good kicks per speed {dict(zip(meta['speeds'], meta['success']))} "
            f"({'PASS' if ok else 'below the gate of 0.75'})")
        if not (ok or FORCE_TRACK):  # imitating a plan that does not kick teaches the wrong thing: keep the GPU quota
            log("the library missed the gate: no tracking (FORCE_TRACK=1 to track anyway); it is in plans/ of the "
                "output: commit it and continue with TASK=kick-plan KICK_PLAN_INIT=1")
            log("done")
            return
        left = SESSION_HOURS * 60 - (time.time() - KERNEL_START) / 60 - MARGIN_MINUTES
        if HOURS > 0:
            left = min(left, HOURS * 60 - (time.time() - KERNEL_START) / 60)
        log(f"tracking the kick library: {left:.0f} min left")
        go(kick_track_cmd(f"{run_name}-track", left, str(p)), "training")
    elif TASK == "getup-plan":
        go(plan_cmd(run_name, budget), "planning")
    elif TASK == "getup-track":
        go(track_cmd(run_name, budget, PLANS), "training")
    elif TASK == "getup-mbrl":
        # 1. plan with the model, 2. learn to track the fresh plans with the time that is left (two W&B runs)
        go(plan_cmd(f"{run_name}-plan", min(PLAN_HOURS * 60, budget - 60)), "planning")
        plans, kinds = [], []
        for kind in [k.strip() for k in PLAN_KINDS.split(",") if k.strip()]:
            p = WORK / "plans" / f"getup-{kind}-{PLAN_TAG}.pt"
            meta = json.loads(p.with_suffix(".json").read_text())
            ok = meta["success"] >= 0.75
            log(f"plan {kind}: stands without assist in {meta['success']:.2f} of the variants "
                f"({'PASS' if ok else 'below the gate'}; nominal {meta['success_nominal']:.2f}, "
                f"fitted {meta['success_fitted']})")
            if ok or FORCE_TRACK:
                plans.append(str(p))
                kinds.append(kind)
        if not plans:  # tracking a plan that falls teaches falling: keep the GPU quota for the next planning run
            log("no plan passed the gate: no tracking (FORCE_TRACK=1 to track anyway); plans are in plans/ of the "
                "output: commit them and continue with TASK=getup-plan PLAN_INIT=plans/getup-{kind}-<tag>.pt")
            log("done")
            return
        left = SESSION_HOURS * 60 - (time.time() - KERNEL_START) / 60 - MARGIN_MINUTES
        if HOURS > 0:
            left = min(left, HOURS * 60 - (time.time() - KERNEL_START) / 60)
        log(f"tracking {kinds}: {left:.0f} min left")
        go(track_cmd(f"{run_name}-track", left, ",".join(plans), ",".join(kinds)), "training")
    else:
        go([py, "-m", "mbrl.train", TASK, "--fit", FIT, "--mode", MODE,
            *(["--residual", RESIDUAL] if RESIDUAL else []), "--", *harness(run_name, budget), *TRAIN_ARGS.split()],
           "training")
    log("done")


if __name__ == "__main__":
    main()

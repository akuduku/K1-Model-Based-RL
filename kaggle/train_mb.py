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

TASK = "stand"                            # stand | getup
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
TRAIN_ARGS = ""                           # extra arguments for Full Body's rl/<task>/train.py
RESUME = ""                               # W&B run (name or id) in WANDB_PROJECT to continue from; "" = fresh
SESSION_HOURS = 11.0                      # our own cap, 1 h under Kaggle's 12 h (kaggle/run.sh pushes with the same
                                          # timeout): the kernel never runs into Kaggle's limit
MARGIN_MINUTES = 30                       # kept free after the training: final checkpoint, deploy.bin, W&B artifact

TASK_DEFAULTS = {"stand": {"iterations": 3000, "video_every": 500, "video_every_min": 0.0},
                 "getup": {"iterations": 100000, "video_every": 0, "video_every_min": 30.0}}
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
        raise SystemExit(f"TASK={TASK!r}: stand or getup")
    d = TASK_DEFAULTS[TASK]
    run_name = RUN_NAME or f"{TASK}-mb-v1"
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
    cmd = [py, "-m", "mbrl.train", TASK, "--fit", FIT, "--mode", MODE, *(["--residual", RESIDUAL] if RESIDUAL else []),
           "--", "--run-name", run_name, "--num-envs", NUM_ENVS, "--iterations", iterations,
           "--save-every", SAVE_EVERY, "--save-every-min", SAVE_EVERY_MIN,
           "--video-every", video_every, "--video-every-min", video_every_min,
           *(["--first-video-min", FIRST_VIDEO_MIN] if video_every_min > 0 else []),
           "--wandb-project", WANDB_PROJECT, "--wandb-entity", WANDB_ENTITY, "--log-root", WORK / "runs",
           "--max-minutes", f"{max(budget, 1):.1f}", *(["--resume-wandb", RESUME] if RESUME else []),
           *TRAIN_ARGS.split()]
    start = time.time()
    code = run(cmd, "train.log", env=env, cwd=SRC, check=False)
    log(f"training exit code {code} after {(time.time() - start) / 60:.1f} min")
    if code:
        raise SystemExit(f"training failed (exit code {code}), see logs/train.log")
    log("done")


if __name__ == "__main__":
    main()

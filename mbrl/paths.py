r"""Where the Full Body pH plant (package ph_k1, the RL envs rl.stand / rl.getup) and the robot logs are.

Full Body: $PH_K1_ROOT, else the sibling folder ../Full Body (local development), else the pinned submodule
external/full-body (Kaggle/Colab clone with --recursive).  It is put on sys.path when mbrl is imported.

Robot logs: $MBRL_LOGS, else MA/Karls Repo/Logs next to this repository (the logs are not in git).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def full_body_root() -> Path:
    candidates = [os.environ.get("PH_K1_ROOT"), REPO.parent / "Full Body", REPO / "external" / "full-body"]
    for c in candidates:
        if c and (Path(c) / "ph_k1" / "plant.py").exists():
            return Path(c).resolve()
    raise ImportError("Full Body (ph_k1) not found: set PH_K1_ROOT, or `git submodule update --init` "
                      "(external/full-body)")


def logs_root() -> Path:
    env = os.environ.get("MBRL_LOGS")
    return Path(env) if env else REPO.parents[1] / "Karls Repo" / "Logs"


def setup() -> Path:
    root = full_body_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


FULL_BODY = setup()

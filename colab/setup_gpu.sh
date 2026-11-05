#!/bin/bash
# A CUDA machine (Colab, Kaggle) ready to train on the fitted model, from a clone of this repo:
#   1 the Full Body submodule: its own setup (packages next to the preinstalled CUDA torch, the C++/CUDA plant
#     built for this GPU, its tests ON THE GPU: C++ == Python reference, command path, get-up task)
#   2 this repo's tests (parameter rows, replay self-consistency, ensemble/residual attach, the env wrapper)
# PYTHON=... picks the interpreter; PH_K1_CUDA=0 PH_TEST_DEVICE=cpu: a dry run without a GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
git submodule update --init --recursive
PYTHON="$PY" bash external/full-body/colab/setup_gpu.sh
$PY -m pip install -q matplotlib
PH_K1_ROOT="$PWD/external/full-body" $PY -m pytest tests -q -p no:cacheprovider
echo "[setup] model-based RL tests passed"

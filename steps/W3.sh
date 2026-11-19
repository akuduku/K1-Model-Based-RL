#!/bin/bash
# W3: Can a learned term 2 take the place of the fixed one at the window's end, and is standing as good with it?
# Runs:   steps/W3.sh check                         the tests (laptop, 1 min)
#         steps/W3.sh laptop                        40 min of standing from scratch with learned weights on one core
#                                                   (the settings of 16 Nov's fixed-term-2 run), then its scoring (8 min)
#         steps/W3.sh launch [--yes]                show the Kaggle run grad-stand-weights (T4, 6 h, seed 1: the settings
#                                                   of grad-stand-v1c); --yes starts it.  HOURS=3 steps/W3.sh launch: shorter
#         steps/W3.sh fetch | score                 download it when it has ended | score it (as W1)
# Needs:  for launch: everything committed and pushed to GitHub, GPU quota
# How:    term 2 = sum_k s_k c_k(x(T)): the costs the end state causes, weights s_k >= 0 learned from TD(lambda)
#         returns, starting at the fixed term 2's (--term2 weights, mbrl/grad/value.py).  A free value network
#         (--term2 value) failed twice on the laptop: its slope is unrelated to the physics (docs/EXPERIMENTS.md W3)
# Then:   17 Nov 2026, laptop (logs/grad-stand-weights-laptop): in-run falls with pushes 1.6, 0, 3.1, 1.6 % (fixed
#         term 2 on 16 Nov: 26.6, 3.1, 26.6, 4.7 %); scored update 48: 5 of 384 fell (1.3 %), 3.1 W, quiet 0 %
#         0.9 W, MuJoCo stood 20 s quiet and pushed (fixed, update 41: 20 of 384 fell (5.2 %), 2.6 W).
#         term2_diag on model_48: action gradient against the fixed term 2's, cosine +0.97.
#         19 Nov 2026 (tag W3), Kaggle grad-stand-weights: best in-run update 40 (0.82 M steps, 1.4 h): 2 of 384 fell
#         (0.5 %), 2.4 W, tilt 0.30 deg, quiet 0 % 1.0 W, MuJoCo stood 20 s (0.1 / 0.7 deg); end (update 164):
#         0.5 %, 3.5 W.  Fixed term 2 (grad-stand-v1c, update 80, 1.64 M steps, 2.9 h): 1.0 %, 2.3 W, 0.46 deg
set -euo pipefail
. "$(dirname "$0")/common.sh"
NAME=grad-stand-weights
SLUG=mbrl-k1-grad-stand-weights

case "${1:-}" in
    check)
        begin
        run "$PY" -m pytest tests/test_grad_value.py tests/test_grad.py tests/test_grad_walk.py -q -p no:cacheprovider
        finish "passed|failed"
        ;;
    laptop)
        STEP="W3-laptop"
        begin
        run "$PY" -m mbrl.grad.train --robots 16 --steps 20 --minutes 40 --eval-every-min 10 --eval-robots 64 \
            --threads 1 --dtype float64 --seed 1 --term2 weights --no-wandb --run-name "$NAME-laptop-$(date +%m%d-%H%M)"
        run "$PY" -m mbrl.grad.score "$(ls -td logs/$NAME-laptop-*/ | head -1)"
        finish "stopped|done after|of [0-9]+ fell|pH plant, quiet|MuJoCo|best in-run"
        ;;
    launch)
        cd "$REPO"
        if [ "${2:-}" = "--yes" ]; then
            KERNEL_SLUG=$SLUG TASK=grad-stand kaggle/run.sh HOURS=${HOURS:-6} RUN_NAME=$NAME TRAIN_ARGS="--term2 weights --seed 1"
            echo "started.  When it has ended: steps/W3.sh fetch && steps/W3.sh score"
        else
            echo "This would start a ${HOURS:-6} h Kaggle GPU run '$NAME' (term 2 with learned weights, seed 1) on kernel $SLUG."
            echo "Nothing was started.  To start it: steps/W3.sh launch --yes"
        fi
        ;;
    fetch)
        cd "$REPO"
        KERNEL_SLUG=$SLUG TASK=grad-stand kaggle/fetch.sh "$NAME"
        ;;
    score)
        STEP="W3-score"
        begin
        run "$PY" -m mbrl.grad.score "$NAME"
        finish "of [0-9]+ fell|pH plant, quiet|MuJoCo|best in-run"
        ;;
    *)
        header
        ;;
esac

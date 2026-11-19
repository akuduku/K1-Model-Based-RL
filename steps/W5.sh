#!/bin/bash
# W5: Does the professor's formula learn to walk from scratch (with W3's learned term 2 and a curriculum on the requests)?
# Runs:   steps/W5.sh check                         the tests (laptop, 5 min)
#         steps/W5.sh laptop                        30 min of walking from scratch on one core (16 robots): a smoke run
#                                                   that shows everything runs and moves the right way; not a result
#         steps/W5.sh launch [--yes]                show the Kaggle run grad-walk-v1 (T4, 6 h, seed 1); --yes starts it.
#                                                   HOURS=3 steps/W5.sh launch: shorter
#         steps/W5.sh fetch | score                 download it when it has ended | score it per command (about 30 min)
# Needs:  for launch: both repositories committed and pushed to GitHub (the kernel clones Full Body as a submodule),
#         GPU quota
# How:    mbrl/grad/train.py --task walk: Full Body's rl/walk task on mbrl/grad/walk_env.py, the firmware's sprint agent
#         (75 -> 22), term 2 with learned weights (W3).  Curriculum: every request x 0.25 at the start (forward up to
#         0.5 m/s), one step of 0.05 up when over 20 updates at most 0.2 % of robot-windows ended in a fall and the
#         speed error was at most 30 % of the requested speed, up to Eric's ranges (x 1, sprints up to 2 m/s).
#         In-run evaluation and scoring: six fixed commands (stand, forward 0.5 and 1.0, back 0.5, sideways 0.5 m/s,
#         turn 1 rad/s) on the C++ plant; scoring also in MuJoCo
# Then:   (filled in when the step is finished)
set -euo pipefail
. "$(dirname "$0")/common.sh"
NAME=grad-walk-v1
SLUG=mbrl-k1-grad-walk

case "${1:-}" in
    check)
        begin
        run "$PY" -m pytest tests/test_grad_train.py tests/test_grad_walk.py tests/test_grad_value.py -q -p no:cacheprovider
        run "$PY" -m pytest "$FULL_BODY/rl/walk/tests" -q -p no:cacheprovider
        finish "passed|failed"
        ;;
    laptop)
        STEP="W5-laptop"
        begin
        run "$PY" -m mbrl.grad.train --task walk --robots 16 --steps 20 --minutes 30 --eval-every-min 10 --eval-robots 16 \
            --threads 1 --dtype float64 --seed 1 --no-wandb --run-name "$NAME-laptop-$(date +%m%d-%H%M)"
        finish "stopped|done after|curriculum:|\] it +(1|10|20|30|40|50|60|70) "
        ;;
    launch)
        cd "$REPO"
        if [ "${2:-}" = "--yes" ]; then
            KERNEL_SLUG=$SLUG TASK=grad-walk kaggle/run.sh HOURS=${HOURS:-6} RUN_NAME=$NAME TRAIN_ARGS="--seed 1"
            echo "started.  When it has ended: steps/W5.sh fetch && steps/W5.sh score"
        else
            echo "This would start a ${HOURS:-6} h Kaggle GPU run '$NAME' (walking from scratch, seed 1) on kernel $SLUG."
            echo "Nothing was started.  To start it: steps/W5.sh launch --yes"
            DRY_RUN=1 KERNEL_SLUG=$SLUG TASK=grad-walk kaggle/run.sh HOURS=${HOURS:-6} RUN_NAME=$NAME TRAIN_ARGS="--seed 1" | tail -n 8
        fi
        ;;
    fetch)
        cd "$REPO"
        KERNEL_SLUG=$SLUG TASK=grad-walk kaggle/fetch.sh "$NAME"
        ;;
    score)
        STEP="W5-score"
        begin
        run "$PY" -m mbrl.grad.score "$NAME"
        finish "best in-run|^ +(command|stand|fwd|back|side|turn)|quiet|MuJoCo|command [+-]|^\|"
        ;;
    *)
        header
        ;;
esac

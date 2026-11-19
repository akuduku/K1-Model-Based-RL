#!/bin/bash
# W5b: Does the professor's formula step when lifting a foot is visible to its derivative (air_time's smooth stand-in)?
# Runs:   steps/W5b.sh check                        the tests (laptop, 2 min)
#         steps/W5b.sh gait                         the stand-in on a real gait: Eric's walk_v9 (a check only, part of no
#                                                   training) on the pH plant, against his air_time count (laptop, 1 min)
#         steps/W5b.sh lift                         a standing robot's next 0.8 s of actions optimised through the pH model
#                                                   with the learner's own window, without and with the stand-in: does a
#                                                   foot come up?  (laptop, about 18 min per weight; WEIGHTS="0 0.1 0.5")
#         steps/W5b.sh laptop                       30 min of walking from scratch with the stand-in (16 robots): a smoke
#                                                   run to compare with W5's (not a result)
#         steps/W5b.sh launch [--yes]               show the Kaggle run grad-walk-v2 (T4, 6 h, seed 1: W5's settings and
#                                                   only --air-time-smooth W); --yes starts it.  W=0.5 steps/W5b.sh launch
#         steps/W5b.sh fetch | score                download it when it has ended | score it per command (about 30 min)
# Needs:  for launch: both repositories committed and pushed to GitHub, GPU quota
# Why:    W5 (grad-walk-v1) stood and did not step.  Eric's air_time is the only term that pays for lifting a foot; it
#         counts yes/no events, so its derivative is zero, and the learned term 2 neither learns it nor can express it.
# How:    Full Body rl/walk/task.py air_time_smooth (STAND_INS): per foot unload = 1 - ground force / half the weight,
#         clamped to [0, 1]; paid while the foot has been nearly without weight (under a quarter of the robot) for less
#         than 0.5 s, and only for requests above 0.2 (Eric's limit and gate).  train.py --air-time-smooth W adds it with
#         weight W to what is differentiated; the logged full reward and the scoring stay Eric's.  All else as in W5.
# Then:   (filled in when the step is finished)
set -euo pipefail
. "$(dirname "$0")/common.sh"
NAME=grad-walk-v2
SLUG=mbrl-k1-grad-walk-v2
W=${W:-0.1}

case "${1:-}" in
    check)
        begin
        run "$PY" -m pytest tests/test_grad_train.py tests/test_grad_walk.py tests/test_grad_value.py -q -p no:cacheprovider
        run "$PY" -m pytest "$FULL_BODY/rl/walk/tests" -q -p no:cacheprovider
        finish "passed|failed"
        ;;
    gait)
        STEP="W5b-gait"
        begin
        run "$PY" -m mbrl.grad.probe_step gait --out "$REPO/outputs/steps/W5b"
        finish "^\[gait\]"
        ;;
    lift)
        STEP="W5b-lift"
        begin
        # shellcheck disable=SC2086   (the weights are separate arguments)
        run "$PY" -m mbrl.grad.probe_step lift --weights ${WEIGHTS:-0 0.1} --iterations 40 --threads "${THREADS:-1}" \
            --out "$REPO/outputs/steps/W5b"
        finish "^\[lift\] (after|figure)|^ +(weight|[0-9])"
        ;;
    laptop)
        STEP="W5b-laptop"
        begin
        run "$PY" -m mbrl.grad.train --task walk --robots 16 --steps 20 --minutes 30 --eval-every-min 10 --eval-robots 16 \
            --threads 1 --dtype float64 --seed 1 --no-wandb --air-time-smooth "$W" --run-name "$NAME-laptop-$(date +%m%d-%H%M)"
        finish "stopped|done after|curriculum:|stand-in|\] it +(1|10|20|30|40|50|60|70) "
        ;;
    launch)
        cd "$REPO"
        if [ "${2:-}" = "--yes" ]; then
            KERNEL_SLUG=$SLUG TASK=grad-walk kaggle/run.sh HOURS=${HOURS:-6} RUN_NAME=$NAME TRAIN_ARGS="--seed 1 --air-time-smooth $W"
            echo "started.  When it has ended: steps/W5b.sh fetch && steps/W5b.sh score"
        else
            echo "This would start a ${HOURS:-6} h Kaggle GPU run '$NAME' (walking from scratch, seed 1, stand-in weight $W) on kernel $SLUG."
            echo "Nothing was started.  To start it: steps/W5b.sh launch --yes"
            DRY_RUN=1 KERNEL_SLUG=$SLUG TASK=grad-walk kaggle/run.sh HOURS=${HOURS:-6} RUN_NAME=$NAME \
                TRAIN_ARGS="--seed 1 --air-time-smooth $W" | tail -n 8
        fi
        ;;
    fetch)
        cd "$REPO"
        KERNEL_SLUG=$SLUG TASK=grad-walk kaggle/fetch.sh "$NAME"
        ;;
    score)
        STEP="W5b-score"
        begin
        run "$PY" -m mbrl.grad.score "$NAME"
        finish "best in-run|^ +(command|stand|fwd|back|side|turn)|quiet|MuJoCo|command [+-]|^\|"
        ;;
    *)
        header
        ;;
esac

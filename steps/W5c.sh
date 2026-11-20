#!/bin/bash
# W5c: Does the professor's formula step with a gait clock (two clock inputs and a reward for stepping in time)?
# Runs:   steps/W5c.sh check                        the tests (laptop, 3 min)
#         steps/W5c.sh lift                         the stepping check with the clock's reward instead of the stand-in
#                                                   (laptop, about 25 min per weight and 40 Adam steps; WEIGHTS="2 4"
#                                                   ITERATIONS=100)
#         steps/W5c.sh laptop                       30 min of walking from scratch with the clock (16 robots, W=4): a smoke
#                                                   run to compare with W5's and W5b's (not a result); no in-run evaluation
#         steps/W5c.sh launch [--yes]               show the Kaggle run grad-walk-clock-v1 (T4, 6 h, seed 1: W5's settings
#                                                   and only --gait-clock W, W=4); --yes starts it
#         steps/W5c.sh fetch | score                download it when it has ended | score it per command, on the pH plant
#                                                   and in MuJoCo, with the clock (the walk_clock agent's 77 inputs)
# Needs:  for the twin test, K1_Deploy next to Full Body on its branch walk-clock (else that test is skipped); for
#         launch: both repositories committed and pushed to GitHub, GPU quota
# Why:    W5b gave stepping a slope; the learner spent it on shifting weight, never on a step, even in a quiet world.
#         Walking needs a beat, and the learner cannot find one from Eric's rewards.
# How:    Full Body rl/walk/task.py: a clock that runs while the request asks for a move (|(vx, vy)| + |wz| above 0.2,
#         Eric's air_time gate), walk_v9's own cycle on the pH plant (0.46 s, each foot 36 % of it in the air), 0 while
#         standing (Booster Gym's rule); clock() gives the network sin and cos (77 inputs), the term gait_clock pays each
#         foot for unloading while the clock says swing and for carrying while it says stand (SHAPING: never in Eric's
#         full reward).  mbrl/grad: GradWalk(gait_clock=W), train.py --gait-clock W, probe_step lift --term clock.
#         The firmware: a draft for Eric, K1_Deploy branch walk-clock (WalkClockAgent, the twin's WalkClock).
# Then:   20 Nov 2026, laptop (no tag yet).  lift (4 robots, 0.3 m/s; standing still scores 0.64 in time): clock 2,
#         40 steps: in time 0.74, 59 % off a foot; 100 steps: 0.78, 80 %; clock 4, 100 steps: 0.88, 94 %, a foot up
#         0.12 s in 2 of 4 robots, no fall (the kept unload of W5b at 2: 0.57, 86 %, 0.02 s).  laptop (W=4, 30 min):
#         feet in the air 1.5 % (W5: 1-3 %), the clock's reward 0.037 -> 0.031; no Kaggle run.
set -euo pipefail
. "$(dirname "$0")/common.sh"
NAME=grad-walk-clock-v1
SLUG=mbrl-k1-grad-walk-clock
W=${W:-4}

case "${1:-}" in
    check)
        begin
        run "$PY" -m pytest tests/test_grad_train.py tests/test_grad_walk.py tests/test_grad_value.py -q -p no:cacheprovider
        run "$PY" -m pytest "$FULL_BODY/rl/walk/tests" -q -p no:cacheprovider
        finish "passed|failed"
        ;;
    lift)
        STEP="W5c-lift"
        begin
        # shellcheck disable=SC2086   (the weights are separate arguments)
        run "$PY" -m mbrl.grad.probe_step lift --term clock --weights ${WEIGHTS:-2 4} --iterations "${ITERATIONS:-40}" \
            --threads "${THREADS:-1}" --out "$REPO/outputs/steps/W5c"
        finish "^\[lift\] (after|figure)|^ +(weight|[0-9])"
        ;;
    laptop)
        STEP="W5c-laptop"
        begin
        run "$PY" -m mbrl.grad.train --task walk --robots 16 --steps 20 --minutes 30 --eval-robots 0 --threads 1 \
            --dtype float64 --seed 1 --no-wandb --gait-clock "$W" --run-name "grad-walk-clock-laptop-$(date +%m%d-%H%M)"
        finish "stopped|done after|curriculum:|gait clock|\] it +(1|10|20|30|40|50|60|70) "
        ;;
    launch)
        cd "$REPO"
        if [ "${2:-}" = "--yes" ]; then
            KERNEL_SLUG=$SLUG TASK=grad-walk kaggle/run.sh HOURS=${HOURS:-6} RUN_NAME=$NAME TRAIN_ARGS="--seed 1 --gait-clock $W"
            echo "started.  When it has ended: steps/W5c.sh fetch && steps/W5c.sh score"
        else
            echo "This would start a ${HOURS:-6} h Kaggle GPU run '$NAME' (walking from scratch with the gait clock, seed 1, weight $W) on kernel $SLUG."
            echo "Nothing was started.  To start it: steps/W5c.sh launch --yes"
            DRY_RUN=1 KERNEL_SLUG=$SLUG TASK=grad-walk kaggle/run.sh HOURS=${HOURS:-6} RUN_NAME=$NAME \
                TRAIN_ARGS="--seed 1 --gait-clock $W" | tail -n 8
        fi
        ;;
    fetch)
        cd "$REPO"
        KERNEL_SLUG=$SLUG TASK=grad-walk kaggle/fetch.sh "$NAME"
        ;;
    score)
        STEP="W5c-score"
        begin
        run "$PY" -m mbrl.grad.score "$NAME"
        finish "best in-run|^ +(command|stand|fwd|back|side|turn)|quiet|MuJoCo|command [+-]|^\|"
        ;;
    *)
        header
        ;;
esac

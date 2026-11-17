#!/bin/bash
# W2: Is the walk task the same on both plants, and is the derivative right while the robot walks?
# Runs:   laptop, one CPU core, about 25 min.     steps/W2.sh        (steps/W2.sh quick: windows of 5 and 20 only, 8 min)
# Needs:  k1_sprint/checkpoints/walk_v9.pt beside the repositories: Eric's released walk makes the robot step and
#         checks the environment; it is never part of the method
# How:    1. the task and the C++ environment (Full Body rl/walk/tests): signals against MuJoCo, single reward
#            terms against Eric's formulas, walk_v9 following 0.5 m/s
#         2. the task on the differentiable plant (tests/test_grad_walk.py): both plants pay the same for the same
#            motion; the learner's window, step by step = one graph
#         3. walk_v9 scored per command on the C++ plant
#         4. the derivative against finite differences while walking: windows of 5, 20 and 50 control steps
# Then:   17 Nov 2026 (tag W2): 13 + 8 tests passed; both plants pay the same (every term within 1e-5);
#         walk_v9 on the C++ plant: no falls on all six commands, speed error 0.05 to 0.10 m/s;
#         derivative while walking, worst relative error 3.8e-08 (5 control steps, 1 foot event), 3.1e-07 (20, 3
#         events), 5.6e-07 (50, 8 events); gradient norm 0.12, 0.29, 0.73: it grows with the window, it does not blow up
set -euo pipefail
. "$(dirname "$0")/common.sh"
begin
windows="5 20 50"
[ "${1:-}" = "quick" ] && windows="5 20"
(cd "$FULL_BODY" && run "$PY" -m pytest rl/walk/tests/test_walk_env.py -q -p no:cacheprovider)
run "$PY" -m pytest tests/test_grad_walk.py tests/test_grad.py -q -p no:cacheprovider
(cd "$FULL_BODY" && run "$PY" rl/walk/evaluate.py --policy walk_v9 -B 16 --quiet)
run "$PY" -m mbrl.grad.check --task walk --steps $windows
finish "passed|failed|worst relative error|gradient: finite|foot landings|^ +-?[0-9.]+ +-?[0-9.]+ +-?[0-9.]+ +[0-9.]+%"

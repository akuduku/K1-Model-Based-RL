#!/bin/bash
# W0: Is the derivative through the pH model right when a foot lands inside the window?
# Runs:   laptop, one CPU core, about 5 min.      steps/W0.sh
# Needs:  nothing
# How:    the robot starts 2 cm above the floor and lands after about 64 ms; autograd against finite differences
# Then:   16 Nov 2026 (tag W0): worst relative error 6.6e-08 (5 control steps) and 2.8e-08 (20); gradient norm 9.3 and 14.2
#         (repeat of 17 Nov on one core: 6.4e-08 and 3.3e-07, the same norms; anything below 1e-5 is agreement)
set -euo pipefail
. "$(dirname "$0")/common.sh"
begin
run "$PY" -m mbrl.grad.check --steps 5 20 --drop 0.02
finish "worst relative error|gradient: finite"

#!/bin/bash
# P1: Which physics predicts the real K1 better, MuJoCo or the pH plant, and how sure is that?
# Runs:   laptop, one CPU core: check 2 min, robot 15 min, energy 2 min, view 5 min (then open it in Rerun)
#           steps/P1.sh check | robot | energy | view [--no-camera]
# Needs:  the full-rate robot log MA/Karls Repo/Repo_Eric/rec_kick_walk_20261113_222701.csv with its .k1low (and
#         .k1cam for the camera), K1 Debug beside this repository, the walk policy k1_sprint sprint-repro-001
# How:    robot: from the robot's measured state both engines replay 372 windows of 0.4 s with the logged commands
#         and gains; once with the start used until 20 Nov (as-before), once corrected (flat-feet: ankle offsets
#         that make the soles parallel, the trunk turned onto flat soles in windows that start while the robot
#         stands still); energy: one walk policy on both engines, booked every 1 ms and sampled every 20 ms as on
#         8 Nov; view: the comparison in Rerun (real robot, MuJoCo, pH; outputs/sim2real/<run>/flat-feet/sim2real.rrd)
# Then:   20 Nov 2026, joint error median over the windows [deg], MuJoCo / pH, and pH - MuJoCo (95 % interval):
#           as-before   1.147 / 1.113, -0.078 (-0.088 ... -0.067)   (the 16 Nov numbers exactly)
#           flat-feet   1.080 / 1.051, -0.047 (-0.058 ... -0.036)   still windows 0.762 / 0.749 ("nothing moves" 0.023)
#         the old get-ups (flat-feet): back 5.128 / 5.610, +0.310 (+0.125 ... +0.516); front 6.308 / 6.379, +0.325 (+0.079 ... +0.722)
#         energy at 1 m/s, battery power pH / MuJoCo: booked 177 / 190 W, sampled 129 / 144 W; pH ledger residual -11 %
set -euo pipefail
. "$(dirname "$0")/common.sh"
REC=${REC:-$REPO/../../Karls Repo/Repo_Eric/rec_kick_walk_20261113_222701.csv}
GETUPS=("$REPO/../../Karls Repo/Logs/30102026/laufen-schuss-20261031T132350Z-1-001/laufen-schuss/karl/logs/getup_2236.rec.csv"
        "$REPO/../../Karls Repo/Logs/30102026/laufen-schuss-20261031T132350Z-1-001/laufen-schuss/karl/logs/getup_front_2250.rec.csv")
what=${1:-check}
shift || true
begin
case "$what" in
    check)
        run "$PY" -m pytest -q -p no:cacheprovider tests/test_calib.py tests/test_sim2real.py tests/test_lowlog.py \
            tests/test_replay_mujoco.py tests/test_motor_check.py
        finish "passed|failed" ;;
    robot)
        for start in as-before flat-feet; do
            run "$PY" -m mbrl.sim2real "$REC" "${GETUPS[@]}" --start "$start"
        done
        finish "^\[sim2real\].*start" ;;
    energy)
        run "$PY" -m mbrl.energy_compare --policy sprint-repro-001
        finish "^\[energy\].* 1.0 m/s" ;;
    view)
        RRD="$REPO/outputs/sim2real/$(basename "$REC" .csv)/flat-feet/sim2real.rrd"
        run "$PY" -m mbrl.sim2real "$REC" --start flat-feet --save "$RRD" "$@"
        echo "open it:  $(dirname "$PY")/rerun \"$RRD\""
        finish "^\[sim2real\]" ;;
    *)
        echo "steps/P1.sh check | robot | energy | view [--no-camera]"; exit 2 ;;
esac

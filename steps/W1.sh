#!/bin/bash
# W1: Does the professor's formula learn to stand from a random network, and how good is the stand?
# Runs:   training on Kaggle (T4, 6 h, started by the user); fetch and score on the laptop (about 8 min)
#           steps/W1.sh score grad-stand-v1c         score a fetched run again (the best in-run checkpoint)
#           steps/W1.sh launch 2                     show the Kaggle launch for seed 2; add --yes to start it
#           steps/W1.sh fetch grad-stand-s2          wait for the run and download it
# Needs:  for launch: everything committed and pushed to GitHub, GPU quota for 6 h
# Then:   17 Nov 2026 (tag W1-seed1), grad-stand-v1c, seed 1, 1024 robots: update 80 (1.64 M steps, 2.9 h):
#           4 of 384 fell (1.0 %), 2.3 W with pushes; quiet 0 %, 1.1 W; stands 20 s in MuJoCo (tilt 0.7 / 1.6 deg)
set -euo pipefail
. "$(dirname "$0")/common.sh"

# one kernel per seed, so runs do not overwrite each other's output: grad-stand-s2 -> kernel mbrl-k1-grad-stand-s2
slug() { case "$1" in grad-stand-s[0-9]*) echo "mbrl-k1-$1" ;; *) echo "mbrl-k1-grad-stand" ;; esac; }

case "${1:-}" in
    score)
        [ -n "${2:-}" ] || { echo "steps/W1.sh score <run name> [--update N]" >&2; exit 1; }
        STEP="W1-score-$2"
        begin
        run "$PY" -m mbrl.grad.score "${@:2}"
        finish "of [0-9]+ fell|pH plant, quiet|MuJoCo|best in-run"
        ;;
    launch)
        seed=${2:?steps/W1.sh launch <seed> [--yes]}
        name="grad-stand-s$seed"
        cd "$REPO"
        if [ "${3:-}" = "--yes" ]; then
            KERNEL_SLUG=$(slug "$name") TASK=grad-stand kaggle/run.sh HOURS=${HOURS:-6} RUN_NAME="$name" TRAIN_ARGS="--seed $seed"
            echo "started.  When it has ended (about ${HOURS:-6} h): steps/W1.sh fetch $name && steps/W1.sh score $name"
        else
            echo "This would start a ${HOURS:-6} h Kaggle GPU run '$name' (seed $seed) on kernel $(slug "$name")."
            echo "Nothing was started.  To start it: steps/W1.sh launch $seed --yes"
            DRY_RUN=1 KERNEL_SLUG=$(slug "$name") TASK=grad-stand kaggle/run.sh HOURS=${HOURS:-6} RUN_NAME="$name" TRAIN_ARGS="--seed $seed" | tail -5
        fi
        ;;
    fetch)
        name=${2:?steps/W1.sh fetch <run name>}
        cd "$REPO"
        KERNEL_SLUG=$(slug "$name") TASK=grad-stand kaggle/fetch.sh "$name"
        ;;
    *)
        header
        ;;
esac

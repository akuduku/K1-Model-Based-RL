# Sourced by the Kaggle helpers: paths, the Kaggle CLI, the user name and the kernel id.
KIT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$KIT_DIR/.." && pwd)
KAGGLE=${KAGGLE:-kaggle}   # e.g. KAGGLE=/path/to/venv/bin/kaggle
command -v "$KAGGLE" > /dev/null || { echo "kaggle CLI not found (pip install kaggle, or set KAGGLE=...)" >&2; exit 1; }
KAGGLE_USER=${KAGGLE_USER:-$("$KAGGLE" config view | sed -n 's/^- username: //p')}
[ -n "$KAGGLE_USER" ] || { echo "no Kaggle user name (set KAGGLE_USER)" >&2; exit 1; }
# TASK=stand (default) / getup: train that skill on the fitted model; one kernel per task
# TASK=getup-plan: the model-based get-up planner (mbrl.getup.plan); TASK=getup-track: learn to track its plans;
# TASK=getup-mbrl: both in one session
TASK=${TASK:-stand}
case "$TASK" in
    stand|getup|getup-plan|getup-track|getup-mbrl) SCRIPT=train_mb.py; DEFAULT_SLUG=mbrl-k1-$TASK; DEFAULT_TIMEOUT=39600 ;;   # 11 h = SESSION_HOURS
    *) echo "TASK=$TASK: use stand, getup, getup-plan, getup-track or getup-mbrl" >&2; exit 1 ;;
esac
KERNEL_SLUG=${KERNEL_SLUG:-$DEFAULT_SLUG}
KERNEL="$KAGGLE_USER/$KERNEL_SLUG"

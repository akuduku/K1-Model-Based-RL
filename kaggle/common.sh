# Sourced by the Kaggle helpers: paths, the Kaggle CLI, the user name and the kernel id.
KIT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$KIT_DIR/.." && pwd)
KAGGLE=${KAGGLE:-kaggle}   # e.g. KAGGLE=/path/to/venv/bin/kaggle
command -v "$KAGGLE" > /dev/null || { echo "kaggle CLI not found (pip install kaggle, or set KAGGLE=...)" >&2; exit 1; }
KAGGLE_USER=${KAGGLE_USER:-$("$KAGGLE" config view | sed -n 's/^- username: //p')}
[ -n "$KAGGLE_USER" ] || { echo "no Kaggle user name (set KAGGLE_USER)" >&2; exit 1; }
# TASK=stand (default) / getup: train that skill on the fitted model; one kernel per task
TASK=${TASK:-stand}
case "$TASK" in
    stand|getup) SCRIPT=train_mb.py; DEFAULT_SLUG=mbrl-k1-$TASK; DEFAULT_TIMEOUT=43200 ;;   # Kaggle's 12 h cap
    *) echo "TASK=$TASK: use stand or getup" >&2; exit 1 ;;
esac
KERNEL_SLUG=${KERNEL_SLUG:-$DEFAULT_SLUG}
KERNEL="$KAGGLE_USER/$KERNEL_SLUG"

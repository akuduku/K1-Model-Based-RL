# Sourced by every step script: where things are, one CPU thread, a log of every repeat.
STEPS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$STEPS_DIR/.." && pwd)
FULL_BODY=${PH_K1_ROOT:-$REPO/../Full Body}
[ -d "$FULL_BODY/ph_k1" ] || FULL_BODY=$REPO/external/full-body
PY=${PYTHON:-$HOME/Dokumente/Robotics/venv/bin/python}
[ -x "$PY" ] || PY=python3
export OMP_NUM_THREADS=${THREADS:-1} MKL_NUM_THREADS=${THREADS:-1}   # the laptop: one job, one core
STEP=$(basename "$0" .sh)

# the comment block at the top of a step script: its question, where it runs, what it needs, the result we got
header() { sed -n '2,${/^#/!q;s/^# \{0,1\}//;p}' "${1:-$0}"; }

begin() {
    cd "$REPO"
    LOG="$REPO/outputs/steps/$STEP/$(date +%Y%m%d-%H%M)_$(git rev-parse --short HEAD 2>/dev/null || echo nogit).log"
    mkdir -p "$(dirname "$LOG")"
    { header; echo "Code:   Model Based RL $(git describe --tags --always --dirty 2>/dev/null), Full Body $(git -C "$FULL_BODY" describe --tags --always --dirty 2>/dev/null)"; echo; } | tee "$LOG"
    HEADER_LINES=$(wc -l < "$LOG")
}

# run a command, show it, keep its output in the log
run() {
    echo "+ $*" | tee -a "$LOG"
    "$@" 2>&1 | tee -a "$LOG"
    return "${PIPESTATUS[0]}"
}

# the lines of this repeat that carry the result (a grep pattern), beside the result we got then
finish() {
    { echo; echo "Now:"; tail -n +$((HEADER_LINES + 1)) "$LOG" | grep -E "$1" | grep -v '^+ ' | sed 's/^ */    /'; } > "$LOG.now" || true
    tee -a "$LOG" < "$LOG.now"; rm -f "$LOG.now"
    header | sed -n '/^Then:/,$p'
    echo "log: $LOG"
}

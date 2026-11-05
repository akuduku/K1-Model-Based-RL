#!/bin/bash
# Wait for the kernel to finish, download its output to outputs/kaggle/<name>/ and print the summary.
#   TASK=stand kaggle/fetch.sh [name]     (the stand training on the fitted model; name defaults to a timestamp)
#   TASK=getup kaggle/fetch.sh [name]
set -euo pipefail
. "$(dirname "$0")/common.sh"
OUT="$REPO/outputs/kaggle/${1:-$(date +%Y%m%d-%H%M)}"
while true; do
    status=$("$KAGGLE" kernels status "$KERNEL" 2>&1 | tail -1)
    echo "$(date +%H:%M:%S) $status"
    case "$status" in
        *[Rr]unning*|*RUNNING*|*[Qq]ueued*|*QUEUED*) sleep "${POLL:-60}" ;;
        *) break ;;
    esac
done
mkdir -p "$OUT"
"$KAGGLE" kernels output "$KERNEL" -p "$OUT" -o -q
echo "downloaded to $OUT"
[ -f "$OUT/summary.txt" ] && cat "$OUT/summary.txt"
case "$status" in *[Cc]omplete*|*COMPLETE*) ;; *) echo "kernel ended with: $status" >&2; exit 1 ;; esac

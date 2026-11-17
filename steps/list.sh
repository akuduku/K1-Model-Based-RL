#!/bin/bash
# Every step: its question, where it runs, what it needs, the result we got.   steps/list.sh
. "$(dirname "$0")/common.sh"
for f in "$STEPS_DIR"/W*.sh; do
    echo "== steps/$(basename "$f")"
    header "$f" | sed 's/^/   /'
    echo
done

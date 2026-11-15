#!/bin/bash
# All tests of this repository on the CPU (about a minute).  Full Body is found by mbrl/paths.py.
#   scripts/test_all.sh            PYTHON=/path/to/python scripts/test_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-$HOME/Dokumente/Robotics/venv/bin/python}
[ -x "$PY" ] || PY=python3
"$PY" -m pytest tests -q -p no:cacheprovider "$@"

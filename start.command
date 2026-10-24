#!/usr/bin/env bash
# Cortex launcher for macOS / Linux. Double-click (rename to start.command on
# macOS to open in Terminal) or run: ./start.sh
cd "$(dirname "$0")" || exit 1
PY="$(command -v python3 || command -v python)"
if [ -z "$PY" ]; then
  echo "Python 3 not found. Install it from https://python.org (or: brew install python / apt install python3)."
  read -r -p "Press Enter to close." _
  exit 1
fi
exec "$PY" run.py "$@"

#!/usr/bin/env bash
# Cortex uninstaller for macOS / Linux.
cd "$(dirname "$0")" || exit 1
PY="$(command -v python3 || command -v python)"
if [ -z "$PY" ]; then echo "Python 3 not found."; read -r -p "Press Enter to close." _; exit 1; fi
exec "$PY" uninstall.py "$@"

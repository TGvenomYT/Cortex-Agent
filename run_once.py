#!/usr/bin/env python3
"""
run_once.py — thin root shim for the scheduled job runner.

The real implementation lives in cortex/server/run_once.py. This shim is kept at
the project root because the OS scheduler and the tick invoke this path. It just
puts the project root on sys.path and delegates.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# tick and alarm run all day, so they skip the imports the real jobs need.
if sys.argv[1:2] == ["tick"]:
    from cortex.core.tick import main
elif sys.argv[1:2] == ["alarm"]:
    from cortex.core.alarm import main
else:
    from cortex.server.run_once import main

if __name__ == "__main__":
    sys.exit(main())

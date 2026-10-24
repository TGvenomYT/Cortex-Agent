#!/usr/bin/env python3
"""
run_once.py — thin root shim for the scheduled job runner.

The real implementation lives in cortex/server/run_once.py. This shim is kept at
the project root so the installed launchd jobs (which invoke this path) keep
working after the codebase was organized into the cortex/ package. It just puts
the project root on sys.path and delegates.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cortex.server.run_once import main

if __name__ == "__main__":
    sys.exit(main())

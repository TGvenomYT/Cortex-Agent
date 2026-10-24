"""Cortex — on-device AI assistant. Application package."""
import os

# Absolute path to the project root (one level above this package), used by
# modules that need to reach config.yaml, assets/, or the run_once shim
# regardless of the current working directory.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

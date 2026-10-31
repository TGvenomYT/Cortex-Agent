"""
viz.py — render agent-generated matplotlib code into a PNG (the visualization
canvas). The model writes plotting code (plt/np/pd available) to visualize math
or data; we run it in an isolated subprocess with a timeout and save the figure.

The code comes from a model, and a model can be steered by anything it has read, so it
is treated as hostile: `check_code()` only passes code that sticks to plotting (no
imports beyond maths, no files, no network, no reaching through dunders), the subprocess
gets no secrets, and on macOS it also runs under a sandbox profile with no network and
no writes outside its scratch folder. `-I` alone is not a sandbox.
"""
from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
import tempfile

_HARNESS_HEAD = (
    "import matplotlib\n"
    "matplotlib.use('Agg')\n"
    "import matplotlib.pyplot as plt\n"
    "import numpy as np\n"
    "try:\n import pandas as pd\nexcept Exception:\n pd=None\n"
    "plt.rcParams.update({'figure.facecolor':'white','savefig.facecolor':'white','font.size':11})\n"
)

_OK_IMPORTS = {"math", "cmath", "statistics", "random", "datetime", "calendar", "itertools", "functools",
               "collections", "fractions", "decimal", "numpy", "pandas", "matplotlib", "seaborn", "scipy"}
_BAD_NAMES = {"open", "exec", "eval", "compile", "__import__", "getattr", "setattr", "delattr", "globals",
              "locals", "vars", "input", "breakpoint", "memoryview", "help", "exit", "quit", "os", "sys",
              "subprocess", "socket", "shutil", "pathlib", "importlib", "ctypes", "pickle", "marshal"}
# numpy/pandas/matplotlib calls that touch the disk or network.
_BAD_ATTRS = {"savefig", "read_csv", "read_json", "read_html", "read_excel", "read_pickle", "read_parquet",
              "read_sql", "read_table", "read_fwf", "read_xml", "read_clipboard", "read_feather",
              "read_hdf", "read_orc", "read_sas", "read_spss", "read_stata", "to_csv", "to_json",
              "to_html", "to_excel", "to_pickle", "to_parquet", "to_sql", "to_hdf", "to_feather",
              "to_clipboard", "to_xml", "to_latex", "to_stata", "tofile", "load", "save", "savez",
              "savez_compressed", "loadtxt", "savetxt", "genfromtxt", "fromfile", "memmap", "DataSource"}


def check_code(code: str) -> tuple[bool, str]:
    """(ok, why_not). Passes code that only computes and plots."""
    try:
        tree = ast.parse(code or "")
    except SyntaxError as e:
        return True, f"syntax error: {e}"       # it can't run, so it can't do harm
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in _OK_IMPORTS:
                    return False, f"it imports {a.name}"
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in _OK_IMPORTS:
                return False, f"it imports {node.module}"
        elif isinstance(node, ast.Name):
            if node.id in _BAD_NAMES or node.id.startswith("__"):
                return False, f"it uses {node.id}"
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                return False, f"it reaches into {node.attr}"
            if node.attr in _BAD_ATTRS:
                return False, f"it calls {node.attr}"
    return True, ""


def clean_env(base: dict | None = None) -> dict:
    """The environment for anything Cortex runs: everything except its own secrets."""
    env = dict(base if base is not None else os.environ)
    for k in list(env):
        if k == "CORTEX_DB_PASSWORD" or k.endswith(("_API_KEY", "_TOKEN", "_SECRET", "_PASSWORD")):
            env.pop(k, None)
    return env


def _sandbox_prefix(scratch: str, out_path: str) -> list[str]:
    """macOS: no network, and writes only to the scratch folder + the output PNG."""
    sb = shutil.which("sandbox-exec") or ("/usr/bin/sandbox-exec" if os.path.exists("/usr/bin/sandbox-exec") else None)
    if not sb:
        return []
    real = os.path.realpath(scratch)
    out_dir = os.path.realpath(os.path.dirname(out_path) or ".")
    profile = (
        "(version 1)(allow default)(deny network*)(deny file-write*)"
        f'(allow file-write* (subpath "{real}") (subpath "{out_dir}") (literal "/dev/null"))'
    )
    return [sb, "-p", profile]


def render_matplotlib(code: str, out_path: str, timeout: int = 30) -> tuple[bool, str]:
    script = (_HARNESS_HEAD + "\n" + (code or "") + "\n"
              + "if plt.get_fignums():\n"
              + "    plt.savefig(" + repr(out_path) + ", dpi=130, bbox_inches='tight')\n")
    scratch = tempfile.mkdtemp(prefix="cortex-chart-")
    path = os.path.join(scratch, "chart.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(script)
    env = clean_env()
    env["MPLCONFIGDIR"] = scratch
    env["HOME"] = scratch
    try:
        r = subprocess.run(_sandbox_prefix(scratch, out_path) + [sys.executable, "-I", path],
                           capture_output=True, text=True, timeout=timeout, env=env, cwd=scratch)
    except subprocess.TimeoutExpired:
        return False, "chart code timed out"
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return True, ""
    return False, (r.stderr or "no figure produced").strip()[-400:]

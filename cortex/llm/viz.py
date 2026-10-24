"""
viz.py — render agent-generated matplotlib code into a PNG (the visualization
canvas). The model writes plotting code (plt/np/pd available) to visualize math
or data; we run it in an isolated subprocess with a timeout and save the figure.

Local, single-user tool — the code runs as the user, but in a separate `-I`
(isolated) interpreter with a hard timeout so it can't hang the app.
"""
from __future__ import annotations

import os
import sys
import tempfile
import subprocess

_HARNESS_HEAD = (
    "import matplotlib\n"
    "matplotlib.use('Agg')\n"
    "import matplotlib.pyplot as plt\n"
    "import numpy as np\n"
    "try:\n import pandas as pd\nexcept Exception:\n pd=None\n"
    "plt.rcParams.update({'figure.facecolor':'white','savefig.facecolor':'white','font.size':11})\n"
)


def render_matplotlib(code: str, out_path: str, timeout: int = 30) -> tuple[bool, str]:
    script = (_HARNESS_HEAD + "\n" + (code or "") + "\n"
              + "if plt.get_fignums():\n"
              + "    plt.savefig(" + repr(out_path) + ", dpi=130, bbox_inches='tight')\n")
    tmp = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
    tmp.write(script)
    tmp.close()
    try:
        r = subprocess.run([sys.executable, "-I", tmp.name],
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "chart code timed out"
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return True, ""
    return False, (r.stderr or "no figure produced").strip()[-400:]

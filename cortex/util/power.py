"""
power.py — battery / thermal gating for scheduled Cortex jobs.

In the scheduled (wake-run-exit) model, every launchd job calls should_run()
at the top to decide whether this invocation is worth the energy. The goal is
"100% battery friendly": heavy, skippable work (the autonomy LLM session,
briefing generation) backs off or skips entirely when the Mac is on battery
and low, asleep-ish, or thermally throttled. Cheap work (reminder checks) can
pass a low `cost` and run almost always.

Pure standard library + `pmset`/`sysctl`; no installs.
"""
from __future__ import annotations

import subprocess
import logging

logger = logging.getLogger("cortex.power")


def _run(cmd: list[str], timeout: float = 4.0) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return out.stdout or ""
    except Exception:
        return ""


def battery_state() -> dict:
    """
    Returns {"on_ac": bool, "percent": int|None, "charging": bool}.
    Falls back to on_ac=True (desktop/unknown) so we never over-throttle a
    machine that has no battery.
    """
    txt = _run(["pmset", "-g", "batt"])
    if not txt:
        return {"on_ac": True, "percent": None, "charging": False}
    low = txt.lower()
    on_ac = "ac power" in low
    charging = "charging" in low and "discharging" not in low
    percent = None
    for tok in txt.replace(";", " ").split():
        if tok.endswith("%"):
            try:
                percent = int(tok[:-1])
            except ValueError:
                pass
            break
    return {"on_ac": on_ac, "percent": percent, "charging": charging}


def thermal_pressure() -> str:
    """'nominal' | 'fair' | 'serious' | 'critical' | 'unknown'."""
    txt = _run(["pmset", "-g", "therm"]).lower()
    if "critical" in txt:
        return "critical"
    if "serious" in txt:
        return "serious"
    if "fair" in txt:
        return "fair"
    if txt:
        return "nominal"
    # sysctl fallback (Apple silicon)
    s = _run(["sysctl", "-n", "machdep.xcpm.cpu_thermal_level"]).strip()
    try:
        lvl = int(s)
        return "nominal" if lvl == 0 else ("fair" if lvl < 40 else "serious")
    except ValueError:
        return "unknown"


def should_run(cost: str = "heavy", min_battery: int = 30) -> tuple[bool, str]:
    """
    Decide whether a job of the given cost should run this tick.

    cost="cheap"  → runs unless the machine is thermally critical.
    cost="heavy"  → skips when on battery AND below `min_battery`, or when
                    thermal pressure is serious/critical.

    Returns (run: bool, reason: str). The reason is logged by the caller.
    """
    therm = thermal_pressure()
    if therm == "critical":
        return False, "thermal pressure critical"

    if cost == "cheap":
        return True, f"cheap job (thermal={therm})"

    batt = battery_state()
    if therm == "serious":
        return False, "thermal pressure serious"

    if not batt["on_ac"]:
        pct = batt["percent"]
        if pct is not None and pct < min_battery:
            return False, f"on battery and low ({pct}% < {min_battery}%)"
        return True, f"on battery but ok ({pct}%)"

    return True, f"on AC (thermal={therm})"


if __name__ == "__main__":
    # Quick manual check: `python3 power.py`
    print("battery:", battery_state())
    print("thermal:", thermal_pressure())
    print("heavy should_run:", should_run("heavy"))
    print("cheap should_run:", should_run("cheap"))

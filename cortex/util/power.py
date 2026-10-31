"""
power.py — battery / thermal gating for background Cortex work.

The tick (cortex/core/tick.py) asks here how the machine is powered before it runs
anything. Heavy, skippable work (the autonomy LLM session, briefing generation) backs
off or skips entirely when the machine is on battery and low, or thermally throttled.
Cheap work can pass a low `cost` and run almost always.

Pure standard library + `pmset`/`sysctl`; no installs. The battery reading works on
macOS, Linux and Windows; thermal and memory readings are macOS only.
"""
from __future__ import annotations

import os
import sys
import time
import subprocess
import logging

logger = logging.getLogger("cortex.power")


def _run(cmd: list[str], timeout: float = 4.0) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return out.stdout or ""
    except Exception:
        return ""


_NO_BATTERY = {"on_ac": True, "percent": None, "charging": False, "has_battery": False}
_SYS_POWER = "/sys/class/power_supply"


def battery_state() -> dict:
    """
    Returns {"on_ac": bool, "percent": int|None, "charging": bool, "has_battery": bool}.
    Falls back to on_ac=True and no battery (desktop/unknown) so we never over-throttle
    a machine that has no battery.
    """
    if sys.platform.startswith("linux"):
        return _linux_battery()
    if os.name == "nt":
        return _windows_battery()
    txt = _run(["pmset", "-g", "batt"])
    if not txt:
        return dict(_NO_BATTERY)
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
    return {"on_ac": on_ac, "percent": percent, "charging": charging, "has_battery": "internalbattery" in low}


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""


def _linux_battery() -> dict:
    out = dict(_NO_BATTERY)
    try:
        names = sorted(os.listdir(_SYS_POWER))
    except OSError:
        return out
    for name in names:
        d = os.path.join(_SYS_POWER, name)
        # scope=Device is a wireless mouse or headset, not what's powering this machine
        if _read(f"{d}/type").lower() != "battery" or _read(f"{d}/scope").lower() == "device":
            continue
        status = _read(f"{d}/status").lower()
        out["has_battery"] = True
        out["on_ac"] = out["on_ac"] and status != "discharging"
        out["charging"] = out["charging"] or status == "charging"
        if out["percent"] is None and _read(f"{d}/capacity").isdigit():
            out["percent"] = int(_read(f"{d}/capacity"))
    return out


def _windows_reading(ac: int, flag: int, percent: int) -> dict:
    if flag & 128:      # 128 = no system battery, 255 = can't tell
        return dict(_NO_BATTERY)
    return {"on_ac": ac != 0, "percent": percent if percent <= 100 else None,
            "charging": bool(flag & 8), "has_battery": True}


def _windows_battery() -> dict:
    import ctypes

    class Status(ctypes.Structure):     # SYSTEM_POWER_STATUS
        _fields_ = [("ac", ctypes.c_ubyte), ("flag", ctypes.c_ubyte), ("percent", ctypes.c_ubyte),
                    ("saver", ctypes.c_ubyte), ("life", ctypes.c_ulong), ("full", ctypes.c_ulong)]
    s = Status()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s)):
        return dict(_NO_BATTERY)
    return _windows_reading(s.ac, s.flag, s.percent)


_paced = {"checked": None, "battery": False}


def pace(seconds: float, on_battery: float = 30) -> float:
    """How long a waiting loop should sleep. Waking the CPU often is what stops it resting,
    so on battery loops wait longer. The check itself runs at most every 5 minutes."""
    now = time.monotonic()
    if _paced["checked"] is None or now - _paced["checked"] > 300:
        _paced.update(checked=now, battery=not battery_state()["on_ac"])
    return max(seconds, on_battery) if _paced["battery"] else seconds


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


def free_memory_pct() -> int | None:
    """% of RAM macOS considers free/reclaimable (kern.memorystatus_level); None elsewhere."""
    out = _run(["sysctl", "-n", "kern.memorystatus_level"]).strip()
    try:
        return int(out)
    except ValueError:
        return None


def snapshot() -> dict:
    """One-glance machine state used to decide how much work Cortex should take on."""
    b = battery_state()
    return {"on_ac": b["on_ac"], "battery": b["percent"], "thermal": thermal_pressure(),
            "free_mem_pct": free_memory_pct()}


def is_constrained(snap: dict | None = None) -> tuple[bool, str]:
    """True when the Mac is short on power/thermal/memory headroom — the moment to
    skip optional AI work (follow-up suggestions, parallel helpers)."""
    snap = snap or snapshot()
    if snap["thermal"] in ("serious", "critical"):
        return True, f"thermal pressure {snap['thermal']}"
    if not snap["on_ac"] and snap["battery"] is not None and snap["battery"] < 25:
        return True, f"battery low ({snap['battery']}%)"
    if snap["free_mem_pct"] is not None and snap["free_mem_pct"] < 15:
        return True, f"memory tight ({snap['free_mem_pct']}% free)"
    return False, "ok"


def ai_jobs_only_on_ac() -> bool:
    try:
        import cortex.storage.settings_store as settings_store
        return bool(settings_store.section("resources").get("ai_jobs_only_on_ac", True))
    except Exception:
        return True


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
        if ai_jobs_only_on_ac():
            return False, f"on battery ({pct}%) — AI jobs only run when plugged in (Settings → Background)"
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

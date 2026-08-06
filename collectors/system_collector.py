"""
System Collector — runs every 5 minutes.
Captures disk, RAM, CPU, uptime, network interfaces, battery, and macOS version.
"""

import os
import sys
import re
import logging

sys.path.insert(0, os.path.dirname(__file__))
from base import load_config, get_db_conn, upsert_context, run_cmd

logger = logging.getLogger("collector.system")


def collect_disk(conn):
    """Disk usage for all mounted volumes."""
    out, _, _ = run_cmd("df -h | grep -v tmpfs | grep -v devfs | awk 'NR>1 {print $1,$2,$3,$4,$5,$6}'")
    disks = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 6:
            disks.append({
                "filesystem": parts[0],
                "size": parts[1],
                "used": parts[2],
                "available": parts[3],
                "use_pct": parts[4],
                "mount": parts[5],
            })

    # Main disk summary
    main_disk = next((d for d in disks if d["mount"] == "/"), None)

    upsert_context(conn, "system", "disk", {
        "main": main_disk,
        "all_volumes": disks,
    }, expires_minutes=10)
    if main_disk:
        logger.info(f"Disk: {main_disk['used']}/{main_disk['size']} ({main_disk['use_pct']})")


def collect_memory(conn):
    """RAM usage via vm_stat (macOS native)."""
    out, _, _ = run_cmd("vm_stat")
    page_size = 16384  # macOS default 16KB pages on Apple Silicon, 4096 on Intel
    # Try to parse page size from output
    ps_match = re.search(r"page size of (\d+) bytes", out)
    if ps_match:
        page_size = int(ps_match.group(1))

    stats = {}
    for line in out.splitlines():
        m = re.match(r"(.+?):\s+(\d+)", line)
        if m:
            stats[m.group(1).strip()] = int(m.group(2))

    pages_free = stats.get("Pages free", 0)
    pages_active = stats.get("Pages active", 0)
    pages_wired = stats.get("Pages wired down", 0)
    pages_compressed = stats.get("Pages occupied by compressor", 0)

    to_mb = lambda pages: round(pages * page_size / (1024 ** 2), 1)

    # Total physical RAM
    total_out, _, _ = run_cmd("sysctl -n hw.memsize")
    total_mb = round(int(total_out.strip()) / (1024 ** 2), 1) if total_out.strip().isdigit() else 0

    used_mb = to_mb(pages_active + pages_wired + pages_compressed)
    free_mb = to_mb(pages_free)

    upsert_context(conn, "system", "memory", {
        "total_mb": total_mb,
        "used_mb": used_mb,
        "free_mb": free_mb,
        "use_pct": round((used_mb / total_mb * 100), 1) if total_mb else 0,
        "wired_mb": to_mb(pages_wired),
        "compressed_mb": to_mb(pages_compressed),
    }, expires_minutes=10)
    logger.info(f"Memory: {used_mb}MB / {total_mb}MB used")


def collect_cpu(conn):
    """CPU info and current load."""
    # CPU model
    cpu_model, _, _ = run_cmd("sysctl -n machdep.cpu.brand_string 2>/dev/null || sysctl -n hw.model")
    # Core count
    cores, _, _ = run_cmd("sysctl -n hw.logicalcpu")
    # Load averages
    load_out, _, _ = run_cmd("sysctl -n vm.loadavg")
    # Parse: { 0.12 0.15 0.18 }
    load_vals = re.findall(r"[\d.]+", load_out)
    load_1 = float(load_vals[0]) if len(load_vals) > 0 else 0
    load_5 = float(load_vals[1]) if len(load_vals) > 1 else 0
    load_15 = float(load_vals[2]) if len(load_vals) > 2 else 0

    upsert_context(conn, "system", "cpu", {
        "model": cpu_model.strip(),
        "logical_cores": int(cores.strip()) if cores.strip().isdigit() else 0,
        "load_1min": load_1,
        "load_5min": load_5,
        "load_15min": load_15,
        "load_pct_estimate": round(load_1 / max(int(cores.strip() or 1), 1) * 100, 1),
    }, expires_minutes=10)
    logger.info(f"CPU: load={load_1} ({cores.strip()} cores)")


def collect_uptime(conn):
    """System uptime and boot time."""
    uptime_out, _, _ = run_cmd("uptime")
    boot_out, _, _ = run_cmd("sysctl -n kern.boottime")
    # Parse boot time: { sec = 1234567890, usec = 0 }
    boot_match = re.search(r"sec = (\d+)", boot_out)
    boot_sec = int(boot_match.group(1)) if boot_match else 0

    # Parse uptime string — just store it as-is plus the raw number
    sysctl_up, _, _ = run_cmd("sysctl -n kern.boottime")

    upsert_context(conn, "system", "uptime", {
        "uptime_string": uptime_out.strip(),
        "boot_epoch": boot_sec,
    }, expires_minutes=10)
    logger.info(f"Uptime: {uptime_out.strip()[:60]}")


def collect_network(conn):
    """Network interfaces and connectivity."""
    # Active interfaces with IPs
    ifconfig_out, _, _ = run_cmd(
        "ifconfig | grep -E '^[a-z]|inet ' | grep -v '127.0.0.1' | grep -v '::1'"
    )

    interfaces = []
    current_iface = None
    for line in ifconfig_out.splitlines():
        if not line.startswith(" ") and not line.startswith("\t"):
            current_iface = line.split(":")[0]
        elif "inet " in line and current_iface:
            parts = line.strip().split()
            ip = parts[1] if len(parts) > 1 else ""
            if ip and not ip.startswith("127.") and not ip.startswith("169.254."):
                interfaces.append({"interface": current_iface, "ip": ip})

    # DNS check — quick ping
    ping_out, _, ping_rc = run_cmd("ping -c1 -W1 8.8.8.8 2>/dev/null | tail -1")
    connected = ping_rc == 0

    # External IP
    ext_ip, _, _ = run_cmd("curl -s --max-time 3 https://api.ipify.org 2>/dev/null")

    upsert_context(conn, "system", "network", {
        "interfaces": interfaces,
        "internet_connected": connected,
        "external_ip": ext_ip.strip() if ext_ip else "",
    }, expires_minutes=10)
    logger.info(f"Network: {len(interfaces)} interfaces, connected={connected}")


def collect_macos_info(conn):
    """macOS version and machine info."""
    sw_vers, _, _ = run_cmd("sw_vers")
    hostname, _, _ = run_cmd("hostname")
    hw_model, _, _ = run_cmd("sysctl -n hw.model")
    username, _, _ = run_cmd("whoami")

    info = {}
    for line in sw_vers.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            info[k.strip()] = v.strip()

    upsert_context(conn, "system", "macos_info", {
        "os_version": info.get("ProductVersion", ""),
        "os_build": info.get("BuildVersion", ""),
        "os_name": info.get("ProductName", "macOS"),
        "hostname": hostname.strip(),
        "hw_model": hw_model.strip(),
        "username": username.strip(),
    }, expires_minutes=60)  # rarely changes
    logger.info(f"macOS: {info.get('ProductVersion', '?')} on {hw_model.strip()}")


def main():
    config = load_config()
    conn = get_db_conn(config)

    collect_disk(conn)
    collect_memory(conn)
    collect_cpu(conn)
    collect_uptime(conn)
    collect_network(conn)
    collect_macos_info(conn)

    conn.close()
    logger.info("System collection done")


if __name__ == "__main__":
    main()

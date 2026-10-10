"""
Remote Server Collector — Collects system state from the Linux server via SSH.
Runs every 2 minutes. Stores in context_store with category 'server'.
"""

import os
import sys
import json
import subprocess
import logging

sys.path.insert(0, os.path.dirname(__file__))
from base import load_config, get_db_conn, upsert_context

logger = logging.getLogger("collector.remote")

SSH_CMD = 'ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -i ~/.ssh/id_ed25519_myserver rathinaswamy@100.100.35.38'


def _ssh(command: str, timeout: int = 10) -> str:
    """Run a command on the remote server via SSH."""
    try:
        result = subprocess.run(
            f'{SSH_CMD} "{command}"',
            shell=True, capture_output=True, text=True, timeout=timeout
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return ""
    except Exception as e:
        logger.error(f"SSH failed: {e}")
        return ""


def collect_server_cpu(conn):
    """Remote CPU info and load."""
    load = _ssh("cat /proc/loadavg")
    cpu_info = _ssh("nproc")
    cpu_model = _ssh("grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2")
    cpu_usage = _ssh("top -bn1 | grep 'Cpu(s)' | awk '{print $2}' | tr -d '%us,'")

    if load:
        parts = load.split()
        cores = int(cpu_info) if cpu_info.isdigit() else 1
        load_1 = float(parts[0]) if parts else 0

        try:
            usage = float(cpu_usage) if cpu_usage else round(load_1 / cores * 100, 1)
        except (ValueError, TypeError):
            usage = round(load_1 / cores * 100, 1)

        upsert_context(conn, "server", "cpu", {
            "model": cpu_model.strip() if cpu_model else "",
            "cores": cores,
            "load_1min": load_1,
            "load_5min": float(parts[1]) if len(parts) > 1 else 0,
            "load_15min": float(parts[2]) if len(parts) > 2 else 0,
            "usage_pct": usage,
        }, expires_minutes=5)
        logger.info(f"Server CPU: load={load_1}, cores={cores}")


def collect_server_memory(conn):
    """Remote memory usage."""
    mem = _ssh("free -b | grep Mem")
    if not mem:
        return

    parts = mem.split()
    if len(parts) >= 4:
        total = int(parts[1])
        used = int(parts[2])
        available = int(parts[6]) if len(parts) > 6 else total - used

        total_gb = round(total / (1024**3), 1)
        used_gb = round(used / (1024**3), 1)
        avail_gb = round(available / (1024**3), 1)
        pct = round(used / total * 100, 1) if total > 0 else 0

        upsert_context(conn, "server", "memory", {
            "total_gb": total_gb,
            "used_gb": used_gb,
            "available_gb": avail_gb,
            "usage_pct": pct,
        }, expires_minutes=5)
        logger.info(f"Server Memory: {used_gb}GB / {total_gb}GB ({pct}%)")


def collect_server_disk(conn):
    """Remote disk usage."""
    disk = _ssh("df -h / | tail -1")
    if not disk:
        return

    parts = disk.split()
    if len(parts) >= 5:
        upsert_context(conn, "server", "disk", {
            "filesystem": parts[0],
            "size": parts[1],
            "used": parts[2],
            "available": parts[3],
            "usage_pct": parts[4].rstrip('%'),
        }, expires_minutes=10)
        logger.info(f"Server Disk: {parts[2]} used of {parts[1]} ({parts[4]})")


def collect_server_docker(conn):
    """Remote docker containers."""
    output = _ssh("docker ps --format '{{.Names}}|{{.Status}}|{{.Image}}'", timeout=15)
    if not output:
        upsert_context(conn, "server", "docker", {"running": False, "containers": []}, expires_minutes=5)
        return

    containers = []
    for line in output.splitlines():
        parts = line.split("|")
        if len(parts) >= 2:
            containers.append({
                "name": parts[0],
                "status": parts[1],
                "image": parts[2] if len(parts) > 2 else "",
            })

    upsert_context(conn, "server", "docker", {
        "running": True,
        "container_count": len(containers),
        "containers": containers,
    }, expires_minutes=5)
    logger.info(f"Server Docker: {len(containers)} containers")


def collect_server_uptime(conn):
    """Remote uptime and OS info."""
    uptime = _ssh("uptime -p")
    os_info = _ssh("cat /etc/os-release | grep PRETTY_NAME | cut -d= -f2 | tr -d '\"'")
    kernel = _ssh("uname -r")
    hostname = _ssh("hostname")

    upsert_context(conn, "server", "info", {
        "hostname": hostname,
        "os": os_info,
        "kernel": kernel,
        "uptime": uptime,
    }, expires_minutes=30)
    logger.info(f"Server: {hostname} — {os_info} — {uptime}")


def collect_server_network(conn):
    """Remote network — listening ports and connections."""
    ports = _ssh("ss -tlnp | awk 'NR>1 {print $4}' | sort -u | head -20")
    connections = _ssh("ss -s | head -5")

    port_list = [p.split(":")[-1] for p in ports.splitlines() if p] if ports else []

    upsert_context(conn, "server", "network", {
        "listening_ports": port_list,
        "port_count": len(port_list),
        "connections_summary": connections,
    }, expires_minutes=5)
    logger.info(f"Server Network: {len(port_list)} listening ports")


def collect_server_services(conn):
    """Remote running services — systemd top services."""
    services = _ssh("systemctl list-units --type=service --state=running --no-pager --plain | head -20")

    service_list = []
    if services:
        for line in services.splitlines():
            parts = line.split()
            if parts and ".service" in parts[0]:
                service_list.append(parts[0].replace(".service", ""))

    upsert_context(conn, "server", "services", {
        "running": service_list[:15],
        "count": len(service_list),
    }, expires_minutes=10)
    logger.info(f"Server Services: {len(service_list)} running")


def main():
    config = load_config()
    conn = get_db_conn(config)

    # Quick connectivity check
    test = _ssh("echo ok")
    if test != "ok":
        logger.error("Cannot reach remote server — skipping collection")
        conn.close()
        return

    collect_server_cpu(conn)
    collect_server_memory(conn)
    collect_server_disk(conn)
    collect_server_docker(conn)
    collect_server_uptime(conn)
    collect_server_network(conn)
    collect_server_services(conn)

    conn.close()
    logger.info("Remote server collection done")


if __name__ == "__main__":
    main()

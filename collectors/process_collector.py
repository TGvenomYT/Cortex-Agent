"""
Process Collector — runs every 1 minute.
Captures running processes, Docker containers, active ports,
and resource-heavy processes.
"""

import os
import sys
import json
import logging

sys.path.insert(0, os.path.dirname(__file__))
from base import load_config, get_db_conn, upsert_context, run_cmd

logger = logging.getLogger("collector.process")


def collect_docker(conn):
    """Capture Docker containers if Docker is running."""
    out, _, rc = run_cmd("docker ps --format '{{json .}}' 2>/dev/null")
    if rc != 0 or not out:
        upsert_context(conn, "process", "docker", {"running": False, "containers": []}, expires_minutes=3)
        return

    containers = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            c = json.loads(line)
            containers.append({
                "id": c.get("ID", "")[:12],
                "name": c.get("Names", ""),
                "image": c.get("Image", ""),
                "status": c.get("Status", ""),
                "ports": c.get("Ports", ""),
            })
        except Exception:
            pass

    upsert_context(conn, "process", "docker", {
        "running": True,
        "container_count": len(containers),
        "containers": containers,
    }, expires_minutes=3)
    logger.info(f"Docker: {len(containers)} containers running")


def collect_ports(conn):
    """Capture active listening ports."""
    out, _, rc = run_cmd("lsof -iTCP -sTCP:LISTEN -n -P 2>/dev/null | awk 'NR>1 {print $1,$2,$9}' | sort -u")
    if rc != 0 or not out:
        return

    ports = []
    seen = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        process_name = parts[0]
        addr = parts[2]
        port = addr.split(":")[-1] if ":" in addr else ""

        if port and port not in seen:
            seen.add(port)
            ports.append({"port": port, "process": process_name, "address": addr})

    # Sort by port number
    try:
        ports.sort(key=lambda x: int(x["port"]))
    except Exception:
        pass

    upsert_context(conn, "process", "listening_ports", {
        "count": len(ports),
        "ports": ports[:50],  # cap at 50
    }, expires_minutes=3)
    logger.info(f"Ports: {len(ports)} active")


def collect_top_processes(conn):
    """Capture CPU/RAM-heavy processes."""
    # Top 10 by CPU
    out, _, _ = run_cmd(
        "ps aux | sort -rk3 | head -11 | awk 'NR>1 {print $1,$2,$3,$4,$11}'"
    )
    top_cpu = []
    for line in out.splitlines():
        parts = line.split(None, 4)
        if len(parts) == 5:
            top_cpu.append({
                "user": parts[0],
                "pid": parts[1],
                "cpu": parts[2],
                "mem": parts[3],
                "cmd": parts[4][:80],
            })

    # Top 10 by memory
    out_mem, _, _ = run_cmd(
        "ps aux | sort -rk4 | head -11 | awk 'NR>1 {print $1,$2,$3,$4,$11}'"
    )
    top_mem = []
    for line in out_mem.splitlines():
        parts = line.split(None, 4)
        if len(parts) == 5:
            top_mem.append({
                "user": parts[0],
                "pid": parts[1],
                "cpu": parts[2],
                "mem": parts[3],
                "cmd": parts[4][:80],
            })

    upsert_context(conn, "process", "top_processes", {
        "top_cpu": top_cpu,
        "top_memory": top_mem,
    }, expires_minutes=3)
    logger.info(f"Processes: captured top CPU/memory processes")


def collect_services(conn):
    """Capture known dev services (node, python servers, etc.)."""
    out, _, _ = run_cmd(
        "ps aux | grep -E '(node|python|ruby|rails|flask|gunicorn|uvicorn|nginx|caddy|postgres|redis|mysql|mongo)' "
        "| grep -v grep | awk '{print $1,$2,$3,$4,$11,$12,$13}'"
    )
    services = []
    for line in out.splitlines():
        parts = line.split(None, 6)
        if len(parts) >= 5:
            services.append({
                "user": parts[0],
                "pid": parts[1],
                "cpu": parts[2],
                "mem": parts[3],
                "cmd": " ".join(parts[4:])[:100],
            })

    upsert_context(conn, "process", "dev_services", {
        "count": len(services),
        "services": services,
    }, expires_minutes=3)
    logger.info(f"Services: {len(services)} dev processes")


def main():
    config = load_config()
    conn = get_db_conn(config)

    collect_docker(conn)
    collect_ports(conn)
    collect_top_processes(conn)
    collect_services(conn)

    conn.close()
    logger.info("Process collection done")


if __name__ == "__main__":
    main()

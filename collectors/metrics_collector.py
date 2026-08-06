"""
Metrics Collector — Pulls real-time metrics from Prometheus (devops-agent stack).
Stores CPU, memory, container stats, disk, network data in context_store.
Runs every 1 minute.
"""

import os
import sys
import json
import logging

import urllib.request
import urllib.error

sys.path.insert(0, os.path.dirname(__file__))
from base import load_config, get_db_conn, upsert_context

logger = logging.getLogger("collector.metrics")

PROMETHEUS_URL = "http://localhost:9090"


def prom_query(promql: str) -> list:
    """Query Prometheus, return result list."""
    try:
        url = f"{PROMETHEUS_URL}/api/v1/query"
        data = f"query={urllib.parse.quote(promql)}".encode()
        req = urllib.request.Request(url, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = json.loads(resp.read())
        return body.get("data", {}).get("result", [])
    except Exception as e:
        logger.debug(f"Prometheus query failed: {e}")
        return []


import urllib.parse


def collect_host_cpu(conn):
    """Host CPU usage percentage."""
    results = prom_query('100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)')
    if results:
        cpu_pct = round(float(results[0]["value"][1]), 1)
        upsert_context(conn, "metrics", "host_cpu", {
            "usage_pct": cpu_pct,
            "status": "critical" if cpu_pct > 90 else "warning" if cpu_pct > 70 else "ok"
        }, expires_minutes=3)
        logger.info(f"Host CPU: {cpu_pct}%")


def collect_host_memory(conn):
    """Host memory usage."""
    results = prom_query("(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100")
    if results:
        mem_pct = round(float(results[0]["value"][1]), 1)

    total = prom_query("node_memory_MemTotal_bytes")
    avail = prom_query("node_memory_MemAvailable_bytes")

    total_gb = round(float(total[0]["value"][1]) / (1024**3), 1) if total else 0
    avail_gb = round(float(avail[0]["value"][1]) / (1024**3), 1) if avail else 0
    used_gb = round(total_gb - avail_gb, 1)

    upsert_context(conn, "metrics", "host_memory", {
        "usage_pct": mem_pct if results else 0,
        "total_gb": total_gb,
        "used_gb": used_gb,
        "available_gb": avail_gb,
        "status": "critical" if (results and mem_pct > 90) else "warning" if (results and mem_pct > 80) else "ok"
    }, expires_minutes=3)
    logger.info(f"Host Memory: {used_gb}GB / {total_gb}GB ({mem_pct if results else '?'}%)")


def collect_container_resources(conn):
    """Per-container CPU and memory."""
    cpu_results = prom_query('rate(container_cpu_usage_seconds_total{name!=""}[5m]) * 100')
    mem_results = prom_query('container_memory_usage_bytes{name!=""}')

    containers = {}
    for r in cpu_results:
        name = r["metric"].get("name", "")
        if name:
            containers.setdefault(name, {})["cpu_pct"] = round(float(r["value"][1]), 2)

    for r in mem_results:
        name = r["metric"].get("name", "")
        if name:
            containers.setdefault(name, {})["memory_mb"] = round(float(r["value"][1]) / (1024 * 1024), 1)

    upsert_context(conn, "metrics", "container_resources", {
        "containers": containers,
        "count": len(containers),
    }, expires_minutes=3)
    logger.info(f"Container metrics: {len(containers)} containers tracked")


def collect_disk(conn):
    """Disk usage from Prometheus node_exporter."""
    results = prom_query('100 - ((node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"}) * 100)')
    if results:
        disk_pct = round(float(results[0]["value"][1]), 1)

        size = prom_query('node_filesystem_size_bytes{mountpoint="/"}')
        avail = prom_query('node_filesystem_avail_bytes{mountpoint="/"}')
        size_gb = round(float(size[0]["value"][1]) / (1024**3), 1) if size else 0
        avail_gb = round(float(avail[0]["value"][1]) / (1024**3), 1) if avail else 0

        upsert_context(conn, "metrics", "host_disk", {
            "usage_pct": disk_pct,
            "total_gb": size_gb,
            "available_gb": avail_gb,
            "status": "critical" if disk_pct > 90 else "warning" if disk_pct > 80 else "ok"
        }, expires_minutes=10)
        logger.info(f"Disk: {disk_pct}% used, {avail_gb}GB free")


def collect_network(conn):
    """Network throughput."""
    rx = prom_query('rate(node_network_receive_bytes_total{device!="lo"}[5m])')
    tx = prom_query('rate(node_network_transmit_bytes_total{device!="lo"}[5m])')

    rx_kbs = round(sum(float(r["value"][1]) for r in rx) / 1024, 2) if rx else 0
    tx_kbs = round(sum(float(r["value"][1]) for r in tx) / 1024, 2) if tx else 0

    upsert_context(conn, "metrics", "host_network", {
        "rx_kb_s": rx_kbs,
        "tx_kb_s": tx_kbs,
    }, expires_minutes=3)
    logger.info(f"Network: rx={rx_kbs}KB/s, tx={tx_kbs}KB/s")


def main():
    config = load_config()
    conn = get_db_conn(config)

    collect_host_cpu(conn)
    collect_host_memory(conn)
    collect_container_resources(conn)
    collect_disk(conn)
    collect_network(conn)

    conn.close()
    logger.info("Metrics collection done")


if __name__ == "__main__":
    main()

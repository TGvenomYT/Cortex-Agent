"""
Terminal Dashboard — Rich-powered system visualization.
Shows CPU, memory, disk, containers, network as bar charts and tables.
Called from `cx` when user wants a visual overview.
"""

import json
import urllib.request
import urllib.error

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich.bar import Bar
from rich.layout import Layout
from rich.live import Live
from rich import box

console = Console()
BASE_URL = "http://localhost:7800"


def _get(endpoint: str) -> dict:
    try:
        with urllib.request.urlopen(f"{BASE_URL}{endpoint}", timeout=5) as resp:
            return json.loads(resp.read())
    except Exception:
        return {}


def _post(endpoint: str, data: dict) -> dict:
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        f"{BASE_URL}{endpoint}", data=body,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception:
        return {}


def _bar(pct: float, width: int = 20) -> str:
    """Generate a text-based bar chart."""
    filled = int(pct / 100 * width)
    empty = width - filled
    if pct > 90:
        color = "red"
    elif pct > 70:
        color = "yellow"
    else:
        color = "green"
    return f"[{color}]{'█' * filled}[/{color}][dim]{'░' * empty}[/dim] {pct:.1f}%"


def _status_dot(status: str) -> str:
    colors = {"ok": "green", "warning": "yellow", "critical": "red", "unknown": "dim"}
    color = colors.get(status, "dim")
    return f"[{color}]●[/{color}]"


def render_dashboard():
    """Render full terminal dashboard."""
    # Fetch metrics from Cortex API
    import subprocess, os

    # Get metrics directly from DB via a quick Python call
    env = os.environ.copy()
    env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:" + env.get("PATH", "")

    # Query Prometheus directly for live data
    metrics = _query_prometheus()
    docker = _query_docker()
    status = _get("/status")

    console.print()

    # ─── Header ───
    console.print(Panel(
        "[bold]Cortex Dashboard[/bold]",
        border_style="cyan",
        box=box.ROUNDED
    ))

    # ─── System Resources ───
    resource_table = Table(title="System Resources", box=box.SIMPLE_HEAVY, show_header=True)
    resource_table.add_column("Metric", style="bold", width=14)
    resource_table.add_column("Usage", width=30)
    resource_table.add_column("Details", style="dim")

    cpu = metrics.get("cpu", {})
    mem = metrics.get("memory", {})
    disk = metrics.get("disk", {})
    net = metrics.get("network", {})

    if cpu:
        resource_table.add_row("CPU", _bar(cpu.get("pct", 0)), f"{cpu.get('pct', 0):.1f}%")
    if mem:
        resource_table.add_row("Memory", _bar(mem.get("pct", 0)),
                               f"{mem.get('used_gb', 0)}GB / {mem.get('total_gb', 0)}GB")
    if disk:
        resource_table.add_row("Disk", _bar(disk.get("pct", 0)),
                               f"{disk.get('available_gb', 0)}GB free")
    if net:
        resource_table.add_row("Network", "",
                               f"↓ {net.get('rx_kb_s', 0)} KB/s  ↑ {net.get('tx_kb_s', 0)} KB/s")

    console.print(resource_table)

    # ─── Docker Containers ───
    if docker:
        docker_table = Table(title="Docker Containers", box=box.SIMPLE_HEAVY, show_header=True)
        docker_table.add_column("Container", style="bold", width=28)
        docker_table.add_column("Status", width=12)
        docker_table.add_column("CPU", width=8)
        docker_table.add_column("Memory", width=10)

        container_metrics = metrics.get("containers", {})

        for c in docker:
            name = c.get("name", "")
            status_str = c.get("status", "")

            if "Up" in status_str:
                if "healthy" in status_str:
                    st = "[green]healthy[/green]"
                else:
                    st = "[green]running[/green]"
            elif "Exited" in status_str:
                if "(0)" in status_str:
                    st = "[dim]stopped[/dim]"
                else:
                    st = "[red]crashed[/red]"
            else:
                st = f"[dim]{status_str[:10]}[/dim]"

            cm = container_metrics.get(name, {})
            cpu_str = f"{cm.get('cpu_pct', 0):.1f}%" if cm.get('cpu_pct') else "[dim]-[/dim]"
            mem_str = f"{cm.get('memory_mb', 0):.0f}MB" if cm.get('memory_mb') else "[dim]-[/dim]"

            docker_table.add_row(name, st, cpu_str, mem_str)

        console.print(docker_table)

    # ─── Status bar ───
    st = status or {}
    projects = st.get("tracked_projects", 0)
    reminders = st.get("active_reminders", 0)
    console.print(
        f"\n  [dim]Tracked projects:[/dim] {projects}  "
        f"[dim]Active reminders:[/dim] {reminders}  "
        f"[dim]LLM:[/dim] {st.get('llm_primary', '?')} → {st.get('llm_fallback', 'none')}"
    )
    console.print()


def _query_prometheus() -> dict:
    """Pull live metrics from Prometheus."""
    import urllib.parse

    def pq(query: str):
        try:
            url = f"http://localhost:9090/api/v1/query"
            data = f"query={urllib.parse.quote(query)}".encode()
            req = urllib.request.Request(url, data=data, method="POST")
            with urllib.request.urlopen(req, timeout=3) as resp:
                body = json.loads(resp.read())
            results = body.get("data", {}).get("result", [])
            return float(results[0]["value"][1]) if results else None
        except Exception:
            return None

    metrics = {}

    # CPU
    cpu_pct = pq('100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)')
    if cpu_pct is not None:
        metrics["cpu"] = {"pct": round(cpu_pct, 1)}

    # Memory
    mem_pct = pq("(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100")
    total = pq("node_memory_MemTotal_bytes")
    avail = pq("node_memory_MemAvailable_bytes")
    if mem_pct is not None:
        total_gb = round(total / (1024**3), 1) if total else 0
        avail_gb = round(avail / (1024**3), 1) if avail else 0
        metrics["memory"] = {
            "pct": round(mem_pct, 1),
            "total_gb": total_gb,
            "used_gb": round(total_gb - avail_gb, 1),
            "available_gb": avail_gb,
        }

    # Disk
    disk_pct = pq('100 - ((node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"}) * 100)')
    disk_avail = pq('node_filesystem_avail_bytes{mountpoint="/"}')
    if disk_pct is not None:
        metrics["disk"] = {
            "pct": round(disk_pct, 1),
            "available_gb": round(disk_avail / (1024**3), 1) if disk_avail else 0,
        }

    # Network
    rx = pq('sum(rate(node_network_receive_bytes_total{device!="lo"}[5m]))')
    tx = pq('sum(rate(node_network_transmit_bytes_total{device!="lo"}[5m]))')
    metrics["network"] = {
        "rx_kb_s": round(rx / 1024, 1) if rx else 0,
        "tx_kb_s": round(tx / 1024, 1) if tx else 0,
    }

    # Container CPU/Memory
    try:
        url = "http://localhost:9090/api/v1/query"
        cpu_query = 'rate(container_cpu_usage_seconds_total{name!=""}[5m]) * 100'
        data = f"query={urllib.parse.quote(cpu_query)}".encode()
        req = urllib.request.Request(url, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=3) as resp:
            body = json.loads(resp.read())
        cpu_results = body.get("data", {}).get("result", [])

        mem_query = 'container_memory_usage_bytes{name!=""}'
        data = f"query={urllib.parse.quote(mem_query)}".encode()
        req = urllib.request.Request(url, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=3) as resp:
            body = json.loads(resp.read())
        mem_results = body.get("data", {}).get("result", [])

        containers = {}
        for r in cpu_results:
            name = r["metric"].get("name", "")
            if name:
                containers.setdefault(name, {})["cpu_pct"] = round(float(r["value"][1]), 2)
        for r in mem_results:
            name = r["metric"].get("name", "")
            if name:
                containers.setdefault(name, {})["memory_mb"] = round(float(r["value"][1]) / (1024**2), 1)

        metrics["containers"] = containers
    except Exception:
        metrics["containers"] = {}

    return metrics


def _query_docker() -> list:
    """Get container list from docker."""
    import subprocess
    env = {"PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"}
    try:
        r = subprocess.run(
            "docker ps -a --format '{{json .}}'",
            shell=True, capture_output=True, text=True, timeout=5, env=env
        )
        containers = []
        for line in r.stdout.splitlines():
            line = line.strip()
            if line:
                try:
                    c = json.loads(line)
                    containers.append({
                        "name": c.get("Names", ""),
                        "image": c.get("Image", ""),
                        "status": c.get("Status", ""),
                    })
                except Exception:
                    pass
        return containers
    except Exception:
        return []


if __name__ == "__main__":
    render_dashboard()


def render_chart(request: str):
    """Render a specific chart based on what the user asked for."""
    request_lower = request.lower()
    metrics = _query_prometheus()
    docker = _query_docker()

    console.print()

    if any(kw in request_lower for kw in ["cpu", "processor", "load"]):
        _render_cpu_chart(metrics)
    elif any(kw in request_lower for kw in ["memory", "ram", "mem"]):
        _render_memory_chart(metrics)
    elif any(kw in request_lower for kw in ["disk", "storage", "space"]):
        _render_disk_chart(metrics)
    elif any(kw in request_lower for kw in ["container", "docker"]):
        _render_container_chart(metrics, docker)
    elif any(kw in request_lower for kw in ["network", "traffic", "bandwidth"]):
        _render_network_chart(metrics)
    else:
        # Default: show all
        _render_cpu_chart(metrics)
        _render_memory_chart(metrics)
        _render_container_chart(metrics, docker)

    console.print()


def _render_cpu_chart(metrics: dict):
    """Render CPU usage bar chart."""
    cpu = metrics.get("cpu", {})
    pct = cpu.get("pct", 0)

    console.print(Panel(
        _wide_bar(pct, label="CPU"),
        title="[bold]CPU Usage[/bold]",
        border_style="cyan",
        box=box.ROUNDED
    ))


def _render_memory_chart(metrics: dict):
    """Render memory usage visualization."""
    mem = metrics.get("memory", {})
    pct = mem.get("pct", 0)
    used = mem.get("used_gb", 0)
    total = mem.get("total_gb", 0)
    avail = mem.get("available_gb", 0)

    content = (
        f"{_wide_bar(pct, label='RAM')}\n"
        f"\n"
        f"  [bold]{used}GB[/bold] used  /  [bold]{total}GB[/bold] total  /  [green]{avail}GB[/green] free"
    )
    console.print(Panel(
        content,
        title="[bold]Memory Usage[/bold]",
        border_style="cyan",
        box=box.ROUNDED
    ))


def _render_disk_chart(metrics: dict):
    """Render disk usage."""
    disk = metrics.get("disk", {})
    pct = disk.get("pct", 0)
    avail = disk.get("available_gb", 0)

    content = (
        f"{_wide_bar(pct, label='Disk')}\n"
        f"\n"
        f"  [green]{avail}GB[/green] available"
    )
    console.print(Panel(
        content,
        title="[bold]Disk Usage[/bold]",
        border_style="cyan",
        box=box.ROUNDED
    ))


def _render_container_chart(metrics: dict, docker: list):
    """Render per-container resource usage as horizontal bars."""
    container_metrics = metrics.get("containers", {})

    if not docker and not container_metrics:
        console.print("[dim]  No container metrics available (cAdvisor may be down)[/dim]")
        return

    table = Table(
        title="Container Resources",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold"
    )
    table.add_column("Container", width=26)
    table.add_column("CPU", width=30)
    table.add_column("Memory", width=16)
    table.add_column("Status", width=10)

    for c in docker:
        name = c.get("name", "")
        status_str = c.get("status", "")
        cm = container_metrics.get(name, {})

        # CPU bar
        cpu_pct = cm.get("cpu_pct", 0)
        cpu_bar = _mini_bar(min(cpu_pct * 10, 100))  # scale up for visibility
        cpu_str = f"{cpu_bar} {cpu_pct:.1f}%" if cpu_pct else "[dim]—[/dim]"

        # Memory
        mem_mb = cm.get("memory_mb", 0)
        mem_str = f"{mem_mb:.0f} MB" if mem_mb else "[dim]—[/dim]"

        # Status
        if "Up" in status_str:
            st = "[green]●[/green] up"
        elif "Exited (0)" in status_str:
            st = "[dim]● stopped[/dim]"
        elif "Exited" in status_str:
            st = "[red]● crash[/red]"
        else:
            st = "[dim]●[/dim]"

        table.add_row(name, cpu_str, mem_str, st)

    console.print(table)


def _render_network_chart(metrics: dict):
    """Render network throughput."""
    net = metrics.get("network", {})
    rx = net.get("rx_kb_s", 0)
    tx = net.get("tx_kb_s", 0)

    content = (
        f"  [green]↓ Download:[/green]  {rx:.1f} KB/s\n"
        f"  [blue]↑ Upload:  [/blue]  {tx:.1f} KB/s"
    )
    console.print(Panel(
        content,
        title="[bold]Network[/bold]",
        border_style="cyan",
        box=box.ROUNDED
    ))


def _wide_bar(pct: float, width: int = 40, label: str = "") -> str:
    """Wide bar chart for panels."""
    filled = int(pct / 100 * width)
    empty = width - filled
    if pct > 90:
        color = "red"
    elif pct > 70:
        color = "yellow"
    else:
        color = "green"

    bar = f"[{color}]{'█' * filled}[/{color}][dim]{'░' * empty}[/dim]"
    return f"  {label + ':' if label else '':<6} {bar} [bold]{pct:.1f}%[/bold]"


def _mini_bar(pct: float, width: int = 15) -> str:
    """Small inline bar."""
    filled = int(pct / 100 * width)
    empty = width - filled
    if pct > 80:
        color = "red"
    elif pct > 50:
        color = "yellow"
    else:
        color = "green"
    return f"[{color}]{'▓' * filled}[/{color}][dim]{'░' * empty}[/dim]"

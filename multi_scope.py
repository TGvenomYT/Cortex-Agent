"""
Multi-Scope Query Handler — Parallel fan-out for complex queries.

When a query spans multiple domains (docker + git + system), this module:
1. Classifies the query scope
2. Fans out real commands in parallel (subprocess, not LLM)
3. Merges results into one context block
4. Sends to a single LLM call for a unified explanation

This is faster and cheaper than LLM sub-agents.
"""

import os
import json
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

logger = logging.getLogger("cortex.multiscope")

# ─── Command templates for each scope ───

SCOPE_COMMANDS = {
    "docker": [
        ("containers_running", "docker ps --format '{{.Names}}: {{.Status}}'"),
        ("containers_exited", "docker ps -a --filter status=exited --format '{{.Names}}: {{.Status}}'"),
    ],
    "git": [],  # Dynamic — one per tracked project
    "system": [
        ("disk", "df -h / | tail -1"),
        ("memory", "vm_stat | head -5"),
        ("load", "sysctl -n vm.loadavg"),
        ("uptime", "uptime"),
    ],
    "network": [
        ("ports", "lsof -iTCP -sTCP:LISTEN -n -P 2>/dev/null | awk 'NR>1 {print $1,$9}' | sort -u | head -20"),
        ("connectivity", "ping -c1 -W2 8.8.8.8 >/dev/null 2>&1 && echo 'internet: connected' || echo 'internet: disconnected'"),
    ],
    "processes": [
        ("top_cpu", "ps aux --sort=-%cpu | head -6 | awk '{print $11, $3\"%\"}'"),
        ("dev_services", "ps aux | grep -E '(node|python|docker|redis|postgres|nginx)' | grep -v grep | awk '{print $11}' | sort -u"),
    ],
}

# Keywords that trigger each scope
SCOPE_KEYWORDS = {
    "docker": ["docker", "container", "compose", "image", "crash", "exited", "healthy", "restart"],
    "git": ["git", "commit", "branch", "uncommitted", "push", "pull", "repo", "changes", "diff"],
    "system": ["disk", "memory", "ram", "cpu", "load", "space", "storage", "uptime", "system"],
    "network": ["port", "network", "listen", "connection", "internet", "ip"],
    "processes": ["process", "running", "service", "pid", "kill"],
}


def _run_cmd(cmd: str, cwd: str = None, timeout: int = 10) -> str:
    """Run a command, return stdout or error string."""
    env = os.environ.copy()
    env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:" + env.get("PATH", "")
    try:
        r = subprocess.run(
            cmd, shell=True, cwd=cwd,
            capture_output=True, text=True,
            timeout=timeout, env=env
        )
        if r.returncode == 0:
            return r.stdout.strip()
        else:
            return f"[error: {r.stderr.strip()[:200]}]"
    except subprocess.TimeoutExpired:
        return "[timeout]"
    except Exception as e:
        return f"[failed: {e}]"


def detect_scopes(query: str) -> list[str]:
    """Detect which scopes a query touches based on keywords."""
    query_lower = query.lower()
    scopes = []
    for scope, keywords in SCOPE_KEYWORDS.items():
        if any(kw in query_lower for kw in keywords):
            scopes.append(scope)

    # "health report", "status", "overview" → all scopes
    broad_keywords = ["health", "report", "status", "overview", "everything", "all", "full"]
    if any(kw in query_lower for kw in broad_keywords):
        scopes = list(SCOPE_KEYWORDS.keys())

    return scopes or ["docker", "system"]  # default to docker + system


def fan_out(scopes: list[str], tracked_projects: list = None) -> dict:
    """
    Execute all commands for the given scopes in parallel.
    Returns: {scope: {label: output_string}}
    """
    tasks = []  # (scope, label, cmd, cwd)

    for scope in scopes:
        if scope == "git" and tracked_projects:
            for proj in tracked_projects:
                path = proj.get("path", "")
                name = proj.get("name", os.path.basename(path))
                if os.path.isdir(os.path.join(path, ".git")):
                    tasks.append(("git", f"git_{name}", f"git -C '{path}' status --short && echo '---' && git -C '{path}' log -1 --oneline", None))
        elif scope in SCOPE_COMMANDS:
            for label, cmd in SCOPE_COMMANDS[scope]:
                tasks.append((scope, label, cmd, None))

    # Execute in parallel
    results = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {}
        for scope, label, cmd, cwd in tasks:
            future = pool.submit(_run_cmd, cmd, cwd)
            futures[future] = (scope, label)

        for future in as_completed(futures):
            scope, label = futures[future]
            output = future.result()
            if scope not in results:
                results[scope] = {}
            results[scope][label] = output

    return results


def format_results(results: dict) -> str:
    """Format parallel results into a context block for the LLM."""
    sections = []
    for scope, data in results.items():
        lines = [f"[{scope.upper()} — LIVE]"]
        for label, output in data.items():
            if output and output not in ("[timeout]", "[failed]"):
                lines.append(f"  {label}:")
                for line in output.splitlines()[:15]:
                    lines.append(f"    {line}")
        if len(lines) > 1:
            sections.append("\n".join(lines))

    return "\n\n".join(sections)


def is_multi_scope(query: str) -> bool:
    """Determine if a query would benefit from parallel fan-out.
    Only triggers for genuinely broad queries, not single-topic ones."""
    query_lower = query.lower()

    # Broad keywords that indicate a multi-scope request
    broad_keywords = ["health", "report", "status update", "overview", "everything", "full check", "all"]
    is_broad = any(kw in query_lower for kw in broad_keywords)

    if is_broad:
        return True

    # Only multi-scope if 2+ domains are explicitly mentioned
    scopes = detect_scopes(query)
    return len(scopes) >= 3


def execute_multi_scope(query: str, tracked_projects: list = None) -> str:
    """
    Full pipeline: detect scopes → fan out → format results.
    Returns a context string to feed to the LLM.
    """
    scopes = detect_scopes(query)
    logger.info(f"Multi-scope query detected: {scopes}")

    results = fan_out(scopes, tracked_projects=tracked_projects)
    formatted = format_results(results)

    logger.info(f"Multi-scope fan-out complete: {sum(len(v) for v in results.values())} results gathered")
    return formatted

"""
Multi-Machine — Remote Cortex access over Tailscale/network.
Allows managing remote servers by proxying commands through Cortex.
Also supports receiving requests from other machines on the tailnet.
"""

import subprocess
import json
import logging
import os

logger = logging.getLogger("cortex.multimachine")


class RemoteExecutor:
    """Execute commands on remote machines via SSH over Tailscale."""

    def __init__(self, hosts: dict):
        """
        hosts: {name: {ip: "100.x.x.x", user: "root", key: "~/.ssh/id_ed25519"}}
        """
        self.hosts = hosts

    def execute(self, host_name: str, command: str, timeout: int = 30) -> dict:
        """Run a command on a remote host via SSH."""
        host = self.hosts.get(host_name)
        if not host:
            return {"error": f"Unknown host: {host_name}", "exit_code": -1}

        ip = host.get("ip", "")
        user = host.get("user", "root")
        key = os.path.expanduser(host.get("key", "~/.ssh/id_ed25519"))

        ssh_cmd = [
            "ssh", "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=5",
            "-i", key,
            f"{user}@{ip}",
            command
        ]

        try:
            result = subprocess.run(
                ssh_cmd, capture_output=True, text=True, timeout=timeout
            )
            return {
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "host": host_name,
            }
        except subprocess.TimeoutExpired:
            return {"exit_code": -1, "stderr": f"Timeout after {timeout}s", "host": host_name}
        except Exception as e:
            return {"exit_code": -1, "stderr": str(e), "host": host_name}

    def execute_all(self, command: str, timeout: int = 30) -> dict:
        """Run a command on ALL hosts."""
        results = {}
        for name in self.hosts:
            results[name] = self.execute(name, command, timeout)
        return results

    def get_host_status(self, host_name: str) -> dict:
        """Quick health check on a remote host."""
        result = self.execute(host_name, "uptime && df -h / | tail -1 && free -h | head -2", timeout=10)
        return result

    def list_hosts(self) -> list:
        """List configured remote hosts."""
        return [{"name": k, "ip": v.get("ip", ""), "user": v.get("user", "")}
                for k, v in self.hosts.items()]


def get_tailscale_peers() -> list:
    """Get list of machines on the tailnet."""
    try:
        result = subprocess.run(
            ["tailscale", "status", "--json"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            peers = []
            for peer_id, peer in data.get("Peer", {}).items():
                peers.append({
                    "name": peer.get("HostName", ""),
                    "ip": peer.get("TailscaleIPs", [""])[0],
                    "os": peer.get("OS", ""),
                    "online": peer.get("Online", False),
                })
            return peers
    except Exception as e:
        logger.error(f"Tailscale status failed: {e}")
    return []

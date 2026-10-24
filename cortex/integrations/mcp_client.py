"""
mcp_client.py — a minimal, standard MCP (Model Context Protocol) client.

Lets Cortex (the local harness / gemma agent) connect to ANY MCP server over
stdio, discover its tools, and call them — the same protocol Claude Code uses.
Configure servers in Settings → MCP (name + command line). Tools are exposed to
the model as `mcp__<server>__<tool>` and routed back here.

JSON-RPC 2.0 over stdio, newline-delimited (the MCP stdio transport). Stdlib only.
"""
from __future__ import annotations

import os
import json
import shlex
import queue
import threading
import subprocess
import logging

logger = logging.getLogger("cortex.mcp")

PROTOCOL = "2025-06-18"


class MCPServer:
    def __init__(self, name: str, command: str, env: dict | None = None):
        self.name = name
        self.argv = shlex.split(command)
        self.env = {**os.environ, **(env or {})}
        self.proc: subprocess.Popen | None = None
        self.tools: list[dict] = []
        self._id = 0
        self._pending: dict[int, queue.Queue] = {}
        self._lock = threading.Lock()
        self._alive = False

    # ── lifecycle ──
    def start(self, timeout: float = 20) -> bool:
        try:
            self.proc = subprocess.Popen(
                self.argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, env=self.env, text=True, bufsize=1,
            )
        except Exception as e:
            logger.warning(f"[{self.name}] spawn failed: {e}")
            return False
        self._alive = True
        threading.Thread(target=self._reader, daemon=True, name=f"mcp-{self.name}").start()
        try:
            self._request("initialize", {
                "protocolVersion": PROTOCOL, "capabilities": {},
                "clientInfo": {"name": "cortex", "version": "2.0"},
            }, timeout=timeout)
            self._notify("notifications/initialized", {})
            res = self._request("tools/list", {}, timeout=timeout)
            self.tools = (res or {}).get("tools", []) or []
            logger.info(f"[{self.name}] connected · {len(self.tools)} tools")
            return True
        except Exception as e:
            logger.warning(f"[{self.name}] handshake failed: {e}")
            self.stop()
            return False

    def stop(self):
        self._alive = False
        if self.proc:
            try:
                self.proc.terminate()
            except Exception:
                pass
            self.proc = None

    # ── io ──
    def _reader(self):
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            mid = msg.get("id")
            if mid is not None and mid in self._pending:
                self._pending[mid].put(msg)
        self._alive = False

    def _send(self, obj: dict):
        if not self.proc or not self.proc.stdin:
            raise RuntimeError("server not running")
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()

    def _notify(self, method: str, params: dict):
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def _request(self, method: str, params: dict, timeout: float = 30):
        with self._lock:
            self._id += 1
            mid = self._id
        q: queue.Queue = queue.Queue(maxsize=1)
        self._pending[mid] = q
        self._send({"jsonrpc": "2.0", "id": mid, "method": method, "params": params})
        try:
            msg = q.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f"{method} timed out")
        finally:
            self._pending.pop(mid, None)
        if "error" in msg:
            raise RuntimeError(msg["error"].get("message", "mcp error"))
        return msg.get("result", {})

    def call(self, tool: str, args: dict, timeout: float = 60) -> str:
        res = self._request("tools/call", {"name": tool, "arguments": args or {}}, timeout=timeout)
        # MCP returns content: [{type:"text",text:...}, ...]
        parts = []
        for c in (res or {}).get("content", []) or []:
            if c.get("type") == "text":
                parts.append(c.get("text", ""))
            else:
                parts.append(json.dumps(c, default=str))
        if res.get("isError"):
            return "tool error: " + (" ".join(parts) or "unknown")
        return "\n".join(parts) or json.dumps(res, default=str)


class MCPManager:
    """Loads configured MCP servers and exposes/routes their tools."""

    def __init__(self, configs: list[dict]):
        self.configs = configs or []
        self.servers: dict[str, MCPServer] = {}
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        self._loaded = True
        for c in self.configs:
            name = c.get("name")
            cmd = c.get("command")
            if not name or not cmd:
                continue
            srv = MCPServer(name, cmd, c.get("env"))
            if srv.start():
                self.servers[name] = srv

    def openai_tools(self) -> list[dict]:
        """Merged tool schemas, namespaced mcp__<server>__<tool>."""
        self.load()
        out = []
        for sname, srv in self.servers.items():
            for t in srv.tools:
                out.append({"type": "function", "function": {
                    "name": f"mcp__{sname}__{t['name']}",
                    "description": (t.get("description") or "")[:900],
                    "parameters": t.get("inputSchema") or {"type": "object", "properties": {}},
                }})
        return out

    def call(self, fqname: str, args: dict) -> str:
        # fqname = mcp__<server>__<tool>
        try:
            _, rest = fqname.split("mcp__", 1)
            sname, tool = rest.split("__", 1)
        except ValueError:
            return f"bad mcp tool name '{fqname}'"
        srv = self.servers.get(sname)
        if not srv:
            return f"mcp server '{sname}' not connected"
        try:
            return srv.call(tool, args)
        except Exception as e:
            return f"mcp call failed: {e}"

    def status(self) -> list[dict]:
        self.load()
        done = {s.name for s in self.servers.values()}
        out = [{"name": n, "connected": True, "tools": len(s.tools)}
               for n, s in self.servers.items()]
        for c in self.configs:
            if c.get("name") not in done:
                out.append({"name": c.get("name"), "connected": False, "tools": 0})
        return out

    def stop(self):
        for s in self.servers.values():
            s.stop()

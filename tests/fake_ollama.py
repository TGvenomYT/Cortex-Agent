"""A small stand-in for an Ollama server: real HTTP, scripted replies."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeOllama:
    def __init__(self, port: int = 0, reply=None, models=("fake:1b",)):
        self.requests, self.tag_headers = [], []
        self.hold = 0
        self.reply = reply or (lambda body: {"content": "hello"})
        self.models = list(models)
        self.chat_status = 200
        self.chunk_delay = 0.0
        self.parameters = ""
        self.capabilities = ["completion", "tools"]
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, obj, status=200):
                data = json.dumps(obj).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/api/tags":
                    outer.tag_headers.append(dict(self.headers))
                    self._json({"models": [{"name": m, "model": m, "capabilities": ["completion", "tools"]}
                                           for m in outer.models]})
                elif self.path == "/api/ps":
                    self._json({"models": []})
                else:
                    self._json({}, 404)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if self.path == "/api/embed":
                    return self._json({"embeddings": [[0.1] * 768]})
                if self.path == "/api/show":
                    return self._json({"parameters": outer.parameters, "capabilities": outer.capabilities})
                body["_auth"] = self.headers.get("Authorization")
                outer.requests.append(body)
                if outer.chat_status != 200:
                    return self._json({"error": "nope"}, outer.chat_status)
                if body.get("tools") and "no tools" in str(body.get("model")):
                    return self._json({"error": "registry.ollama.ai/x does not support tools"}, 400)
                r = outer.reply(body)
                if not body.get("stream"):
                    return self._json({"message": {"role": "assistant", **r}, "done": True})
                time.sleep(outer.hold)
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.end_headers()
                text = r.get("content") or ""
                for i in range(0, len(text), 8):
                    self.wfile.write((json.dumps({"message": {"content": text[i:i + 8]}, "done": False}) + "\n").encode())
                    self.wfile.flush()
                    time.sleep(outer.chunk_delay)
                final = {"message": {"content": "", "tool_calls": r.get("tool_calls") or []}, "done": True}
                self.wfile.write((json.dumps(final) + "\n").encode())

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.port = self.server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()

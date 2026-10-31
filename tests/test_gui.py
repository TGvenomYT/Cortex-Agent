"""The server's door. Flask's test client: no socket, stubbed agent."""
import json

import pytest


@pytest.fixture
def gui(tmp_path, monkeypatch):
    import cortex.server.gui_server as g
    import cortex.core.tasks as tasks
    import cortex.core.workers as workers
    calls = []
    monkeypatch.setattr(g, "_TOKEN", None)
    monkeypatch.setattr(g, "_TOKEN_FILE", str(tmp_path / ".cortex" / "gui_token"))
    monkeypatch.setattr(tasks, "_install", lambda t: calls.append(("install", t["prompt"])))
    monkeypatch.setattr(workers, "apply_profile", lambda n: calls.append(("profile", n)) or {"ok": True})

    class Job:
        done = True

        def tail(self, start=0):
            return iter(())

    monkeypatch.setattr(g, "_start_job", lambda s, make: calls.append(("turn", s)) or Job())
    monkeypatch.setattr(g, "core", lambda: type("C", (), {"mcp": None, "config": {}})())
    g.calls = calls
    return g


def headers(g, **extra):
    return {"X-Cortex-Token": g.gui_token(), **extra}


EVIL = {"Origin": "https://attacker.example", "Sec-Fetch-Site": "cross-site"}


@pytest.mark.parametrize("path,body", [
    ("/tasks/add", {"prompt": "evil", "schedule": "hourly"}), ("/workers/profile", {"profile": "full"}),
    ("/mcp/add", {"name": "x", "command": "id"}), ("/keys/set", {"provider": "openai", "key": "k"}),
    ("/approve", {"session": "s", "id": 1, "decision": "approve"}),
])
def test_a_post_with_no_token_is_refused(gui, path, body):
    r = gui.app.test_client().post(path, data=json.dumps(body), content_type="text/plain")
    assert r.status_code == 403 and gui.calls == []


def test_a_cross_site_page_cannot_start_the_agent(gui):
    r = gui.app.test_client().get("/stream", query_string={"q": "run this", "session": "x"}, headers=EVIL)
    assert r.status_code == 403 and gui.calls == []


def test_a_cross_site_page_cannot_use_a_stolen_token(gui):
    r = gui.app.test_client().post("/tasks/add", json={"prompt": "p", "schedule": "hourly"},
                                   headers=headers(gui, **EVIL))
    assert r.status_code == 403 and gui.calls == []


def test_another_local_port_is_not_this_origin(gui):
    r = gui.app.test_client().get("/settings", headers=headers(gui, Origin="http://localhost:8099",
                                                               **{"Sec-Fetch-Site": "same-site"}))
    assert r.status_code == 403


@pytest.mark.parametrize("host", ["attacker.example", "attacker.example:7860", "127.0.0.1.nip.io", "192.168.1.5:7860"])
def test_dns_rebinding_is_refused_even_on_the_public_page(gui, host):
    c = gui.app.test_client()
    assert c.get("/", headers={"Host": host}).status_code == 403
    assert c.get("/settings", headers=headers(gui, Host=host)).status_code == 403


def test_a_null_origin_is_refused(gui):
    assert gui.app.test_client().get("/settings", headers=headers(gui, Origin="null")).status_code == 403


def test_the_page_itself_can_use_everything(gui):
    c = gui.app.test_client()
    r = c.post("/tasks/add", json={"prompt": "p", "schedule": "hourly"},
               headers=headers(gui, Origin="http://localhost", **{"Sec-Fetch-Site": "same-origin"}))
    assert r.status_code == 200 and ("install", "p") in gui.calls


def test_stream_accepts_the_token_in_the_url(gui):
    r = gui.app.test_client().get("/stream", query_string={"q": "hi", "session": "x", "t": gui.gui_token()})
    assert r.status_code == 200 and ("turn", "x") in gui.calls


def test_the_token_in_the_url_only_works_for_the_stream(gui):
    assert gui.app.test_client().get("/settings", query_string={"t": gui.gui_token()}).status_code == 403


def test_a_bad_schedule_is_a_clear_400(gui):
    r = gui.app.test_client().post("/tasks/add", json={"prompt": "p", "schedule": "whenever"}, headers=headers(gui))
    assert r.status_code == 400 and "can't read" in r.get_json()["error"]


def test_settings_route_cannot_write_tasks_or_mcp(gui):
    import cortex.storage.settings_store as ss
    gui.app.test_client().post("/settings", json={"name": "A", "mcp_servers": [{"name": "x", "command": "id"}],
                                                  "tasks": [{"id": "z"}]}, headers=headers(gui))
    s = ss.load()
    assert s["name"] == "A" and s["mcp_servers"] == [] and s["tasks"] == []


def test_the_page_has_a_strict_policy_and_no_inline_handlers(gui, monkeypatch):
    import re
    monkeypatch.setattr("cortex.core.context_reader.ContextReader.owner_name", staticmethod(lambda: "A"))
    r = gui.app.test_client().get("/", headers={"Sec-Fetch-Dest": "document", "Sec-Fetch-Site": "none"})
    csp = r.headers["Content-Security-Policy"]
    nonce = re.search(r"'nonce-([^']+)'", csp).group(1)
    assert "script-src-attr 'none'" in csp and "unsafe-inline" not in csp.split("style-src")[0]
    html = r.get_data(as_text=True)
    assert f'<script nonce="{nonce}">' in html
    assert not re.search(r"\son(click|change|input|error|load|focus|mouseover)=", html)


def test_opening_the_app_from_a_link_still_works(gui, monkeypatch):
    monkeypatch.setattr("cortex.core.context_reader.ContextReader.owner_name", staticmethod(lambda: "A"))
    r = gui.app.test_client().get("/", headers={"Sec-Fetch-Dest": "document", "Sec-Fetch-Site": "cross-site"})
    assert r.status_code == 200


def test_a_name_cannot_inject_html(gui, monkeypatch):
    monkeypatch.setattr("cortex.core.context_reader.ContextReader.owner_name",
                        staticmethod(lambda: '<img src=x onerror=alert(1)>'))
    html = gui.app.test_client().get("/").get_data(as_text=True)
    assert "<img src=x" not in html

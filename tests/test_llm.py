"""Routing and the wire format, against a real HTTP server standing in for Ollama."""
import pytest

from cortex.llm import llm
from fake_ollama import FakeOllama


@pytest.fixture
def fake():
    f = FakeOllama()
    yield f
    f.close()


@pytest.fixture
def cloud(fake, monkeypatch):
    monkeypatch.setattr(llm, "OLLAMA_CLOUD_URL", fake.url)
    fake.models = ["gemma4:31b", "gpt-oss:120b", "minimax-m3"]
    return fake


CFG = {"ollama": {"model": "local:1b", "base_url": "http://127.0.0.1:9", "num_ctx": 16384}}


@pytest.mark.parametrize("model,provider", [
    ("gemma4-agent:12b-mlx", "ollama"), ("ollama-cloud/gemma4:31b", "ollama_cloud"),
    ("ollama-cloud/minimax-m3", "ollama_cloud"), ("gpt-oss:120b-cloud", "ollama"), ("minimax-m3:cloud", "ollama"),
    ("claude-opus-5-5", "anthropic"), ("gpt-4o", "openai"), ("gemini-2.5-pro", "gemini"),
])
def test_routing_without_a_key(model, provider):
    assert llm.provider_for_model(model) == provider


def test_daemon_cloud_tags_take_the_direct_route_with_a_key():
    llm.set_key("ollama", "k")
    assert llm.provider_for_model("gpt-oss:120b-cloud") == "ollama_cloud"
    assert llm.provider_for_model("minimax-m3:cloud") == "ollama_cloud"


def test_cloud_list_is_public_and_needs_no_local_ollama(cloud):
    assert "ollama-cloud/gemma4:31b" in llm.list_provider_models("ollama")
    assert "Authorization" not in cloud.tag_headers[0]


def test_the_windows_case_cloud_models_show_with_no_ollama_and_no_key(cloud):
    groups = llm.available_models(CFG)
    assert [g["label"] for g in groups] == ["Ollama Cloud"]
    assert all(m["needs_key"] for m in groups[0]["models"])


def test_with_a_key_they_are_usable(cloud):
    llm.set_key("ollama", "k")
    llm._MODEL_CACHE.clear()
    assert not any(m["needs_key"] for m in llm.available_models(CFG)[0]["models"])


def test_a_cloud_model_cannot_be_chosen_without_a_key():
    assert "key" in llm.choose_model("ollama-cloud/gemma4:31b")
    assert llm.choose_model("local:1b") is None


@pytest.mark.parametrize("given,wire", [
    ("ollama-cloud/gemma4:31b", "gemma4:31b"), ("gpt-oss:120b-cloud", "gpt-oss:120b"),
    ("gemma4:cloud", "gemma4:31b"), ("minimax-m3:cloud", "minimax-m3"),
])
def test_cloud_names_become_what_the_api_expects(cloud, given, wire):
    assert llm.cloud_model_name(given) == wire


def test_cloud_call_uses_the_key_and_no_local_knobs(cloud):
    llm.set_key("ollama", "secret-key")
    b = llm.backend_for_model("ollama-cloud/gemma4:31b", CFG)
    b.complete([{"role": "user", "content": "hi"}])
    req = cloud.requests[-1]
    assert req["_auth"] == "Bearer secret-key" and req["model"] == "gemma4:31b"
    assert "keep_alive" not in req and "options" not in req


def test_local_call_sends_context_size_and_no_key(fake):
    b = llm.OllamaBackend(fake.url, "m:1", num_ctx=16384)
    b.complete([{"role": "user", "content": "hi"}])
    req = fake.requests[-1]
    assert req["_auth"] is None and req["options"]["num_ctx"] == 16384 and req["keep_alive"]


def test_context_size_only_ever_raises_a_models_own(fake):
    fake.parameters = "num_ctx                        32768\ntemperature 1"
    b = llm.OllamaBackend(fake.url, "m:1", num_ctx=16384)
    b.complete([{"role": "user", "content": "hi"}])
    assert "options" not in fake.requests[-1]
    fake.parameters = "num_ctx 4096"
    b = llm.OllamaBackend(fake.url, "m:2", num_ctx=16384)
    b.complete([{"role": "user", "content": "hi"}])
    assert fake.requests[-1]["options"]["num_ctx"] == 16384


def test_thinking_is_off_and_output_capped_locally(fake):
    fake.capabilities = ["completion", "tools", "thinking"]
    b = llm.OllamaBackend(fake.url, "m:1", think=False, max_tokens=2048)
    b.complete([{"role": "user", "content": "hi"}])
    req = fake.requests[-1]
    assert req["think"] is False and req["options"]["num_predict"] == 2048


def test_think_is_not_sent_to_a_model_that_cannot_think(fake):
    b = llm.OllamaBackend(fake.url, "m:1", think=False)
    b.complete([{"role": "user", "content": "hi"}])
    assert "think" not in fake.requests[-1]


def test_a_stalled_local_model_does_not_load_a_second_local_model(monkeypatch):
    import httpx

    class Stalls(llm.OllamaBackend):
        tried = 0

        def complete(self, *a, **k):
            type(self).tried += 1
            raise httpx.ReadTimeout("slow")

    class Rescue(llm.LLMBackend):
        def complete(self, *a, **k):
            return "from the cloud"

    router = llm.LLMRouter(Stalls(model="a"), Stalls(model="b"), Rescue())
    assert router.complete([]) == "from the cloud" and Stalls.tried == 1


def test_a_missing_local_model_still_tries_the_fallback():
    import httpx

    class Refuses(llm.OllamaBackend):
        def complete(self, *a, **k):
            raise httpx.ConnectError("no")

    class Works(llm.OllamaBackend):
        def complete(self, *a, **k):
            return "ok"

    assert llm.LLMRouter(Refuses(model="a"), Works(model="b")).complete([]) == "ok"


def test_saved_choice_survives_into_a_new_router(fake, monkeypatch):
    monkeypatch.setattr(llm, "OLLAMA_CLOUD_URL", fake.url)
    llm.choose_model("other:2b")
    assert llm.create_router(CFG).primary.model == "other:2b"


def test_saved_cloud_choice_is_ignored_once_its_key_is_gone(fake, monkeypatch):
    monkeypatch.setattr(llm, "OLLAMA_CLOUD_URL", fake.url)
    llm.set_key("ollama", "k")
    llm.choose_model("ollama-cloud/gemma4:31b")
    llm.delete_key("ollama")
    assert llm.create_router(CFG).primary.model == "local:1b"


def test_nothing_local_running_falls_over_to_the_cloud(cloud):
    llm.set_key("ollama", "k")
    cloud.reply = lambda body: {"content": "from the cloud"}
    router = llm.create_router(CFG)
    assert router.is_local() and router.rescue.model == "gemma4:31b"
    out = list(router.stream_with_tools([{"role": "user", "content": "hi"}], []))
    assert out[-1][1]["content"] == "from the cloud"
    assert cloud.requests[-1]["_auth"] == "Bearer k"


def test_no_rescue_without_a_key():
    assert llm.create_router(CFG).rescue is None


def test_router_reports_the_chosen_models_error():
    router = llm.LLMRouter(llm.OllamaBackend("http://127.0.0.1:9", "a"), llm.OllamaBackend("http://127.0.0.1:9", "b"))
    with pytest.raises(Exception):
        router.complete([{"role": "user", "content": "x"}])


def test_streaming_delivers_pieces_then_the_whole(fake):
    fake.reply = lambda body: {"content": "a reply that comes in pieces"}
    out = list(llm.OllamaBackend(fake.url, "m:1").stream_with_tools([{"role": "user", "content": "x"}], []))
    deltas = [p for k, p in out if k == "delta"]
    assert len(deltas) > 1 and "".join(deltas) == "a reply that comes in pieces"
    assert out[-1][0] == "done" and out[-1][1]["content"] == "a reply that comes in pieces"


def test_streaming_returns_tool_calls(fake):
    fake.reply = lambda body: {"tool_calls": [{"function": {"name": "execute_shell", "arguments": {"command": "ls"}}}]}
    done = list(llm.OllamaBackend(fake.url, "m:1").stream_with_tools([{"role": "user", "content": "x"}], [{}]))[-1][1]
    assert done["tool_calls"][0]["function"] == {"name": "execute_shell", "arguments": {"command": "ls"}}


def test_tool_results_go_out_in_ollamas_format(fake):
    b = llm.OllamaBackend(fake.url, "m:1")
    b.complete([{"role": "assistant", "content": "", "tool_calls": [
        {"id": "c0", "function": {"name": "execute_shell", "arguments": {"command": "ls"}}}]},
        {"role": "tool", "tool_call_id": "c0", "tool_name": "execute_shell", "content": "ok"}])
    sent = fake.requests[-1]["messages"]
    assert sent[1] == {"role": "tool", "content": "ok", "tool_name": "execute_shell"}
    assert "id" not in sent[0]["tool_calls"][0]


def test_a_model_without_tools_still_answers(fake):
    fake.reply = lambda body: {"content": "plain answer"}
    b = llm.OllamaBackend(fake.url, "no tools:1")
    out = list(b.stream_with_tools([{"role": "user", "content": "x"}], [{"type": "function"}]))
    assert out[-1][1]["content"] == "plain answer" and b._no_tools
    assert b.complete_with_tools([{"role": "user", "content": "x"}], [{}])["content"] == "plain answer"


def test_a_rejected_key_gives_a_readable_error(fake, monkeypatch, make_core):
    import httpx
    fake.chat_status = 401
    b = llm.OllamaBackend(fake.url, "m:1", api_key="bad")
    with pytest.raises(httpx.HTTPStatusError) as e:
        list(b.stream_with_tools([{"role": "user", "content": "x"}], []))
    assert "API key was rejected" in make_core()._llm_error_message(e.value)


def _route(fake, text):
    fake.reply = lambda body: {"content": text}
    out = list(llm.LLMRouter(llm.OllamaBackend(fake.url, "m:1")).stream_with_tools([{"role": "user", "content": "x"}], []))
    return "".join(p for k, p in out if k == "delta"), out[-1][1]["content"]


@pytest.mark.parametrize("leak", ["thought\n<channel|>", "<|channel>thought\nThe user wants this.<channel|>"])
def test_the_models_reasoning_never_reaches_the_reply(fake, leak):
    assert _route(fake, leak + "Done, all checked.") == ("Done, all checked.", "Done, all checked.")


def test_a_reply_that_merely_starts_like_reasoning_is_kept(fake):
    assert _route(fake, "thought so too, honestly") == ("thought so too, honestly", "thought so too, honestly")


def test_an_ordinary_reply_still_comes_in_pieces(fake):
    fake.reply = lambda body: {"content": "a reply that comes in pieces"}
    out = list(llm.LLMRouter(llm.OllamaBackend(fake.url, "m:1")).stream_with_tools([{"role": "user", "content": "x"}], []))
    assert len([p for k, p in out if k == "delta"]) > 1


def test_a_reply_written_in_one_go_loses_the_reasoning_too(fake):
    fake.reply = lambda body: {"content": "thought\n<channel|>Hello."}
    router = llm.LLMRouter(llm.OllamaBackend(fake.url, "m:1"))
    assert router.complete([{"role": "user", "content": "x"}]) == "Hello."
    assert router.complete_with_tools([{"role": "user", "content": "x"}], [])["content"] == "Hello."


@pytest.fixture
def limit():
    token = llm.quiet_limit.set(0.3)
    yield
    llm.quiet_limit.reset(token)


ASK = [{"role": "user", "content": "x"}]


def test_a_model_showing_nothing_in_time_is_dropped(fake, limit):
    fake.hold, fake.reply = 1, lambda body: {"content": "too late"}
    with pytest.raises(llm.WentQuiet):
        list(llm.LLMRouter(llm.OllamaBackend(fake.url, "m:1")).stream_with_tools(ASK, []))


def test_going_quiet_doesnt_wake_a_second_model(fake, limit):
    fake.hold, fake.reply = 1, lambda body: {"content": "too late"}
    router = llm.LLMRouter(llm.OllamaBackend(fake.url, "a:1"), llm.OllamaBackend(fake.url, "b:1"))
    with pytest.raises(llm.WentQuiet):
        list(router.stream_with_tools(ASK, []))
    assert [r["model"] for r in fake.requests] == ["a:1"]


def test_a_slow_reply_already_under_way_is_left_alone(fake, limit):
    fake.chunk_delay, fake.reply = 0.5, lambda body: {"content": "slow but it gets there"}
    out = list(llm.LLMRouter(llm.OllamaBackend(fake.url, "m:1")).stream_with_tools(ASK, []))
    assert out[-1][1]["content"] == "slow but it gets there"


def test_without_a_limit_a_slow_start_is_fine(fake):
    fake.hold, fake.reply = 0.5, lambda body: {"content": "in my own time"}
    assert list(llm.OllamaBackend(fake.url, "m:1").stream_with_tools(ASK, []))[-1][1]["content"] == "in my own time"


def test_a_cloud_model_has_no_limit(fake, limit):
    fake.hold, fake.reply = 0.5, lambda body: {"content": "from far away"}
    backend = llm.OllamaBackend(fake.url, "m:1", api_key="k")
    assert list(backend.stream_with_tools(ASK, []))[-1][1]["content"] == "from far away"


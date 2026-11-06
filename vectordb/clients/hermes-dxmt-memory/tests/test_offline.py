"""The whole provider loop against an in-memory stand-in for the server. No network."""
from __future__ import annotations

import json
import time
from typing import Any

import pytest


class FakeDXMT:
    """Enough of ``dxmt.DXMTClient``'s memory API for the provider: id-addressed memories, tag
    filters that mean "any of", kind filter, duplicate detection by text, supersede, newest-first
    listing. Scores are keyword overlap so recall has an order."""

    def __init__(self) -> None:
        self.memories: dict[str, dict[str, Any]] = {}
        self.calls: list[str] = []
        self.closed = False

    def remember(self, text, kind=None, tags=None, source=None, supersedes=None, memory_id=None, metadata=None):
        self.calls.append("remember")
        for m in self.memories.values():  # the server deduplicates within one kind
            if m["text"] == text and m["kind"] == (kind or "note") and not m.get("superseded_by") and memory_id is None:
                return {"id": m["id"], "superseded": [], "duplicate": True}
        mid = memory_id or f"mem-{len(self.memories) + 1:04d}"
        row = {"id": mid, "text": text, "kind": kind or "note", "tags": list(tags or []), "source": source,
               "at": int(time.time()) + len(self.memories), **(metadata or {})}
        self.memories[mid] = row
        done = []
        for old in supersedes or []:
            if old in self.memories:
                self.memories[old]["superseded_by"] = mid
                done.append(old)
        return {"id": mid, "superseded": done}

    def recall(self, query, limit=None, kind=None, tags=None, since=None, include_superseded=False):
        self.calls.append("recall")
        words = set(query.lower().split())
        hits = []
        for m in self.memories.values():
            if m.get("superseded_by") and not include_superseded:
                continue
            if kind and m["kind"] != kind:
                continue
            if tags and not set(tags) & set(m["tags"]):
                continue
            overlap = len(words & set(m["text"].lower().split()))
            hits.append({**m, "score": overlap / max(len(words), 1)})
        hits.sort(key=lambda h: (-h["score"], -h["at"]))
        return hits[: limit or 25]

    def forget(self, memory_id):
        self.calls.append("forget")
        return 1 if self.memories.pop(memory_id, None) else 0

    def list_memories(self, kind=None, limit=50, offset=0):
        self.calls.append("list")
        rows = [m for m in sorted(self.memories.values(), key=lambda m: -m["at"]) if not kind or m["kind"] == kind]
        return {"memories": rows[offset: offset + limit], "total": len(rows)}

    def health_check(self):
        return {"status": "Healthy"}

    def close(self):
        self.closed = True


@pytest.fixture
def provider(plugin, tmp_path, monkeypatch):
    monkeypatch.delenv("DXMT_URL", raising=False)
    store_mod = __import__(f"{plugin.__name__}.src.store", fromlist=["client_from_url"])
    fake = FakeDXMT()
    seen = {}

    def fake_client(url, *, api_key, timeout):
        seen["url"], seen["api_key"], seen["timeout"] = url, api_key, timeout
        return fake

    monkeypatch.setattr(store_mod, "client_from_url", fake_client)
    monkeypatch.setattr(f"{plugin.__name__}.src.provider.client_from_url", fake_client)
    p = plugin.DXMTMemoryProvider()
    p.initialize("session-1", hermes_home=str(tmp_path), platform="cli", agent_context="primary",
                 agent_identity="tester", agent_workspace="ws-a", user_id="u1")
    p.fake, p.seen = fake, seen
    yield p
    p.shutdown()


def test_registers_with_the_plugin_context(plugin):
    class Collector:  # the memory activation path: register_* only, no llm
        def __init__(self):
            self.providers = []

        def register_memory_provider(self, provider):
            self.providers.append(provider)

        def register_auxiliary_task(self, key, **kwargs):
            self.task = (key, kwargs)

        def __getattr__(self, attr):
            raise AttributeError(attr)

    ctx = Collector()
    plugin.register(ctx)
    assert len(ctx.providers) == 1 and ctx.providers[0].name == "dxmt"
    assert ctx.providers[0].is_available() is True
    assert ctx.providers[0].llm is None
    assert ctx.task[0] == "dxmt_extraction" and ctx.task[1]["display_name"]

    class Full(Collector):  # a real PluginContext offers the facade
        llm = object()

    ctx = Full()
    plugin.register(ctx)
    assert ctx.providers[0].llm is Full.llm


def test_extraction_prefers_the_host_llm_facade(provider):
    seen = {}

    class Facade:
        def complete_structured(self, **kwargs):
            seen.update(kwargs)
            parsed = {"facts": [{"content": "The user writes Rust.", "category": "pattern", "evidence": [0]}]}
            return type("R", (), {"parsed": parsed, "text": json.dumps(parsed)})()

    provider.llm = Facade()
    messages = [{"role": "user", "content": f"turn {i}"} for i in range(3)]
    assert provider.on_pre_compress(messages).endswith("- The user writes Rust.")
    assert seen["task"] == "dxmt_extraction" and seen["json_mode"] is True
    assert "[0] user: turn 0" in seen["input"][0]["text"]
    assert any(m["text"] == "The user writes Rust." for m in provider.fake.memories.values())


def test_initialize_binds_the_configured_server(provider, tmp_path):
    assert provider.seen["url"] == "http://localhost:9999"
    assert provider.seen["api_key"] is None
    assert provider.seen["timeout"] == 15.0


def test_remember_recall_read_forget_loop(provider):
    stored = json.loads(provider.handle_tool_call("dxmt_remember", {
        "content": "The user prefers pytest over unittest.", "category": "preference", "tags": ["testing"]}))
    assert stored["status"] == "stored"
    mid = stored["id"]
    row = provider.fake.memories[mid]
    assert row["kind"] == "preference"
    assert set(row["tags"]) >= {"hermes", "hermes:ws:ws-a", "hermes:user:u1", "hermes:agent:tester", "testing"}
    assert row["origin"] == "remember" and row["session_id"] == "session-1"

    same = {"content": "The user prefers pytest over unittest.", "category": "preference"}
    again = json.loads(provider.handle_tool_call("dxmt_remember", same))
    assert again["status"] == "exists" and again["id"] == mid
    # Dedup is per category on the server: the same words filed elsewhere are another memory.
    other = json.loads(provider.handle_tool_call("dxmt_remember", {**same, "category": "general"}))
    assert other["status"] == "stored" and other["id"] != mid
    provider.store.forget(other["id"])

    recalled = json.loads(provider.handle_tool_call("dxmt_recall", {"query": "pytest preference"}))
    assert recalled["total"] == 1
    hit = recalled["results"][0]
    assert hit["id"] == mid and hit["kind"] == "fact" and hit["category"] == "preference"
    assert hit["tags"] == ["testing"]  # scope tags are not shown to the model
    assert hit["score"] > 0

    read = json.loads(provider.handle_tool_call("dxmt_read", {"id": mid}))
    assert read["memory"]["content"] == "The user prefers pytest over unittest."

    preview = json.loads(provider.handle_tool_call("dxmt_forget", {"action": "preview", "query": "pytest"}))
    assert preview["candidates"][0]["id"] == mid
    deleted = json.loads(provider.handle_tool_call("dxmt_forget", {"action": "delete", "id": mid}))
    assert deleted["deleted"]["id"] == mid
    assert mid not in provider.fake.memories
    missing = json.loads(provider.handle_tool_call("dxmt_read", {"id": mid}))
    assert "error" in missing


def test_supersede_hides_the_old_fact(provider):
    berlin = {"content": "The user lives in Berlin.", "category": "entity"}
    old = json.loads(provider.handle_tool_call("dxmt_remember", berlin))
    new = json.loads(provider.handle_tool_call("dxmt_remember", {
        "content": "The user lives in Munich.", "category": "entity", "supersedes": [old["id"]]}))
    assert new["superseded"] == [old["id"]]
    ids = [r["id"] for r in json.loads(provider.handle_tool_call("dxmt_recall", {"query": "user lives"}))["results"]]
    assert ids == [new["id"]]


def test_recall_is_scoped_to_the_workspace_and_excludes_turns(provider):
    provider.fake.remember("Other workspace secret plan", kind="general", tags=["hermes", "hermes:ws:ws-b"])
    provider.fake.remember("Shared note from REST with no tags", kind="note", tags=[])
    provider.handle_tool_call("dxmt_remember", {"content": "Our plan is a secret plan", "category": "general"})
    provider.sync_turn("tell me the plan", "the plan is secret", session_id="session-1")
    provider.store.writer.flush()

    facts = json.loads(provider.handle_tool_call("dxmt_recall", {"query": "secret plan"}))["results"]
    assert [f["snippet"] for f in facts] == ["Our plan is a secret plan"]
    turns = json.loads(provider.handle_tool_call("dxmt_recall", {"query": "plan", "kind": "turn"}))["results"]
    assert {t["snippet"] for t in turns} == {"tell me the plan", "the plan is secret"}
    assert all(t["kind"] == "turn" for t in turns)

    provider.config["retrieval"]["isolate_workspace"] = False
    everything = json.loads(provider.handle_tool_call("dxmt_recall", {"query": "secret plan", "kind": "any"}))
    everything = everything["results"]
    assert len(everything) == 5


def test_turns_are_idempotent_and_carry_provenance_ids(provider, plugin):
    store_mod = __import__(f"{plugin.__name__}.src.store", fromlist=["stable_turn_id"])
    provider.sync_turn("hello", "hi there", session_id="session-1")
    provider.store.writer.flush()
    assert set(provider.fake.memories) == {
        store_mod.stable_turn_id("session-1", 0, "user", "hello"),
        store_mod.stable_turn_id("session-1", 1, "assistant", "hi there"),
    }
    turn = provider.fake.memories[store_mod.stable_turn_id("session-1", 0, "user", "hello")]
    assert (turn["kind"], turn["role"], turn["turn_index"], turn["source"]) == ("turn", "user", 0, "session-1")


def test_non_primary_contexts_do_not_write(plugin, tmp_path, monkeypatch):
    fake = FakeDXMT()
    monkeypatch.setattr(f"{plugin.__name__}.src.provider.client_from_url", lambda *a, **k: fake)
    p = plugin.DXMTMemoryProvider()
    p.initialize("s", hermes_home=str(tmp_path), agent_context="cron")
    p.sync_turn("a", "b")
    p.on_memory_write("add", "user", "likes tea")
    p.on_session_end([{"role": "user", "content": "x"}] * 5)
    p.shutdown()
    assert fake.memories == {}


def test_prefetch_is_asynchronous_and_formatted(provider):
    tea = "The user drinks oolong tea every morning."
    provider.handle_tool_call("dxmt_remember", {"content": tea, "category": "preference"})
    assert provider.prefetch("anything") == ""
    provider.queue_prefetch("what tea does the user drink")
    block = provider.prefetch("what tea does the user drink")
    assert block.startswith("## DXMT Memory")
    assert f"(preference, id=mem-0001) {tea}" in block
    assert provider.prefetch("again") == ""  # consumed once


def test_extraction_stores_facts_with_provenance(provider, plugin, monkeypatch):
    messages = [
        {"role": "user", "content": "I moved to Lisbon last month."},
        {"role": "assistant", "content": "Noted."},
        {"role": "user", "content": "And I prefer short answers."},
        {"role": "assistant", "content": "Sure."},
        {"role": "user", "content": "Thanks."},
    ]
    seen = {}

    def fake_call_llm(task, *, messages, timeout, extra_body):
        seen.update(task=task, extra_body=extra_body, prompt=messages[1]["content"])
        return type("R", (), {"content": json.dumps({"facts": [
            {"content": "The user lives in Lisbon.", "category": "entity", "tags": ["home"], "evidence": [0]},
            {"content": "The user prefers short answers.", "category": "preference", "evidence": [2, 99]},
            {"content": "", "category": "general"},
        ]})})()

    import sys
    import types
    aux = types.ModuleType("agent.auxiliary_client")
    aux.call_llm = fake_call_llm
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", aux)

    summary = provider.on_pre_compress(messages)
    assert seen["task"] == "dxmt_extraction"
    assert seen["extra_body"] == {"response_format": {"type": "json_object"}}
    assert "[0] user: I moved to Lisbon last month." in seen["prompt"]
    assert summary.splitlines()[1:] == ["- The user lives in Lisbon.", "- The user prefers short answers."]

    store_mod = __import__(f"{plugin.__name__}.src.store", fromlist=["stable_turn_id"])
    lisbon = next(m for m in provider.fake.memories.values() if "Lisbon" in m["text"])
    assert lisbon["kind"] == "entity" and lisbon["origin"] == "pre_compress"
    assert lisbon["provenance"] == [store_mod.stable_turn_id("session-1", 0, "user", "I moved to Lisbon last month.")]

    # The same session ending again adds nothing: the server reports duplicates.
    provider.on_session_end(messages)
    assert sum(1 for m in provider.fake.memories.values() if m["kind"] != "turn") == 2
    assert provider.on_pre_compress(messages) == ""


def test_extraction_skips_short_sessions_and_bad_json(provider, plugin, monkeypatch):
    import sys
    import types
    calls = []
    aux = types.ModuleType("agent.auxiliary_client")
    aux.call_llm = lambda *a, **k: (calls.append(1), type("R", (), {"content": "not json"})())[1]
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", aux)
    provider.on_session_end([{"role": "user", "content": "hi"}])
    assert calls == []
    provider.on_session_end([{"role": "user", "content": f"m{i}"} for i in range(3)])
    assert calls == [1]
    assert provider.fake.memories == {}


def test_memory_write_mirror(provider):
    provider.on_memory_write("add", "user", "Prefers dark mode")
    provider.on_memory_write("replace", "memory", "ignored", {"previous_content": "x"})
    rows = list(provider.fake.memories.values())
    assert len(rows) == 1 and rows[0]["kind"] == "preference" and rows[0]["origin"] == "memory_write_mirror"


def test_server_errors_become_tool_errors(provider):
    def boom(*a, **k):
        raise ConnectionError("server down")

    provider.fake.recall = boom
    out = json.loads(provider.handle_tool_call("dxmt_recall", {"query": "x"}))
    assert "server down" in out["error"]
    assert json.loads(provider.handle_tool_call("dxmt_nope", {}))["error"].startswith("Unknown")
    assert "required" in json.loads(provider.handle_tool_call("dxmt_remember", {}))["error"]


def test_shutdown_flushes_and_closes(provider):
    provider.sync_turn("a", "b")
    provider.shutdown()
    assert provider.fake.closed is True
    assert len(provider.fake.memories) == 2


def test_setup_schema_and_save_config(provider, tmp_path):
    schema = {f["key"]: f for f in provider.get_config_schema()}
    assert schema["url"]["default"] == "http://localhost:9999"
    assert schema["api_key"]["secret"] is True and schema["api_key"]["env_var"] == "DXMT_API_KEY"
    provider.save_config({"url": "http://db.example:7000"}, str(tmp_path))
    assert provider.config["server"]["url"] == "http://db.example:7000"


def test_url_parsing(plugin, monkeypatch):
    store_mod = __import__(f"{plugin.__name__}.src.store", fromlist=["client_from_url"])
    import dxmt
    seen = {}
    monkeypatch.setattr(dxmt, "DXMTClient", lambda **kw: seen.update(kw) or object())
    store_mod.client_from_url("https://db.example", api_key="k", timeout=3)
    assert seen == {"host": "db.example", "port": 443, "ssl": True, "api_key": "k", "timeout": 3}
    store_mod.client_from_url("localhost:9999", api_key=None, timeout=3)
    assert (seen["host"], seen["port"], seen["ssl"]) == ("localhost", 9999, False)
    with pytest.raises(ValueError):
        store_mod.client_from_url("ftp://x", api_key=None, timeout=3)

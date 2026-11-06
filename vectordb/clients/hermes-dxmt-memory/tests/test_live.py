"""The same loop against a real DXMT server (``DXMT_TEST_URL``) with an embedding model, driven
through Hermes's own ``MemoryManager`` so the provider is exercised the way the agent calls it."""
from __future__ import annotations

import json
import uuid

import pytest

pytestmark = pytest.mark.live


@pytest.fixture
def manager(plugin, live_url, tmp_path, monkeypatch):
    from agent.memory_manager import MemoryManager

    monkeypatch.setenv("DXMT_URL", live_url)
    # The manager refuses a tool call whose caller profile is not the one the provider was
    # initialized for, so the test's profile home is the one passed to initialize_all.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    provider = plugin.DXMTMemoryProvider()
    mgr = MemoryManager()
    mgr.add_provider(provider)
    ws = f"ws-{uuid.uuid4().hex[:8]}"
    mgr.initialize_all(f"session-{ws}", hermes_home=str(tmp_path), platform="cli", agent_context="primary",
                       agent_identity="live-test", agent_workspace=ws, user_id="alice")
    yield mgr, provider
    # Clean up what this test wrote.
    for row in provider.store.client.recall("the", limit=100, tags=[f"hermes:ws:{ws}"]):
        provider.store.forget(row["id"])
    mgr.shutdown_all()


def test_live_loop_through_memory_manager(manager):
    mgr, provider = manager
    schemas = {s["name"] for s in provider.get_tool_schemas()}
    assert schemas == {"dxmt_recall", "dxmt_remember", "dxmt_read", "dxmt_forget"}

    bees = "The user keeps bees on a rooftop in Lisbon and sells the honey at a Saturday market."
    args = {"content": bees, "category": "entity", "tags": ["hobby"]}
    stored = json.loads(mgr.handle_tool_call("dxmt_remember", args))
    assert stored["status"] == "stored"
    mid = stored["id"]
    assert json.loads(mgr.handle_tool_call("dxmt_remember", args))["status"] == "exists"

    # Recall by meaning: no word of the query appears in the memory.
    hits = json.loads(mgr.handle_tool_call("dxmt_recall", {"query": "apiculture and weekend trading"}))["results"]
    assert hits and hits[0]["id"] == mid and 0 < hits[0]["score"] <= 1.0

    # Turns go through the writer and come back under kind turn, scoped to this workspace.
    sid = provider._session_id
    mgr.sync_all("do bees sleep?", "They rest at night but do not sleep as mammals do.", session_id=sid)
    provider.store.writer.flush()
    turns = json.loads(mgr.handle_tool_call("dxmt_recall", {"query": "bees sleeping at night", "kind": "turn"}))
    turns = turns["results"]
    assert {t["snippet"] for t in turns} >= {"do bees sleep?"}

    # Prefetch injects facts, not turns, when a fact matches.
    provider.queue_prefetch("what does the user sell at the market")
    block = provider.prefetch("what does the user sell at the market")
    assert block.startswith("## DXMT Memory") and mid in block

    # Supersede: the old fact leaves recall and stays readable as history.
    newer = json.loads(mgr.handle_tool_call("dxmt_remember", {
        "content": "The user moved the beehives to Porto.", "category": "entity", "supersedes": [mid]}))
    assert newer["superseded"] == [mid]
    hits = json.loads(mgr.handle_tool_call("dxmt_recall", {"query": "where are the beehives"}))["results"]
    ids = [h["id"] for h in hits]
    assert newer["id"] in ids and mid not in ids

    read = json.loads(mgr.handle_tool_call("dxmt_read", {"id": newer["id"], "include_provenance": True}))
    assert read["memory"]["content"] == "The user moved the beehives to Porto."

    deleted = json.loads(mgr.handle_tool_call("dxmt_forget", {"action": "delete", "id": newer["id"]}))
    assert deleted["deleted"]["id"] == newer["id"]
    assert "error" in json.loads(mgr.handle_tool_call("dxmt_read", {"id": newer["id"]}))


def test_live_workspace_isolation(plugin, live_url, tmp_path, monkeypatch):
    monkeypatch.setenv("DXMT_URL", live_url)
    a, b = plugin.DXMTMemoryProvider(), plugin.DXMTMemoryProvider()
    wa, wb = f"ws-{uuid.uuid4().hex[:8]}", f"ws-{uuid.uuid4().hex[:8]}"
    a.initialize("sa", hermes_home=str(tmp_path), agent_workspace=wa)
    b.initialize("sb", hermes_home=str(tmp_path), agent_workspace=wb)
    try:
        ida = a.remember_fact("Workspace A's deploy key rotates on Mondays.", category="pattern")["id"]
        def ids(p):
            out = json.loads(p.handle_tool_call("dxmt_recall", {"query": "deploy key rotation"}))
            return [h["id"] for h in out["results"]]

        assert ids(b) == []
        assert ids(a) == [ida]
        b.config["retrieval"]["isolate_workspace"] = False
        assert ida in ids(b)
    finally:
        for p in (a, b):
            for row in p.store.client.recall("deploy", limit=50, tags=[f"hermes:ws:{p._agent_workspace}"]):
                p.store.forget(row["id"])
            p.shutdown()

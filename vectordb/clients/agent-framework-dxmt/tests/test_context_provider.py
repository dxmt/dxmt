"""The context provider: unit tests with a mocked client, and one run against a server."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock

import pytest
from agent_framework import AgentResponse, AgentSession, Message, SessionContext
from dxmt import DXMTClient

from agent_framework_dxmt import DXMTContextProvider


def _client() -> MagicMock:
    client = MagicMock(spec=DXMTClient)
    client.arecall = AsyncMock(return_value=[])
    client.aremember = AsyncMock(return_value={"id": "mem-1", "superseded": []})
    client.aforget = AsyncMock(return_value=1)  # one id per call, as the SDK's aforget takes
    client.aclose = AsyncMock()
    return client


def _context(user_text: str, assistant_text: str | None = None) -> SessionContext:
    context = SessionContext(input_messages=[Message(role="user", contents=[user_text])])
    if assistant_text is not None:
        context._response = AgentResponse(messages=[Message(role="assistant", contents=[assistant_text])])
    return context


def test_scope_is_required():
    with pytest.raises(ValueError, match="scope"):
        DXMTContextProvider(client=_client())


async def test_before_run_recalls_within_the_search_scope_and_adds_a_context_message():
    client = _client()
    client.arecall.return_value = [
        {"id": "m1", "text": "Prefers answers in Chinese"},
        {"id": "m2", "text": "Lives in Berlin"},
    ]
    provider = DXMTContextProvider(client=client, user_id="alice", search_user_id="alice", search_agent_id="helper")
    context = _context("how should you answer me?")
    await provider.before_run(agent=MagicMock(), session=AgentSession(), context=context, state={})
    client.arecall.assert_awaited_once_with("how should you answer me?", limit=10, tags=["user:alice", "agent:helper"])
    added = context.context_messages[provider.source_id]
    assert len(added) == 1
    assert added[0].role == "user"
    assert added[0].text.startswith("## Memories") and "Lives in Berlin" in added[0].text


async def test_before_run_without_a_search_scope_recalls_nothing_and_warns_once(caplog):
    client = _client()
    provider = DXMTContextProvider(client=client, user_id="alice")
    for _ in range(2):
        await provider.before_run(agent=MagicMock(), session=AgentSession(), context=_context("hi"), state={})
    client.arecall.assert_not_called()
    assert sum("no retrieval scope" in r.message for r in caplog.records) == 1


async def test_after_run_remembers_user_and_assistant_turns_under_the_storage_scope():
    client = _client()
    provider = DXMTContextProvider(
        client=client, user_id="alice", agent_id="helper", application_id="app", kind="event"
    )
    session = AgentSession(session_id="s-1")
    await provider.after_run(
        agent=MagicMock(), session=session, context=_context("I moved to Berlin", "Noted."), state={}
    )
    assert client.aremember.await_count == 2
    first = client.aremember.await_args_list[0]
    assert first.args == ("user: I moved to Berlin",)
    assert first.kwargs == {"kind": "event", "tags": ["user:alice", "agent:helper", "app:app"], "source": "session:s-1"}
    assert client.aremember.await_args_list[1].args == ("assistant: Noted.",)


async def test_assistant_turns_can_be_left_out():
    client = _client()
    provider = DXMTContextProvider(client=client, user_id="alice", store_assistant_messages=False)
    await provider.after_run(agent=MagicMock(), session=AgentSession(), context=_context("q", "a"), state={})
    assert client.aremember.await_count == 1


async def test_explicit_operations():
    client = _client()
    provider = DXMTContextProvider(client=client, user_id="alice", search_user_id="alice")
    assert (await provider.remember("Lives in Berlin", kind="fact", supersedes=["m0"]))["id"] == "mem-1"
    client.aremember.assert_awaited_with("Lives in Berlin", kind="fact", tags=["user:alice"], supersedes=["m0"])
    await provider.recall("where", limit=3)
    client.arecall.assert_awaited_with("where", limit=3, tags=["user:alice"])
    assert await provider.forget(["mem-1", "mem-2"]) == 2
    assert client.aforget.await_count == 2


async def test_a_recall_failure_is_logged_not_raised(caplog):
    from dxmt.exceptions import DXMTError

    client = _client()
    client.arecall.side_effect = DXMTError("503")
    provider = DXMTContextProvider(client=client, user_id="alice", search_user_id="alice")
    context = _context("hi")
    await provider.before_run(agent=MagicMock(), session=AgentSession(), context=context, state={})
    assert provider.source_id not in context.context_messages
    assert any("recall failed" in r.message for r in caplog.records)


@pytest.mark.skipif(not os.getenv("DXMT_TEST_EMBEDDING"), reason="memory needs a server with an embedding model")
@pytest.mark.integration
async def test_remember_then_recall_on_a_server(server_url):
    import uuid

    user = f"u-{uuid.uuid4().hex[:8]}"
    async with DXMTContextProvider(
        url=server_url, api_key=os.getenv("DXMT_TEST_API_KEY"), user_id=user, search_user_id=user
    ) as provider:
        session = AgentSession(session_id="s-live")
        await provider.after_run(
            agent=MagicMock(),
            session=session,
            context=_context("I prefer answers in Chinese.", "Understood."),
            state={},
        )
        context = _context("how should you answer me")
        await provider.before_run(agent=MagicMock(), session=session, context=context, state={})
        text = context.context_messages[provider.source_id][0].text
        assert "Chinese" in text
        ids = [m["id"] for m in await provider.recall("Chinese")]
        assert await provider.forget(ids) == len(ids)

"""DXMT memory as an Agent Framework context provider.

Before a run, the user's input is sent to DXMT's ``recall`` (hybrid: by meaning and by
keyword) and the memories come back into the context under a short heading; after a run,
the user and assistant messages are kept with ``remember``. Scope is tags: a memory is
tagged with the ``user_id``, ``agent_id`` and ``application_id`` it was written under, and
recall asks for the ``search_*`` ids -- kept separate from the storage scope, as the mem0
provider keeps them, so a memory written under a shared agent id is not read back for an
unrelated user unless ``search_agent_id`` says so.

DXMT's memory semantics come along: a memory said twice is kept once, a new fact can
``supersede`` an old one (kept as history, left out of recall), and ``forget`` removes on
request. The same store is reachable from every other agent on the machine over MCP and REST.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar, Self

from agent_framework import AgentSession, ContextProvider, Message, SecretString, SessionContext
from dxmt import DXMTClient
from dxmt.exceptions import DXMTError

from ._settings import _resolve_client

if TYPE_CHECKING:
    from agent_framework._agents import SupportsAgentRun

logger = logging.getLogger(__name__)


class DXMTContextProvider(ContextProvider):
    """Recall DXMT memories before a run and remember the exchange after it."""

    DEFAULT_CONTEXT_PROMPT = "## Memories\nConsider the following memories when answering user questions:"
    DEFAULT_SOURCE_ID: ClassVar[str] = "dxmt"

    def __init__(
        self,
        source_id: str = DEFAULT_SOURCE_ID,
        *,
        client: DXMTClient | None = None,
        url: str | None = None,
        api_key: str | SecretString | None = None,
        managed_client: bool | None = None,
        env_file_path: str | None = None,
        env_file_encoding: str | None = None,
        application_id: str | None = None,
        agent_id: str | None = None,
        user_id: str | None = None,
        search_application_id: str | None = None,
        search_agent_id: str | None = None,
        search_user_id: str | None = None,
        context_prompt: str | None = None,
        limit: int = 10,
        kind: str | None = None,
        store_user_messages: bool = True,
        store_assistant_messages: bool = True,
    ) -> None:
        """Initialize the provider.

        Args:
            source_id: Identifier of this provider in the session context.
            client: Existing ``DXMTClient``; otherwise one is built from ``url`` / ``api_key`` /
                ``DXMT_URL`` / ``DXMT_API_KEY`` and closed with the provider.
            url, api_key, managed_client, env_file_path, env_file_encoding: As on ``DXMTCollection``.
            application_id, agent_id, user_id: The storage scope, stamped as tags on every memory written.
            search_application_id, search_agent_id, search_user_id: The retrieval scope. Memories
                carrying any of these tags are recalled; none set means nothing is recalled.
            context_prompt: Heading placed above the recalled memories.
            limit: How many memories to recall per run.
            kind: Memory kind to write (``fact``, ``preference``, ``event``, ``instruction``, ``note``);
                default lets the server file user turns as ``note``.
            store_user_messages, store_assistant_messages: Which sides of the exchange to remember.
        """
        super().__init__(source_id)
        resolved, owns = _resolve_client(
            client,
            url=url,
            api_key=api_key,
            managed_client=managed_client,
            env_file_path=env_file_path,
            env_file_encoding=env_file_encoding,
        )
        self.client = resolved
        self.managed_client = owns
        self.application_id = application_id
        self.agent_id = agent_id
        self.user_id = user_id
        self.search_application_id = search_application_id
        self.search_agent_id = search_agent_id
        self.search_user_id = search_user_id
        self.context_prompt = context_prompt or self.DEFAULT_CONTEXT_PROMPT
        self.limit = limit
        self.kind = kind
        self.store_user_messages = store_user_messages
        self.store_assistant_messages = store_assistant_messages
        self._warned_no_search_scope = False
        self._closed = False
        if not (application_id or agent_id or user_id):
            raise ValueError("At least one of application_id, agent_id or user_id is required to scope memories.")

    # ---- scope as tags --------------------------------------------------------------

    @staticmethod
    def _tags(application_id: str | None, agent_id: str | None, user_id: str | None) -> list[str]:
        tags = []
        if user_id:
            tags.append(f"user:{user_id}")
        if agent_id:
            tags.append(f"agent:{agent_id}")
        if application_id:
            tags.append(f"app:{application_id}")
        return tags

    @property
    def storage_tags(self) -> list[str]:
        return self._tags(self.application_id, self.agent_id, self.user_id)

    @property
    def search_tags(self) -> list[str]:
        return self._tags(self.search_application_id, self.search_agent_id, self.search_user_id)

    # ---- hooks -----------------------------------------------------------------------

    async def before_run(
        self, *, agent: SupportsAgentRun, session: AgentSession, context: SessionContext, state: dict[str, Any]
    ) -> None:
        tags = self.search_tags
        if not tags:
            if not self._warned_no_search_scope:
                self._warned_no_search_scope = True
                logger.warning(
                    "DXMTContextProvider has no retrieval scope, so no memories will be recalled. "
                    "Set search_user_id, search_agent_id and/or search_application_id."
                )
            return
        query = "\n".join(m.text for m in context.input_messages if m and m.text and m.text.strip())
        if not query.strip():
            return
        try:
            memories = await self.client.arecall(query, limit=self.limit, tags=tags)
        except DXMTError as exc:
            logger.error("DXMT recall failed: %s", exc)
            return
        lines = [str(m.get("text", "")) for m in memories if m.get("text")]
        if lines:
            context.extend_messages(
                self.source_id, [Message(role="user", contents=[f"{self.context_prompt}\n" + "\n".join(lines)])]
            )

    async def after_run(
        self, *, agent: SupportsAgentRun, session: AgentSession, context: SessionContext, state: dict[str, Any]
    ) -> None:
        messages: list[Message] = list(context.input_messages)
        if context.response and context.response.messages:
            messages.extend(context.response.messages)
        tags = self.storage_tags
        source = f"session:{session.session_id}" if session.session_id else self.agent_id or self.source_id
        for message in messages:
            role = message.role.value if hasattr(message.role, "value") else str(message.role)
            if not message.text or not message.text.strip():
                continue
            if role == "user" and not self.store_user_messages:
                continue
            if role == "assistant" and not self.store_assistant_messages:
                continue
            if role not in {"user", "assistant"}:
                continue
            try:
                await self.client.aremember(f"{role}: {message.text.strip()}", kind=self.kind, tags=tags, source=source)
            except DXMTError as exc:
                logger.error("DXMT remember failed: %s", exc)

    # ---- explicit memory operations ---------------------------------------------------

    async def remember(
        self, text: str, *, kind: str | None = None, supersedes: list[str] | None = None
    ) -> dict[str, Any]:
        """Keep a memory under the storage scope; returns the server's answer (``id``, ``superseded``)."""
        return await self.client.aremember(text, kind=kind or self.kind, tags=self.storage_tags, supersedes=supersedes)

    async def recall(self, query: str, *, limit: int | None = None) -> list[dict[str, Any]]:
        """Memories matching ``query`` within the retrieval scope."""
        return await self.client.arecall(query, limit=limit or self.limit, tags=self.search_tags or None)

    async def forget(self, ids: list[str]) -> int:
        """Delete memories by id; returns how many there were."""
        forgotten = 0
        for memory_id in ids:
            forgotten += await self.client.aforget(memory_id)
        return forgotten

    # ---- lifecycle -------------------------------------------------------------------

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        if self.managed_client and not self._closed:
            await self.client.aclose()
            self._closed = True

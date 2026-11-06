"""The storage backend under CrewAI's real ``Memory`` engine, against a live server.

The LLM is a mock (so scope, categories and importance are given explicitly, as the upstream
unified-memory tests do); the embedder is deterministic. What is being checked is that the
backend answers every protocol call the engine makes the way the LanceDB default does.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest
from crewai import Memory
from crewai.memory.storage.backend import EmbeddingDimensionMismatchError, StorageBackend
from crewai.memory.types import MemoryRecord

from conftest import API_KEY, DIM, HOST, PORT, embed
from crewai_dxmt import DXMTStorageBackend

pytestmark = pytest.mark.integration


@pytest.fixture
def backend(table_name):
    return DXMTStorageBackend(table_name, host=HOST, port=PORT, api_key=API_KEY)


@pytest.fixture
def memory(backend):
    return Memory(storage=backend, llm=MagicMock(), embedder=embed)


def rec(content, scope="/", categories=(), **kw):
    r = MemoryRecord(content=content, scope=scope, categories=list(categories), **kw)
    r.embedding = embed([content])[0]
    return r


class TestProtocol:
    def test_is_a_storage_backend(self, backend):
        assert isinstance(backend, StorageBackend)

    def test_save_search_get_update_delete(self, backend):
        a, b = rec("postgres is the user database", "/project", ["decision"]), rec("the api rate limit is 1000 per minute", "/project/api", ["limit"])
        backend.save([a, b])
        assert backend.count() == 2
        hits = backend.search(embed(["which database holds users"])[0], limit=2)
        assert hits[0][0].id == a.id and 0 < hits[0][1] <= 1.0
        got = backend.get_record(b.id)
        assert got is not None and got.scope == "/project/api" and got.categories == ["limit"]
        assert got.embedding is not None and len(got.embedding) == DIM
        b.content = "the api rate limit is 2000 per minute"
        b.importance = 0.9
        backend.update(b)
        assert backend.get_record(b.id).content.endswith("2000 per minute")
        assert backend.get_record(b.id).importance == 0.9
        assert backend.delete(record_ids=[a.id]) == 1
        assert backend.get_record(a.id) is None and backend.count() == 1

    def test_scope_prefix_is_a_path_not_a_string_prefix(self, backend):
        backend.save([rec("one", "/a"), rec("two", "/a/b"), rec("three", "/ab"), rec("four", "/")])
        assert backend.count("/a") == 2
        assert {r.scope for r in backend.list_records("/a")} == {"/a", "/a/b"}
        assert backend.count("/ab") == 1
        assert backend.count() == 4 and backend.count("/") == 4
        q = embed(["one two three four"])[0]
        assert {r.scope for r, _ in backend.search(q, scope_prefix="/a", limit=10)} == {"/a", "/a/b"}

    def test_categories_metadata_and_older_than(self, backend):
        old = datetime.utcnow() - timedelta(days=30)
        backend.save([
            rec("alpha", "/p", ["x", "y"], metadata={"team": "core"}),
            rec("beta", "/p", ["y"], metadata={"team": "edge"}),
            rec("gamma", "/p", ["z"], metadata={"team": "core"}, created_at=old),
        ])
        q = embed(["alpha beta gamma"])[0]
        assert {r.content for r, _ in backend.search(q, categories=["x"], limit=10)} == {"alpha"}
        assert {r.content for r, _ in backend.search(q, categories=["y", "z"], limit=10)} == {"alpha", "beta", "gamma"}
        assert {r.content for r, _ in backend.search(q, metadata_filter={"team": "core"}, limit=10)} == {"alpha", "gamma"}
        assert backend.list_categories("/p") == {"x": 1, "y": 2, "z": 1}
        assert backend.delete(older_than=datetime.utcnow() - timedelta(days=1)) == 1
        assert backend.delete(scope_prefix="/p", categories=["x"]) == 1
        assert [r.content for r in backend.list_records()] == ["beta"]

    def test_scope_info_and_list_scopes(self, backend):
        backend.save([rec("a", "/company"), rec("b", "/company/eng"), rec("c", "/company/eng/infra"), rec("d", "/company/sales")])
        info = backend.get_scope_info("/company")
        assert info.record_count == 4 and info.child_scopes == ["/company/eng", "/company/sales"]
        assert info.oldest_record is not None and info.newest_record is not None
        assert backend.list_scopes("/company/eng") == ["/company/eng/infra"]
        assert backend.list_scopes("/") == ["/company"]
        assert backend.get_scope_info("/nowhere").record_count == 0

    def test_list_records_is_newest_first_and_pages(self, backend):
        t0 = datetime.utcnow()
        backend.save([rec(f"n{i}", "/s", created_at=t0 + timedelta(seconds=i)) for i in range(5)])
        assert [r.content for r in backend.list_records("/s", limit=2)] == ["n4", "n3"]
        assert [r.content for r in backend.list_records("/s", limit=2, offset=2)] == ["n2", "n1"]

    def test_reset_scope_and_reset_all(self, backend):
        backend.save([rec("a", "/x"), rec("b", "/y")])
        backend.reset("/x")
        assert backend.count() == 1
        backend.reset()
        assert backend.count() == 0 and backend.search(embed(["a"])[0]) == []

    def test_dimension_mismatch_is_the_actionable_error(self, backend):
        backend.save([rec("a")])
        with pytest.raises(EmbeddingDimensionMismatchError):
            backend.search([0.1] * (DIM + 1))

    def test_touch_records_moves_last_accessed_only(self, backend):
        r = rec("touched", "/t")
        backend.save([r])
        before = backend.get_record(r.id)
        backend.touch_records([r.id, "no-such-id"])
        after = backend.get_record(r.id)
        assert after.last_accessed >= before.last_accessed and after.content == "touched" and after.embedding is not None

    def test_empty_store_answers_quietly(self, backend):
        assert backend.count() == 0 and backend.search(embed(["x"])[0]) == [] and backend.list_records() == []
        assert backend.get_record("nope") is None and backend.delete(record_ids=["nope"]) == 0
        backend.reset()

    async def test_async_variants(self, backend):
        await backend.asave([rec("async one", "/a")])
        hits = await backend.asearch(embed(["async one"])[0], limit=1)
        assert hits[0][0].content == "async one"
        assert await backend.adelete(scope_prefix="/a") == 1


class TestUnderCrewAIsMemoryEngine:
    def test_remember_recall_forget(self, memory):
        memory.remember("We decided to use PostgreSQL for the user database.", scope="/project", categories=["decision"], importance=0.9)
        memory.remember("The staging environment uses port 8080.", scope="/project/infra", categories=["config"])
        matches = memory.recall("What database did we choose?", limit=2)
        assert matches and "PostgreSQL" in matches[0].record.content
        assert 0.0 < matches[0].score <= 1.0
        assert memory.recall("port", scope="/project/infra")[0].record.content.endswith("8080.")
        memory.forget(scope="/project/infra")
        assert memory.recall("staging port", limit=5) and all("8080" not in m.record.content for m in memory.recall("staging port", limit=5))

    def test_scopes_tree_and_info(self, memory):
        memory.remember("alpha", scope="/co/eng", categories=["a"])
        memory.remember("beta", scope="/co/sales", categories=["b"])
        assert memory.list_scopes("/co") == ["/co/eng", "/co/sales"]
        assert memory.info("/co").record_count == 2
        assert set(memory.list_categories()) == {"a", "b"}
        assert "/co" in memory.tree()

    def test_update_importance(self, memory):
        memory.remember("keep this", scope="/u", importance=0.2)
        rid = memory.list_records(scope="/u")[0].id
        memory.update(rid, importance=0.95)
        assert memory.list_records(scope="/u")[0].importance == 0.95

    def test_two_memory_instances_share_the_store(self, table_name):
        a = Memory(storage=DXMTStorageBackend(table_name, host=HOST, port=PORT, api_key=API_KEY), llm=MagicMock(), embedder=embed)
        b = Memory(storage=DXMTStorageBackend(table_name, host=HOST, port=PORT, api_key=API_KEY), llm=MagicMock(), embedder=embed)
        a.remember("shared fact about feature flags", scope="/s", categories=["x"])
        assert "feature flags" in b.recall("feature flags")[0].record.content


class TestInstall:
    def test_install_routes_memory_without_a_storage_argument(self, table_name):
        import crewai_dxmt

        crewai_dxmt.install(host=HOST, port=PORT, api_key=API_KEY, memory_table=table_name, knowledge=False)
        try:
            m = Memory(llm=MagicMock(), embedder=embed)   # the spec defaults to "lancedb"
            assert isinstance(m._storage, DXMTStorageBackend) and m._storage.table_name == table_name
            m.remember("routed", scope="/r")
            assert DXMTStorageBackend(table_name, host=HOST, port=PORT, api_key=API_KEY).count() == 1
        finally:
            crewai_dxmt.uninstall()

"""Knowledge storage and the search tool, against a live server."""

import json
import os

import pytest
from crewai.knowledge.storage.base_knowledge_storage import BaseKnowledgeStorage
from crewai.tools import BaseTool

from conftest import API_KEY, HOST, PORT, embed
from crewai_dxmt import DXMTKnowledgeStorage, DXMTVectorSearchTool

pytestmark = pytest.mark.integration

DOCS = [
    "rust borrows memory instead of collecting it",
    "go collects garbage instead of borrowing",
    "c frees memory by hand",
]


@pytest.fixture
def storage(table_name):
    return DXMTKnowledgeStorage(table_name=table_name, host=HOST, port=PORT, api_key=API_KEY,
                                embedder=embed, enable_full_text_search=True)


class TestKnowledge:
    def test_is_a_base_knowledge_storage(self, storage):
        assert isinstance(storage, BaseKnowledgeStorage)

    def test_needs_a_way_to_embed(self, table_name):
        with pytest.raises(ValueError, match="embed"):
            DXMTKnowledgeStorage(table_name=table_name, host=HOST, port=PORT)

    def test_save_search_reset(self, storage):
        storage.save(DOCS)
        hits = storage.search(["go collects garbage"], limit=2, score_threshold=0.0)
        assert hits and hits[0]["content"].startswith("go collects")
        assert set(hits[0]) == {"id", "content", "metadata", "score"}
        # Two queries are merged by best score, de-duplicated by id.
        both = storage.search(["rust borrows", "c frees"], limit=5, score_threshold=0.0)
        assert len({h["id"] for h in both}) == len(both) and len(both) >= 2
        assert storage.search(["anything"], limit=5, score_threshold=1.01) == []
        storage.save(DOCS)          # the same chunk again is the same id: no duplicate
        assert storage.client.count(storage.table_name) == 3
        storage.reset()
        assert storage.search(["rust"], limit=5, score_threshold=0.0) == []

    async def test_async(self, storage):
        await storage.asave(DOCS[:1])
        hits = await storage.asearch(["rust"], limit=1, score_threshold=0.0)
        assert hits[0]["content"].startswith("rust")
        await storage.areset()

    @pytest.mark.skipif(not os.environ.get("DXMT_EMBEDDING_MODEL"), reason="needs a server with an embedding model")
    def test_server_side_embedding(self, table_name):
        # "server": the model the server was started with; a dict names one for the table.
        s = DXMTKnowledgeStorage(table_name=table_name, host=HOST, port=PORT, api_key=API_KEY, embedding="server")
        s.save(DOCS)
        hits = s.search(["garbage collection"], limit=1, score_threshold=0.0)
        assert hits[0]["content"].startswith("go collects")


class TestTool:
    def test_is_a_base_tool_with_a_schema(self, table_name):
        tool = DXMTVectorSearchTool(table_name=table_name, host=HOST, port=PORT, api_key=API_KEY, embedder=lambda q: embed([q])[0])
        assert isinstance(tool, BaseTool)
        assert set(tool.args_schema.model_json_schema()["properties"]) == {"query", "filter_by", "filter_value"}

    def test_run_returns_json_hits_and_honours_filters(self, storage, table_name):
        storage.save(DOCS)
        storage.client.update_record(table_name, storage._id(DOCS[1]), payload={"lang": "go"}, merge_mode="merge")
        tool = DXMTVectorSearchTool(table_name=table_name, host=HOST, port=PORT, api_key=API_KEY,
                                    embedder=lambda q: embed([q])[0], limit=3)
        out = json.loads(tool.run(query="memory management"))
        assert len(out) == 3 and {"id", "score", "payload"} <= set(out[0])
        only_go = json.loads(tool.run(query="memory", filter_by="lang", filter_value="go"))
        assert len(only_go) == 1 and only_go[0]["payload"]["lang"] == "go"
        hybrid = DXMTVectorSearchTool(table_name=table_name, host=HOST, port=PORT, api_key=API_KEY,
                                      embedder=lambda q: embed([q])[0], search_mode="hybrid", limit=1)
        assert json.loads(hybrid.run(query="garbage"))[0]["payload"]["content"].startswith("go")
        text = DXMTVectorSearchTool(table_name=table_name, host=HOST, port=PORT, api_key=API_KEY, search_mode="text", limit=1)
        assert json.loads(text.run(query="borrows"))[0]["payload"]["content"].startswith("rust")

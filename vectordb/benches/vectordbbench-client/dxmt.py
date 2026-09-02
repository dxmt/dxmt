"""VectorDBBench client for DXMT VectorDB, over the dxmt-python-sdk REST client."""

import logging
from contextlib import contextmanager

import httpx

from ..api import VectorDB
from .config import DXMTIndexConfig

log = logging.getLogger(__name__)

# The server rejects a request carrying more than 10,000 records. VectorDBBench hands over
# 5,000 at a time by default, but a case is free to change that, so chunk here rather than
# trusting the caller to stay under the limit.
MAX_RECORDS_PER_REQUEST = 2000


class DXMT(VectorDB):
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: DXMTIndexConfig | None,
        collection_name: str = "VectorDBBenchCollection",
        drop_old: bool = False,
        **kwargs,
    ):
        from dxmt import DXMTClient

        self.dim = dim
        self.db_config = db_config
        self.case_config = db_case_config
        self.table_name = collection_name
        self.host = db_config.get("host", "localhost")
        self.port = int(db_config.get("port", 9999))
        self.client = None

        index_param = db_case_config.index_param() if db_case_config else {}
        self.metric = index_param.get("metric_type", "cosine")
        self.index_type = index_param.get("index_type", "hnsw")
        search_param = db_case_config.search_param() if db_case_config else {}
        self.search_mode = search_param.get("search_mode")

        conn = DXMTClient(host=self.host, port=self.port, timeout=600.0)
        try:
            if drop_old:
                try:
                    conn.delete_table(self.table_name)
                    log.info("dropped existing table %s", self.table_name)
                except Exception as e:
                    log.info("no table to drop (%s)", e)

            existing = {t.get("name") for t in conn.list_tables()}
            if self.table_name not in existing:
                create_kwargs = {}
                # "auto" is this client word for "do not pin an index type". The DXMT
                # selector then picks one from the table size and re-checks it on a timer,
                # which is the behaviour a user who configures nothing actually gets.
                if self.index_type != "auto":
                    create_kwargs["index_type"] = self.index_type
                conn.create_table(
                    name=self.table_name,
                    dimension=dim,
                    distance_metric=self.metric,
                    **create_kwargs,
                )
                log.info(
                    "created table %s dim=%d metric=%s index=%s",
                    self.table_name,
                    dim,
                    self.metric,
                    self.index_type,
                )
        finally:
            conn.close()

    @contextmanager
    def init(self):
        from dxmt import DXMTClient

        self.client = DXMTClient(host=self.host, port=self.port, timeout=600.0)
        try:
            yield
        finally:
            try:
                self.client.close()
            finally:
                self.client = None

    def need_normalize_cosine(self) -> bool:
        # DXMT computes cosine itself. Handing it pre-normalised vectors would measure
        # something other than what a user gets.
        return False

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **kwargs,
    ) -> tuple[int, Exception]:
        inserted = 0
        try:
            for start in range(0, len(embeddings), MAX_RECORDS_PER_REQUEST):
                chunk_vecs = embeddings[start : start + MAX_RECORDS_PER_REQUEST]
                chunk_ids = metadata[start : start + MAX_RECORDS_PER_REQUEST]
                # VectorDBBench keys rows by int; DXMT record ids are strings.
                records = [
                    {"id": str(pk), "vector": list(vec)}
                    for pk, vec in zip(chunk_ids, chunk_vecs)
                ]
                self.client.insert_records(self.table_name, records)
                inserted += len(records)
        except Exception as e:
            log.warning("insert failed after %d rows: %s", inserted, e)
            return inserted, e
        return inserted, None

    def search_embedding(self, query: list[float], k: int = 100, **kwargs) -> list[int]:
        hits = self.client.search(
            self.table_name, vector=list(query), k=k, search_mode=self.search_mode
        )
        return [int(h["id"]) for h in hits]

    def optimize(self, data_size: int | None = None):
        # DXMT_BENCH_SKIP_REBUILD=1 answers one question: is the index that insert already
        # built good enough to serve, or does the from-scratch rebuild actually buy recall?
        # DXMT builds the graph twice -- once incrementally as records arrive, then again
        # here from zero -- while Qdrant writes unindexed segments on insert and builds once.
        # If recall holds without this call, the second build is pure waste.
        import os

        if os.environ.get("DXMT_BENCH_SKIP_REBUILD") == "1":
            log.info("skipping index rebuild (DXMT_BENCH_SKIP_REBUILD=1)")
            return

        # DXMT re-evaluates its index choice on a 900s timer. Without forcing it here the
        # search phase races that sweep and measures whichever index happened to be current,
        # which is not reproducible run to run.
        url = "http://%s:%d/api/v1/tables/%s/index/rebuild" % (
            self.host,
            self.port,
            self.table_name,
        )
        try:
            resp = httpx.post(url, timeout=7200.0)
            log.info("index rebuild returned %s", resp.status_code)
        except Exception as e:
            # Best effort: a rebuild that cannot be forced still leaves a benchmarkable
            # server, it just measures whatever index the sweep last chose.
            log.warning("could not force index rebuild: %s", e)

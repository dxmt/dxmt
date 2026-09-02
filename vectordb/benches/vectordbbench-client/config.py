"""VectorDBBench configuration for DXMT VectorDB.

This lives inside the VectorDBBench tree rather than in the DXMT repository, because
VectorDBBench discovers clients from its own DB enum and imports them from its own package.
"""

from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, MetricType


class DXMTConfig(DBConfig):
    host: str = "localhost"
    port: int = 9999
    api_key: SecretStr | None = None

    def to_dict(self) -> dict:
        config = {"host": self.host, "port": self.port}
        if self.api_key:
            config["api_key"] = self.api_key.get_secret_value()
        return config


class DXMTIndexConfig(BaseModel, DBCaseConfig):
    """DXMT names its metrics cosine / euclidean / dot."""

    metric_type: MetricType | None = None
    # How hard to search. Qdrant's config in this suite pins hnsw_ef=256; DXMT's pinned
    # nothing, so every DXMT number here was taken at the server default while the
    # comparison claimed aligned graph parameters. None keeps that default.
    search_mode: str | None = None

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.L2:
            return "euclidean"
        if self.metric_type == MetricType.IP:
            return "dot"
        return "cosine"


class DXMTAutoConfig(DXMTIndexConfig):
    """Let DXMT pick the index.

    This is what a user who configures nothing actually gets, so it is the configuration
    whose numbers are worth reporting.
    """

    def index_param(self) -> dict:
        return {"index_type": "auto", "metric_type": self.parse_metric()}

    def search_param(self) -> dict:
        return {"search_mode": self.search_mode} if self.search_mode else {}


class DXMTHNSWConfig(DXMTIndexConfig):
    m: int = 16
    ef_construction: int = 200
    ef_search: int = 128

    def index_param(self) -> dict:
        return {
            "index_type": "hnsw",
            "metric_type": self.parse_metric(),
            "m": self.m,
            "ef_construction": self.ef_construction,
        }

    def search_param(self) -> dict:
        return {"ef_search": self.ef_search}


class DXMTHNSWPQConfig(DXMTIndexConfig):
    m: int = 16
    ef_construction: int = 200
    ef_search: int = 128
    pq_m: int = 8
    pq_nbits: int = 8

    def index_param(self) -> dict:
        return {
            "index_type": "hnsw_pq",
            "metric_type": self.parse_metric(),
            "m": self.m,
            "ef_construction": self.ef_construction,
            "pq_m": self.pq_m,
            "pq_nbits": self.pq_nbits,
        }

    def search_param(self) -> dict:
        return {"ef_search": self.ef_search}

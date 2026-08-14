"""DXMT Python SDK Type Definitions.

This module defines type aliases and data structures used throughout the SDK.
"""

from typing import Any, Literal

from typing_extensions import TypeAlias

# Vector types
DenseVector: TypeAlias = list[float]
SparseVector: TypeAlias = dict[int, float]

# Distance metrics
DistanceMetric: TypeAlias = Literal["cosine", "euclidean", "manhattan", "dot"]

# Index types
IndexType: TypeAlias = Literal[
    "hnsw", "ivf", "ivf_pq", "lsh", "brute_force", "optimized"
]

# Insert modes
InsertMode: TypeAlias = Literal[
    "standard",
    "lock_free",
    "streaming",
    "deferred_index",
    "ultimate",
    "batch",
    "parallel",
    "chunked",
    "async_write",
    "bulk",
]

# Quantization types
QuantizationType: TypeAlias = Literal["scalar", "product", "binary"]

# Record structure
Record: TypeAlias = dict[str, Any]

# Search result structure
SearchResult: TypeAlias = dict[str, Any]

# Table configuration
TableConfig: TypeAlias = dict[str, Any]

# Filter structure
Filter: TypeAlias = dict[str, Any]

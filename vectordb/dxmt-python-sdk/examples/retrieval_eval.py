"""Retrieval quality evaluation — DXMT port of Chroma experimentation-playground pattern.

Measures Recall@K and MRR (Mean Reciprocal Rank) on a labelled dataset where
ground-truth relevant documents are known for each query.

Reference:
    reference/examples/chroma-cookbooks/experimentation-playground/
        src/experimentation_playground/eval/simple_eval.py

Evaluation metrics:
    Recall@K  — fraction of relevant docs returned in top-K results
    MRR@K     — mean 1/rank of the first relevant result within top-K
    Precision@K — fraction of returned docs that are relevant

This script is self-contained: no external APIs required.
sentence-transformers is used for text embeddings if available.

Usage:
    python retrieval_eval.py
    python retrieval_eval.py --k 5 --k 10 --verbose
"""

from __future__ import annotations

import argparse
import math
import hashlib
from dataclasses import dataclass, field
from typing import Any

from dxmt import Client

try:
    from sentence_transformers import SentenceTransformer as _ST
    _SMODEL = _ST("all-MiniLM-L6-v2")
    EMBED_DIM = _SMODEL.get_sentence_embedding_dimension()
    _HAS_ST = True
except ImportError:
    _SMODEL = None
    EMBED_DIM = 384
    _HAS_ST = False


# ── Embedding ─────────────────────────────────────────────────────────────────

def _synthetic_vector(seed: str, dim: int) -> list[float]:
    raw = []
    for i in range(dim):
        h = int(hashlib.md5(f"{seed}:{i}".encode()).hexdigest(), 16)
        raw.append((h % 10000) / 10000.0 * 2 - 1)
    norm = math.sqrt(sum(x * x for x in raw)) or 1.0
    return [x / norm for x in raw]


def embed(texts: list[str]) -> list[list[float]]:
    if _HAS_ST and _SMODEL is not None:
        return _SMODEL.encode(texts, convert_to_numpy=True).tolist()
    return [_synthetic_vector(t, EMBED_DIM) for t in texts]


# ── Dataset ───────────────────────────────────────────────────────────────────
# Each document has an id and text.
# Each query has a text and a set of relevant document ids (ground truth).

DOCUMENTS: list[dict[str, str]] = [
    {"id": "d01", "text": "HNSW builds a layered graph for approximate nearest neighbour search."},
    {"id": "d02", "text": "IVF-PQ combines inverted file index with product quantisation."},
    {"id": "d03", "text": "Cosine similarity measures the angle between two vectors."},
    {"id": "d04", "text": "Euclidean distance is the straight-line distance between points."},
    {"id": "d05", "text": "RAG retrieves documents from a vector store to ground LLM answers."},
    {"id": "d06", "text": "Payload filters restrict search results to records matching metadata conditions."},
    {"id": "d07", "text": "Sentence-transformers produce fixed-size embeddings from variable-length text."},
    {"id": "d08", "text": "Approximate nearest neighbour search trades recall for speed."},
    {"id": "d09", "text": "Hybrid search combines dense vector search with sparse keyword matching."},
    {"id": "d10", "text": "Product quantisation compresses vectors into sub-vector codes."},
    {"id": "d11", "text": "ef_search controls the HNSW beam width during query traversal."},
    {"id": "d12", "text": "Chunking splits long documents into overlapping passages for better retrieval."},
    {"id": "d13", "text": "BM25 is a classical sparse retrieval model based on term frequency."},
    {"id": "d14", "text": "Re-ranking applies a cross-encoder to re-score candidate documents."},
    {"id": "d15", "text": "DiskANN stores the graph on SSD to support billion-scale vector search."},
    {"id": "d16", "text": "Dot product is the most efficient similarity metric for normalised vectors."},
    {"id": "d17", "text": "FAISS is a library for efficient similarity search and clustering of dense vectors."},
    {"id": "d18", "text": "Vector databases are optimised for unstructured data like text and images."},
    {"id": "d19", "text": "Multi-vector search stores multiple embeddings per document for different fields."},
    {"id": "d20", "text": "LangChain integrates vector stores with LLM chains for RAG applications."},
]

# Labelled queries: each has one or more relevant document ids (ground truth)
QUERIES: list[dict[str, Any]] = [
    {
        "query": "How does HNSW approximate nearest neighbour search work?",
        "relevant_ids": {"d01", "d11", "d08"},
    },
    {
        "query": "What distance metrics are commonly used for text embeddings?",
        "relevant_ids": {"d03", "d04", "d16"},
    },
    {
        "query": "How is RAG used with language models?",
        "relevant_ids": {"d05", "d12", "d20"},
    },
    {
        "query": "What are methods for compressing high-dimensional vectors?",
        "relevant_ids": {"d02", "d10"},
    },
    {
        "query": "How do payload filters work in vector search?",
        "relevant_ids": {"d06"},
    },
    {
        "query": "What is hybrid search and why is it useful?",
        "relevant_ids": {"d09", "d13"},
    },
    {
        "query": "How can vector search scale to billions of documents?",
        "relevant_ids": {"d15", "d17"},
    },
    {
        "query": "What is re-ranking and how does it improve retrieval?",
        "relevant_ids": {"d14"},
    },
    {
        "query": "How do embedding models encode text into vectors?",
        "relevant_ids": {"d07", "d18"},
    },
    {
        "query": "What is multi-vector search?",
        "relevant_ids": {"d19"},
    },
]


# ── Metrics ───────────────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    query: str
    relevant_ids: set[str]
    retrieved_ids: list[str]  # ordered
    k: int

    def recall_at_k(self) -> float:
        hits = sum(1 for r in self.retrieved_ids[:self.k] if r in self.relevant_ids)
        return hits / len(self.relevant_ids) if self.relevant_ids else 1.0

    def precision_at_k(self) -> float:
        if self.k == 0:
            return 0.0
        hits = sum(1 for r in self.retrieved_ids[:self.k] if r in self.relevant_ids)
        return hits / self.k

    def reciprocal_rank(self) -> float:
        for rank, rid in enumerate(self.retrieved_ids[:self.k], start=1):
            if rid in self.relevant_ids:
                return 1.0 / rank
        return 0.0


@dataclass
class EvalReport:
    k: int
    results: list[QueryResult] = field(default_factory=list)

    def mean_recall(self) -> float:
        return sum(r.recall_at_k() for r in self.results) / len(self.results) if self.results else 0.0

    def mean_precision(self) -> float:
        return sum(r.precision_at_k() for r in self.results) / len(self.results) if self.results else 0.0

    def mrr(self) -> float:
        return sum(r.reciprocal_rank() for r in self.results) / len(self.results) if self.results else 0.0


# ── Evaluator ─────────────────────────────────────────────────────────────────

TABLE = "eval_docs"


class RetrievalEvaluator:
    def __init__(self) -> None:
        self.client = Client(host="localhost", port=9999)

    def _setup(self) -> None:
        try:
            self.client.delete_table(TABLE)
        except Exception:
            pass
        self.client.create_table(name=TABLE, dimension=EMBED_DIM, distance_metric="cosine")

    def _index(self) -> None:
        texts = [d["text"] for d in DOCUMENTS]
        vectors = embed(texts)
        records = [
            {
                "id": doc["id"],
                "vector": vec,
                "payload": {"text": doc["text"]},
            }
            for doc, vec in zip(DOCUMENTS, vectors)
        ]
        self.client.insert_records(table_name=TABLE, records=records, mode="batch")

    def _retrieve(self, query: str, k: int) -> list[str]:
        q_vec = embed([query])[0]
        hits = self.client.search(table_name=TABLE, vector=q_vec, k=k)
        return [h["id"] for h in hits]

    def evaluate(self, k_values: list[int], verbose: bool = False) -> dict[int, EvalReport]:
        self._setup()
        self._index()
        mode = "sentence-transformers" if _HAS_ST else "synthetic"
        print(f"[eval] indexed {len(DOCUMENTS)} documents (embedding: {mode}, dim={EMBED_DIM})")

        reports: dict[int, EvalReport] = {k: EvalReport(k=k) for k in k_values}
        max_k = max(k_values)

        for q_item in QUERIES:
            retrieved = self._retrieve(q_item["query"], k=max_k)

            for k in k_values:
                qr = QueryResult(
                    query=q_item["query"],
                    relevant_ids=q_item["relevant_ids"],
                    retrieved_ids=retrieved,
                    k=k,
                )
                reports[k].results.append(qr)

                if verbose:
                    hits = [r for r in retrieved[:k] if r in q_item["relevant_ids"]]
                    print(
                        f"  [{k:2d}] recall={qr.recall_at_k():.2f}  "
                        f"rr={qr.reciprocal_rank():.2f}  "
                        f"hits={hits}  "
                        f'Q="{q_item["query"][:50]}"'
                    )

        return reports

    def close(self) -> None:
        try:
            self.client.delete_table(TABLE)
        except Exception:
            pass
        self.client.close()


# ── Report printing ───────────────────────────────────────────────────────────

def print_report(reports: dict[int, EvalReport]) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {'K':>4}  {'Recall@K':>10}  {'Precision@K':>12}  {'MRR@K':>8}")
    print(f"{'─' * 60}")
    for k in sorted(reports):
        r = reports[k]
        print(f"  {k:>4}  {r.mean_recall():>10.4f}  {r.mean_precision():>12.4f}  {r.mrr():>8.4f}")
    print(f"{'─' * 60}")

    # Interpretation guide
    best = reports[max(reports)]
    recall = best.mean_recall()
    mrr = best.mrr()

    print("\nInterpretation:")
    if recall >= 0.90:
        print(f"  Recall@{max(reports)}: {recall:.2%} — excellent coverage")
    elif recall >= 0.70:
        print(f"  Recall@{max(reports)}: {recall:.2%} — good coverage")
    else:
        print(f"  Recall@{max(reports)}: {recall:.2%} — consider larger K or better embeddings")

    if mrr >= 0.70:
        print(f"  MRR:      {mrr:.4f} — relevant results rank near the top")
    else:
        print(f"  MRR:      {mrr:.4f} — relevant results may appear lower in ranking")

    note = "Note: synthetic vectors produce low recall — install sentence-transformers for realistic numbers." if not _HAS_ST else ""
    if note:
        print(f"\n  {note}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="DXMT retrieval quality evaluation")
    parser.add_argument("--k", type=int, action="append", dest="k_values",
                        help="K value(s) to evaluate (repeatable). Default: 1 3 5 10")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-query results")
    args = parser.parse_args()

    k_values = sorted(set(args.k_values or [1, 3, 5, 10]))

    print("=" * 60)
    print("  DXMT Retrieval Evaluation")
    print("  (Chroma experimentation-playground pattern)")
    print("=" * 60)
    print(f"  Queries:    {len(QUERIES)}")
    print(f"  Documents:  {len(DOCUMENTS)}")
    print(f"  K values:   {k_values}")

    evaluator = RetrievalEvaluator()
    try:
        if args.verbose:
            print(f"\n{'─' * 60}")
            print("  Per-query results")
            print(f"{'─' * 60}")

        reports = evaluator.evaluate(k_values=k_values, verbose=args.verbose)
        print_report(reports)

    finally:
        evaluator.close()
        print("\n[done]")


if __name__ == "__main__":
    main()

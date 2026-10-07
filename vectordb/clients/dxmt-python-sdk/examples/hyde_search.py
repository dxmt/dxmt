"""HyDE (Hypothetical Document Embeddings) search — DXMT implementation.

Instead of embedding the raw query, HyDE:
    1. Prompts an LLM to write a hypothetical ideal answer (a "document" that
       would perfectly answer the question).
    2. Embeds that hypothetical document.
    3. Uses that embedding as the search vector.

This technique dramatically improves recall for complex factual queries because
the embedding of a relevant-style answer is closer to the stored document
embeddings than the embedding of the short question itself.

Pattern from the "Advance-RAG-with-HyDE" recipe in LanceDB's vectordb-recipes
(https://github.com/lancedb/vectordb-recipes), reimplemented against DXMT.

Modes:
    No LLM (default)   — "hypothetical doc" is a rule-based expansion of the query.
                          Useful to understand the pattern without API keys.
    --openai           — GPT-3.5 generates the hypothetical document.

Usage:
    python hyde_search.py
    python hyde_search.py --openai
"""

from __future__ import annotations

import argparse
import os
import textwrap
from typing import Any

from dxmt import Client
from environment import client_kwargs
from common_utils import embed, EMBED_DIM

try:
    import openai as _openai
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False


# ── Knowledge corpus ──────────────────────────────────────────────────────────

CORPUS = [
    {
        "id": "doc_hnsw_basics",
        "text": (
            "HNSW (Hierarchical Navigable Small World) is a graph-based ANN algorithm. "
            "It builds a multi-layer graph where the top layers have sparse long-range edges "
            "and the bottom layer has dense local connections. Search starts at the top layer "
            "and greedily descends, using ef_search as the beam width."
        ),
        "topic": "indexing",
    },
    {
        "id": "doc_hnsw_params",
        "text": (
            "Key HNSW parameters: M controls the number of bidirectional connections per node. "
            "Higher M improves recall but uses more memory. ef_construction sets the beam width "
            "during index build. ef_search sets it during query. Typical values: M=16, ef=200."
        ),
        "topic": "indexing",
    },
    {
        "id": "doc_ivf_pq",
        "text": (
            "IVF-PQ combines an inverted file index with product quantisation. "
            "Vectors are first clustered (IVF), then each cluster's residuals are "
            "compressed using PQ sub-codebooks. Reduces memory 10-50x with a small "
            "recall penalty. Good for 100K–10M scale datasets."
        ),
        "topic": "indexing",
    },
    {
        "id": "doc_cosine",
        "text": (
            "Cosine similarity measures the angle between two vectors, ignoring magnitude. "
            "For normalised unit vectors, dot product equals cosine similarity. "
            "Most text embedding models output unit-normalised vectors, making dot product "
            "the preferred and cheapest metric."
        ),
        "topic": "distance",
    },
    {
        "id": "doc_euclidean",
        "text": (
            "Euclidean (L2) distance is the straight-line distance between two points. "
            "It is sensitive to vector magnitude, unlike cosine. Common for image feature "
            "vectors. Squared L2 avoids the sqrt and is monotonically equivalent for ranking."
        ),
        "topic": "distance",
    },
    {
        "id": "doc_rag",
        "text": (
            "Retrieval-Augmented Generation (RAG) grounds LLM answers in retrieved evidence. "
            "Pipeline: chunk documents → embed each chunk → store in vector DB → at query time "
            "embed the question → retrieve top-k chunks → pass chunks + question to LLM. "
            "RAG reduces hallucination and keeps knowledge current without retraining."
        ),
        "topic": "rag",
    },
    {
        "id": "doc_rag_chunking",
        "text": (
            "Chunking strategy matters for RAG quality. Fixed-size chunks are simple. "
            "Sentence-boundary chunking preserves semantic units. Overlapping chunks (10-20 % "
            "overlap) reduce the chance of splitting a key sentence. Late chunking (embed first, "
            "then chunk) preserves full-document context in embeddings."
        ),
        "topic": "rag",
    },
    {
        "id": "doc_filters",
        "text": (
            "Payload filters restrict vector search to a subset of records matching metadata "
            "conditions. DXMT supports: Equal, In, GreaterThan, LessThan, Contains, StartsWith. "
            "Filters are evaluated per-candidate during HNSW traversal (in-graph) for high "
            "selectivity or as a post-filter for low selectivity queries."
        ),
        "topic": "filtering",
    },
    {
        "id": "doc_quantisation",
        "text": (
            "Vector quantisation reduces memory usage by compressing floating-point vectors. "
            "Scalar quantisation maps each dimension to an 8-bit integer. Binary quantisation "
            "maps to 1 bit per dimension (32x compression). Product quantisation splits the "
            "vector into sub-vectors and codes each independently."
        ),
        "topic": "compression",
    },
    {
        "id": "doc_diskann",
        "text": (
            "DiskANN stores the HNSW-like graph on SSD rather than RAM, enabling billion-scale "
            "search on commodity hardware. Only the working set (hot nodes) is cached in memory. "
            "Achieves >95% recall@10 at <5ms latency on 1B vectors with 64GB RAM."
        ),
        "topic": "scalability",
    },
    {
        "id": "doc_hybrid",
        "text": (
            "Hybrid search combines dense vector retrieval (semantic) with sparse keyword "
            "retrieval (BM25/TF-IDF). Results are merged via Reciprocal Rank Fusion (RRF) "
            "or a learned linear combination. Hybrid consistently outperforms either method "
            "alone on domain-specific retrieval benchmarks."
        ),
        "topic": "search",
    },
    {
        "id": "doc_reranking",
        "text": (
            "Re-ranking applies a cross-encoder model to re-score the top-k candidates from "
            "an initial retrieval pass. Cross-encoders attend to both query and document jointly, "
            "giving much higher ranking quality than bi-encoders. The two-stage pipeline keeps "
            "latency acceptable by limiting cross-encoder calls to the top 50-100 candidates."
        ),
        "topic": "search",
    },
]

# Queries used in the demo — each has both a raw and an expanded (HyDE) form
DEMO_QUERIES = [
    {
        "query": "How does HNSW handle search at large scale?",
        "topic": "indexing",
    },
    {
        "query": "What metric should I use for text similarity?",
        "topic": "distance",
    },
    {
        "query": "How can I ground LLM answers in retrieved evidence?",
        "topic": "rag",
    },
    {
        "query": "What options exist for reducing memory used by vectors?",
        "topic": "compression",
    },
]

TABLE = "hyde_corpus"


# ── HyDE helpers ──────────────────────────────────────────────────────────────

_RULE_EXPANSIONS = {
    "hnsw": (
        "HNSW search uses a multi-layer graph. At each layer the algorithm "
        "greedily moves toward the query vector. The bottom layer contains all "
        "nodes; ef_search controls how many candidates are considered at each step."
    ),
    "metric": (
        "Common similarity metrics for text embeddings include cosine similarity "
        "and dot product. For normalised vectors these are equivalent. Cosine is "
        "insensitive to magnitude, making it the standard choice for text."
    ),
    "rag": (
        "RAG (Retrieval-Augmented Generation) works by embedding documents and "
        "queries into a shared vector space, retrieving the nearest passages, "
        "and providing them to an LLM as context before generating an answer."
    ),
    "compress": (
        "Vector compression techniques include product quantisation, scalar "
        "quantisation, and binary quantisation. PQ divides vectors into sub-vectors "
        "and codes each with a small codebook, achieving 10-50x compression."
    ),
    "memory": (
        "Reducing memory for vector indexes: use PQ or scalar quantisation to "
        "compress vectors, use DiskANN to store the graph on SSD, or limit "
        "ef_construction to reduce HNSW layer sizes."
    ),
}


def _rule_expand(query: str) -> str:
    """Return a keyword-expanded 'hypothetical document' without any LLM."""
    q_lower = query.lower()
    for keyword, expansion in _RULE_EXPANSIONS.items():
        if keyword in q_lower:
            return expansion
    # Fallback: repeat and slightly paraphrase the query
    return (
        f"A detailed explanation about the topic: {query}. "
        "This covers the key concepts, mechanisms, and practical considerations."
    )


def _llm_expand(query: str) -> str:
    """Generate a hypothetical ideal answer using GPT."""
    client = _openai.OpenAI()
    prompt = (
        "Write a short, dense paragraph (3-5 sentences) that would be a perfect "
        "answer to the following question. Do not mention the question itself. "
        "Write it as if it were a passage from a technical document.\n\n"
        f"Question: {query}"
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=120,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def hypothetical_document(query: str, use_openai: bool) -> str:
    if use_openai and _HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
        return _llm_expand(query)
    # NOTE: rule expansion is NOT true HyDE. It uses pre-written keyword mappings,
    # so similarity gains here reflect the quality of the hand-crafted expansions,
    # not the HyDE algorithm. Use --openai to see the real technique.
    return _rule_expand(query)


# ── DXMT helpers ──────────────────────────────────────────────────────────────

def setup_table(client: Client) -> None:
    try:
        client.delete_table(TABLE)
    except Exception:
        pass
    client.create_table(name=TABLE, dimension=EMBED_DIM, distance_metric="cosine")


def index_corpus(client: Client) -> None:
    texts = [doc["text"] for doc in CORPUS]
    vectors = embed(texts)
    records = [
        {
            "id": doc["id"],
            "vector": vec,
            "payload": {"text": doc["text"], "topic": doc["topic"]},
        }
        for doc, vec in zip(CORPUS, vectors)
    ]
    client.insert_records(table_name=TABLE, records=records)
    print(f"[hyde] indexed {len(CORPUS)} documents")


def search_raw(client: Client, query: str, k: int = 3) -> list[dict[str, Any]]:
    vec = embed([query])[0]
    return client.search(table_name=TABLE, vector=vec, k=k)


def search_hyde(
    client: Client, query: str, use_openai: bool, k: int = 3
) -> tuple[str, list[dict[str, Any]]]:
    hyp_doc = hypothetical_document(query, use_openai)
    vec = embed([hyp_doc])[0]
    return hyp_doc, client.search(table_name=TABLE, vector=vec, k=k)


# ── Display helpers ───────────────────────────────────────────────────────────

def _doc_text(hit: dict[str, Any]) -> str:
    doc_id = hit.get("id", "")
    for d in CORPUS:
        if d["id"] == doc_id:
            return d["text"]
    return hit.get("id", "")


def _show_hits(hits: list[dict[str, Any]], label: str) -> None:
    print(f"\n  {label}:")
    for i, h in enumerate(hits, 1):
        score = h.get("score", 0.0)
        text = textwrap.shorten(_doc_text(h), width=90, placeholder="…")
        print(f"    {i}. [{score:.4f}] {text}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="HyDE search demo — DXMT")
    parser.add_argument(
        "--openai", action="store_true",
        help="Use GPT-3.5 to generate hypothetical documents (requires OPENAI_API_KEY)"
    )
    parser.add_argument("--k", type=int, default=3, help="Number of results (default 3)")
    args = parser.parse_args()

    use_openai = args.openai and _HAS_OPENAI and bool(os.getenv("OPENAI_API_KEY"))
    if args.openai and not use_openai:
        print("[warn] --openai set but openai not installed or OPENAI_API_KEY missing — using rule expansion")

    hyde_mode = "OpenAI GPT-3.5 (true HyDE)" if use_openai else "rule expansion (NOT true HyDE — use --openai for real results)"
    from common_utils import _HAS_ST
    embed_mode = "sentence-transformers" if _HAS_ST else "bag-of-words"
    print("=" * 70)
    print("  DXMT HyDE (Hypothetical Document Embeddings) Demo")
    print(f"  Embedding:    {embed_mode}  (dim={EMBED_DIM})")
    print(f"  HyDE expand:  {hyde_mode}")
    print("=" * 70)
    if not use_openai:
        print("\n  [note] Without --openai, keyword expansion replaces the LLM step.")
        print("         Similarity gains reflect the hand-crafted expansions, not HyDE.")
        print("         Run with --openai and OPENAI_API_KEY for the genuine algorithm.\n")

    client = Client(**client_kwargs())
    try:
        setup_table(client)
        index_corpus(client)

        for item in DEMO_QUERIES:
            q = item["query"]
            print(f"\n{'─' * 70}")
            print(f"Query: {q}")

            # Baseline: embed the raw query
            raw_hits = search_raw(client, q, k=args.k)

            # HyDE: embed a hypothetical ideal answer
            hyp_doc, hyde_hits = search_hyde(client, q, use_openai=use_openai, k=args.k)

            print(f"\n  Hypothetical document (truncated):")
            print(f"    {textwrap.shorten(hyp_doc, width=100, placeholder='…')}")

            _show_hits(raw_hits, f"Raw query (top {args.k})")
            _show_hits(hyde_hits, f"HyDE query (top {args.k})")

            # Compare: show which docs changed
            raw_ids = {h.get("id") for h in raw_hits}
            hyde_ids = {h.get("id") for h in hyde_hits}
            gained = hyde_ids - raw_ids
            lost = raw_ids - hyde_ids
            if gained:
                print(f"\n  HyDE surfaced: {gained}")
            if lost:
                print(f"  HyDE dropped:  {lost}")

    finally:
        try:
            client.delete_table(TABLE)
        except Exception:
            pass
        client.close()
        print("\n[done]")


if __name__ == "__main__":
    main()

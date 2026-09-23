"""Contextual RAG with Context Enrichment Window — DXMT implementation.

Standard RAG retrieves only the matching chunk. But an individual chunk is often
too short to provide complete context. The "Context Enrichment Window" pattern:

    1. When a chunk is retrieved, also fetch the N neighbouring chunks
       (chunk_index - N … chunk_index + N).
    2. Concatenate them in order to form a wider context window.
    3. Pass the enriched window to the LLM (or return it directly).

This dramatically improves answer quality when the relevant information spans
chunk boundaries, without requiring re-chunking or late chunking.

Reference:
    reference/examples/vectordb-recipes/examples/
        Advanced_RAG_Context_Enrichment_Window/

Modes:
    No LLM (default)  — returns the enriched context window directly.
    --openai          — sends window + question to GPT-3.5.

Usage:
    python contextual_rag.py
    python contextual_rag.py --window 2
    python contextual_rag.py --openai
"""

from __future__ import annotations

import argparse
import os
import re
import textwrap
from typing import Any

from dxmt import Client
from common_utils import embed, EMBED_DIM

try:
    import openai as _openai
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False


# ── Source document ───────────────────────────────────────────────────────────

DOCUMENT = """
# The History of Vector Databases

## Early Foundations (1990s – 2010s)

The concept of similarity search in high-dimensional spaces has roots in the
1990s when researchers began developing efficient data structures for nearest
neighbour queries. Early work centred on KD-trees and ball-trees, which worked
well in low dimensions but degraded quickly beyond 10–20 dimensions due to the
curse of dimensionality.

The introduction of Locality-Sensitive Hashing (LSH) in 1998 provided the
first practical approximate nearest neighbour (ANN) technique for high-dimensional
data. LSH projects vectors onto random hyperplanes and hashes similar vectors into
the same buckets with high probability. It became the de facto standard for
large-scale image and document retrieval throughout the 2000s.

Facebook's FAISS library (2017) dramatically accelerated the field by providing
GPU-accelerated implementations of IVF-PQ and HNSW on a single machine. FAISS
remains a foundational library that many production systems build upon.

## The Deep Learning Era (2017 – 2020)

The transformer revolution transformed retrieval. Dense bi-encoder models such
as Sentence-BERT (2019) learned to map semantically similar sentences to nearby
points in embedding space, far outperforming keyword-based BM25 on semantic
similarity tasks.

This created demand for purpose-built vector databases that could handle
real-time insert-and-query workloads at scale, with rich metadata filtering —
requirements that FAISS alone could not satisfy.

Pinecone launched as the first fully-managed vector database service in 2021,
followed quickly by Weaviate, Qdrant, and Chroma, each with distinct
architectural philosophies.

## HNSW and Graph-Based Indexes

HNSW (Hierarchical Navigable Small World), introduced by Malkov and Yashunin in
2016, became the dominant in-memory ANN algorithm. It builds a layered navigable
graph where upper layers contain long-range edges for fast coarse navigation and
the bottom layer contains all nodes with short-range edges for fine-grained
search.

The key insight is that starting navigation from a sparse upper layer avoids
getting trapped in local minima early — a failure mode of the earlier NSW
(non-hierarchical) algorithm. HNSW achieves >95 % recall@10 at single-digit
millisecond latency for datasets up to ~50 million vectors in RAM.

Parameter M controls the number of bidirectional links per node at each layer.
Higher M improves recall but increases memory usage quadratically. ef_construction
determines beam width during index build (quality vs build time). ef_search sets
beam width at query time (recall vs latency).

## Hybrid Search

Pure dense retrieval misses exact keyword matches and named entities. Pure BM25
misses paraphrase and conceptual similarity. Hybrid search merges both signals.

The standard fusion technique is Reciprocal Rank Fusion (RRF): given ranked lists
from dense and sparse retrieval, each document's score is the sum of 1/(rank + k)
across both lists, where k=60 is a smoothing constant. RRF is robust, parameter-
free, and typically outperforms linear combination of raw scores.

## Production Workloads

Production deployments face challenges not present in benchmarks: incremental
inserts without full reindexing, concurrent readers/writers, payload filter
selectivity, and multi-tenancy. DXMT addresses these with channel-based insertion
(sequential writes through a background task), an RwLock-wrapped index for
concurrent reads, and payload indexing for O(1) filter evaluation per candidate.

Storage tiering — hot (memory) → warm (SSD) → cold (S3) — allows unbounded
data capacity while keeping frequently accessed vectors in low-latency tiers.
Tier promotion is driven by a hotness_score: decay × recency × frequency.
"""


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_document(text: str, max_sentences: int = 4, overlap_sentences: int = 1) -> list[str]:
    """Split on sentence boundaries. Each chunk holds up to max_sentences
    complete sentences; overlap_sentences trailing sentences from the previous
    chunk are prepended so context is never cut at a boundary mid-thought."""
    # Split into sentences, filtering empty strings
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in raw if s.strip() and len(s.strip()) > 10]

    chunks: list[str] = []
    i = 0
    while i < len(sentences):
        window = sentences[i: i + max_sentences]
        chunks.append(" ".join(window))
        i += max_sentences - overlap_sentences  # step forward leaving overlap

    return chunks


# ── Storage helpers ───────────────────────────────────────────────────────────

TABLE = "ctx_rag_chunks"


def setup_and_index(client: Client, chunks: list[str]) -> None:
    try:
        client.delete_table(TABLE)
    except Exception:
        pass
    client.create_table(name=TABLE, dimension=EMBED_DIM, distance_metric="cosine")

    vectors = embed(chunks)
    records = [
        {
            "id": f"chunk_{i:04d}",
            "vector": vec,
            "payload": {"text": chunk, "chunk_index": i},
        }
        for i, (chunk, vec) in enumerate(zip(chunks, vectors))
    ]
    client.insert_records(table_name=TABLE, records=records)
    print(f"[index] {len(chunks)} chunks indexed (dim={EMBED_DIM})")


# ── Context enrichment ────────────────────────────────────────────────────────

def _parse_chunk_index(chunk_id: str) -> int:
    """Extract integer index from id like 'chunk_0007'. No HTTP call needed."""
    try:
        return int(chunk_id.split("_")[-1])
    except ValueError:
        return 0


def retrieve_with_context(
    client: Client,
    query: str,
    chunks: list[str],
    k: int = 2,
    window: int = 1,
) -> list[dict[str, Any]]:
    """
    Retrieve top-k chunks and expand each with ±window neighbouring chunks.
    chunk index is decoded directly from the record ID — no extra HTTP round-trips.
    """
    q_vec = embed([query])[0]
    hits = client.search(table_name=TABLE, vector=q_vec, k=k)

    enriched = []
    seen_windows: set[tuple[int, int]] = set()

    for hit in hits:
        chunk_id = hit.get("id", "")
        center_idx = _parse_chunk_index(chunk_id)

        start = max(0, center_idx - window)
        end = min(len(chunks) - 1, center_idx + window)
        span = (start, end)

        if span in seen_windows:
            continue
        seen_windows.add(span)

        # Collect the surrounding chunks in order
        window_texts = [
            chunks[i] for i in range(start, end + 1)
        ]
        context_text = "\n\n".join(window_texts)

        enriched.append({
            "center_chunk_id": chunk_id,
            "center_idx": center_idx,
            "window_start": start,
            "window_end": end,
            "score": hit.get("score", 0.0),
            "retrieved_text": chunks[center_idx],
            "context_text": context_text,
            "span_chunks": end - start + 1,
        })

    return enriched


# ── LLM answer ────────────────────────────────────────────────────────────────

def llm_answer(question: str, context: str) -> str:
    client = _openai.OpenAI()
    prompt = (
        "Answer the question using ONLY the provided context. "
        "Be concise (2-3 sentences). "
        "If the answer is not in the context, say 'Not found in context.'\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


# ── Demo queries ──────────────────────────────────────────────────────────────

DEMO_QUERIES = [
    "What is HNSW and what parameters control its recall?",
    "How does Reciprocal Rank Fusion work in hybrid search?",
    "How does DXMT handle concurrent reads and writes?",
    "What was the first fully-managed vector database service?",
    "How does storage tiering work in production vector databases?",
]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Contextual RAG with context enrichment window")
    parser.add_argument("--window", type=int, default=1,
                        help="Chunks to include on each side of retrieved chunk (default 1)")
    parser.add_argument("--k", type=int, default=2,
                        help="Number of retrieval hits before window expansion (default 2)")
    parser.add_argument("--openai", action="store_true",
                        help="Use GPT-3.5 to answer (requires OPENAI_API_KEY)")
    args = parser.parse_args()

    use_openai = args.openai and _HAS_OPENAI and bool(os.getenv("OPENAI_API_KEY"))
    if args.openai and not use_openai:
        print("[warn] --openai requested but not available — showing context only")

    from common_utils import _HAS_ST
    embed_mode = "sentence-transformers" if _HAS_ST else "synthetic"
    print("=" * 70)
    print("  DXMT Contextual RAG — Context Enrichment Window")
    print(f"  Embedding: {embed_mode}  (dim={EMBED_DIM})")
    print(f"  Window: ±{args.window} chunks around each retrieved chunk")
    print("=" * 70)

    chunks = chunk_document(DOCUMENT)
    print(f"\n[chunking] document → {len(chunks)} chunks (sentence-boundary, 4 sentences/chunk, 1-sentence overlap)")

    client = Client(host="localhost", port=9999)
    try:
        setup_and_index(client, chunks)

        for query in DEMO_QUERIES:
            print(f"\n{'─' * 70}")
            print(f"Query: {query}")

            enriched = retrieve_with_context(
                client, query, chunks, k=args.k, window=args.window
            )

            if not enriched:
                print("  (no results)")
                continue

            for i, item in enumerate(enriched, 1):
                print(f"\n  Hit {i}  [score={item['score']:.4f}]  "
                      f"chunk {item['center_idx']} — window [{item['window_start']}..{item['window_end']}] "
                      f"({item['span_chunks']} chunks)")

                print(f"\n  Retrieved chunk (what standard RAG would use):")
                excerpt = textwrap.fill(
                    item["retrieved_text"], width=68,
                    initial_indent="    ", subsequent_indent="    "
                )
                print(excerpt)

                if item["span_chunks"] > 1:
                    print(f"\n  Enriched context window ({item['span_chunks']} chunks):")
                    full = textwrap.fill(
                        item["context_text"].replace("\n\n", " | "),
                        width=68,
                        initial_indent="    ",
                        subsequent_indent="    ",
                    )
                    print(full)

            # Use first result's context for answer
            best_context = enriched[0]["context_text"]

            if use_openai:
                answer = llm_answer(query, best_context)
                print(f"\n  GPT answer:\n  {answer}")
            else:
                snippet = textwrap.shorten(best_context, width=200, placeholder="…")
                print(f"\n  [no LLM] best context (enriched): {snippet}")

    finally:
        try:
            client.delete_table(TABLE)
        except Exception:
            pass
        client.close()
        print("\n[done]")


if __name__ == "__main__":
    main()

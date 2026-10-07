"""Local RAG pipeline — retrieval-augmented generation against a local corpus.

End-to-end Retrieval-Augmented Generation without any cloud dependency:
    1. Chunk documents
    2. Generate embeddings (sentence-transformers or synthetic fallback)
    3. Store in DXMT
    4. Retrieve top-k chunks for a query
    5. Assemble a grounded prompt (LLM call is optional / stubbed)

Pattern from the "Local-RAG-from-Scratch" tutorial in LanceDB's vectordb-recipes
(https://github.com/lancedb/vectordb-recipes), reimplemented against DXMT.

Usage:
    python rag_pipeline.py                    # synthetic embeddings, no LLM
    python rag_pipeline.py --openai           # OpenAI embeddings + GPT answer
    OPENAI_API_KEY=sk-... python rag_pipeline.py --openai
"""

from __future__ import annotations

import argparse
import math
import hashlib
import os
import re
import textwrap
from typing import Any

from dxmt import Client
from environment import client_kwargs

try:
    from sentence_transformers import SentenceTransformer as _ST
    _SMODEL = _ST("all-MiniLM-L6-v2")
    EMBED_DIM = _SMODEL.get_sentence_embedding_dimension()
    _HAS_ST = True
except ImportError:
    _SMODEL = None
    EMBED_DIM = 384
    _HAS_ST = False

try:
    import openai as _openai_mod
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False

OPENAI_EMBED_DIM = 1536
OPENAI_EMBED_MODEL = "text-embedding-3-small"
OPENAI_CHAT_MODEL = "gpt-3.5-turbo"

# ── Knowledge base ─────────────────────────────────────────────────────────────

KNOWLEDGE_BASE = """
# Vector Databases

A vector database stores and retrieves high-dimensional vectors efficiently.
It supports approximate nearest neighbour (ANN) search to find the most similar
vectors to a given query. Unlike traditional relational databases, vector databases
are optimised for unstructured data like text, images, and audio.

## Core Operations

Indexing: vectors are inserted and organised into an index structure such as
HNSW (Hierarchical Navigable Small World) or IVF-PQ (Inverted File with Product
Quantisation). These structures allow fast ANN search at the cost of a small
approximation error.

Search: given a query vector, the database returns the k nearest neighbours
using a distance metric such as cosine similarity or Euclidean distance. The
result set can be further filtered by metadata predicates (payload filters).

## Distance Metrics

Cosine similarity measures the angle between two unit vectors. It is the most
common metric for text embeddings because it is invariant to vector magnitude,
focusing purely on directional similarity.

Euclidean (L2) distance measures the straight-line distance between two points
in high-dimensional space. It is sensitive to magnitude and is commonly used
for image feature vectors.

Dot product is equivalent to cosine similarity for unit-normalised vectors and
is the most computationally efficient metric. Many modern embedding models
output normalised vectors specifically to allow dot-product search.

## HNSW Algorithm

HNSW builds a multi-layered graph where each node represents a vector. The top
layers are sparse long-range edges (highway network), while the bottom layer
(layer 0) contains all nodes with dense local connections. During search, the
algorithm starts from an entry point at the top layer, greedily descends to
closer nodes, and finally performs a beam search at the bottom layer.

Two key parameters control the quality-speed tradeoff: M (number of connections
per node) and ef_search (size of the candidate list during search). Higher
values improve recall but increase memory and latency.

## Payload Filtering

DXMT supports attaching structured metadata (payload) to each vector.
Filters can be applied at search time to restrict results to a subset of
records. For example, filtering by {"category": "science"} returns only vectors
whose payload contains that field.

When the filter selectivity is low (< 5 % of records match), filtering inside
the graph (Acorn-style) is more efficient than post-filtering. When selectivity
is high, post-filtering is cheaper because most candidates already pass the test.

## RAG Pattern

Retrieval-Augmented Generation (RAG) combines a vector store with a language
model. The pipeline:
    1. Chunk documents into passages of ~200-500 tokens.
    2. Embed each chunk with a text encoder (e.g. sentence-transformers).
    3. Store embeddings + chunk text in the vector database.
    4. At query time, embed the question and retrieve the top-k chunks.
    5. Concatenate the retrieved passages into a context window.
    6. Ask the language model to answer based only on the provided context.

RAG reduces hallucination because the model can cite specific retrieved passages.
It also enables knowledge to be updated without retraining the model.
"""

# ── Text chunking ──────────────────────────────────────────────────────────────

def chunk_text(text: str, max_chars: int = 500, overlap: int = 80) -> list[str]:
    """Split text into overlapping chunks on sentence boundaries."""
    # Split on sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks: list[str] = []
    current = ""

    for sent in sentences:
        if len(current) + len(sent) + 1 <= max_chars:
            current = (current + " " + sent).strip()
        else:
            if current:
                chunks.append(current)
            # Carry overlap from end of previous chunk
            overlap_text = current[-overlap:] if len(current) > overlap else current
            current = (overlap_text + " " + sent).strip()

    if current:
        chunks.append(current)

    return chunks


# ── Embedding ─────────────────────────────────────────────────────────────────

def _synthetic_vector(seed: str, dim: int) -> list[float]:
    raw = []
    for i in range(dim):
        h = int(hashlib.md5(f"{seed}:{i}".encode()).hexdigest(), 16)
        raw.append((h % 10000) / 10000.0 * 2 - 1)
    norm = math.sqrt(sum(x * x for x in raw)) or 1.0
    return [x / norm for x in raw]


def embed_texts_local(texts: list[str]) -> list[list[float]]:
    if _HAS_ST and _SMODEL is not None:
        return _SMODEL.encode(texts, convert_to_numpy=True).tolist()
    return [_synthetic_vector(t, EMBED_DIM) for t in texts]


def embed_texts_openai(texts: list[str]) -> list[list[float]]:
    client = _openai_mod.OpenAI()
    response = client.embeddings.create(input=texts, model=OPENAI_EMBED_MODEL)
    return [item.embedding for item in response.data]


# ── DXMT helpers ──────────────────────────────────────────────────────────────

TABLE = "rag_chunks"


class RAGPipeline:
    def __init__(self, use_openai: bool = False) -> None:
        self.use_openai = use_openai and _HAS_OPENAI and bool(os.getenv("OPENAI_API_KEY"))
        self.dim = OPENAI_EMBED_DIM if self.use_openai else EMBED_DIM
        self.client = Client(**client_kwargs())

        if use_openai and not self.use_openai:
            print("[warn] --openai requested but openai not installed or OPENAI_API_KEY not set")
            print("[warn] falling back to local embeddings")

        mode = "OpenAI" if self.use_openai else ("sentence-transformers" if _HAS_ST else "synthetic")
        print(f"[rag] embedding mode: {mode} (dim={self.dim})")

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if self.use_openai:
            return embed_texts_openai(texts)
        return embed_texts_local(texts)

    # ── Index ─────────────────────────────────────────────────────────────────

    def build_index(self, text: str, chunk_size: int = 500) -> int:
        """Chunk, embed, and store a document. Returns chunk count."""
        # Reset table
        try:
            self.client.delete_table(TABLE)
        except Exception:
            pass
        self.client.create_table(name=TABLE, dimension=self.dim, distance_metric="cosine")

        chunks = chunk_text(text, max_chars=chunk_size)
        if not chunks:
            return 0

        print(f"[rag] chunked document into {len(chunks)} passages")
        vectors = self._embed(chunks)

        records = [
            {
                "id": f"chunk_{i:04d}",
                "vector": vec,
                "payload": {
                    "text": chunk,
                    "chunk_index": i,
                    "char_len": len(chunk),
                },
            }
            for i, (chunk, vec) in enumerate(zip(chunks, vectors))
        ]
        self.client.insert_records(table_name=TABLE, records=records)
        print(f"[rag] indexed {len(records)} chunks")
        return len(records)

    # ── Retrieve ──────────────────────────────────────────────────────────────

    def retrieve(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        """Return the top-k most relevant chunks for a query."""
        q_vec = self._embed([query])[0]
        hits = self.client.search(table_name=TABLE, vector=q_vec, k=k)
        # Enrich hits with chunk text from stored payload
        enriched = []
        for hit in hits:
            chunk_id = hit.get("id")
            record = self._get_chunk_text(chunk_id)
            enriched.append({
                "id": chunk_id,
                "score": hit.get("score", 0.0),
                "text": record,
            })
        return enriched

    def _get_chunk_text(self, chunk_id: str) -> str:
        """Fetch chunk text from the table via get_records."""
        try:
            records = self.client.get_records(table_name=TABLE, ids=[chunk_id])
            if records:
                import json
                r = records[0]
                pj = r.get("payload_json", "{}")
                if pj and pj != "null":
                    payload = json.loads(pj)
                    return payload.get("text", "")
        except Exception:
            pass
        return ""

    # ── Answer ────────────────────────────────────────────────────────────────

    def answer(self, query: str, k: int = 3) -> str:
        """Retrieve context and generate an answer (or stub)."""
        chunks = self.retrieve(query, k=k)

        if not chunks:
            return "No relevant context found."

        context = "\n\n---\n\n".join(
            f"[chunk {i+1}] {c['text']}" for i, c in enumerate(chunks)
        )

        if self.use_openai:
            return self._llm_answer(query, context)

        # Stub: return the best-matching chunk text
        best = chunks[0]
        excerpt = textwrap.shorten(best["text"], width=300, placeholder="…")
        return f"[no LLM — top chunk, score={best['score']:.4f}]\n{excerpt}"

    def _llm_answer(self, question: str, context: str) -> str:
        oai = _openai_mod.OpenAI()
        prompt = (
            "Answer the question based ONLY on the provided context. "
            "If the context does not contain the answer, say 'I don't know'.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}"
        )
        response = oai.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0,
        )
        return response.choices[0].message.content.strip()

    def close(self) -> None:
        try:
            self.client.delete_table(TABLE)
        except Exception:
            pass
        self.client.close()


# ── Demo ──────────────────────────────────────────────────────────────────────

DEMO_QUERIES = [
    "What is HNSW and how does it work?",
    "What distance metrics are available?",
    "How does RAG reduce hallucination?",
    "When should I use cosine vs euclidean distance?",
    "What are payload filters?",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="DXMT local RAG pipeline demo")
    parser.add_argument("--openai", action="store_true", help="Use OpenAI embeddings and GPT")
    args = parser.parse_args()

    print("=" * 70)
    print("  DXMT Local RAG Pipeline")
    print("=" * 70)

    rag = RAGPipeline(use_openai=args.openai)

    try:
        # Index the built-in knowledge base
        n = rag.build_index(KNOWLEDGE_BASE)
        print(f"\n[index] {n} chunks indexed from knowledge base\n")

        # Ask each demo query
        for query in DEMO_QUERIES:
            print(f"\n{'─' * 70}")
            print(f"Q: {query}")
            print(f"{'─' * 70}")

            # Show retrieved chunks
            chunks = rag.retrieve(query, k=2)
            print("Retrieved context:")
            for c in chunks:
                excerpt = textwrap.shorten(c["text"], width=120, placeholder="…")
                print(f"  [{c['score']:.4f}] {excerpt}")

            # Generate answer
            answer = rag.answer(query, k=3)
            print(f"\nAnswer:\n{textwrap.fill(answer, width=70, initial_indent='  ', subsequent_indent='  ')}")

    finally:
        rag.close()
        print("\n[done]")


if __name__ == "__main__":
    main()

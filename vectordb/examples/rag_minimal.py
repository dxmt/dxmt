#!/usr/bin/env python3
"""Retrieval-augmented generation, the retrieval half, end to end and offline.

A RAG demo that needs an API key is a demo most readers cannot run. This one embeds locally with
`all-MiniLM-L6-v2` and answers by quoting what it retrieved, so the whole loop -- chunk, embed,
store, retrieve, ground the answer -- runs with nothing but pip installs and a server.

    pip install sentence-transformers
    cd vectordb && cargo run --release -- serve &
    python3 examples/rag_minimal.py
    python3 examples/rag_minimal.py "how does the graph get built"

**On the fallback.** `dxmt-python-sdk/examples/common_utils.embed` falls back to a deterministic
md5-derived vector when sentence-transformers is absent, which keeps code running offline. It
does not keep *retrieval* working: an md5 of the whole string carries no meaning, so similar
questions land nowhere near their answers. This example says so loudly and keeps going, because
a RAG demo that silently retrieves nonsense while printing a confident answer is worse than one
that refuses -- and being able to see the plumbing work without the model is still worth
something.

Generation is left out on purpose. Adding an LLM would require a key and would put the
interesting failure -- did retrieval find the right passage? -- behind a model's ability to
paper over it. What is printed is the retrieved context, which is the part this database is
responsible for.
"""

import os
import sys
import textwrap

try:
    import httpx
except ImportError:
    sys.exit("this example needs httpx:  pip install httpx")

BASE = os.environ.get("DXMT_URL", "http://localhost:9999").rstrip("/") + "/api/v1"
TABLE = os.environ.get("DXMT_TABLE", "rag_minimal_demo")
TOP_K = 3

HTTP = httpx.Client(timeout=60.0)

#: A small corpus about this database, so the retrieved passage can be checked by eye.
#: A small corpus about this database, so the retrieved passage can be checked by eye.
#:
#: Each passage sticks to one subject. A first draft had a "storage" passage that mentioned
#: disk, memory *and* the WAL directory, and it won two questions it should have lost -- one
#: about compression and one about surviving a restart -- because it touched every topic a
#: little. The retrieval was working; the corpus was the problem. Worth knowing before blaming
#: a database for a bad answer.
DOCUMENTS = [
    ("hnsw", "The index is a hierarchical navigable small world graph. Each record becomes a "
             "node linked to its nearest neighbours, and a search walks from an entry point "
             "towards the query, keeping a candidate list of size ef_search."),
    ("pq", "Product quantization compresses vectors to save memory. Each vector is split into "
           "subspaces and each subspace replaced by the index of its nearest centroid, so "
           "3,072 bytes of floats become 192 bytes of codes. A table does this to itself once "
           "it passes 65,536 rows."),
    ("rerank", "A quantized graph returns approximate distances, so the best candidates are "
               "rescored against their exact vectors read from the row file. The rerank pool "
               "is five times the requested k, at least 100."),
    ("filters", "A filter on a numeric field uses a sorted array and answers a range with two "
                "binary searches. A filter on a string field uses an inverted map keyed by "
                "distinct value. Both are checked against the record itself before a result "
                "is returned."),
    ("wal", "Surviving a restart is the job of the write-ahead log. Every mutation is appended "
            "to it before it is acknowledged, and periodic snapshots let recovery replay only "
            "the tail. A crash costs at most the entries written since the last snapshot."),
    ("rowfiles", "Records and full vectors live in row files next to the log rather than inside "
                 "the snapshot, which is why a snapshot of ten million rows takes seconds. "
                 "DB_RECORD_STORE and DB_VECTOR_STORE choose between those files and memory."),
]


QUESTIONS = [
    "how are vectors compressed to save memory",
    "what happens when I filter on a number",
    "how does the database survive a restart",
]


FALLBACK_DIM = 384


def synthetic_vectors(texts):
    """Deterministic unit vectors from a hash. Same idea as `common_utils.synthetic_vector`,
    written out here rather than imported because that module loads a model at import time --
    a fallback that depends on the thing it is a fallback for is no fallback."""
    import hashlib
    import math

    out = []
    for text in texts:
        raw = []
        for i in range(FALLBACK_DIM):
            h = int(hashlib.md5(f"{text}:{i}".encode()).hexdigest(), 16)
            raw.append((h % 10000) / 10000.0 * 2 - 1)
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        out.append([x / norm for x in raw])
    return out


def load_embedder():
    """The local model, or the fallback with a warning that names what it costs."""
    try:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
    except ImportError:
        print(
            "!! sentence-transformers is not installed.\n"
            "!! Falling back to deterministic hash vectors: the plumbing below will work and\n"
            "!! the retrieved passages will be meaningless, because a hash of a sentence\n"
            "!! carries none of its meaning. Install it to see real retrieval:\n"
            "!!     pip install sentence-transformers\n"
        )
        return synthetic_vectors, FALLBACK_DIM, False

    model = SentenceTransformer("all-MiniLM-L6-v2")
    return (
        lambda texts: model.encode(texts, convert_to_numpy=True).tolist(),
        model.get_sentence_embedding_dimension(),
        True,
    )


def api(method, path, body=None, allow=()):
    response = HTTP.request(method, BASE + path, json=body)
    if response.status_code >= 300 and response.status_code not in allow:
        sys.exit(f"{method} {path} -> {response.status_code}\n{response.text[:400]}")
    return response.json()


def main():
    questions = sys.argv[1:] or QUESTIONS
    embed, dim, real = load_embedder()

    try:
        HTTP.get(BASE + "/health").raise_for_status()
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"no server at {BASE} ({exc})\n\n    cd vectordb && cargo run --release -- serve")

    api("DELETE", f"/tables/{TABLE}", allow=(404,))
    api("POST", "/tables", {"name": TABLE, "dimension": dim, "distance_metric": "cosine"})

    vectors = embed([text for _, text in DOCUMENTS])
    api(
        "POST",
        f"/tables/{TABLE}/records",
        {
            "records": [
                {"id": doc_id, "vector": vector, "payload": {"text": text}}
                for (doc_id, text), vector in zip(DOCUMENTS, vectors)
            ]
        },
    )
    print(f"indexed {len(DOCUMENTS)} passages at {dim} dimensions\n")

    for question in questions:
        query = embed([question])[0]
        hits = api("POST", f"/tables/{TABLE}/search", {"vector": query, "top_k": TOP_K})
        hits = hits["data"]["hits"]

        print(f"Q  {question}")
        best = hits[0]
        print(f"   retrieved '{best['id']}' (score {best['score']:.3f})")
        for line in textwrap.wrap(best["payload"]["text"], 76):
            print(f"   | {line}")
        others = ", ".join(f"{h['id']} {h['score']:.2f}" for h in hits[1:])
        print(f"   also considered: {others}\n")

    if real:
        print("Each answer above was grounded in a passage this database returned. Check the")
        print("passage against the question -- that judgement is the whole point of RAG, and")
        print("it is the retrieval step, not the language model, that decides it.")
    else:
        print("The passages above were retrieved by md5 similarity and mean nothing. The")
        print("plumbing works; the retrieval does not. Install sentence-transformers.")

    print(f"\ntable '{TABLE}' left in place; DELETE /api/v1/tables/{TABLE} to remove it")


if __name__ == "__main__":
    main()

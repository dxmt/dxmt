#!/usr/bin/env python3
"""Score this database on the RAG dimensions a database actually decides.

RAGAS evaluates a RAG system on four metrics, and they split in half. **Context Precision** and
**Context Recall** measure whether retrieval put the passage that answers the question in front
of the model, and at what rank. **Faithfulness** and **Answer Relevancy** measure what the model
then did with it. Only the first pair is this database's to answer for, and a vector store that
reports the second pair is reporting its LLM.

So this measures the first pair, and the things around it that decide whether a real RAG
pipeline works: filtered retrieval, freshness after an edit, and latency. No LLM, no API key,
and no judge model -- the gold passage for each question is known, so precision and recall are
counted rather than estimated, which also makes the result the same on every run.

    pip install sentence-transformers
    cd vectordb && cargo run --release -- serve &
    python3 examples/rag_eval.py

`DXMT_URL` points it elsewhere. The corpus is this repository's own documentation, chunked; the
questions are hand-written, and each names a phrase that only its answering passage contains, so
the labels survive the documents being edited.
"""

import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE = os.environ.get("DXMT_URL", "http://localhost:9999").rstrip("/") + "/api/v1"
TABLE = "rag_eval"
REPO = Path(__file__).resolve().parent.parent

#: Markdown that ships with the repository. Real prose, written for people, with the mix of
#: prose, tables and code blocks a RAG corpus actually contains.
SOURCES = [
    "README.md",
    "CHANGELOG.md",
    "tests/README.md",
    "examples/README.md",
    "docs/operations/backup-restore.md",
    "../CONTRIBUTING.md",
    "../SECURITY.md",
]

#: (question, a phrase only the answering passage contains). The phrase is the label: whichever
#: chunk holds it is the gold passage, so re-chunking or an edit to the surrounding text cannot
#: silently move the answer somewhere else without this failing loudly.
QUESTIONS = [
    ("how do I take a backup and restore it", "backup"),
    ("what happens to my data when the process is killed", "write-ahead log"),
    ("how many dimensions does a table need to declare", "dimension"),
    ("which features are not compiled into a default build", "opt-in"),
    ("how do I report a security problem", "vulnerability"),
    ("what should a pull request include", "pull request"),
    ("how fast is bulk loading", "rows/s"),
    ("how much memory does a row cost", "KB"),
    ("what does the functional test suite run against", "both storage modes"),
    ("which examples need an API key", "OpenAI"),
    ("what is the quantization threshold", "65,536"),
    ("how is dead space in the row files reclaimed", "punch"),
]


def req(method, path, body=None, timeout=120):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(BASE + path, data=data, method=method,
                               headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, e.read()[:300].decode("utf8", "replace")
    except Exception as e:  # noqa: BLE001
        return None, repr(e)


def chunks_of(path: Path, source: str):
    """Split on blank lines, then glue small pieces together.

    Paragraph-per-chunk is the obvious split and a bad one: a table row or a one-line heading
    carries no context of its own, and a corpus of them retrieves noise. Chunks are grown to at
    least `MIN` characters so each one can stand alone.
    """
    MIN, MAX = 320, 1100
    text = path.read_text(encoding="utf8", errors="replace")
    out, buf = [], ""
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        buf = f"{buf}\n\n{para}" if buf else para
        if len(buf) >= MIN:
            out.append(buf[:MAX])
            buf = ""
    if len(buf) >= 120:
        out.append(buf)
    return [(f"{source}#{i}", c, source) for i, c in enumerate(out)]


def main():
    code, _ = req("GET", "/tables")
    if code != 200:
        print(f"no server at {BASE}\n\n    cd vectordb && cargo run --release -- serve\n",
              file=sys.stderr)
        return 1

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("this needs real embeddings: pip install sentence-transformers", file=sys.stderr)
        return 1

    # ── corpus ────────────────────────────────────────────────────────────────────────────
    passages = []
    for source in SOURCES:
        path = REPO / source
        if path.exists():
            passages.extend(chunks_of(path, source.replace("../", "")))
    if len(passages) < 40:
        print(f"only {len(passages)} passages; the repository's docs are missing", file=sys.stderr)
        return 1

    model = SentenceTransformer("all-MiniLM-L6-v2")
    dim = model.get_sentence_embedding_dimension()
    vectors = model.encode([p[1] for p in passages], batch_size=64, show_progress_bar=False)

    req("DELETE", f"/tables/{TABLE}")
    req("POST", "/tables", {"name": TABLE, "dimension": dim, "distance_metric": "cosine"})
    records = [
        {"id": pid, "vector": [float(x) for x in vec],
         "payload": {"source": source, "text": text[:600]}}
        for (pid, text, source), vec in zip(passages, vectors)
    ]
    for i in range(0, len(records), 200):
        status, body = req("POST", f"/tables/{TABLE}/records", {"records": records[i:i + 200]})
        if status != 200:
            print(f"insert failed: {status} {body}", file=sys.stderr)
            return 1

    # Resolve each question's gold passage by its label phrase.
    labelled = []
    for question, marker in QUESTIONS:
        gold = [pid for pid, text, _ in passages if marker.lower() in text.lower()]
        if gold:
            labelled.append((question, gold))
        else:
            print(f"  (dropped '{question}': no passage contains {marker!r})")
    print(f"corpus: {len(passages)} passages from {len(SOURCES)} documents, {dim} dimensions")
    print(f"labelled questions: {len(labelled)}\n")

    def search(vector, k, filt=None):
        body = {"vector": [float(x) for x in vector], "top_k": k, "limit": k}
        if filt:
            body["filter"] = filt
        t = time.time()
        status, resp = req("POST", f"/tables/{TABLE}/search", body)
        took = (time.time() - t) * 1000
        hits = resp.get("data", {}).get("hits", []) if status == 200 else []
        return [h["id"] for h in hits], took

    qvecs = model.encode([q for q, _ in labelled], show_progress_bar=False)

    # ── 1. context recall / precision, the two RAGAS metrics a database decides ──────────
    print("── Context recall and precision (no LLM: the gold passage is known) ──")
    lat = []
    ranks = []
    for (question, gold), qv in zip(labelled, qvecs):
        ids, took = search(qv, 10)
        lat.append(took)
        rank = next((i + 1 for i, pid in enumerate(ids) if pid in gold), None)
        ranks.append(rank)
    for k in (1, 3, 5, 10):
        hit = sum(1 for r in ranks if r and r <= k)
        print(f"   recall@{k:<2}  {hit}/{len(ranks)}  {hit / len(ranks):.2f}")
    mrr = statistics.mean(1 / r if r else 0.0 for r in ranks)
    print(f"   MRR       {mrr:.3f}   (1.0 = the answer is always the first hit)")
    misses = [q for (q, _), r in zip(labelled, ranks) if not r]
    for q in misses:
        print(f"   missed: {q}")

    # ── 2. filtered retrieval: RAG over one tenant, one document, one permission scope ────
    print("\n── Filtered retrieval (the same query, restricted to one document) ──")
    target = "README.md"
    in_scope = sum(1 for _, _, s in passages if s == target)
    ids, _ = search(qvecs[0], 5, {"must": [{"key": "source",
                                            "operation": {"type": "Equal", "value": target}}]})
    leaked = [pid for pid in ids if not pid.startswith(target)]
    print(f"   {in_scope} of {len(passages)} passages are in scope")
    print(f"   returned {len(ids)} hits, {len(leaked)} from outside the filter")
    print("   " + ("PASS: nothing leaked across the filter" if not leaked
                   else f"FAIL: leaked {leaked}"))
    print(f"   (a filter that returns nothing is the other failure: {'PASS' if ids else 'FAIL'})")

    # ── 3. freshness: a corpus that changes has to retrieve what it says now ─────────────
    print("\n── Freshness (a passage is edited; retrieval must follow) ──")
    novel = "The quorum ledger reconciles against the tide table every fortnight."
    victim = passages[0][0]
    nv = model.encode([novel], show_progress_bar=False)[0]
    # The id is `README.md#0`; unencoded, the `#` truncates the path at the fragment and the
    # update lands on a record nobody asked for. This is the shape a RAG corpus produces by
    # default, which is why it is worth a line here and not only in the SDK tests.
    status, _ = req("PUT", f"/tables/{TABLE}/records/{urllib.parse.quote(victim, safe='')}",
                    {"vector": [float(x) for x in nv], "payload": {"source": passages[0][2],
                                                                   "text": novel}})
    ids, _ = search(nv, 1)
    print(f"   update returned {status}; searching for the new text finds {ids[:1]}")
    print("   " + ("PASS: the edit is what answers now" if ids[:1] == [victim]
                   else f"FAIL: expected {victim}"))

    # ── 4. latency, which decides whether RAG is interactive ────────────────────────────
    print("\n── Retrieval latency, top-10 over the whole corpus ──")
    warm = [search(qv, 10)[1] for qv in qvecs for _ in range(3)]
    warm.sort()
    print(f"   p50 {warm[len(warm) // 2]:.1f} ms    p95 {warm[int(len(warm) * 0.95)]:.1f} ms"
          f"    max {warm[-1]:.1f} ms   over {len(warm)} queries")

    print(f"\nNot measured here, on purpose: faithfulness and answer relevancy. Both score a\n"
          f"language model's output, not a retrieval engine, and a vector database that\n"
          f"reports them is reporting its LLM.\n")
    print(f"table '{TABLE}' left in place; DELETE {BASE}/tables/{TABLE} to remove it")
    return 0


if __name__ == "__main__":
    sys.exit(main())

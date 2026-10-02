"""Agent memory — conversation history stored as vectors in DXMT.

Implements an agentic memory pattern:
each conversation turn (user message + agent reply) is embedded and stored.
At recall time, memories are scored by a hybrid of:

    score = α × semantic_similarity + (1 - α) × recency_score

where recency_score = exp(-λ × turns_ago).

Optionally, "consolidation" periodically merges redundant memories (simulated
here with a dedup pass by cosine threshold, since we don't have an LLM in the
default mode).

This example is self-contained: no external APIs required.

Usage:
    python agent_memory.py                  # interactive REPL
    python agent_memory.py --demo           # run a scripted demo
    python agent_memory.py --demo --openai  # demo + GPT answers
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any

from dxmt import Client
from common_utils import embed, EMBED_DIM

try:
    import openai as _openai
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False


# ── Memory types ──────────────────────────────────────────────────────────────

@dataclass
class Memory:
    memory_id: str
    turn: int
    user_message: str
    agent_reply: str
    summary: str
    vector: list[float]  # cached at insert time; no re-embed needed for consolidation
    timestamp: float = field(default_factory=time.time)
    importance: float = 1.0

    def text_for_embedding(self) -> str:
        return f"User: {self.user_message}\nAgent: {self.agent_reply}"

    def display(self, current_turn: int) -> str:
        turns_ago = current_turn - self.turn
        age = f"{turns_ago} turn{'s' if turns_ago != 1 else ''} ago"
        return (
            f"  [turn {self.turn:3d} | {age:15s} | importance={self.importance:.2f}]  "
            f"U: {self.user_message[:40]}  A: {self.agent_reply[:40]}"
        )


# ── Agent memory store ────────────────────────────────────────────────────────

TABLE = "agent_memory"
DEDUP_THRESHOLD = 0.80   # cosine similarity above which two memories are considered duplicates
RECENCY_LAMBDA = 0.15    # recency decay rate (higher = faster decay)
ALPHA = 0.7              # weight on semantic similarity vs recency


class AgentMemoryStore:
    """
    Stores conversation memories as vectors and retrieves them by hybrid score.
    """

    def __init__(self, client: Client) -> None:
        self.client = client
        self._memories: list[Memory] = []   # local registry for recency scoring
        self._turn = 0
        self._setup()

    def _setup(self) -> None:
        try:
            self.client.delete_table(TABLE)
        except Exception:
            pass
        self.client.create_table(name=TABLE, dimension=EMBED_DIM, distance_metric="cosine")

    # ── Write ─────────────────────────────────────────────────────────────────

    def add(
        self,
        user_message: str,
        agent_reply: str,
        importance: float = 1.0,
    ) -> Memory:
        self._turn += 1
        mem_id = f"mem_{self._turn:04d}"
        summary = _summarize(user_message, agent_reply)

        text = f"User: {user_message}\nAgent: {agent_reply}"
        vec = embed([text])[0]  # embed once; cache in Memory to avoid re-embedding

        memory = Memory(
            memory_id=mem_id,
            turn=self._turn,
            user_message=user_message,
            agent_reply=agent_reply,
            summary=summary,
            vector=vec,
            importance=importance,
        )
        self.client.insert_records(
            table_name=TABLE,
            records=[{
                "id": mem_id,
                "vector": vec,
                "payload": {
                    "turn": self._turn,
                    "user": user_message,
                    "agent": agent_reply,
                    "summary": summary,
                    "timestamp": memory.timestamp,
                    "importance": importance,
                },
            }],
        )
        self._memories.append(memory)
        return memory

    # ── Read ──────────────────────────────────────────────────────────────────

    def recall(
        self,
        query: str,
        k: int = 5,
        alpha: float = ALPHA,
    ) -> list[tuple[Memory, float]]:
        """
        Retrieve memories relevant to `query` scored by:
            final_score = alpha * semantic_sim + (1 - alpha) * recency_score
        """
        if not self._memories:
            return []

        q_vec = embed([query])[0]
        # Fetch more candidates than needed; we'll re-rank
        fetch_k = min(len(self._memories), max(k * 3, 10))
        hits = self.client.search(table_name=TABLE, vector=q_vec, k=fetch_k)

        # Build id → memory lookup
        mem_by_id = {m.memory_id: m for m in self._memories}
        current_turn = self._turn

        scored: list[tuple[Memory, float]] = []
        for hit in hits:
            mem_id = hit.get("id")
            sem_sim = hit.get("score", 0.0)
            mem = mem_by_id.get(mem_id)
            if mem is None:
                continue

            turns_ago = current_turn - mem.turn
            recency = math.exp(-RECENCY_LAMBDA * turns_ago)
            hybrid = alpha * sem_sim + (1 - alpha) * recency
            # Boost by importance
            hybrid *= mem.importance
            scored.append((mem, hybrid))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    # ── Consolidation ─────────────────────────────────────────────────────────

    def consolidate(self, sim_threshold: float = DEDUP_THRESHOLD) -> int:
        """
        Remove near-duplicate memories (cosine sim > threshold).
        In production this would use an LLM to merge; here we simply drop the
        older of any duplicate pair.
        Returns the number of memories removed.
        """
        if len(self._memories) < 2:
            return 0

        # Use cached vectors — no HTTP calls, no re-embedding
        vecs = [m.vector for m in self._memories]
        n = len(vecs)
        to_remove: set[str] = set()

        for i in range(n):
            if self._memories[i].memory_id in to_remove:
                continue
            for j in range(i + 1, n):
                if self._memories[j].memory_id in to_remove:
                    continue
                sim = _cosine(vecs[i], vecs[j])
                if sim >= sim_threshold:
                    # Drop the older one (lower turn)
                    older = self._memories[i] if self._memories[i].turn < self._memories[j].turn else self._memories[j]
                    to_remove.add(older.memory_id)

        removed = 0
        for mem_id in to_remove:
            try:
                self.client.delete_records(table_name=TABLE, ids=[mem_id])
            except Exception:
                pass
            self._memories = [m for m in self._memories if m.memory_id != mem_id]
            removed += 1

        return removed

    # ── Stats ─────────────────────────────────────────────────────────────────

    def size(self) -> int:
        return len(self._memories)

    def all_memories(self) -> list[Memory]:
        return list(self._memories)

    def close(self) -> None:
        try:
            self.client.delete_table(TABLE)
        except Exception:
            pass


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def _summarize(user: str, agent: str) -> str:
    """Simple rule-based summariser (no LLM required)."""
    u_short = user[:60] + "…" if len(user) > 60 else user
    a_short = agent[:60] + "…" if len(agent) > 60 else agent
    return f"User asked: {u_short} | Agent replied: {a_short}"


def _stub_answer(question: str, memories: list[tuple[Memory, float]]) -> str:
    """Return a canned answer citing any relevant memories."""
    if not memories:
        return "I don't have any relevant memories for that question yet."
    best_mem, best_score = memories[0]
    return (
        f"[recall score={best_score:.3f}] "
        f"Based on turn {best_mem.turn}: {best_mem.agent_reply}"
    )


def _gpt_answer(question: str, memories: list[tuple[Memory, float]]) -> str:
    context = "\n".join(
        f"- [turn {m.turn}] U: {m.user_message} | A: {m.agent_reply}"
        for m, _ in memories
    )
    client = _openai.OpenAI()
    prompt = (
        "You are a helpful assistant with memory. "
        "Use the following past conversation excerpts to answer the new question. "
        "Be concise.\n\n"
        f"Past context:\n{context}\n\n"
        f"New question: {question}"
    )
    resp = client.chat.completions.create(
        model=os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


# ── Demo conversation ─────────────────────────────────────────────────────────

DEMO_CONVERSATION = [
    ("What is your name?", "I am DXMT Assistant, your vector-powered AI agent."),
    ("What programming languages do you know?", "I have expertise in Python, Rust, and TypeScript."),
    ("How does HNSW work?", "HNSW builds a layered graph for approximate nearest-neighbour search."),
    ("What is RAG?", "RAG grounds LLM answers by retrieving relevant passages from a vector store."),
    ("Do you prefer Python or Rust?", "I enjoy both — Python for rapid prototyping, Rust for performance-critical systems."),
    ("What distance metrics does DXMT support?", "DXMT supports cosine similarity, Euclidean (L2), and dot product."),
    ("Can you recall my first question?", "You first asked me for my name."),
    ("What is product quantisation?", "PQ compresses vectors into sub-vector codes, reducing memory 10-50x."),
    ("Tell me more about HNSW parameters.", "Key parameters: M controls connections per node; ef_search controls beam width during query."),
    ("Which vector database features matter most for production?", "Filtered search, incremental inserts, tiered storage, and concurrent read performance."),
]

RECALL_QUERIES = [
    "What did I ask about HNSW?",
    "Tell me about RAG and retrieval",
    "Which programming languages were mentioned?",
    "What can you tell me about memory compression for vectors?",
    "What was the first thing we talked about?",
]


# ── Main ──────────────────────────────────────────────────────────────────────

def run_demo(use_openai: bool) -> None:
    from common_utils import _HAS_ST
    embed_mode = "sentence-transformers" if _HAS_ST else "bag-of-words"
    print("=" * 70)
    print("  DXMT Agent Memory Demo")
    print(f"  Embedding: {embed_mode}  (dim={EMBED_DIM})")
    print(f"  Scoring: α={ALPHA:.2f} × semantic + {1-ALPHA:.2f} × recency")
    print("=" * 70)

    client = Client(host="localhost", port=9999)
    store = AgentMemoryStore(client)

    try:
        # 1. Store conversation turns
        print(f"\n[store] Recording {len(DEMO_CONVERSATION)} conversation turns …")
        for user_msg, agent_reply in DEMO_CONVERSATION:
            mem = store.add(user_msg, agent_reply)
            print(f"  turn {mem.turn:3d}: {mem.summary[:80]}")

        print(f"\n[store] {store.size()} memories stored")

        # 2. Recall with hybrid scoring
        print(f"\n{'─' * 70}")
        print("  Memory recall (hybrid semantic + recency scoring)")
        print(f"{'─' * 70}")

        for query in RECALL_QUERIES:
            print(f"\n  Query: \"{query}\"")
            results = store.recall(query, k=3)
            if not results:
                print("    (no relevant memories)")
                continue
            for mem, score in results:
                print(f"    [{score:.4f}] {mem.display(store._turn)}")

            # Generate an answer
            if use_openai and _HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
                answer = _gpt_answer(query, results)
            else:
                answer = _stub_answer(query, results)
            print(f"\n    Answer: {answer}")

        # 3. Consolidation demo — add near-duplicate turn
        print(f"\n{'─' * 70}")
        print("  Consolidation demo (dedup near-duplicate memories)")
        print(f"{'─' * 70}")

        # A near-duplicate of turn 3, not merely a related question: the same words about the
        # same thing. "How does HNSW search work?" was here, which shares one word with the
        # turn it was meant to duplicate -- so the demo added a memory, removed nothing, and
        # printed "removed 0" under a heading that says it deduplicates.
        store.add(
            "How does HNSW work?",
            "HNSW builds a layered graph for approximate nearest-neighbour search.",
        )
        print(f"  Added a near-duplicate of turn 3. Store size: {store.size()}")
        removed = store.consolidate(sim_threshold=0.80)
        print(f"  Consolidation removed {removed} near-duplicate(s) at a 0.80 threshold. "
              f"Size now: {store.size()}")
        if removed == 0:
            print("  (nothing crossed the threshold -- with the bag-of-words fallback, only "
                  "near-identical wording does)")

    finally:
        store.close()
        client.close()
        print("\n[done]")


def run_repl(use_openai: bool) -> None:
    """Interactive REPL: type messages, get replies, and retrieve from memory."""
    print("=" * 70)
    print("  DXMT Agent Memory — Interactive REPL")
    print("  Commands: /recall <query>  /list  /consolidate  /quit")
    print("=" * 70)

    client = Client(host="localhost", port=9999)
    store = AgentMemoryStore(client)

    try:
        turn = 0
        while True:
            try:
                user_input = input(f"\n[turn {store._turn + 1}] You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input:
                continue

            if user_input.startswith("/quit"):
                break

            if user_input.startswith("/recall "):
                q = user_input[8:].strip()
                results = store.recall(q, k=3)
                if not results:
                    print("  (no relevant memories found)")
                else:
                    for mem, score in results:
                        print(f"  [{score:.4f}] {mem.display(store._turn)}")
                continue

            if user_input == "/list":
                for mem in store.all_memories():
                    print(mem.display(store._turn))
                continue

            if user_input == "/consolidate":
                removed = store.consolidate()
                print(f"  Removed {removed} duplicate(s). {store.size()} memories remain.")
                continue

            # Get an answer using memory
            memories = store.recall(user_input, k=3)
            if use_openai and _HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
                reply = _gpt_answer(user_input, memories)
            else:
                reply = _stub_answer(user_input, memories)

            print(f"Agent: {reply}")

            # Store this exchange
            store.add(user_input, reply)

    finally:
        store.close()
        client.close()
        print("\n[done]")


def main() -> None:
    parser = argparse.ArgumentParser(description="DXMT agent memory demo")
    parser.add_argument("--demo", action="store_true",
                        help="Run scripted demo instead of interactive REPL")
    parser.add_argument("--openai", action="store_true",
                        help="Use GPT-3.5 for answers (requires OPENAI_API_KEY)")
    args = parser.parse_args()

    use_openai = args.openai and _HAS_OPENAI and bool(os.getenv("OPENAI_API_KEY"))
    if args.openai and not use_openai:
        print("[warn] --openai requested but not available — using stub answers")

    if args.demo:
        run_demo(use_openai=use_openai)
    else:
        run_repl(use_openai=use_openai)


if __name__ == "__main__":
    main()

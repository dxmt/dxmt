"""A LangGraph agent whose long-term memory lives in DXMT.

What the graph itself carries is one turn: a message in, a reply out. Anything the agent knows
about the user from *earlier* threads comes back out of DXMT through `DXMTLangGraphStore`,
which implements LangGraph's `BaseStore` -- so `store.search(namespace, query=...)` is a vector
search and `store.put(...)` is an upsert.

A namespace is a table. `("demo_user", "memories")` becomes `langgraph_demo__user_memories`:
the components are joined with an underscore and an underscore inside a component is doubled,
so the mapping round-trips and two different namespaces cannot land on one table.

No API key and no model download: with `sentence-transformers` installed the example uses it,
and without it falls back to a hashed bag of words, which is enough to show recall by meaning.

Usage:
    python langgraph_memory.py                 # against localhost:9999
    DXMT_PORT=19999 python langgraph_memory.py

Requires: pip install langgraph
"""

from __future__ import annotations

import hashlib
import math
import os
import re

from typing_extensions import TypedDict

from dxmt import Client
from dxmt.langgraph_store import DXMTLangGraphStore

try:
    from langgraph.graph import END, START, StateGraph
except ImportError:  # pragma: no cover - the example is the error message
    raise SystemExit("this example needs langgraph: pip install langgraph")

HOST = os.environ.get("DXMT_HOST", "localhost")
PORT = int(os.environ.get("DXMT_PORT", "9999"))
DIMS = 256

_WORD = re.compile(r"[a-z0-9]+")
# Words that carry no meaning here would otherwise dominate a short sentence's direction.
_STOPWORDS = {"a", "an", "the", "i", "my", "me", "you", "do", "does", "did", "have", "has",
              "how", "what", "which", "of", "is", "are", "was", "no", "on", "in", "to", "it"}


def embed(texts: list[str]) -> list[list[float]]:
    """A hashed bag of words: two texts sharing content words share a direction."""
    vectors = []
    for text in texts:
        vector = [0.0] * DIMS
        for word in _WORD.findall(text.lower()):
            if word in _STOPWORDS:
                continue
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIMS] += 1.0
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            # The server refuses a zero vector, and rightly: it has no direction to compare.
            vector[0] = 1.0
            norm = 1.0
        vectors.append([v / norm for v in vector])
    return vectors


class State(TypedDict):
    message: str
    reply: str


def main() -> None:
    store = DXMTLangGraphStore(
        Client(host=HOST, port=PORT),
        embedding_dims=DIMS,
        embed_fn=embed,
        default_index_fields=["text"],
    )
    user = ("demo_user", "memories")

    def respond(state: State) -> State:
        """Recall what this user said before, then record what they just said."""
        recalled = store.search(user, query=state["message"], limit=1)
        if recalled and recalled[0].score and recalled[0].score > 0.1:
            reply = f"I remember: {recalled[0].value['text']!r} ({recalled[0].score:.2f})"
        else:
            reply = "I have nothing on file about that yet."
        key = hashlib.md5(state["message"].encode()).hexdigest()[:12]
        store.put(user, key, {"text": state["message"]})
        return {"message": state["message"], "reply": reply}

    graph = StateGraph(State)
    graph.add_node("respond", respond)
    graph.add_edge(START, "respond")
    graph.add_edge("respond", END)
    app = graph.compile()

    print(f"DXMT at {HOST}:{PORT}, namespace {user} -> table {store._namespace_to_table(user)}\n")

    # One thread teaches it two things.
    for message in ["I prefer my coffee black, no sugar",
                    "my dog is a border collie called Mei"]:
        print(f"[thread 1] {message}\n           {app.invoke({'message': message, 'reply': ''})['reply']}")

    # A separate invocation carries no state, so anything recalled here came from DXMT.
    for message in ["what kind of dog do I have?", "how do I take my coffee?"]:
        print(f"[thread 2] {message}\n           {app.invoke({'message': message, 'reply': ''})['reply']}")

    print(f"\nnamespaces under demo_user: {store.list_namespaces(prefix=('demo_user',))}")


if __name__ == "__main__":
    main()

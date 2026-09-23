"""Multimodal content store — video/image search over embeddings.

Demonstrates storing text transcripts and image embeddings in the same DXMT table,
then querying each modality independently via payload filters.

Pattern:
    - One table per "session" (video / document set)
    - Records tagged with {"type": "text"} or {"type": "image"}
    - Filtered search returns only the requested modality
    - Cross-modal search returns both (no filter)

This example is self-contained: no external APIs required.
sentence-transformers is used for text embeddings if available;
otherwise synthetic unit vectors are generated deterministically.

Usage:
    python multimodal_store.py
"""

from __future__ import annotations

import math
import hashlib
from typing import Any

from dxmt import Client

try:
    from sentence_transformers import SentenceTransformer
    _MODEL: SentenceTransformer | None = SentenceTransformer("all-MiniLM-L6-v2")
    EMBED_DIM = _MODEL.get_sentence_embedding_dimension()
    print(f"[embedding] sentence-transformers loaded (dim={EMBED_DIM})")
except ImportError:
    _MODEL = None
    EMBED_DIM = 384
    print("[embedding] sentence-transformers not found — using synthetic vectors (dim=384)")


# ── embedding helpers ─────────────────────────────────────────────────────────

def embed_text(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts. Falls back to deterministic synthetic vectors."""
    if _MODEL is not None:
        return _MODEL.encode(texts, convert_to_numpy=True).tolist()
    return [_synthetic_vector(t, EMBED_DIM) for t in texts]


def _synthetic_vector(seed: str, dim: int) -> list[float]:
    """Deterministic unit vector derived from a string seed."""
    raw = []
    for i in range(dim):
        h = int(hashlib.md5(f"{seed}:{i}".encode()).hexdigest(), 16)
        raw.append((h % 10000) / 10000.0 * 2 - 1)
    norm = math.sqrt(sum(x * x for x in raw)) or 1.0
    return [x / norm for x in raw]


def embed_image_stub(image_id: str, dim: int = EMBED_DIM) -> list[float]:
    """Simulate an image embedding (CLIP-like output) using a seeded synthetic vector.

    In a real pipeline, replace this with a CLIP / ViT model call.
    """
    return _synthetic_vector(f"img:{image_id}", dim)


# ── dataset ───────────────────────────────────────────────────────────────────

TRANSCRIPT_SEGMENTS: list[dict[str, Any]] = [
    {"frame": "f001", "text": "A researcher explains how neural networks learn from data through backpropagation."},
    {"frame": "f002", "text": "The demonstration shows training loss decreasing as the model converges."},
    {"frame": "f003", "text": "Scene transitions to a robot arm picking objects using a vision model."},
    {"frame": "f004", "text": "The narrator describes how attention mechanisms focus on relevant parts of an image."},
    {"frame": "f005", "text": "Interview with a professor discussing the limitations of current AI systems."},
    {"frame": "f006", "text": "An animation visualises gradient descent navigating a loss landscape."},
    {"frame": "f007", "text": "Code example shows how to define a simple neural network in Python."},
    {"frame": "f008", "text": "Closing segment: future of AI and ethical considerations in deployment."},
]

IMAGE_FRAMES: list[dict[str, Any]] = [
    {"frame": "f001", "description": "scatter_plot_of_training_loss"},
    {"frame": "f002", "description": "robot_arm_close_up"},
    {"frame": "f003", "description": "attention_heatmap_visualisation"},
    {"frame": "f004", "description": "gradient_descent_animation"},
    {"frame": "f005", "description": "interview_professor_headshot"},
    {"frame": "f006", "description": "python_code_screenshot"},
    {"frame": "f007", "description": "neural_network_diagram"},
    {"frame": "f008", "description": "future_city_skyline"},
]


# ── store helpers ─────────────────────────────────────────────────────────────

TABLE = "video_multimodal"


def setup_table(client: Client) -> None:
    try:
        client.delete_table(TABLE)
    except Exception:
        pass
    client.create_table(name=TABLE, dimension=EMBED_DIM, distance_metric="cosine")
    print(f"[store] table '{TABLE}' created (dim={EMBED_DIM})")


def ingest_transcripts(client: Client) -> None:
    texts = [s["text"] for s in TRANSCRIPT_SEGMENTS]
    vectors = embed_text(texts)

    records = [
        {
            "id": f"transcript_{seg['frame']}",
            "vector": vec,
            "payload": {
                "type": "text",
                "frame": seg["frame"],
                "text": seg["text"],
            },
        }
        for seg, vec in zip(TRANSCRIPT_SEGMENTS, vectors)
    ]
    client.insert_records(table_name=TABLE, records=records)
    print(f"[store] ingested {len(records)} transcript segments")


def ingest_images(client: Client) -> None:
    records = [
        {
            "id": f"image_{frame['frame']}",
            "vector": embed_image_stub(frame["description"]),
            "payload": {
                "type": "image",
                "frame": frame["frame"],
                "description": frame["description"],
            },
        }
        for frame in IMAGE_FRAMES
    ]
    client.insert_records(table_name=TABLE, records=records)
    print(f"[store] ingested {len(records)} image frames")


# ── search helpers ────────────────────────────────────────────────────────────

def search_text(client: Client, query: str, n: int = 3) -> list[dict]:
    """Search only text/transcript records."""
    q_vec = embed_text([query])[0]
    return client.search(
        table_name=TABLE,
        vector=q_vec,
        k=n,
        filter={"type": "text"},
    )


def search_images(client: Client, query_image_id: str, n: int = 3) -> list[dict]:
    """Find visually similar frames given an image id (simulation)."""
    q_vec = embed_image_stub(query_image_id)
    return client.search(
        table_name=TABLE,
        vector=q_vec,
        k=n,
        filter={"type": "image"},
    )


def search_cross_modal(client: Client, query: str, n: int = 5) -> list[dict]:
    """Search across both modalities (no filter)."""
    q_vec = embed_text([query])[0]
    return client.search(table_name=TABLE, vector=q_vec, k=n)


# ── display ───────────────────────────────────────────────────────────────────

def display(results: list[dict], title: str) -> None:
    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print(f"{'─' * 70}")
    if not results:
        print("  (no results)")
        return
    for i, r in enumerate(results, 1):
        score = r.get("score", 0.0)
        rid = r.get("id", "?")
        print(f"  {i}. [{score:.4f}]  {rid}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("  DXMT Multimodal Store")
    print("=" * 70)

    client = Client(host="localhost", port=9999)

    try:
        # 1. Setup and ingest
        setup_table(client)
        ingest_transcripts(client)
        ingest_images(client)
        print("\n[store] ingestion complete")

        # 2. Text-only search
        text_queries = [
            "how do neural networks learn?",
            "robot vision and object detection",
            "ethical concerns about artificial intelligence",
        ]
        for q in text_queries:
            results = search_text(client, q, n=2)
            display(results, f'Text search: "{q}"')

        # 3. Image similarity search (within image modality)
        results = search_images(client, "attention_heatmap_visualisation", n=3)
        display(results, "Image search: similar to attention heatmap frame")

        # 4. Cross-modal search — returns mix of text + image results
        results = search_cross_modal(client, "gradient descent optimisation animation", n=5)
        display(results, "Cross-modal search: gradient descent animation")

        print("\n[demo] key insight: same table, multiple modalities, filter by type")

    finally:
        try:
            client.delete_table(TABLE)
            print(f"\n[cleanup] deleted '{TABLE}'")
        except Exception:
            pass
        client.close()
        print("[done]")


if __name__ == "__main__":
    main()

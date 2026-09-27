"""Anomaly detection via kNN distance — DXMT implementation.

Records that lie far from all their neighbours are statistical outliers.
This is the vector-database analogue of the classical kNN-based anomaly score:

    anomaly_score(x) = mean distance to k nearest neighbours

High scores → unusual points that don't fit any cluster → potential anomalies.

Pattern: embedding-distance based anomaly detection.

Algorithm:
    1. Insert all records into DXMT.
    2. For each record, query for its top-(k+1) neighbours (the first hit is
       itself; skip it and take the next k).
    3. Compute mean distance over those k neighbours.
    4. Records ranked by descending mean distance are the most anomalous.

This script is self-contained: no external APIs required.
sentence-transformers is used if available; otherwise synthetic vectors.

Usage:
    python anomaly_detection.py
    python anomaly_detection.py --k 5 --top 10 --verbose
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

from dxmt import Client
from common_utils import embed, EMBED_DIM


# ── Synthetic dataset ─────────────────────────────────────────────────────────
#
# Three tight product clusters (shoes / knives / audio).
# HARD anomalies: clearly out-of-distribution (used to verify the algorithm works).
# BOUNDARY anomalies: superficially related to a cluster but don't belong —
#   these are the realistic case and the harder challenge.

CLUSTER_A = [
    ("prod_a1", "High-performance running shoes with carbon fibre plate and responsive foam."),
    ("prod_a2", "Lightweight trail running shoe with aggressive grip and waterproof membrane."),
    ("prod_a3", "Road running sneaker with cushioned midsole optimised for marathon pace."),
    ("prod_a4", "Sprint spike with removable pins for track and field competitions."),
    ("prod_a5", "Minimalist barefoot runner with wide toe box and zero-drop heel."),
    ("prod_a6", "Stability shoe with medial post for overpronation correction."),
    ("prod_a7", "Cross-training shoe suitable for gym, HIIT, and short-distance running."),
]

CLUSTER_B = [
    ("prod_b1", "Stainless steel chef's knife with full tang and ergonomic handle."),
    ("prod_b2", "Ceramic nakiri knife for precision vegetable slicing."),
    ("prod_b3", "Santoku knife with granton edge to prevent food sticking."),
    ("prod_b4", "Bread knife with offset serrated blade for clean slicing."),
    ("prod_b5", "Paring knife set with pakkawood handles for fine detail work."),
    ("prod_b6", "Honing rod to keep knife edges aligned between sharpenings."),
    ("prod_b7", "Whetstone set (400/1000/3000/8000 grit) for professional sharpening."),
]

CLUSTER_C = [
    ("prod_c1", "Noise-cancelling over-ear headphones with 30-hour battery and aptX HD."),
    ("prod_c2", "True wireless earbuds with active noise cancellation and transparency mode."),
    ("prod_c3", "Open-back planar magnetic headphones for studio reference monitoring."),
    ("prod_c4", "Gaming headset with 7.1 virtual surround and retractable microphone."),
    ("prod_c5", "Bone conduction headphones for safe listening during outdoor activities."),
    ("prod_c6", "USB-C DAC/amp dongle for powering high-impedance headphones from a phone."),
    ("prod_c7", "Balanced XLR cable replacement for audiophile headphone upgrades."),
]

# Hard anomalies — clearly out-of-distribution (easy case)
HARD_ANOMALIES = [
    ("hard_1", "Quantum entanglement protocol for instantaneous data transfer across galaxies."),
    ("hard_2", "404 error: product not found. Please check the catalogue reference number."),
]

# Boundary anomalies — superficially similar to a cluster but don't belong.
# These are the realistic test: will the algorithm flag them despite surface similarity?
#   border_a: uses footwear vocabulary but is a climbing harness (activity gear, not running shoe)
#   border_b: mentions blades but is a food processor (appliance, not kitchen knife)
#   border_c: mentions audio but is a DJ controller (instrument, not headphones/cables)
BOUNDARY_ANOMALIES = [
    ("border_a", "Full-body climbing harness with adjustable leg loops and belay loop for sport climbing."),
    ("border_b", "1200W food processor with stainless blades, dough hook, and 3-litre bowl."),
    ("border_c", "4-channel DJ controller with jog wheels, EQ knobs, and USB audio interface."),
]

ALL_RECORDS = CLUSTER_A + CLUSTER_B + CLUSTER_C + HARD_ANOMALIES + BOUNDARY_ANOMALIES
INJECTED_IDS = {rid for rid, _ in HARD_ANOMALIES + BOUNDARY_ANOMALIES}
HARD_IDS = {rid for rid, _ in HARD_ANOMALIES}
BOUNDARY_IDS = {rid for rid, _ in BOUNDARY_ANOMALIES}

TABLE = "anomaly_detection"


# ── Anomaly scoring ───────────────────────────────────────────────────────────

@dataclass
class AnomalyResult:
    record_id: str
    text: str
    anomaly_score: float
    neighbor_scores: list[float]
    label: str  # "normal" | "hard" | "boundary"

    def summary(self) -> str:
        tag = {"hard": "* HARD ANOMALY *", "boundary": "~ boundary ~   "}.get(self.label, "               ")
        short = self.text[:65] + "…" if len(self.text) > 65 else self.text
        return f"[score={self.anomaly_score:.4f}] {tag} {self.record_id}: {short}"


def score_anomalies(
    client: Client,
    records: list[tuple[str, str]],
    k: int = 5,
) -> list[AnomalyResult]:
    """For each record compute mean cosine distance to its k nearest neighbours."""
    text_by_id = {rid: text for rid, text in records}

    texts = [text for _, text in records]
    vectors = embed(texts)
    vec_by_id = {rid: vec for (rid, _), vec in zip(records, vectors)}

    results: list[AnomalyResult] = []
    for rec_id, _ in records:
        q_vec = vec_by_id[rec_id]
        # k+1 because the top hit is the record itself
        hits = client.search(table_name=TABLE, vector=q_vec, k=k + 1)

        neighbor_dists = []
        for hit in hits:
            if hit.get("id") == rec_id:
                continue
            # DXMT cosine score is in [0,1]; distance = 1 - similarity
            neighbor_dists.append(1.0 - hit.get("score", 0.0))
            if len(neighbor_dists) >= k:
                break

        mean_dist = sum(neighbor_dists) / len(neighbor_dists) if neighbor_dists else 0.0

        if rec_id in HARD_IDS:
            label = "hard"
        elif rec_id in BOUNDARY_IDS:
            label = "boundary"
        else:
            label = "normal"

        results.append(AnomalyResult(
            record_id=rec_id,
            text=text_by_id[rec_id],
            anomaly_score=mean_dist,
            neighbor_scores=neighbor_dists,
            label=label,
        ))

    results.sort(key=lambda r: r.anomaly_score, reverse=True)
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="DXMT kNN anomaly detection demo")
    parser.add_argument("--k", type=int, default=5, help="Neighbours to average (default 5)")
    parser.add_argument("--top", type=int, default=8, help="Top anomalies to display (default 8)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show neighbour distances")
    args = parser.parse_args()

    from common_utils import _HAS_ST
    embed_mode = "sentence-transformers" if _HAS_ST else "bag-of-words"
    n_normal = len(CLUSTER_A) + len(CLUSTER_B) + len(CLUSTER_C)
    print("=" * 70)
    print("  DXMT kNN Anomaly Detection Demo")
    print(f"  Embedding: {embed_mode}  (dim={EMBED_DIM})")
    print(f"  Records:   {len(ALL_RECORDS)}  "
          f"({n_normal} normal | {len(HARD_ANOMALIES)} hard anomalies | "
          f"{len(BOUNDARY_ANOMALIES)} boundary anomalies)")
    print(f"  k (neighbours): {args.k}")
    print("=" * 70)

    client = Client(host="localhost", port=9999)
    try:
        # Setup
        try:
            client.delete_table(TABLE)
        except Exception:
            pass
        client.create_table(name=TABLE, dimension=EMBED_DIM, distance_metric="cosine")

        # Index
        texts = [text for _, text in ALL_RECORDS]
        vectors = embed(texts)
        insert_records = [
            {
                "id": rid,
                "vector": vec,
                "payload": {"text": text},
            }
            for (rid, text), vec in zip(ALL_RECORDS, vectors)
        ]
        client.insert_records(table_name=TABLE, records=insert_records)
        print(f"\n[index] {len(ALL_RECORDS)} records indexed")

        # Score
        print(f"\n[scoring] computing anomaly scores (k={args.k}) …")
        results = score_anomalies(client, ALL_RECORDS, k=args.k)

        # Report
        print(f"\n{'─' * 70}")
        print(f"  Top-{args.top} most anomalous records")
        print(f"{'─' * 70}")
        for r in results[: args.top]:
            print(f"  {r.summary()}")
            if args.verbose:
                dists = ", ".join(f"{d:.4f}" for d in r.neighbor_scores)
                print(f"           neighbours dist: [{dists}]")

        # Rank report for all injected anomalies
        print(f"\n{'─' * 70}")
        print("  Injected anomaly ranks (hard + boundary):")
        for rank, r in enumerate(results, start=1):
            if r.label != "normal":
                print(f"    rank {rank:3d} / {len(results)}  score={r.anomaly_score:.4f}  [{r.label:8s}] {r.record_id}")

        # p95 auto-threshold using only confirmed normal records
        normal_scores = [r.anomaly_score for r in results if r.label == "normal"]
        if normal_scores:
            sorted_ns = sorted(normal_scores)
            p95 = sorted_ns[int(0.95 * len(sorted_ns))]
            flagged = [r for r in results if r.anomaly_score > p95]
            hard_caught = sum(1 for r in flagged if r.label == "hard")
            boundary_caught = sum(1 for r in flagged if r.label == "boundary")
            false_pos = sum(1 for r in flagged if r.label == "normal")
            print(f"\n  Auto-threshold (p95 of normals = {p95:.4f}): {len(flagged)} flagged")
            print(f"    hard caught:     {hard_caught} / {len(HARD_ANOMALIES)}")
            print(f"    boundary caught: {boundary_caught} / {len(BOUNDARY_ANOMALIES)}  ← harder case")
            print(f"    false positives: {false_pos}")
            for r in flagged:
                print(f"    [{r.label:8s}] {r.record_id}  score={r.anomaly_score:.4f}")
            if not _HAS_ST:
                # These numbers measure the embedding, not the database: with the
                # bag-of-words fallback an anomaly is only odd if its *words* are, and the
                # ones here are odd in meaning. Say so rather than let a reader read a low
                # catch rate as DXMT missing them.
                print("\n  NOTE: running on the bag-of-words fallback, which scores word "
                      "overlap and not meaning -- these catch rates say little. "
                      "pip install sentence-transformers for numbers worth reading.")


    finally:
        try:
            client.delete_table(TABLE)
        except Exception:
            pass
        client.close()
        print("\n[done]")


if __name__ == "__main__":
    main()

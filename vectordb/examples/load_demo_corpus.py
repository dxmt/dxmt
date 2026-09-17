#!/usr/bin/env python3
"""Fill a table with real embeddings, so the console has something true to show.

    python3 examples/load_demo_corpus.py                      # 20,000 documents
    python3 examples/load_demo_corpus.py --rows 200000        # a table big enough to be slow
    python3 examples/load_demo_corpus.py --url http://host:9999

Why this exists: a table of random vectors tells you nothing, and the console is honest enough
to say so. Point the visualize page at one and it reports two per cent of the variance and one
per cent of neighbourhoods kept, which is correct and useless. Real embeddings have structure,
and only with structure do the projection and the recall measurement show what they are for.

Two properties make this a better demo than a downloaded corpus would be:

* **It needs nothing but the model.** The text is generated here from a template bank, so there
  is no dataset to fetch, no licence to carry, and the same command produces the same table on
  any machine. `all-MiniLM-L6-v2` is 90 MB and caches after the first run.
* **The answer is known.** Every document carries the topic it was built from. The projection
  can therefore be *checked* rather than admired: colour by `topic` and see whether the plot
  recovers groups you already know are there, and whether the neighbourhood figure agrees.

The vectors are genuine — a real sentence transformer over real English sentences — so the
index behaves as it would on production data. It is the *text* that is synthetic, which is the
right thing to fake: nobody is reading it.
"""

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

# Topics far enough apart that a sentence transformer separates them, and close enough in pairs
# (the two astronomy-adjacent ones, the two legal ones) that the nearest-neighbour question is
# not trivial. A corpus where every group is equidistant makes any index look perfect.
TOPICS = {
    "astronomy": {
        "subject": ["the red dwarf", "a distant quasar", "the outer moon", "this binary pair",
                    "the dust lane", "a rogue planet"],
        "verb": ["drifts past", "eclipses", "outshines", "perturbs", "is occluded by", "orbits"],
        "object": ["the galactic core", "its parent star", "a globular cluster",
                   "the accretion disc", "an ancient nebula", "the ecliptic plane"],
        "tail": ["once every nine months", "in near-infrared", "at perihelion",
                 "during the observing run", "far outside the habitable zone", "before dawn"],
        "note": ["The survey has seen it twice.", "Nobody has confirmed it.",
                 "The photometry is noisy.", "It was missed in the first pass.",
                 "The period is still uncertain.", "Two teams disagree about it."],
    },
    "spaceflight": {
        "subject": ["the second stage", "the docking adapter", "a reentry capsule",
                    "the ion thruster", "this launch window", "the landing burn"],
        "verb": ["was cut short by", "depends on", "was qualified against", "loiters through",
                 "aborts on", "compensates for"],
        "object": ["a propellant margin", "the abort corridor", "atmospheric drag",
                   "the flight computer", "a leaking valve", "the deorbit sequence"],
        "tail": ["with four seconds of reserve", "on the third attempt", "under manual control",
                 "after a scrub", "within the certified envelope", "on the backup string"],
        "note": ["The review board asked about it.", "It held through the vibration test.",
                 "The margin was thinner than planned.", "Telemetry dropped for a moment.",
                 "It was waived for this flight.", "The sim never reproduced it."],
    },
    "cooking": {
        "subject": ["the braise", "a proofed dough", "this reduction", "the emulsion",
                    "a dry brine", "the pan sauce"],
        "verb": ["breaks if you", "wants", "collapses without", "tastes flat until you",
                 "thickens once", "needs"],
        "object": ["more acid", "a colder oven", "twelve hours of rest", "a heavier pan",
                   "the fat skimmed off", "salt earlier than you think"],
        "tail": ["and it is worth the wait", "which nobody tells you", "before the herbs go in",
                 "no matter what the recipe says", "if the kitchen is warm", "on the second day"],
        "note": ["Taste it before you salt again.", "The pan matters more than the recipe.",
                 "Leftovers are better.", "Do not skip the resting time.",
                 "It freezes badly.", "Halve it and it still works."],
    },
    "gardening": {
        "subject": ["the espaliered pear", "a bed of alliums", "this rootstock",
                    "the compost heap", "a shaded border", "the seed tray"],
        "verb": ["sulks in", "thrives on", "needs staking against", "runs to seed after",
                 "resents", "recovers slowly from"],
        "object": ["heavy clay", "a late frost", "the westerly wind", "too much nitrogen",
                   "a second transplant", "three weeks of drought"],
        "tail": ["by the second season", "on this aspect", "unless you mulch",
                 "even in a mild year", "once it is established", "in a wet spring"],
        "note": ["Move it and start again.", "The label is optimistic.",
                 "Cut it back hard in autumn.", "It seeds itself everywhere.",
                 "Slugs find it first.", "Give it more room than you think."],
    },
    "contract law": {
        "subject": ["the indemnity clause", "a liquidated damages provision",
                    "this assignment right", "the termination notice", "an entire agreement clause",
                    "the limitation of liability"],
        "verb": ["survives", "is read down by", "does not displace", "was held to exclude",
                 "operates independently of", "is void against"],
        "object": ["the insolvency of the counterparty", "an implied term",
                   "the parties' prior dealings", "statutory unfairness rules",
                   "a repudiatory breach", "the notice period"],
        "tail": ["absent clear words", "on the ordinary meaning", "as a matter of construction",
                 "in the commercial context", "however the schedule is drafted", "on these facts"],
        "note": ["The point was not argued below.", "Authority is thin either way.",
                 "The drafting is unusual.", "Both parties were advised.",
                 "It was conceded at the hearing.", "The schedule contradicts the body."],
    },
    "criminal procedure": {
        "subject": ["the search warrant", "this identification parade", "a custodial interview",
                    "the disclosure obligation", "an arrest without warrant",
                    "the caution given"],
        "verb": ["was defective because", "must be supported by", "does not extend to",
                 "renders inadmissible", "is vitiated where", "requires"],
        "object": ["reasonable suspicion at the time", "material held by a third party",
                   "anything said before the caution", "the premises actually specified",
                   "contemporaneous notes", "an appropriate adult being present"],
        "tail": ["on the face of the record", "regardless of what was found",
                 "unless the breach is trivial", "at the point of detention",
                 "and the trial judge must direct on it", "on the officer's own account"],
        "note": ["The custody record is incomplete.", "No note was made at the time.",
                 "The defence did not object.", "The point is taken for the first time on appeal.",
                 "The recording stops midway.", "Two officers recall it differently."],
    },
    "distributed systems": {
        "subject": ["the follower replica", "a quorum read", "this partition strategy",
                    "the leader lease", "an anti-entropy sweep", "the write path"],
        "verb": ["stalls when", "is only linearizable if", "amplifies", "diverges under",
                 "cannot make progress during", "trades latency for"],
        "object": ["a clock skew of a few seconds", "one slow disk", "the network partition",
                   "durability on every write", "a hot key", "an unbounded queue"],
        "tail": ["and the symptom is tail latency", "which no dashboard shows",
                 "long before it is noticed", "under exactly the load you tested",
                 "until the node is restarted", "only in the third region"],
        "note": ["It reproduced once in a week.", "The retry made it worse.",
                 "Nothing in the logs said so.", "It went away after a deploy.",
                 "The metric was averaged away.", "Two services blamed each other."],
    },
    "cycling": {
        "subject": ["the breakaway", "a crosswind section", "this gear ratio", "the descent",
                    "a tubeless setup", "the final climb"],
        "verb": ["is decided by", "punishes", "rewards", "falls apart on", "suits",
                 "rarely survives"],
        "object": ["anyone who started too hard", "a rider with no team",
                   "twelve per cent gradients", "wet cobbles", "patience in the first hour",
                   "the wrong tyre pressure"],
        "tail": ["with forty kilometres to go", "in the third week", "on tired legs",
                 "whatever the numbers say", "once the group splits", "into a headwind"],
        "note": ["The gap never came down.", "Nobody wanted to work.",
                 "It was decided in the first hour.", "The radio was useless.",
                 "Two riders sat up.", "The wind turned at the worst moment."],
    },
}


SLOTS = ("subject", "verb", "object", "tail", "note")


def combinations_per_topic():
    """How many distinct sentences a topic can make. Five slots of six is 7,776."""
    return min(
        # Every topic has the same slot shape; take the smallest in case one is edited.
        [len(slots[name]) for name in SLOTS]
        and __import__("math").prod([len(TOPICS[t][n]) for n in SLOTS])
        for t, slots in TOPICS.items()
    )


def documents(count, seed=11):
    """Deterministic, and **distinct**: no two documents carry the same sentence.

    The first draft drew each slot at random, which at 1,296 combinations a topic and 20,000
    documents produced duplicates by the thousand — and duplicates are the one thing this
    corpus must not have. Identical vectors tie at similarity 1.0, every search mode finds them,
    and the recall page reports a perfect index measuring nothing. (The page's own test covers
    the same hazard from the other side: a duplicate neighbour must not be dropped from a
    comparison.)

    So the slot combinations are *enumerated* rather than sampled, in an order scrambled by
    multiplying the index by a number coprime with the space size. That is a permutation, so
    the first N are N distinct sentences, for any N up to the space itself — 7,776 a topic,
    62,208 across eight.
    """
    names = sorted(TOPICS)
    space = combinations_per_topic()
    # Coprime with 7,776 = 2^5 x 3^5, so multiplication modulo the space is a bijection.
    stride = 6_121 + (seed % 2)
    seen_per_topic = {name: 0 for name in names}

    for i in range(count):
        topic = names[i % len(names)]
        nth = seen_per_topic[topic]
        seen_per_topic[topic] += 1
        if nth >= space:
            raise SystemExit(
                f"{count:,} documents needs more than the {space * len(names):,} distinct "
                f"sentences this corpus can make. Ask for fewer rows, or add options to the "
                f"slots in TOPICS."
            )

        scrambled = (nth * stride + seed) % space
        slots = TOPICS[topic]
        parts = []
        for name in SLOTS[:-1]:
            options = slots[name]
            parts.append(options[scrambled % len(options)])
            scrambled //= len(options)
        note = slots["note"][scrambled % len(slots["note"])]

        sentence = " ".join(parts)
        yield {
            "id": f"doc-{i:06d}",
            "text": sentence[0].upper() + sentence[1:] + ". " + note,
            "topic": topic,
            "index": i,
        }


def call(url, method, path, body=None, timeout=600):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        url.rstrip("/") + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read() or b"null")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()[:400]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:9999")
    parser.add_argument("--table", default="articles")
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--batch", type=int, default=2_000, help="records per insert request")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--keep", action="store_true", help="add to the table instead of replacing it")
    args = parser.parse_args()

    api = args.url.rstrip("/") + "/api/v1"
    status, _ = call(api, "GET", "/health")
    if status != 200:
        sys.exit(f"no server at {args.url} (health said {status})")

    # Imported here so `--help` works without torch on the path.
    from sentence_transformers import SentenceTransformer

    print(f"loading {args.model}…", flush=True)
    model = SentenceTransformer(args.model)
    dimension = model.get_sentence_embedding_dimension()

    if not args.keep:
        call(api, "DELETE", f"/tables/{urllib.parse.quote(args.table)}")
    status, body = call(api, "POST", "/tables", {
        "name": args.table,
        "dimension": dimension,
        "distance_metric": "cosine",
        "description": f"{args.rows} sentences over {len(TOPICS)} topics, embedded with {args.model}",
        # So the full-text and hybrid routes have something to work with, when the server was
        # built with `text-bm25`. The engine indexes `content` when no text_fields are declared.
        "enable_full_text_search": True,
    })
    if status != 200 and "ALREADY_EXISTS" not in str(body):
        sys.exit(f"could not create {args.table}: {body}")

    print(f"embedding {args.rows:,} documents at {dimension} dimensions…", flush=True)
    started = time.time()
    inserted = 0
    batch = []

    def flush():
        nonlocal inserted, batch
        if not batch:
            return
        texts = [doc["text"] for doc in batch]
        # Normalised, because the table is cosine: unit vectors make the score a plain dot
        # product and make `score == 1.0` mean what the console says it means.
        vectors = model.encode(texts, batch_size=256, normalize_embeddings=True,
                               show_progress_bar=False)
        records = [{
            "id": doc["id"],
            "vector": [float(x) for x in vector],
            "payload": {"topic": doc["topic"], "index": doc["index"], "content": doc["text"]},
        } for doc, vector in zip(batch, vectors)]
        status, body = call(api, "POST", f"/tables/{urllib.parse.quote(args.table)}/records",
                            {"records": records})
        if status != 200:
            sys.exit(f"insert failed at {inserted:,}: {status} {body}")
        inserted += len(records)
        rate = inserted / max(time.time() - started, 1e-9)
        print(f"\r  {inserted:,} / {args.rows:,}  ({rate:,.0f}/s)", end="", flush=True)
        batch = []

    for doc in documents(args.rows):
        batch.append(doc)
        if len(batch) >= args.batch:
            flush()
    flush()
    print()

    elapsed = time.time() - started
    print(f"done: {inserted:,} records in {elapsed:.1f}s ({inserted / elapsed:,.0f}/s)")

    # The numbers the console is about to show, printed here so the two can be compared.
    status, detail = call(api, "GET", f"/tables/{urllib.parse.quote(args.table)}")
    if status == 200:
        print(f"table '{args.table}': {detail['data']['record_count']:,} rows, "
              f"{dimension}d, index {detail['data']['table']['index_type']}")

    probe = next(documents(1))
    vector = [float(x) for x in model.encode([probe["text"]], normalize_embeddings=True)[0]]
    _, exact = call(api, "POST", f"/tables/{urllib.parse.quote(args.table)}/search",
                    {"vector": vector, "top_k": 6, "limit": 6, "search_mode": "exact"})
    if isinstance(exact, dict) and exact.get("data"):
        print(f"\nnearest to: {probe['text']!r}")
        for hit in exact["data"]["hits"]:
            payload = hit.get("payload") or {}
            print(f"  {hit['score']:.4f}  [{payload.get('topic', '?'):<20}] {payload.get('content', '')[:74]}")

    print(f"\nnow open  {args.url}/dashboard/tables/{urllib.parse.quote(args.table)}/visualize")
    print("colour by `topic` and see whether the projection recovers groups you know are there.")


if __name__ == "__main__":
    main()

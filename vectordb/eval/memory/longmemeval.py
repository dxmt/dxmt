#!/usr/bin/env python3
"""Recall on LongMemEval-S: does `recall` find the sessions a question's answer is in?

LongMemEval (Wu et al., 2024; `longmemeval_s_cleaned.json` from the xiaowu0162/longmemeval-cleaned
dataset) gives each of its 500 questions its own history of about 50 chat sessions, ~550 turns,
with the sessions that hold the answer named (`answer_session_ids`) and the turns marked
(`has_answer`). Each question's turns are written into the memory table tagged with the
question, the question is asked through `POST /memories/recall` restricted to that tag, and it
scores whether the sessions of the turns it got back cover every answer session (session
recall, all) and how many of the marked turns came back (turn recall).

The turns go in through the table's batch insert rather than one `POST /memories` each -- the
same payload `remember` writes (`text`, `kind`, `tags`, `at`), embedded by the same model, in a
fraction of the calls -- because one question's history is 550 memories and the whole set is
275,000. `--per-type` samples that many questions of each of the six types.

    python3 eval/memory/longmemeval.py --data longmemeval_s_cleaned.json --per-type 10

A turn's text is cut at 4,000 characters (some are pasted documents); `has_answer` turns are
short. Abstention questions (`_abs`) have no answer session and are left out.
"""

import argparse
import collections
import concurrent.futures
import json
import random
import sys
import time

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from locomo import call  # noqa: E402

KS = (5, 10)
CUT = 4000


def memories(question):
    """(id, payload) for every turn of one question's history."""
    tag = question["question_id"]
    for sid, when, session in zip(question["haystack_session_ids"], question["haystack_dates"],
                                  question["haystack_sessions"]):
        for n, turn in enumerate(session):
            text = f"[{when}] {turn['role']}: {turn['content']}"[:CUT]
            yield f"{tag}:{sid}:{n}", text, bool(turn.get("has_answer"))


def ingest(url, table, question, batch=64):
    rows = list(memories(question))
    for i in range(0, len(rows), batch):
        records = [{"id": mid, "text": text,
                    "payload": {"kind": "event", "tags": [question["question_id"]], "at": 0}}
                   for mid, text, _ in rows[i:i + batch]]
        call(url, "POST", f"/api/v1/tables/{table}/records", {"records": records})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:9999")
    parser.add_argument("--table", default="memory")
    parser.add_argument("--per-type", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    data = [q for q in json.load(open(args.data)) if not q["question_id"].endswith("_abs")]
    by_type = collections.defaultdict(list)
    for q in data:
        by_type[q["question_type"]].append(q)
    rng = random.Random(args.seed)
    sample = [q for t in sorted(by_type) for q in rng.sample(by_type[t], min(args.per_type, len(by_type[t])))]

    # Make the memory table (first use) before writing into it directly.
    call(args.url, "GET", "/api/v1/memories?limit=1")

    def run(question):
        rows = ingest(args.url, args.table, question)
        marked = {mid for mid, _, has in rows if has}
        hits = call(args.url, "POST", "/api/v1/memories/recall",
                    {"query": question["question"], "tags": [question["question_id"]],
                     "limit": max(KS)})["memories"]
        ids = [h["id"] for h in hits]
        out = {"type": question["question_type"], "turns": len(rows)}
        wanted = set(question["answer_session_ids"])
        for k in KS:
            sessions = {i.split(":")[1] for i in ids[:k]}
            out[f"session_all@{k}"] = float(wanted <= sessions)
            out[f"session_any@{k}"] = float(bool(wanted & sessions))
            if marked:
                out[f"turn@{k}"] = len(marked & set(ids[:k])) / len(marked)
        return out

    started = time.time()
    results = []
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        for i, out in enumerate(pool.map(run, sample), 1):
            results.append(out)
            print(f"  {i}/{len(sample)} questions, {time.time() - started:.0f} s",
                  file=sys.stderr, flush=True)

    metrics = [f"session_all@{k}" for k in KS] + [f"session_any@{k}" for k in KS] + [f"turn@{k}" for k in KS]
    print(f"\nLongMemEval-S, {len(results)} questions ({args.per_type} per type, seed {args.seed}), "
          f"{sum(r['turns'] for r in results)} turns\n")
    print(f"{'type':28} {'n':>3} " + " ".join(f"{m:>15}" for m in metrics))
    for t in sorted(by_type) + ["all"]:
        rs = [r for r in results if t in ("all", r["type"])]
        cells = []
        for m in metrics:
            vs = [r[m] for r in rs if m in r]
            cells.append(f"{sum(vs) / len(vs):15.3f}" if vs else f"{'-':>15}")
        print(f"{t:28} {len(rs):3} " + " ".join(cells))


if __name__ == "__main__":
    main()

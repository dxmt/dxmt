#!/usr/bin/env python3
"""Which hybrid fusion, at which weight, on real questions -- without asking the server twice.

`/search/hybrid` fetches `3 x limit` candidates from the dense index and from BM25 and fuses
them. This fetches the same two lists once per question (`/search` and `/search/text`, with
their scores), caches them, and replays every fusion offline, so a sweep over methods and
weights costs nothing after the first pass. `--check` first compares the offline fusion with
what the server returns for the shipped methods, so the replay is known to be the server's.

    python3 eval/fusion_sweep.py --table cmrc --dataset cmrc --check
    python3 eval/fusion_sweep.py --table dureader --dataset dureader

Datasets and tables are the ones `eval/ragas/cmrc_eval.py` loads.
"""

import argparse
import json
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ragas"))
import cmrc_eval  # noqa: E402

LIMIT = 10
FETCH = 3 * LIMIT  # IntelligentIndexedStorage::hybrid_search, no filter


def candidates(url, table, question):
    dense = cmrc_eval.call(url, "POST", f"/api/v1/tables/{table}/search", {"text": question, "top_k": FETCH})
    text = cmrc_eval.call(url, "POST", f"/api/v1/tables/{table}/search/text", {"query": question, "limit": FETCH})
    return (
        [(h["id"], h["score"]) for h in dense["data"]["hits"]],
        [(h["id"], h["score"]) for h in text["data"]["hits"]],
    )


# The server's fusions, as in intelligent_indexed_storage.rs, plus weighted DBSF.
def dbsf_norm(xs):
    if not xs:
        return {}
    s = [v for _, v in xs]
    mean = sum(s) / len(s)
    std = max((sum((v - mean) ** 2 for v in s) / len(s)) ** 0.5, 1e-8)
    lo, hi = mean - 3 * std, mean + 3 * std
    return {i: (min(max(v, lo), hi) - lo) / max(hi - lo, 1e-8) for i, v in xs}


def minmax(xs):
    if not xs:
        return {}
    s = [v for _, v in xs]
    lo, hi = min(s), max(s)
    return {i: ((v - lo) / (hi - lo) if hi > lo else 1.0) for i, v in xs}


def fuse(method, w, dense, text):
    out = {}
    if method == "dbsf":
        for i, v in dbsf_norm(dense).items():
            out[i] = out.get(i, 0) + w * v
        for i, v in dbsf_norm(text).items():
            out[i] = out.get(i, 0) + (1 - w) * v
    elif method == "weighted_sum":
        for i, v in minmax(dense).items():
            out[i] = out.get(i, 0) + w * v
        for i, v in minmax(text).items():
            out[i] = out.get(i, 0) + (1 - w) * v
    elif method == "rrf":
        for r, (i, _) in enumerate(dense):
            out[i] = out.get(i, 0) + w / (60 + r + 1)
        for r, (i, _) in enumerate(text):
            out[i] = out.get(i, 0) + (1 - w) / (60 + r + 1)
    return [i for i, _ in sorted(out.items(), key=lambda kv: (-kv[1], kv[0]))[:LIMIT]]


def score(ranked_lists, questions):
    hit1 = mrr = recall = 0.0
    for ids, q in zip(ranked_lists, questions):
        gold = set(q["gold"])
        ranks = [k + 1 for k, x in enumerate(ids) if x in gold]
        if ranks:
            hit1 += ranks[0] == 1
            mrr += 1 / ranks[0]
        recall += len(ranks) / min(len(gold), LIMIT)
    n = len(questions)
    return hit1 / n, mrr / n, recall / n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:19997")
    ap.add_argument("--table", required=True)
    ap.add_argument("--dataset", choices=["cmrc", "dureader", "beir"], required=True)
    ap.add_argument("--data", default=cmrc_eval.DATA)
    ap.add_argument("--duretrieval", default=cmrc_eval.DURETRIEVAL)
    ap.add_argument("--beir", help="beir: a directory with corpus.jsonl, queries.jsonl, qrels/test.tsv")
    ap.add_argument("--limit", type=int, default=1000, help="questions, a seeded random subset")
    ap.add_argument("--cache", default=os.path.expanduser("~/.cache/dxmt-eval/fusion"))
    ap.add_argument("--check", action="store_true", help="compare the replay with the server")
    args = ap.parse_args()

    if args.dataset == "beir":
        _, questions = cmrc_eval.load_beir(args.beir)
    elif args.dataset == "dureader":
        _, questions = cmrc_eval.load_duretrieval(args.duretrieval)
    else:
        _, questions = cmrc_eval.load_dataset(args.data)
    random.Random(7).shuffle(questions)
    questions = questions[: args.limit]

    os.makedirs(args.cache, exist_ok=True)
    path = os.path.join(args.cache, f"{args.table}.json")
    cache = json.load(open(path)) if os.path.exists(path) else {}
    for n, q in enumerate(questions):
        if q["id"] not in cache:
            cache[q["id"]] = candidates(args.url, args.table, q["question"])
            if n % 100 == 0:
                json.dump(cache, open(path, "w"))
                print(f"\r  fetched {n}/{len(questions)}", end="", file=sys.stderr, flush=True)
    json.dump(cache, open(path, "w"))
    print(file=sys.stderr)

    if args.check:
        agree = {}
        for method, server in [("dbsf", "dbsf"), ("weighted_sum", "weighted_sum"), ("rrf", "reciprocal_rank_fusion")]:
            same = 0
            for q in questions[:100]:
                dense, text = cache[q["id"]]
                r = cmrc_eval.call(
                    args.url, "POST", f"/api/v1/tables/{args.table}/search/hybrid",
                    {"text": q["question"], "limit": LIMIT, "fusion_method": server},
                )
                got = [h["id"] for h in r["data"]["hits"]]
                w = 0.5 if method == "dbsf" else 0.7  # dbsf: equal weights; others: the default
                same += got == fuse(method, w, dense, text)
            agree[method] = same
        print(f"replay matches the server on {agree} of 100 questions")

    print(f"{args.table}: {len(questions)} questions, top {LIMIT}, {FETCH} candidates a side\n")
    print("| fusion | vector weight | Hit@1 | MRR@10 | Recall@10 |")
    print("| --- | --- | --- | --- | --- |")
    rows = []
    only = lambda side: [[i for i, _ in cache[q["id"]][side][:LIMIT]] for q in questions]
    for name, lists in [("vector only", only(0)), ("bm25 only", only(1))]:
        rows.append((name, "-", *score(lists, questions)))
    for method in ["dbsf", "weighted_sum", "rrf"]:
        for w in [0.5, 0.6, 0.7, 0.8, 0.9]:
            lists = [fuse(method, w, *cache[q["id"]]) for q in questions]
            rows.append((method, w, *score(lists, questions)))
    for name, w, h, m, r in rows:
        print(f"| {name} | {w} | {h:.3f} | {m:.3f} | {r:.3f} |")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""What a cross-encoder adds to recall on LoCoMo, measured without restarting the server.

For each question it fetches the memory table's dense and keyword candidates (as
`locomo.py` loaded them), fuses them the way `recall` does (min-max, vector weight 0.6), and
has a reranker reorder the first `--pool`; it scores both orders by the share of evidence turns
in the top k. The reranker is text-embeddings-inference's `/rerank` (a server with
`DXMT_RERANK_FORMAT=tei` calls the same route), e.g.

    docker run -p 18081:80 ghcr.io/huggingface/text-embeddings-inference:cpu-1.8 \\
        --model-id BAAI/bge-reranker-v2-m3
    python3 eval/memory/locomo_rerank.py --data locomo10.json --url http://127.0.0.1:9999 \\
        --reranker http://127.0.0.1:18081 --questions 400

`--questions` takes the first n questions with evidence, in file order (conversation by
conversation), not a random sample.
"""

import argparse
import collections
import concurrent.futures
import json
import sys
import time
import urllib.request

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from locomo import CATEGORIES, call, evidence_ids, tag_filter, turns  # noqa: E402

KS = (5, 10, 25)


def minmax(hits):
    if not hits:
        return {}
    xs = [s for _, s in hits]
    lo, hi = min(xs), max(xs)
    return {i: (s - lo) / (hi - lo) if hi > lo else 1.0 for i, s in hits}


def fused(dense, keyword, weight=0.6):
    a, b = minmax(dense), minmax(keyword)
    scores = {i: weight * a.get(i, 0) + (1 - weight) * b.get(i, 0) for i in set(a) | set(b)}
    return [i for i, _ in sorted(scores.items(), key=lambda x: -x[1])]


def rerank(url, query, ids, text):
    body = json.dumps({"query": query, "texts": [text[i] for i in ids], "truncate": True}).encode()
    request = urllib.request.Request(url + "/rerank", data=body,
                                     headers={"content-type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        out = json.load(response)
    return [ids[o["index"]] for o in sorted(out, key=lambda o: -o["score"])]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:9999")
    parser.add_argument("--table", default="memory")
    parser.add_argument("--reranker", required=True, help="text-embeddings-inference base URL")
    parser.add_argument("--pool", type=int, default=30)
    parser.add_argument("--questions", type=int)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    data = json.load(open(args.data))
    text = {i: t for sample in data for i, t in turns(sample)}
    questions = [(s, q) for s in data for q in s["qa"]
                 if q["category"] in CATEGORIES and evidence_ids(s, q)][: args.questions]
    fetch = 2 * args.pool

    def run(item):
        sample, qa = item
        where = tag_filter(sample["sample_id"])
        dense = call(args.url, "POST", f"/api/v1/tables/{args.table}/search",
                     {"text": qa["question"], "top_k": fetch, "filter": where})["hits"]
        keyword = call(args.url, "POST", f"/api/v1/tables/{args.table}/search/text",
                       {"query": qa["question"], "limit": fetch, "filter": where})["hits"]
        base = fused([(h["id"], h["score"]) for h in dense], [(h["id"], h["score"]) for h in keyword])
        started = time.perf_counter()
        reordered = rerank(args.reranker, qa["question"], base[: args.pool], text)
        return qa["category"], evidence_ids(sample, qa), base, reordered, time.perf_counter() - started

    scores = collections.defaultdict(lambda: collections.defaultdict(list))
    seconds = []
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        for category, wanted, base, reordered, took in pool.map(run, questions):
            seconds.append(took)
            for name, order in (("hybrid", base), ("reranked", reordered)):
                for k in KS:
                    found = len(wanted & set(order[:k])) / len(wanted)
                    scores[(name, k)][category].append(found)
                    scores[(name, k)]["all"].append(found)

    print(f"\nLoCoMo, first {len(questions)} questions with evidence; reranking the top {args.pool}\n")
    print(f"{'order':9} {'k':>3} " + " ".join(f"{CATEGORIES.get(c, c):>12}" for c in list(CATEGORIES) + ["all"]))
    for (name, k) in sorted(scores):
        cells = scores[(name, k)]
        print(f"{name:9} {k:3} " + " ".join(
            f"{sum(cells[c]) / len(cells[c]):12.3f}" if cells[c] else f"{'-':>12}"
            for c in list(CATEGORIES) + ["all"]))
    seconds.sort()
    print(f"\nrerank p50 {seconds[len(seconds) // 2] * 1000:.0f} ms, "
          f"p95 {seconds[int(len(seconds) * .95)] * 1000:.0f} ms")


if __name__ == "__main__":
    main()

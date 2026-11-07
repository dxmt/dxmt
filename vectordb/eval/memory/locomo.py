#!/usr/bin/env python3
"""Recall on LoCoMo: does `recall` bring back the turns a question's answer is in?

LoCoMo (snap-research/locomo, `data/locomo10.json`) is ten long multi-session conversations
with about 2,000 questions, each labelled with the dialogue turns that hold its answer
(`evidence`). Every turn is kept through `POST /memories` -- one memory, its speaker and the
session's date in the text, tagged with its conversation -- and every question is asked through
`POST /memories/recall` restricted to that conversation's tag. A question scores the share of its
evidence turns in the top k. No language model judges anything, so the number is exact and
repeatable; it measures retrieval, which is the database's half of an agent's memory, and not
answer quality, which is the model's.

Alongside `recall` (hybrid), the same questions go to the memory table's dense search and its
keyword search, so the number says what fusion is worth here.

    python3 eval/memory/locomo.py --data locomo10.json --url http://127.0.0.1:9999

The server needs an embedding model (`DXMT_EMBEDDING_MODEL`, or an ollama it finds), and should
start empty: the run writes into the memory table. Category 5 (adversarial: the answer is not in
the conversation) has no evidence to find and is left out, as in the papers that report LoCoMo.
"""

import argparse
import collections
import concurrent.futures
import json
import os
import sys
import time
import urllib.request

CATEGORIES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}
KS = (1, 5, 10)


def call(url, method, path, body=None):
    headers = {"content-type": "application/json"}
    if os.environ.get("DXMT_API_KEY"):
        headers["api-key"] = os.environ["DXMT_API_KEY"]
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.load(response)["data"]
    except urllib.error.HTTPError as e:
        # The status alone ("400: Bad Request") does not say which call or why.
        raise RuntimeError(f"{method} {path}: {e.code} {e.read()[:500]!r}") from e


def turns(sample):
    """(memory id, text) for every turn of one conversation."""
    conversation = sample["conversation"]
    n = 1
    while f"session_{n}" in conversation:
        when = conversation.get(f"session_{n}_date_time", "")
        for turn in conversation[f"session_{n}"]:
            text = f"[{when}] {turn['speaker']}: {turn['text']}"
            if turn.get("blip_caption"):
                text += f" (shares a photo: {turn['blip_caption']})"
            yield f"{sample['sample_id']}:{turn['dia_id']}", text
        n += 1


def evidence_ids(sample, qa):
    out = set()
    for e in qa.get("evidence", []):
        # A few labels hold several ids in one string ("D8:6; D9:17").
        for part in e.replace(",", ";").split(";"):
            part = part.strip()
            if part:
                out.add(f"{sample['sample_id']}:{part}")
    return out


def ingest(url, data, workers):
    jobs = [
        (memory_id, text, sample["sample_id"])
        for sample in data
        for memory_id, text in turns(sample)
    ]

    def keep(job):
        memory_id, text, tag = job
        # An explicit id: a turn is a turn, even when two say the same thing ("Thanks!").
        call(url, "POST", "/api/v1/memories",
             {"id": memory_id, "text": text, "kind": "event", "tags": [tag], "source": "locomo"})

    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        for i, _ in enumerate(pool.map(keep, jobs), 1):
            if i % 500 == 0:
                print(f"  kept {i}/{len(jobs)}", file=sys.stderr, flush=True)
    return len(jobs), time.time() - started


KIND = None  # --kind: only memories of this kind


def tag_filter(tag):
    must = [{"key": "tags", "operation": {"type": "In", "value": [tag]}}]
    if KIND:
        must.append({"key": "kind", "operation": {"type": "Equal", "value": KIND}})
    return {"must": must}


def ranked(url, table, method, question, tag, k):
    if method == "recall":
        body = {"query": question, "tags": [tag], "limit": k}
        if KIND:
            body["kind"] = KIND
        hits = call(url, "POST", "/api/v1/memories/recall", body)["memories"]
    elif method == "dense":
        hits = call(url, "POST", f"/api/v1/tables/{table}/search",
                    {"text": question, "top_k": k, "filter": tag_filter(tag)})["hits"]
    else:
        hits = call(url, "POST", f"/api/v1/tables/{table}/search/text",
                    {"query": question, "limit": k, "filter": tag_filter(tag)})["hits"]
    return [h["id"] for h in hits]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", required=True, help="locomo10.json")
    parser.add_argument("--url", default="http://127.0.0.1:9999")
    parser.add_argument("--table", default="memory", help="the server's memory table")
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--methods", default="recall,dense,keyword")
    parser.add_argument("--json", help="write the per-question results here")
    parser.add_argument("--samples", help="comma-separated sample indexes to keep (default: all ten)")
    parser.add_argument("--kind", help="only memories of this kind (the turns are `event`)")
    args = parser.parse_args()
    global KIND
    KIND = args.kind

    data = json.load(open(args.data))
    if args.samples:
        keep = {int(i) for i in args.samples.split(",")}
        data = [s for i, s in enumerate(data) if i in keep]
    if not args.skip_ingest:
        n, seconds = ingest(args.url, data, args.workers)
        print(f"kept {n} turns in {seconds:.0f} s ({n / seconds:.1f}/s)", file=sys.stderr)

    methods = args.methods.split(",")
    questions = [
        (sample, qa) for sample in data for qa in sample["qa"]
        if qa["category"] in CATEGORIES and evidence_ids(sample, qa)
    ]
    k_max = max(KS)
    # method -> category -> k -> [score per question]
    scores = {m: collections.defaultdict(lambda: collections.defaultdict(list)) for m in methods}
    latency = collections.defaultdict(list)
    rows = []

    def ask(item):
        sample, qa = item
        wanted = evidence_ids(sample, qa)
        out = {"question": qa["question"], "category": qa["category"], "evidence": sorted(wanted)}
        for m in methods:
            t = time.perf_counter()
            ids = ranked(args.url, args.table, m, qa["question"], sample["sample_id"], k_max)
            out[m + "_ms"] = (time.perf_counter() - t) * 1000
            out[m] = ids
        return out

    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        for out in pool.map(ask, questions):
            rows.append(out)
            wanted = set(out["evidence"])
            for m in methods:
                latency[m].append(out[m + "_ms"])
                for k in KS:
                    found = len(wanted & set(out[m][:k])) / len(wanted)
                    scores[m][out["category"]][k].append(found)
                    scores[m]["all"][k].append(found)

    print(f"\nLoCoMo, {len(questions)} questions with evidence (category 5 left out); "
          f"share of evidence turns in the top k\n")
    header = f"{'method':8} {'category':12} {'n':>5} " + " ".join(f"{'R@' + str(k):>6}" for k in KS)
    print(header)
    for m in methods:
        for category in list(CATEGORIES) + ["all"]:
            name = CATEGORIES.get(category, "all")
            per_k = scores[m][category]
            n = len(per_k[KS[0]])
            print(f"{m:8} {name:12} {n:5} "
                  + " ".join(f"{sum(per_k[k]) / n:6.3f}" for k in KS))
        lat = sorted(latency[m])
        print(f"{m:8} latency p50 {lat[len(lat) // 2]:.0f} ms, p95 {lat[int(len(lat) * .95)]:.0f} ms\n")
    if args.json:
        json.dump(rows, open(args.json, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""One table from the M1 runs: J per engine and category, retrieval latency, context size, and
what each engine spent on the model server (from `llm_proxy.py`'s log, by tag).

    python3 eval/memory/m1_report.py --results dxmt=/tmp/m1/m1_dxmt_k10.json honcho=/tmp/m1/m1_honcho_k10.json \\
        --llm-log /tmp/m1/llm.jsonl --since 1793975000

`--since` limits the proxy log to the runs being reported (epoch seconds). Engine server-side
calls carry the proxy's default tag (`honcho`, `openviking`, `dxmt` = DXMT's embeddings); the
harness's own answer and judge calls are `<engine>:answer` / `<engine>:judge`.
"""

import argparse
import collections
import json

CATEGORIES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", nargs="+", required=True, help="engine=path.json")
    parser.add_argument("--llm-log", default="/tmp/m1/llm.jsonl")
    parser.add_argument("--since", type=float, default=0)
    args = parser.parse_args()

    runs = {}
    for spec in args.results:
        name, path = spec.split("=", 1)
        runs[name] = json.load(open(path))

    spent = collections.defaultdict(lambda: {"calls": 0, "prompt": 0, "completion": 0, "errors": 0})
    for line in open(args.llm_log):
        row = json.loads(line)
        if row["at"] < args.since:
            continue
        tag = row.get("tag") or "untagged"
        s = spent[tag]
        s["calls"] += 1
        s["prompt"] += row.get("prompt_tokens") or 0
        s["completion"] += row.get("completion_tokens") or 0
        if row.get("status") != 200:
            s["errors"] += 1

    cats = sorted({r["category"] for rows in runs.values() for r in rows})
    header = f"{'engine':22} {'n':>4} " + " ".join(f"{CATEGORIES.get(c, c):>11}" for c in cats) + f" {'all':>6} {'retr s':>7} {'ctx ch':>7}"
    print(header)
    for name, rows in runs.items():
        cells = []
        for c in cats:
            rs = [r for r in rows if r["category"] == c]
            cells.append(f"{sum(r['correct'] for r in rs) / len(rs):11.3f}" if rs else f"{'-':>11}")
        j = sum(r["correct"] for r in rows) / len(rows)
        rt = sorted(r["retrieved_seconds"] for r in rows)[len(rows) // 2]
        cc = sum(r.get("context_chars", 0) for r in rows) / len(rows)
        print(f"{name:22} {len(rows):4} " + " ".join(cells) + f" {j:6.3f} {rt:7.2f} {cc:7.0f}")

    print(f"\nmodel-server spend since {args.since:.0f} (calls / prompt tokens / completion tokens / non-200):")
    for tag in sorted(spent):
        s = spent[tag]
        print(f"  {tag:22} {s['calls']:6} {s['prompt']:10} {s['completion']:9} {s['errors']:4}")


if __name__ == "__main__":
    main()

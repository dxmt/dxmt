#!/usr/bin/env python3
"""Answers on LoCoMo: given what `recall` returns, does a model answer the question right?

`locomo.py` measures retrieval; this measures what an agent would get out of it. For each
sampled question it recalls the top k memories of that conversation, asks a model to answer
from them alone, and asks a model to judge the answer against LoCoMo's gold answer -- the
"LLM-as-a-judge" score (J) that Mem0, Zep and Letta report. Both prompts are below; the judge is
told to be generous about wording and strict about facts and dates.

Run `locomo.py` first, against the same server, to load the conversations.

    python3 eval/memory/locomo_answer.py --data locomo10.json --url http://127.0.0.1:9999 \\
        --llm http://100.78.106.34:11434 --model qwen3.6:35b --per-category 50

The model is called through ollama's `/api/chat` with thinking off. A judge is a model, so the
number moves a little with the model and the sample; the sample is seeded.
"""

import argparse
import collections
import concurrent.futures
import json
import random
import sys
import time
import urllib.request

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from locomo import CATEGORIES, call, evidence_ids  # noqa: E402

ANSWER = """You are answering a question about a long conversation between {a} and {b}, using
only the memories below, which were retrieved from it. Each memory starts with the date and
time of the session it is from; use those dates to work out when things happened, and give
dates as absolute dates (e.g. "7 May 2023", not "last week"). Answer in a short phrase. If the
memories do not contain the answer, say "I don't know".

Memories:
{memories}

Question: {question}
Answer:"""

JUDGE = """Decide whether a generated answer to a question is correct, given the gold answer.
Be generous about wording: the generated answer is CORRECT if it states the same fact as the
gold answer, even if it is longer, phrased differently, or adds detail. For dates and times,
it is CORRECT if it refers to the same date or period, in any format. It is WRONG if it
contradicts the gold answer, misses its key fact, or says it does not know.

Question: {question}
Gold answer: {gold}
Generated answer: {answer}

Reply with one word: CORRECT or WRONG."""


def chat(llm, model, prompt):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "stream": False, "think": False, "options": {"temperature": 0}}
    request = urllib.request.Request(llm + "/api/chat", data=json.dumps(body).encode(),
                                     headers={"content-type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)["message"]["content"].strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:9999")
    parser.add_argument("--llm", default="http://127.0.0.1:11434")
    parser.add_argument("--model", required=True)
    parser.add_argument("--judge-model", help="defaults to --model")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--per-category", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--json", help="write every question, answer and verdict here")
    args = parser.parse_args()

    data = json.load(open(args.data))
    pool = collections.defaultdict(list)
    for sample in data:
        for qa in sample["qa"]:
            if qa["category"] in CATEGORIES and "answer" in qa:
                pool[qa["category"]].append((sample, qa))
    rng = random.Random(args.seed)
    questions = [item for c in sorted(pool)
                 for item in rng.sample(pool[c], min(args.per_category, len(pool[c])))]

    def run(item):
        sample, qa = item
        conv = sample["conversation"]
        hits = call(args.url, "POST", "/api/v1/memories/recall",
                    {"query": qa["question"], "tags": [sample["sample_id"]], "limit": args.k})["memories"]
        memories = "\n".join(f"- {h['text']}" for h in hits)
        answer = chat(args.llm, args.model, ANSWER.format(
            a=conv["speaker_a"], b=conv["speaker_b"], memories=memories, question=qa["question"]))
        verdict = chat(args.llm, args.judge_model or args.model, JUDGE.format(
            question=qa["question"], gold=qa["answer"], answer=answer))
        found = evidence_ids(sample, qa) & {h["id"] for h in hits}
        return {"category": qa["category"], "question": qa["question"], "gold": str(qa["answer"]),
                "answer": answer, "verdict": verdict,
                "correct": verdict.upper().startswith("CORRECT"),
                "evidence_found": len(found), "evidence": len(evidence_ids(sample, qa))}

    started = time.time()
    rows = []
    with concurrent.futures.ThreadPoolExecutor(args.workers) as executor:
        for i, row in enumerate(executor.map(run, questions), 1):
            rows.append(row)
            if i % 20 == 0:
                print(f"  {i}/{len(questions)}, {time.time() - started:.0f} s", file=sys.stderr, flush=True)

    unparsed = sum(1 for r in rows if r["verdict"].upper()[:5] not in ("CORRE", "WRONG"))
    print(f"\nLoCoMo answers, {len(rows)} questions ({args.per_category} per category, seed "
          f"{args.seed}), top {args.k} memories, answered and judged by {args.model}"
          f"{' / ' + args.judge_model if args.judge_model else ''}; {unparsed} verdicts unparsed\n")
    print(f"{'category':12} {'n':>4} {'J':>6} {'J | all evidence found':>24}")
    for c in list(CATEGORIES) + ["all"]:
        rs = [r for r in rows if c == "all" or r["category"] == c]
        full = [r for r in rs if r["evidence"] and r["evidence_found"] == r["evidence"]]
        j = sum(r["correct"] for r in rs) / len(rs)
        jf = sum(r["correct"] for r in full) / len(full) if full else float("nan")
        print(f"{CATEGORIES.get(c, 'all'):12} {len(rs):4} {j:6.3f} {jf:17.3f} (n={len(full)})")
    if args.json:
        json.dump(rows, open(args.json, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

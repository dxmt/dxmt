#!/usr/bin/env python3
"""Answers on LoCoMo: given what `recall` returns, does a model answer the question right?

`locomo.py` measures retrieval; this measures what an agent would get out of it. For each
sampled question it recalls the top k memories of that conversation, asks a model to answer
from them alone, and asks a model to judge the answer against LoCoMo's gold answer -- the
"LLM-as-a-judge" score (J) that Mem0, Zep and Letta report. Both prompts are below; the judge is
told to be generous about wording and strict about facts and dates.

`--engine` picks what is in front of the model (see `engines.py`): `dxmt` (this server),
`honcho` or `openviking`, each ingesting the conversations its own way; the answer model and
the judge are the same for all of them, which is what makes the J columns comparable.
`--mode recall` hands the model what the engine retrieves; `--mode answer` takes the engine's own
answer (Honcho's dialectic `chat`) and only judges it. `--ingest` loads the conversations first
(for `dxmt`, `locomo.py` does the same thing).

    python3 eval/memory/locomo_answer.py --data locomo10.json --engine dxmt --url http://127.0.0.1:9999 \\
        --llm http://127.0.0.1:1235/v1 --model qwen/qwen3.6-35b-a3b --per-category 50

`--llm` ending in `/v1` is an OpenAI-compatible chat endpoint (LM Studio, vLLM, or
`llm_proxy.py`, which counts the calls); otherwise it is ollama's `/api/chat`. A judge is a
model, so the number moves a little with the model and the sample; the sample is seeded.
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
dates as absolute dates (e.g. "7 May 2023", not "last week"). A memory that places something
relative to its own date -- "yesterday", "last weekend", "last year" -- dates it relative to
that date: said on 17 July 2023, "last weekend" is "the weekend before 17 July 2023", not 17
July. Facts about the people come from the memories; general knowledge may connect them (where
a film was shot, which state a city is in, what suits someone allergic to fur). If the
memories support a reasonable inference, give it. Answer in a short phrase. If the memories do
not contain the answer, say "I don't know".

Memories:
{memories}

Question: {question}
Answer:"""

AGENT = """You are answering a question about a long conversation between {a} and {b}. Below are
memories retrieved from it so far. Each memory starts with the date and time of the session it
is from; give dates as absolute dates (e.g. "7 May 2023", not "last week"). A memory that
places something relative to its own date -- "yesterday", "last weekend", "last year" -- dates
it relative to that date: said on 17 July 2023, "last weekend" is "the weekend before 17 July
2023", not 17 July.

Facts about the people come from the memories; general knowledge may connect them (where a
film was shot, which state a city is in, what suits someone allergic to fur). If the memories
answer the question, or support a reasonable inference, reply `ANSWER: <short phrase>`. Otherwise reply `SEARCH: <query>` to search the conversation again
({left} searches left): use different words -- the specific names, places and things the answer
would mention -- and when the question asks for several things, search for the one still
missing.{paging} Do not answer "I don't know" while you have searches left. Reply with that one
line only.

Memories:
{memories}

Question: {question}"""

FINAL = """You are answering a question about a long conversation between {a} and {b}. The searches
are spent; below is everything they found. Each memory starts with the date of the session it is
from; give dates as absolute dates, and date "last week" and the like relative to the memory's
own date. Facts about the people come from the memories; general knowledge may connect them
(where a film was shot, which state a city is in, what suits someone allergic to fur). Give your
best answer in a short phrase -- an inference the memories support is an answer. Say "I don't
know" only if nothing below bears on the question.

Memories:
{memories}

Question: {question}
Answer:"""

PAGING = """ Searching a query you have already used returns
the next memories in its ranking, so a list that may be longer than what you see can be
continued that way."""

JUDGE = """Decide whether a generated answer to a question is correct, given the gold answer.
Be generous about wording: the generated answer is CORRECT if it states the same fact as the
gold answer, even if it is longer, phrased differently, or adds detail. For dates and times,
it is CORRECT if it refers to the same date or period, in any format. It is WRONG if it
contradicts the gold answer, misses its key fact, or says it does not know.

Question: {question}
Gold answer: {gold}
Generated answer: {answer}

Reply with one word: CORRECT or WRONG."""


def chat(llm, model, prompt, tag="", attempts=4, max_tokens=None):
    """One answer from the model; a 4xx/5xx from a model server mid-reload is retried."""
    for attempt in range(attempts):
        try:
            return _chat(llm, model, prompt, tag, max_tokens)
        except urllib.error.HTTPError as e:
            if attempt == attempts - 1:
                raise
            time.sleep(5 * (attempt + 1))


def _chat(llm, model, prompt, tag="", max_tokens=None):
    headers = {"content-type": "application/json", "X-M1-Tag": tag}
    if llm.rstrip("/").endswith("/v1"):
        body = {"model": model, "messages": [{"role": "user", "content": prompt}],
                "temperature": 0, "stream": False, "reasoning_effort": "none"}
        if max_tokens:
            # A model that falls into repeating itself otherwise generates to the end of its
            # context, holding the model server for an hour.
            body["max_tokens"] = max_tokens
        request = urllib.request.Request(llm.rstrip("/") + "/chat/completions",
                                         data=json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(request, timeout=900) as response:
            content = json.load(response)["choices"][0]["message"]["content"] or ""
    else:
        body = {"model": model, "messages": [{"role": "user", "content": prompt}],
                "stream": False, "think": False, "options": {"temperature": 0}}
        request = urllib.request.Request(llm + "/api/chat", data=json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(request, timeout=900) as response:
            content = json.load(response)["message"]["content"]
    # A thinking model that ignored the switch still answers after its </think>.
    if "</think>" in content:
        content = content.rsplit("</think>", 1)[1]
    return content.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", required=True)
    parser.add_argument("--engine", default="dxmt", choices=["dxmt", "honcho", "openviking"])
    parser.add_argument("--mode", default="recall", choices=["recall", "answer", "agentic"],
                        help="recall: answer from what the engine retrieves; answer: judge the engine's own answer; "
                             "agentic: like recall, but the model may search again (--rounds) before answering")
    parser.add_argument("--rounds", type=int, default=3, help="agentic: extra searches the model may make")
    parser.add_argument("--profiles", action="store_true",
                        help="recall/agentic: put the engine's profiles of the people before the memories")
    parser.add_argument("--url", default="http://127.0.0.1:9999", help="the engine's server")
    parser.add_argument("--ingest", action="store_true", help="load the conversations into the engine first")
    parser.add_argument("--ingest-workers", type=int, default=8)
    parser.add_argument("--samples", help="comma-separated sample indexes to keep (default: all ten)")
    parser.add_argument("--llm", default="http://127.0.0.1:11434")
    parser.add_argument("--model", required=True)
    parser.add_argument("--judge-model", help="defaults to --model")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--per-category", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--json", help="write every question, answer and verdict here")
    args = parser.parse_args()

    import engines  # noqa: E402

    data = json.load(open(args.data))
    if args.samples:
        keep = {int(i) for i in args.samples.split(",")}
        data = [s for i, s in enumerate(data) if i in keep]
    engine = engines.make(args.engine, args.url)
    if args.ingest:
        print(f"ingesting {len(data)} conversations into {args.engine} ...", file=sys.stderr, flush=True)
        stats = engine.ingest(data, args.ingest_workers)
        print(f"ingested: {json.dumps(stats)}", file=sys.stderr, flush=True)
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
        t0 = time.time()
        memories_list, ids, chars, searches = [], None, 0, []
        if args.mode == "answer":
            answer = engine.answer(sample, qa["question"])
            if answer is None:
                raise SystemExit(f"{args.engine} has no answering path; use --mode recall")
            retrieved = time.time() - t0
        elif args.mode == "agentic":
            memories_list, ids = engine.recall(sample, qa["question"], args.k)
            if args.profiles:
                memories_list = list(dict.fromkeys(engine.profiles(sample, qa["question"]) + memories_list))
            ids = set(ids) if ids is not None else None
            answer, searches = None, []
            for left in range(args.rounds, -1, -1):
                memories = "\n".join(f"- {m}" for m in memories_list) or "(nothing was recalled)"
                if left == 0:
                    break
                reply = chat(args.llm, args.model, AGENT.format(
                    a=conv["speaker_a"], b=conv["speaker_b"], memories=memories, question=qa["question"],
                    left=left, paging=PAGING if engine.pages else ""), tag=f"{args.engine}:agent")
                line = reply.strip().splitlines()[0] if reply.strip() else ""
                if not line.upper().startswith("SEARCH:"):
                    answer = line.split(":", 1)[1].strip() if line.upper().startswith("ANSWER:") else reply.strip()
                    break
                query = line.split(":", 1)[1].strip()
                # The same query again asks for the next page, where the engine has pages.
                asked = [q.lower() for q in [qa["question"]] + searches]
                offset = args.k * asked.count(query.lower())
                searches.append(query)
                more, more_ids = engine.recall(sample, query, args.k, offset=offset if engine.pages else 0)
                memories_list += [m for m in more if m not in memories_list]
                if ids is not None and more_ids is not None:
                    ids |= set(more_ids)
            retrieved = time.time() - t0
            chars = len(memories)
            if answer is None:
                answer = chat(args.llm, args.model, FINAL.format(
                    a=conv["speaker_a"], b=conv["speaker_b"], memories=memories, question=qa["question"]),
                    tag=f"{args.engine}:answer")
        else:
            memories_list, ids = engine.recall(sample, qa["question"], args.k)
            if args.profiles:
                memories_list = list(dict.fromkeys(engine.profiles(sample, qa["question"]) + memories_list))
            retrieved = time.time() - t0
            memories = "\n".join(f"- {m}" for m in memories_list) or "(nothing was recalled)"
            chars = len(memories)
            answer = chat(args.llm, args.model, ANSWER.format(
                a=conv["speaker_a"], b=conv["speaker_b"], memories=memories, question=qa["question"]),
                tag=f"{args.engine}:answer")
        verdict = chat(args.llm, args.judge_model or args.model, JUDGE.format(
            question=qa["question"], gold=qa["answer"], answer=answer), tag=f"{args.engine}:judge")
        evidence = evidence_ids(sample, qa)
        found = len(evidence & ids) if ids is not None else None
        return {"category": qa["category"], "sample": sample["sample_id"], "question": qa["question"],
                "gold": str(qa["answer"]), "answer": answer, "verdict": verdict,
                "correct": verdict.upper().startswith("CORRECT"),
                "evidence_found": found, "evidence": len(evidence),
                "retrieved_seconds": round(retrieved, 3), "seconds": round(time.time() - t0, 3),
                "memories": len(memories_list), "context_chars": chars,
                "searches": searches if args.mode == "agentic" else None}

    started = time.time()
    rows = []
    with concurrent.futures.ThreadPoolExecutor(args.workers) as executor:
        for i, row in enumerate(executor.map(run, questions), 1):
            rows.append(row)
            if i % 5 == 0:
                print(f"  {i}/{len(questions)}, {time.time() - started:.0f} s", file=sys.stderr, flush=True)

    unparsed = sum(1 for r in rows if r["verdict"].upper()[:5] not in ("CORRE", "WRONG"))
    print(f"\nLoCoMo answers, engine {args.engine} ({args.mode}), {len(rows)} questions ({args.per_category} "
          f"per category, seed {args.seed}), top {args.k} memories, answered and judged by {args.model}"
          f"{' / ' + args.judge_model if args.judge_model else ''}; {unparsed} verdicts unparsed\n")
    print(f"{'category':12} {'n':>4} {'J':>6} {'J | all evidence found':>24} {'retrieve s':>11} {'ctx chars':>10}")
    for c in list(CATEGORIES) + ["all"]:
        rs = [r for r in rows if c == "all" or r["category"] == c]
        full = [r for r in rs if r["evidence"] and r["evidence_found"] == r["evidence"]]
        j = sum(r["correct"] for r in rs) / len(rs)
        jf = sum(r["correct"] for r in full) / len(full) if full else float("nan")
        rt = sorted(r["retrieved_seconds"] for r in rs)[len(rs) // 2]
        cc = sum(r["context_chars"] for r in rs) / len(rs)
        print(f"{CATEGORIES.get(c, 'all'):12} {len(rs):4} {j:6.3f} {jf:17.3f} (n={len(full):4}) {rt:11.2f} {cc:10.0f}")
    if args.json:
        json.dump(rows, open(args.json, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

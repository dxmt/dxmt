#!/usr/bin/env python3
"""RAG quality on CMRC2018 (Chinese reading comprehension over Chinese Wikipedia) with DXMT.

Two phases, because they cost very different amounts:

* ``retrieval`` -- no LLM. Every paragraph of the dev set goes into one table (848 of them,
  each the others' distractors); every question is asked through vector search, BM25 and
  hybrid search; the dataset says which paragraph holds each answer, so Hit@k and MRR are
  exact. Minutes for all 3,219 questions.
* ``ragas`` -- an LLM answers a sample of questions from the retrieved paragraphs, and RAGAS
  scores context precision, context recall, faithfulness and answer relevancy with an LLM as
  judge. Minutes per question on a CPU, so a sample.

    python3 eval/ragas/cmrc_eval.py load       --url http://127.0.0.1:19997
    python3 eval/ragas/cmrc_eval.py retrieval  --url http://127.0.0.1:19997
    ~/.venvs/dxmt-ragas/bin/python eval/ragas/cmrc_eval.py ragas --url ... --sample 20

The server needs a default embedding model (``DXMT_EMBEDDING_MODEL``, or a local ollama with
``bge-m3``). The RAGAS phase talks to ollama for the answering and judging models.
"""

import argparse
import json
import os
import random
import statistics
import sys
import time
import urllib.request

DATA = os.environ.get("CMRC_DEV", "cmrc2018_dev.json")  # github.com/ymcui/cmrc2018, squad-style/cmrc2018_dev.json
TABLE = "cmrc"


def call(url, method, path, body=None, timeout=600):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url + path, data=data, method=method, headers={"content-type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        return json.load(e)


def load_dataset(path):
    d = json.load(open(path, encoding="utf-8"))
    paragraphs, questions = [], []
    for article in d["data"]:
        for p in article["paragraphs"]:
            paragraphs.append({"id": p["id"], "title": article["title"], "text": p["context"]})
            for qa in p["qas"]:
                questions.append(
                    {
                        "id": qa["id"],
                        "question": qa["question"],
                        "answer": qa["answers"][0]["text"],
                        "gold": p["id"],
                    }
                )
    return paragraphs, questions


def cmd_load(args):
    paragraphs, _ = load_dataset(args.data)
    call(args.url, "DELETE", f"/api/v1/tables/{TABLE}")
    created = call(
        args.url,
        "POST",
        "/api/v1/tables",
        {
            "name": TABLE,
            # No dimension and no model: the server's default model, and its width.
            "enable_full_text_search": True,
            "text_fields": [
                {
                    "name": "content",
                    "source": {"metadata_field": "text"},
                    "analyzer": "chinese",
                    "boost": 1.0,
                }
            ],
        },
    )
    if not created.get("success"):
        sys.exit(f"create failed: {created}")
    model = created["data"]["embedding"]["model"]
    print(f"table {TABLE}: {created['data']['dimension']} dimensions, model {model}")
    started = time.time()
    batch = 8  # a CPU embedding model takes ~2 s a paragraph; the server allows 60 s a call
    for i in range(0, len(paragraphs), batch):
        chunk = paragraphs[i : i + batch]
        r = call(
            args.url,
            "POST",
            f"/api/v1/tables/{TABLE}/records",
            {
                "records": [
                    {"id": p["id"], "text": p["text"], "payload": {"title": p["title"]}}
                    for p in chunk
                ]
            },
        )
        if not r.get("success"):
            sys.exit(f"insert failed at {i}: {r}")
        print(f"\r  {i + len(chunk)}/{len(paragraphs)} paragraphs", end="", flush=True)
    print(f"\nloaded in {time.time() - started:.0f}s")


def search(url, mode, question, k):
    if mode == "vector":
        r = call(url, "POST", f"/api/v1/tables/{TABLE}/search", {"text": question, "top_k": k})
    elif mode == "bm25":
        r = call(url, "POST", f"/api/v1/tables/{TABLE}/search/text", {"query": question, "limit": k})
    else:
        fusion = mode.split(":", 1)[1] if ":" in mode else None
        body = {"text": question, "limit": k}
        if fusion:
            body["fusion_method"] = fusion
        r = call(url, "POST", f"/api/v1/tables/{TABLE}/search/hybrid", body)
    if not r.get("success"):
        raise RuntimeError(f"{mode}: {r}")
    return [h["id"] for h in r["data"]["hits"]]


def cmd_retrieval(args):
    _, questions = load_dataset(args.data)
    if args.limit:
        random.Random(7).shuffle(questions)
        questions = questions[: args.limit]
    ks = [1, 3, 5, 10]
    print(f"{len(questions)} questions, 848 paragraphs\n")
    print("| retrieval | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 | p50 ms |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for mode in args.modes.split(","):
        hits = {k: 0 for k in ks}
        rr = 0.0
        latencies = []
        for q in questions:
            t = time.time()
            ids = search(args.url, mode, q["question"], max(ks))
            latencies.append((time.time() - t) * 1000)
            if q["gold"] in ids:
                rank = ids.index(q["gold"]) + 1
                rr += 1.0 / rank
                for k in ks:
                    if rank <= k:
                        hits[k] += 1
        n = len(questions)
        row = " | ".join(f"{hits[k] / n:.3f}" for k in ks)
        print(f"| {mode} | {row} | {rr / n:.3f} | {statistics.median(latencies):.0f} |", flush=True)


def ollama_answer(model, question, contexts):
    prompt = (
        "请只根据下面的资料回答问题。答案要简洁，资料里没有的信息不要编造。\n\n"
        + "\n\n".join(f"资料{i + 1}：{c}" for i, c in enumerate(contexts))
        + f"\n\n问题：{question}\n答案："
    )
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 256},
    }
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/chat",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)["message"]["content"].strip()


def cmd_ragas(args):
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )

    paragraphs, questions = load_dataset(args.data)
    text_of = {p["id"]: p["text"] for p in paragraphs}
    random.Random(11).shuffle(questions)
    sample = questions[: args.sample]

    samples, rows = [], []
    for i, q in enumerate(sample):
        ids = search(args.url, args.mode, q["question"], args.k)
        contexts = [text_of[x] for x in ids]
        t = time.time()
        response = ollama_answer(args.llm, q["question"], contexts)
        print(
            f"[{i + 1}/{len(sample)}] {q['question']}\n  gold: {q['answer']} | answer: "
            f"{response[:80]} | gold paragraph retrieved: {q['gold'] in ids} ({time.time() - t:.0f}s)",
            flush=True,
        )
        samples.append(
            SingleTurnSample(
                user_input=q["question"],
                retrieved_contexts=contexts,
                response=response,
                reference=q["answer"],
            )
        )
        rows.append({**q, "retrieved": ids, "response": response})

    llm = LangchainLLMWrapper(
        ChatOllama(model=args.llm, temperature=0, reasoning=False, num_ctx=8192)
    )
    embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=args.embedding))
    started = time.time()
    result = evaluate(
        EvaluationDataset(samples=samples),
        metrics=[
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
            Faithfulness(),
            ResponseRelevancy(),
        ],
        llm=llm,
        embeddings=embeddings,
        run_config=RunConfig(timeout=1800, max_workers=2, max_retries=2),
    )
    print(f"\nRAGAS ({args.mode}, top {args.k}, judge {args.llm}, {len(sample)} questions, "
          f"{time.time() - started:.0f}s):")
    print(result)
    frame = result.to_pandas()
    frame.to_csv(args.out, index=False)
    print(f"per-question scores in {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["load", "retrieval", "ragas"])
    ap.add_argument("--url", default="http://127.0.0.1:19997")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--modes", default="vector,bm25,hybrid,hybrid:reciprocal_rank_fusion")
    ap.add_argument("--limit", type=int, default=0, help="retrieval: a random subset of questions")
    ap.add_argument("--mode", default="hybrid", help="ragas: the retrieval to answer from")
    ap.add_argument("--k", type=int, default=3, help="ragas: paragraphs given to the answerer")
    ap.add_argument("--sample", type=int, default=20)
    ap.add_argument("--llm", default="gemma4:e4b")
    ap.add_argument("--embedding", default="bge-m3")
    ap.add_argument("--out", default="/tmp/ragas_cmrc.csv")
    args = ap.parse_args()
    {"load": cmd_load, "retrieval": cmd_retrieval, "ragas": cmd_ragas}[args.phase](args)


if __name__ == "__main__":
    main()

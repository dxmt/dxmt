#!/usr/bin/env python3
"""Prototype of server-side fact extraction (M2), run from outside the server.

For each session of each LoCoMo conversation, one model call turns the dialogue into
self-contained facts about each speaker -- names, things, counts, plans, with relative times
resolved against the session's date -- and keeps them through `POST /memories` as kind `fact`,
tagged with the conversation, beside the turns already there. Recall then searches both. This
measures whether derived facts are worth building into the server before building them.

    python3 eval/memory/derive_facts.py --data locomo10.json --samples 0,1,2,3,4 \\
        --url http://127.0.0.1:9999 --llm http://127.0.0.1:1237/v1 --model qwen/qwen3.6-35b-a3b
"""

import argparse
import concurrent.futures
import json
import re
import sys
import time

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from locomo import call  # noqa: E402
from locomo_answer import chat  # noqa: E402

PROMPT = """Below is one session of a conversation between {a} and {b}, held on {when}.
List the facts it establishes about {a} and {b} that someone might later ask about: what they
did, have, own, made, read, watched, visited, like, feel, plan or decided; the people, pets,
places, books, songs, events and numbers involved, by name; and why, where it was said.

Rules:
- One fact per line, self-contained, starting with the person's name; no pronouns that need
  the conversation to resolve.
- Never leave a relative time in a fact. Rewrite it against the session date ({day}):
  "yesterday" -> "the day before {day}", "last week" -> "the week before {day}",
  "last Friday" -> "the Friday before {day}", "last weekend" -> "the weekend before {day}",
  "last year" -> the year before, "ten years ago" -> that year, "next month" -> that month.
  A fact with no time of its own happened on or just before {day}; say "as of {day}".
- Keep specific names, titles and counts exactly as said. Do not invent anything.
- Skip greetings and small talk.

Session:
{dialogue}

Facts, one per line:"""


def sessions(sample):
    conv = sample["conversation"]
    n = 1
    while f"session_{n}" in conv:
        yield n, conv.get(f"session_{n}_date_time", ""), conv[f"session_{n}"]
        n += 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--samples")
    p.add_argument("--url", required=True)
    p.add_argument("--llm", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--done", help="a file of finished sessions, so a rerun picks up where it stopped")
    p.add_argument("--server", action="store_true",
                   help="have the server extract (POST /memories/extract, DXMT_MEMORY_LLM_MODEL) "
                        "instead of this script calling the model")
    args = p.parse_args()
    data = json.load(open(args.data))
    if args.samples:
        keep = {int(i) for i in args.samples.split(",")}
        data = [s for i, s in enumerate(data) if i in keep]

    done = set(open(args.done).read().split()) if args.done and __import__("os").path.exists(args.done) else set()
    jobs = [(s, n, when, turns) for s in data for n, when, turns in sessions(s)
            if f"{s['sample_id']}:S{n}" not in done]

    def derive(job):
        try:
            return derive_one(job)
        except Exception as e:  # a session that failed is not marked done; a rerun picks it up
            print(f"  {job[0]['sample_id']} session {job[1]} failed: {e!r}", file=sys.stderr, flush=True)
            return 0

    def derive_one(job):
        sample, n, when, turns = job
        conv = sample["conversation"]
        if args.server:
            dialogue = "\n".join(
                f"{t['speaker']}: {t['text']}" + (f" (shares a photo: {t['blip_caption']})" if t.get("blip_caption") else "")
                for t in turns)
            facts = call(args.url, "POST", "/api/v1/memories/extract",
                         {"text": dialogue, "when": when, "tags": [sample["sample_id"]], "source": "locomo",
                          "id_prefix": f"{sample['sample_id']}:S{n}"})["facts"]
            if args.done:
                with open(args.done, "a") as f:
                    f.write(f"{sample['sample_id']}:S{n}\n")
            return len(facts)
        dialogue = "\n".join(
            f"{t['speaker']}: {t['text']}" + (f" (shares a photo: {t['blip_caption']})" if t.get("blip_caption") else "")
            for t in turns)
        day = re.sub(r"^.*? on ", "", when)
        reply = chat(args.llm, args.model, PROMPT.format(
            a=conv["speaker_a"], b=conv["speaker_b"], when=when, day=day, dialogue=dialogue),
            tag="dxmt:derive", max_tokens=2500)
        facts = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip() for line in reply.splitlines()]
        facts = [f for f in facts if len(f) > 12]
        for i, fact in enumerate(facts):
            body = {"id": f"{sample['sample_id']}:S{n}:F{i}", "text": f"[{when}] {fact}", "kind": "fact",
                    "tags": [sample["sample_id"]], "source": "derive"}
            for attempt in range(4):
                try:
                    call(args.url, "POST", "/api/v1/memories", body)
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(10)
        if args.done:
            with open(args.done, "a") as f:
                f.write(f"{sample['sample_id']}:S{n}\n")
        return len(facts)

    started = time.time()
    total = 0
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        for i, k in enumerate(pool.map(derive, jobs), 1):
            total += k
            if i % 10 == 0:
                print(f"  {i}/{len(jobs)} sessions, {total} facts, {time.time() - started:.0f} s", file=sys.stderr, flush=True)
    print(json.dumps({"sessions": len(jobs), "facts": total, "seconds": round(time.time() - started)}))


if __name__ == "__main__":
    main()

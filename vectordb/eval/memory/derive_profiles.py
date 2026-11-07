#!/usr/bin/env python3
"""Prototype of per-person profiles (M4): every fact kept about a person, condensed into one
memory of kind `profile`, the way Honcho keeps a peer card. Run after derive_facts.py.

    python3 eval/memory/derive_profiles.py --data locomo10.json --samples 0,1,2,3,4 \\
        --url http://127.0.0.1:9999 --llm http://127.0.0.1:1237/v1 --model qwen/qwen3.6-35b-a3b
"""

import argparse
import json
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from locomo import call  # noqa: E402
from locomo_answer import chat  # noqa: E402

PROMPT = """Below are facts collected about {name} over many conversations, each with its date.
Write {name}'s profile: everything someone would need to answer questions about {name}, grouped
under these headings, one line per item, keeping names, titles, counts and dates exactly:

Identity and background; Family and relationships; Pets; Work, business and career plans;
Education; Health; Hobbies, sports and exercise; Creative work; Books, films, TV, music and
games (each one by title); Places visited and trips (with dates); Events attended (with dates);
Possessions and purchases; Recommendations given and received (by title, and to or from whom);
Goals, values and beliefs; Opinions and personality as others describe it.

Merge repeats; where a fact changed, give the latest and note the change. Leave out a heading
with nothing under it. Do not invent anything.

Facts:
{facts}

Profile of {name}:"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--samples")
    p.add_argument("--url", required=True)
    p.add_argument("--llm", required=True)
    p.add_argument("--model")
    p.add_argument("--server", action="store_true",
                   help="have the server make each profile (POST /memories/profile)")
    args = p.parse_args()
    data = json.load(open(args.data))
    if args.samples:
        keep = {int(i) for i in args.samples.split(",")}
        data = [s for i, s in enumerate(data) if i in keep]
    for sample in data:
        sid = sample["sample_id"]
        if args.server:
            for name in (sample["conversation"]["speaker_a"], sample["conversation"]["speaker_b"]):
                out = call(args.url, "POST", "/api/v1/memories/profile", {"name": name, "tags": [sid]})
                print(f"{sid} {name}: {out['facts']} facts -> {len(out['text'])} chars", flush=True)
            continue
        facts, offset = [], 0
        while True:  # every fact of this conversation, by paging the list
            page = call(args.url, "GET", f"/api/v1/memories?kind=fact&limit=1000&offset={offset}")["memories"]
            facts += [m["text"] for m in page if sid in m.get("tags", [])]
            if len(page) < 1000:
                break
            offset += 1000
        for name in (sample["conversation"]["speaker_a"], sample["conversation"]["speaker_b"]):
            mine = [f for f in facts if f"] {name}" in f or f" {name}'" in f or f"{name} " in f]
            profile = chat(args.llm, args.model, PROMPT.format(name=name, facts="\n".join(mine)),
                           tag="dxmt:profile", max_tokens=6000)
            call(args.url, "POST", "/api/v1/memories",
                 {"id": f"{sid}:profile:{name}", "text": f"Profile of {name}:\n{profile}", "kind": "profile",
                  "tags": [sid], "source": "derive"})
            print(f"{sid} {name}: {len(mine)} facts -> {len(profile)} chars", flush=True)


if __name__ == "__main__":
    main()

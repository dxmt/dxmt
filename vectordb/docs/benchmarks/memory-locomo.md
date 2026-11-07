# Agent memory on LoCoMo: DXMT, Honcho and OpenViking on one harness

**Result (2026-11-07).** On every question of the first five LoCoMo conversations (762), with one
model answering and judging for every engine, an agent answering from DXMT's memory scored
**0.835**; Honcho's own answering agent scored **0.735** on the same questions. The difference is
0.10 (95% bootstrap interval 0.062-0.135), McNemar exact p = 1.7e-7 on the paired answers. In an
earlier 196-question run of the same harness, OpenViking scored 0.459.

Everything needed to check this is in the repository: the harness (`eval/memory/`), the
per-question answers and verdicts of the final run
(`eval/memory/results/locomo-m1-2026-11-07.json`), and the commands below.

## What was measured

[LoCoMo](https://github.com/snap-research/locomo) is ten long conversations between two people
(19-32 sessions each, months of dated chat) with about 2,000 questions labelled by category. This
run uses conversations 0-4 and **every** question in categories 1-4 (multi-hop, temporal,
open-domain, single-hop); category 5 (adversarial, the answer is not in the conversation) is left
out, as in the papers that report LoCoMo.

For each engine the conversations are ingested the engine's own way, then each question is
answered and the answer judged:

| | how it ingests | how it answers |
| --- | --- | --- |
| **DXMT** | every turn kept as a memory; per session, `POST /memories/extract` (one model call) keeps the facts the session establishes, relative times resolved; per person, `POST /memories/profile` | an agent loop: recall the top 25 memories plus the two profiles, answer or search again (up to 3 more searches), then answer |
| **Honcho** (server 3.6.0) | its own pipeline: messages per peer, the deriver turns them into conclusions, peer cards and representations | its own `chat` (the dialectic agent, `reasoning_level="low"`), which plans its own retrieval |
| **OpenViking** (0.5.3) | its own pipeline: sessions of 8 turns committed, memory extraction by its ReAct loop | the same agent loop as DXMT over its `find` + `read` |

**One model for everything that answers and judges**: qwen3.6-35B-A3B (Q4_K_M), thinking off,
temperature 0, the same weights for every engine's answers, for Honcho's and OpenViking's own
server-side calls, for DXMT's extraction, and for the judge. Embeddings: bge-m3 for all three.
The judge prompt is in `eval/memory/locomo_answer.py`: generous about wording, strict about facts
and dates; a judge is a model, so the absolute numbers move with the model -- the comparison is
what the same model makes of each engine's memory.

## Results

### All 762 questions (final run)

| | multi-hop (142) | temporal (156) | open-domain (46) | single-hop (418) | **all (762)** |
| --- | --- | --- | --- | --- | --- |
| **DXMT** | **0.803** | **0.859** | 0.457 | **0.878** | **0.835** |
| Honcho `chat` | 0.775 | 0.756 | **0.565** | 0.732 | 0.735 |
| p (McNemar, paired) | 0.62 | 0.014 | 0.30 | 2.5e-8 | 1.7e-7 |

Questions only DXMT answered right: 143; only Honcho: 67. DXMT is ahead on each of the five
conversations (0.868/0.750, 0.864/0.778, 0.849/0.783, 0.794/0.754, 0.826/0.640). Honcho is ahead
on open-domain, the category that needs world knowledge on top of the conversation, by a margin
that 46 questions cannot separate from noise.

### 196 questions, all three engines and the ablations

25 questions per category (all 46 open-domain), the same five conversations. DXMT's facts and
profiles here were made by the prototype scripts with the prompts the endpoints now use.

| | multi-hop | temporal | open-domain | single-hop | **all** |
| --- | --- | --- | --- | --- | --- |
| DXMT, facts + profiles, agent loop | 0.760 | 0.880 | 0.457 | 0.860 | **0.745** |
| Honcho `chat` | 0.740 | 0.700 | 0.587 | 0.760 | 0.699 |
| Honcho, same agent loop | 0.680 | 0.700 | 0.478 | 0.660 | 0.633 |
| OpenViking, same agent loop | 0.540 | 0.420 | 0.348 | 0.520 | 0.459 |

And what each part of DXMT's memory is worth (100 questions, same agent loop, k = 25):

| DXMT memory | J |
| --- | --- |
| kept turns only | 0.67 |
| + facts from `extract` | 0.71 |
| + profiles | 0.75 |

## Cost

Model calls counted by a proxy in front of the model server (`eval/memory/llm_proxy.py`):

- **Ingest, all five conversations.** DXMT: 139 calls (128 sessions extracted, 10 profiles),
  306k prompt and 136k completion tokens. OpenViking: 480 calls, 3.4M prompt and 231k completion
  tokens. Honcho: its deriver and the answers of the 196-question run together came to 644 calls
  and 4.2M prompt tokens; its ingest alone was at least several times DXMT's.
- **Answering.** Comparable: about 15k prompt tokens a question for each. DXMT's loop puts two
  profiles (about 25k characters each) and 25 memories in front of the model; Honcho's `chat`
  makes several retrieval calls of its own. `POST /memories/context` exists to cut DXMT's side:
  one block inside a token budget.
- **Without a model at all.** DXMT keeps and recalls memory with an embedding model only; the
  turns-only line above is that configuration. Honcho and OpenViking need a chat model to keep
  anything.

## What could bias this, and what was done about it

- **Same model, two runtimes.** The answer model ran on LM Studio for the first part of the runs
  and on ollama (same GGUF weights, Q4_K_M) for the rest, for both engines; the judge stayed on
  LM Studio for both. Both engines' answers were judged by the identical judge.
- **Honcho lost some derivation.** During its ingest, eleven of its deriver batches (conversations
  30, 42 and 43) failed while another job loaded a model on the same server. On the two
  conversations untouched by that, the result is the same: DXMT 0.859, Honcho 0.766 (p = 0.0034,
  304 questions).
- **OpenViking's own protocol.** OpenViking's published LoCoMo numbers use Doubao models and a
  retrieval limit of 50 with reranking; here it ran with the shared local model, and a run at
  limit 50 without the agent loop also scored 0.15 on 20 questions. Its extraction is the part
  that suffers most from a weaker model; a stronger model would likely raise it.
- **The judge is the answer model.** A second, different judge would harden the numbers; the
  per-question file makes rejudging cheap.
- **One dataset.** LongMemEval is next.

## Reproduce

```bash
# a DXMT server with an embedding model and a memory model
DXMT_EMBEDDING_MODEL=bge-m3 DXMT_EMBEDDING_BASE_URL=http://<host>/v1 \
DXMT_MEMORY_LLM_MODEL=<chat model> DXMT_MEMORY_LLM_BASE_URL=http://<host>/v1 \
  vectordb serve --port 9999

# ingest the turns, then extract and profile -- through the server's endpoints
python3 -c "import json,sys; sys.path.insert(0,'eval/memory'); import engines; \
  engines.make('dxmt','http://127.0.0.1:9999').ingest(json.load(open('locomo10.json'))[:5], 4)"
python3 eval/memory/derive_facts.py    --data locomo10.json --samples 0,1,2,3,4 --url http://127.0.0.1:9999 --server
python3 eval/memory/derive_profiles.py --data locomo10.json --samples 0,1,2,3,4 --url http://127.0.0.1:9999 --server

# answer and judge every question
python3 eval/memory/locomo_answer.py --data locomo10.json --samples 0,1,2,3,4 --per-category 1000 \
  --engine dxmt --url http://127.0.0.1:9999 --llm http://<host>/v1 --model <chat model> \
  --mode agentic --k 25 --profiles --json dxmt.json --resume

# Honcho: --engine honcho --mode answer (its server configured with the same model);
# OpenViking: --engine openviking --mode agentic
```

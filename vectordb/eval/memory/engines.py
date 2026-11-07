"""The memory engines `locomo_answer.py --engine` compares, behind one small interface.

Each engine ingests LoCoMo's conversations the way its own benchmark harness does (the same
dates, the same speakers), then answers two questions per LoCoMo question: *recall* -- the text
an agent would be handed to answer from -- and, where the engine has an answering path of its
own (Honcho's dialectic `chat`), *answer*. The answer model and the judge are the same for every
engine and live in `locomo_answer.py`; what differs is only what each engine puts in front of
the model.

    dxmt       POST /api/v1/memories, recall(query, tags=[sample], limit=k)       -- this repo
    honcho     peers + session per conversation, messages with created_at, wait for the
               deriver; recall = session.context(peer_target, search_query, search_top_k=k)
               rendered as text; answer = peer.chat(question, session)             -- honcho-ai SDK
    openviking a user per conversation, a session per 8 turns, commit, wait for the task;
               recall = find(question, viking://user/<id>/memories, limit=k) + read   -- openviking-sdk

Ingest statistics (turns, seconds) are returned by `ingest`; LLM calls and tokens the engine
spends are not counted here but by `llm_proxy.py`, which every engine's model traffic is
routed through.
"""

import concurrent.futures
import datetime as dt
import os
import re
import time

from locomo import call, turns  # noqa: E402  (sys.path set by the caller)


def parse_locomo_date(text):
    """"1:56 pm on 8 May, 2023" -> datetime; None when the string does not parse."""
    text = (text or "").replace(" on ", " ").strip()
    for fmt in ("%I:%M %p %d %B, %Y", "%I:%M %p %d %B %Y"):
        try:
            return dt.datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def sessions(sample):
    """[(session number, date string, [turn dicts])] in order."""
    conversation = sample["conversation"]
    out, n = [], 1
    while f"session_{n}" in conversation:
        out.append((n, conversation.get(f"session_{n}_date_time", ""), conversation[f"session_{n}"]))
        n += 1
    return out


def question_target(question, speaker_a, speaker_b):
    """Which speaker a question is about: the one named in it, else speaker_a (Honcho's rule)."""
    q = question.lower()
    if speaker_b.lower() in q and speaker_a.lower() not in q:
        return speaker_b
    return speaker_a


class Engine:
    name = ""

    def ingest(self, data, workers):
        raise NotImplementedError

    pages = False

    def profiles(self, sample, question):
        """Standing summaries of the people, given before the recalled memories; none by default."""
        return []

    def recall(self, sample, question, k, offset=0):
        """(list of memory strings for the prompt, set of evidence ids matched or None)."""
        raise NotImplementedError

    def answer(self, sample, question):
        """The engine's own answer, or None when it has no answering path."""
        return None


class Dxmt(Engine):
    name = "dxmt"

    def __init__(self, url):
        self.url = url

    def ingest(self, data, workers):
        jobs = [(mid, text, s["sample_id"]) for s in data for mid, text in turns(s)]

        def keep(job):
            mid, text, tag = job
            call(self.url, "POST", "/api/v1/memories",
                 {"id": mid, "text": text, "kind": "event", "tags": [tag], "source": "locomo"})

        started = time.time()
        with concurrent.futures.ThreadPoolExecutor(workers) as pool:
            list(pool.map(keep, jobs))
        return {"turns": len(jobs), "seconds": time.time() - started}

    pages = True  # recall takes `offset`: the same query again gives the next memories

    def profiles(self, sample, question):
        """The people's profiles (kind `profile`), the most relevant first."""
        hits = call(self.url, "POST", "/api/v1/memories/recall",
                    {"query": question, "tags": [sample["sample_id"]], "kind": "profile", "limit": 2})["memories"]
        return [h["text"] for h in hits]

    def recall(self, sample, question, k, offset=0):
        hits = call(self.url, "POST", "/api/v1/memories/recall",
                    {"query": question, "tags": [sample["sample_id"]], "limit": k,
                     "offset": offset})["memories"]
        return [h["text"] for h in hits], {h["id"] for h in hits}


class Honcho(Engine):
    name = "honcho"

    def __init__(self, url, workspace="locomo-m1", reasoning_level="low", wait_seconds=7200):
        from honcho import Honcho as Client

        self.client = Client(base_url=url, workspace_id=workspace, api_key="none", timeout=600)
        self.reasoning_level = reasoning_level
        self.wait_seconds = wait_seconds

    def _session_id(self, sample):
        return f"locomo-{sample['sample_id']}"

    def ingest(self, data, workers):
        from honcho.session import SessionPeerConfig

        started, count = time.time(), 0
        for sample in data:
            conv = sample["conversation"]
            a, b = conv["speaker_a"], conv["speaker_b"]
            peer_a, peer_b = self.client.peer(a), self.client.peer(b)
            session = self.client.session(self._session_id(sample))
            session.add_peers([(peer_a, SessionPeerConfig(observe_me=True, observe_others=False)),
                               (peer_b, SessionPeerConfig(observe_me=True, observe_others=False))])
            messages = []
            for _, when, turns_ in sessions(sample):
                at = parse_locomo_date(when)
                for turn in turns_:
                    text = turn["text"]
                    if turn.get("blip_caption"):
                        text += f" (shares a photo: {turn['blip_caption']})"
                    peer = peer_a if turn["speaker"] == a else peer_b
                    messages.append(peer.message(text, metadata={"dia_id": turn["dia_id"]}, created_at=at))
            for i in range(0, len(messages), 100):
                session.add_messages(messages[i:i + 100])
            count += len(messages)
        self.wait(data)
        return {"turns": count, "seconds": time.time() - started}

    def wait(self, data):
        """Block until the deriver has processed every session (Honcho's memory is asynchronous)."""
        deadline = time.time() + self.wait_seconds
        for sample in data:
            sid = self._session_id(sample)
            delay = 2.0
            while time.time() < deadline:
                status = self.client.queue_status(session=sid)
                if status.pending_work_units == 0 and status.in_progress_work_units == 0:
                    break
                time.sleep(delay)
                delay = min(delay * 1.5, 30)

    def recall(self, sample, question, k, offset=0):
        conv = sample["conversation"]
        target = question_target(question, conv["speaker_a"], conv["speaker_b"])
        session = self.client.session(self._session_id(sample))
        context = session.context(summary=True, peer_target=target, search_query=question,
                                  search_top_k=k, max_conclusions=k)
        parts = []
        if context.peer_card:
            parts.append(f"[{target}, card] " + "; ".join(str(x) for x in context.peer_card))
        if context.peer_representation:
            parts.append(f"[{target}, representation]\n{context.peer_representation}")
        if context.summary:
            summary = context.summary if isinstance(context.summary, str) else getattr(context.summary, "content", str(context.summary))
            parts.append(f"[conversation summary]\n{summary}")
        return parts, None

    def answer(self, sample, question):
        conv = sample["conversation"]
        target = question_target(question, conv["speaker_a"], conv["speaker_b"])
        peer = self.client.peer(target)
        out = peer.chat(question, session=self._session_id(sample), reasoning_level=self.reasoning_level, timeout=600)
        if out is None:
            return ""
        return out if isinstance(out, str) else getattr(out, "content", str(out))


class OpenViking(Engine):
    name = "openviking"
    CHUNK = 8  # turns per OpenViking session, as benchmark/locomo/openviking/import_to_ov.py does

    def __init__(self, url, wait_seconds=7200):
        import openviking_sdk

        self.sdk = openviking_sdk
        self.url = url
        self.wait_seconds = wait_seconds
        self.clients = {}

    def _client(self, sample):
        user = sample["sample_id"]
        if user not in self.clients:
            client = self.sdk.SyncHTTPClient(url=self.url, user=user, timeout=600)
            client.initialize()
            self.clients[user] = client
        return self.clients[user]

    def ingest(self, data, workers):
        started, count, tasks = time.time(), 0, []
        for sample in data:
            client = self._client(sample)
            conv = sample["conversation"]
            a = conv["speaker_a"]
            for _, when, turns_ in sessions(sample):
                base = parse_locomo_date(when)
                for c in range(0, len(turns_), self.CHUNK):
                    chunk = turns_[c:c + self.CHUNK]
                    sid = client.create_session()["session_id"]
                    for i, turn in enumerate(chunk):
                        text = f"[{turn['speaker']}]: {turn['text']}"
                        if turn.get("blip_caption"):
                            text += f" (shares a photo: {turn['blip_caption']})"
                        options = {}
                        if base:
                            options["created_at"] = (base + dt.timedelta(seconds=c + i)).isoformat()
                        client.add_message(sid, "user" if turn["speaker"] == a else "assistant",
                                           parts=[{"type": "text", "text": text}], options=options or None)
                        count += 1
                    result = client.commit_session(sid)
                    if result.get("task_id"):
                        tasks.append((client, result["task_id"]))
        self.wait(tasks)
        return {"turns": count, "seconds": time.time() - started, "commits": len(tasks)}

    def wait(self, tasks):
        deadline = time.time() + self.wait_seconds
        for client, task_id in tasks:
            delay = 2.0
            while time.time() < deadline:
                task = client.get_task(task_id) or {}
                status = str(task.get("status", "")).lower()
                if status in ("completed", "succeeded", "success", "done", "failed", "error", "cancelled"):
                    break
                time.sleep(delay)
                delay = min(delay * 1.5, 30)

    def recall(self, sample, question, k, offset=0):
        client = self._client(sample)
        result = client.find(question, target_uri=f"viking://user/{sample['sample_id']}/memories", limit=k)
        items = _find_items(result)[:k]
        memories = []
        for item in items:
            uri = item.get("uri") or ""
            text = item.get("content") or ""
            if not text and uri:
                try:
                    text = client.read(uri, offset=0, limit=-1)
                except Exception as exc:  # a directory, or a transient read error
                    text = item.get("abstract") or f"[unreadable: {exc}]"
            when = item.get("created_at") or ""
            memories.append(f"[{when}] {text}" if when else text)
        return memories, None


def _find_items(result):
    """The hits of an OpenViking find response, whatever the envelope: a list, or a dict holding
    `results`, `items`, `contexts`, or per-type lists (`memories`, `resources`, `skills`)."""
    if isinstance(result, list):
        return [x for x in result if isinstance(x, dict)]
    if not isinstance(result, dict):
        return []
    for key in ("results", "items", "contexts", "hits"):
        if isinstance(result.get(key), list):
            return [x for x in result[key] if isinstance(x, dict)]
    items = []
    for key in ("memories", "resources", "skills"):
        if isinstance(result.get(key), list):
            items.extend(x for x in result[key] if isinstance(x, dict))
    items.sort(key=lambda x: -float(x.get("score") or 0))
    return items


def make(name, url):
    if name == "dxmt":
        return Dxmt(url)
    if name == "honcho":
        return Honcho(url)
    if name == "openviking":
        return OpenViking(url)
    raise SystemExit(f"unknown engine {name}")

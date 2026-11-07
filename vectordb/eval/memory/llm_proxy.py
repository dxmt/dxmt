#!/usr/bin/env python3
"""A counting proxy in front of an OpenAI-compatible server.

Every engine under comparison (DXMT's answer model, Honcho's deriver, OpenViking's VLM) is
pointed at this proxy instead of the model server, so the LLM calls and tokens each one spends
are measured the same way, from the `usage` field of the responses, rather than read off their
own logs. One JSON line per upstream call goes to --log: time, path, model, prompt and
completion tokens, latency, and the `tag` taken from the `X-M1-Tag` header or --tag.

    python3 eval/memory/llm_proxy.py --upstream http://100.85.201.58:1234 --port 1235 --log /tmp/m1/llm.jsonl

Streaming (SSE) responses are passed through chunk by chunk; usage is read from the final
chunk when the client asked for `stream_options.include_usage`, else counted as unknown.

The proxy also makes the model server a fair bottleneck: at most --chat-parallel chat
completions and --embed-parallel embedding requests are in flight upstream at once (LM Studio
answers `terminated` or 500 past its own limits), and a 5xx or a `terminated` answer is retried
up to three times. Every engine gets the same queue, so none is penalised for how many calls it
fires at once -- but the wall-clock an engine spends waiting in it is its own.
"""

import argparse
import http.client
import http.server
import json
import sys
import threading
import time
import urllib.parse

LOCK = threading.Lock()
RETRIES = 5


class Gate:
    """One model family upstream at a time. LM Studio keeps answering `terminated` or `Model is
    unloaded` when chat and embedding requests overlap (measured: 8 parallel embeddings fine, 4
    parallel chats fine, 2 chats + 4 embeddings -> two failures), so chat calls and embedding
    calls take turns: up to `chat` chats in flight, or up to `embed` embeddings, never both."""

    def __init__(self, chat, embed):
        self.limits = {"chat": chat, "embed": embed, "other": 8}
        self.mode = None
        self.inflight = 0
        self.waiting = {"chat": 0, "embed": 0, "other": 0}
        self.cv = threading.Condition()

    @staticmethod
    def kind(path):
        if "/embeddings" in path:
            return "embed"
        if "/chat/completions" in path or "/completions" in path or "/responses" in path:
            return "chat"
        return "other"

    def acquire(self, kind):
        # Fair: while the other family waits, the one in flight takes no newcomers and drains.
        # Without it a steady stream of embeddings (a writer storing facts) kept chats out, and
        # one hung upstream call held everything for the full upstream timeout.
        with self.cv:
            self.waiting[kind] += 1
            while not ((self.mode is None or self.mode == kind) and self.inflight < self.limits[kind]
                       and not (self.mode == kind and any(n for k, n in self.waiting.items() if k != kind))):
                self.cv.wait()
            self.waiting[kind] -= 1
            self.mode = kind
            self.inflight += 1

    def release(self):
        with self.cv:
            self.inflight -= 1
            if self.inflight == 0:
                self.mode = None
            self.cv.notify_all()


class Proxy(http.server.BaseHTTPRequestHandler):
    upstream = None
    log = None
    errors = None
    tag = ""
    gate = None
    chat_upstream = None
    chat_gate = None
    chat_model = None
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # quiet
        pass

    def _forward(self):
        length = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(length) if length else b""
        body = self._no_thinking(body)
        started = time.time()
        kind = self.gate.kind(self.path)
        # --chat-upstream: an engine's own chat calls go to a second server running the same
        # model, so two engines' runs stop queueing on one GPU. Judges stay on --upstream, so
        # every engine is judged by the same server.
        tag = self.headers.get("x-m1-tag") or self.tag
        up, gate = self.upstream, self.gate
        if kind == "chat" and self.chat_upstream is not None and not tag.endswith(":judge"):
            up, gate = self.chat_upstream, self.chat_gate
            body = self._rename_model(body)
        headers = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "x-m1-tag", "content-length")}
        headers["Host"] = up.netloc
        headers["Content-Length"] = str(len(body))
        self._upstream_used = up.netloc
        gate.acquire(kind)
        try:
            # An embedding takes well under a second; one that has not answered in two minutes is
            # a hung upstream call, and holding the gate for it stalls every caller.
            upstream_timeout = 120 if kind == "embed" else 300
            for attempt in range(RETRIES):
                conn = http.client.HTTPConnection(up.hostname, up.port, timeout=upstream_timeout)
                try:
                    conn.request(self.command, self.path, body=body, headers=headers)
                    resp = conn.getresponse()
                except (TimeoutError, OSError):
                    conn.close()
                    time.sleep(3.0 * (attempt + 1))
                    continue
                if resp.status < 500 and not (resp.status == 400 and attempt < RETRIES - 1 and self._terminated(resp)):
                    break
                if resp.status >= 500:
                    resp.read()
                conn.close()
                time.sleep(3.0 * (attempt + 1))  # the model reloads in a few seconds
            else:
                conn = http.client.HTTPConnection(up.hostname, up.port, timeout=upstream_timeout)
                conn.request(self.command, self.path, body=body, headers=headers)
                resp = conn.getresponse()
            self._stream_back(resp, conn, body, started)
        finally:
            gate.release()

    def _rename_model(self, body):
        if not self.chat_model:
            return body
        try:
            payload = json.loads(body)
        except Exception:
            return body
        payload["model"] = self.chat_model
        return json.dumps(payload).encode()

    def _no_thinking(self, body):
        """Every chat completion goes out with `reasoning_effort: none`. The Qwen3.6 model on the
        LM Studio host thinks by default and ignores `chat_template_kwargs.enable_thinking`;
        thinking tokens ate the output budget of structured-output calls (Honcho's deriver got
        empty payloads) and multiplied latency. One switch for every engine keeps the comparison
        on the same model behaviour."""
        if "/chat/completions" not in self.path or not body:
            return body
        try:
            payload = json.loads(body)
        except Exception:
            return body
        if not isinstance(payload, dict) or "reasoning_effort" in payload:
            return body
        payload["reasoning_effort"] = "none"
        return json.dumps(payload).encode()

    def _terminated(self, resp):
        """LM Studio's `{"error": "terminated"}` is a load shed, not a bad request; read and retry."""
        peek = resp.read()
        self._peeked = peek
        return b"terminated" in peek or b"unloaded" in peek or b"Failed to load model" in peek

    def _stream_back(self, resp, conn, body, started):
        self.send_response(resp.status)
        chunked = (resp.getheader("transfer-encoding") or "").lower() == "chunked"
        for k, v in resp.getheaders():
            if k.lower() in ("transfer-encoding", "connection", "content-length"):
                continue
            self.send_header(k, v)
        collected = b""
        peeked = getattr(self, "_peeked", None)
        self._peeked = None
        if peeked is not None and resp.status == 400:
            self.send_header("Content-Length", str(len(peeked)))
            self.end_headers()
            self.wfile.write(peeked)
            self.wfile.flush()
            conn.close()
            self._record(body, peeked, resp.status, time.time() - started)
            return
        if chunked or "text/event-stream" in (resp.getheader("content-type") or ""):
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                collected += chunk
                self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
            self.wfile.write(b"0\r\n\r\n")
        else:
            collected = resp.read()
            self.send_header("Content-Length", str(len(collected)))
            self.end_headers()
            self.wfile.write(collected)
        self.wfile.flush()
        conn.close()
        self._record(body, collected, resp.status, time.time() - started)

    def _record(self, body, collected, status, seconds):
        model, usage = None, None
        try:
            model = json.loads(body or b"{}").get("model")
        except Exception:
            pass
        try:
            if collected.startswith(b"data:") or b"\ndata:" in collected:
                for line in collected.split(b"\n"):
                    if line.startswith(b"data:") and b"usage" in line:
                        payload = json.loads(line[5:].strip())
                        if payload.get("usage"):
                            usage = payload["usage"]
            else:
                usage = json.loads(collected).get("usage")
        except Exception:
            pass
        row = {
            "at": time.time(), "tag": self.headers.get("x-m1-tag") or self.tag, "path": self.path,
            "model": model, "status": status, "seconds": round(seconds, 3),
            "upstream": getattr(self, "_upstream_used", None),
            "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "completion_tokens": (usage or {}).get("completion_tokens"),
        }
        with LOCK:
            self.log.write(json.dumps(row) + "\n")
            self.log.flush()
            if status != 200 and self.errors is not None:
                self.errors.write(json.dumps({**row, "request": body[:4000].decode(errors="replace"),
                                              "response": collected[:1000].decode(errors="replace")}) + "\n")
                self.errors.flush()

    do_POST = _forward
    do_GET = _forward


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--upstream", default="http://100.85.201.58:1234")
    parser.add_argument("--port", type=int, default=1235)
    parser.add_argument("--log", required=True)
    parser.add_argument("--tag", default="")
    parser.add_argument("--chat-parallel", type=int, default=2)
    parser.add_argument("--embed-parallel", type=int, default=4)
    parser.add_argument("--chat-upstream", help="send chat completions (not judges) here instead")
    parser.add_argument("--chat-model", help="the model name on --chat-upstream")
    args = parser.parse_args()
    Proxy.upstream = urllib.parse.urlparse(args.upstream)
    Proxy.log = open(args.log, "a")
    Proxy.errors = open(args.log.rsplit(".", 1)[0] + "_errors.jsonl", "a")
    Proxy.tag = args.tag
    Proxy.gate = Gate(args.chat_parallel, args.embed_parallel)
    if args.chat_upstream:
        Proxy.chat_upstream = urllib.parse.urlparse(args.chat_upstream)
        Proxy.chat_gate = Gate(args.chat_parallel, args.embed_parallel)
        Proxy.chat_model = args.chat_model
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Proxy)
    print(f"proxy :{args.port} -> {args.upstream}, log {args.log}", file=sys.stderr, flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run every example in `examples/`, and say which ones did not run and why.

`examples/README.md` says: "Nothing runs them automatically -- they need frameworks, keys and
PDFs that a CI job should not -- so this table is the contract: if one stops working, it is a
bug, not an expected casualty." The table was the contract and nothing enforced it. Seventeen
Python examples, ten of them the framework integrations this project's roadmap says must be
"validated with real cases", and the only thing standing between them and rot was somebody
remembering to run them by hand.

This is what enforces it. The design is one property:

    **A skip is named and counted. A precondition that holds and an example that then fails
    is an error.**

That is the whole difference between this and a script that runs what it can. A runner that
quietly skips is worse than no runner, because the green line at the end says "examples: ok"
either way. So every example is one row in `EXAMPLES` below, carrying the imports it needs,
whether it wants a key, and any other precondition; anything unmet prints as a skip with the
reason next to it, and the summary counts skips separately from passes. `--strict` turns any
skip into a failure, which is what a machine that was supposed to have the dependencies wants.

Usage:

    scripts/run_examples.py                      against a server this script starts
    DXMT_URL=http://host:9999 scripts/run_examples.py --no-server
    scripts/run_examples.py --strict             a skip is a failure (CI with full deps)
    scripts/run_examples.py --only rag_minimal   one example, by substring
    scripts/run_examples.py --list               what would run, and what would not

`OPENAI_API_KEY` is read from the environment and never printed. Examples that want a key are
skipped without one rather than failing, because the key is not this repository's to have.
"""

import argparse
import importlib.util
import os
import pathlib
import signal
import socket
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
BIN = ROOT / "target" / "debug" / "vectordb"
#: The client packages, so an example imports the SDK from *this* checkout.
#:
#: Not a convenience. On the machine this was written, `import dxmt` resolved to a different
#: clone entirely, an older one with no `Client.health_check`, and `basic_crud.py` reported
#: "no server at 127.0.0.1:NNNNN (\'DXMTClient\' object has no attribute \'health_check\')" --
#: an API mismatch with someone else\'s code, wearing the costume of a connection problem. A
#: runner that inherits that is worse than no runner: it goes green against code that is not in
#: this repository. `check_sdk_is_ours` below is what makes the pinning checkable rather than
#: hoped for.
CLIENTS = [
    ROOT / "clients" / "dxmt-python-sdk",
    ROOT / "clients" / "langchain-dxmt",
    ROOT / "clients" / "llama-index-vector-stores-dxmt",
]
#: The port the examples assume. Eleven of them write `port=9999` into a constructor, so the
#: runner cannot pick a free one: it would leave those failing for a reason that is not theirs.
#: Occupied means stop, with the reason, rather than start somewhere else.
PORT = int(os.environ.get("DXMT_EXAMPLES_PORT", "9999"))


def load_dotenv():
    """Read `vectordb/.env` into the environment, the way the server already does.

    `AppConfig::load` pulls this file in through the `dotenv` crate, so the server sees the
    keys without anyone exporting anything. Python did not, which meant the examples and the
    server disagreed about what was configured -- and an example that wants a key would skip
    while the server it is talking to has one. Same file, one place to edit.

    Values already in the environment win, so `OPENAI_API_KEY=… scripts/run_examples.py`
    still overrides the file.
    """
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def resolve_provider():
    """Point the OpenAI-compatible clients at whichever provider `.env` selects.

    Every client the examples use -- the `openai` SDK, `langchain_openai`, mem0's embedder,
    llama-index -- reads `OPENAI_API_KEY` and `OPENAI_BASE_URL`. Setting those two is
    therefore the whole switch, and no example needs to know which provider it is talking to.

    What an example *does* need to know is the model name, because those are deployment names
    and do not carry across: this Azure resource serves `gpt-4.1-mini` and answers
    `DeploymentNotFound` for `gpt-4o-mini`. `DXMT_CHAT_MODEL`, `DXMT_EMBED_MODEL` and
    `DXMT_EMBED_DIMS` carry them, and each example falls back to its old hardcoded value when
    they are unset, so nothing breaks for someone who just has an `OPENAI_API_KEY`.

    Returns a one-line description for the run header, or None when nothing is configured.
    """
    provider = os.environ.get("DXMT_LLM_PROVIDER", "").strip().lower()
    if provider == "azure":
        key, base = os.environ.get("AZURE_OPENAI_KEY"), os.environ.get("AZURE_OPENAI_ENDPOINT")
    elif provider == "gemini":
        key, base = os.environ.get("GEMINI_API_KEY"), os.environ.get("GEMINI_BASE_URL")
    elif provider in ("openai", ""):
        key, base = os.environ.get("OPENAI_API_KEY"), os.environ.get("OPENAI_BASE_URL")
        provider = "openai"
    else:
        sys.exit(f"DXMT_LLM_PROVIDER={provider!r} is not one of azure, gemini, openai")

    if not key:
        return None
    os.environ["OPENAI_API_KEY"] = key
    if base:
        os.environ["OPENAI_BASE_URL"] = base.rstrip("/")
    where = f" at {os.environ['OPENAI_BASE_URL']}" if base else ""
    default = "(the example's own default)"
    chat = os.environ.get("DXMT_CHAT_MODEL", default)
    embed = os.environ.get("DXMT_EMBED_MODEL", default)
    models = f"chat {chat}, embeddings {embed}"
    return f"provider {provider}{where} -- {models}"


class Example:
    """One example, and everything that has to be true before running it is meaningful."""

    def __init__(self, name, imports=(), key=False, feature=None, needs=None, timeout=300,
                 args=()):
        self.name = name
        self.imports = imports          # python modules that must import
        self.key = key                  # needs OPENAI_API_KEY
        self.feature = feature          # cargo feature the server must have been built with
        self.needs = needs              # (description, callable -> bool)
        self.timeout = timeout
        self.args = list(args)

    def blocked_by(self, features):
        missing = [m for m in self.imports if importlib.util.find_spec(m) is None]
        if missing:
            return "no " + " ".join(missing)
        if self.key and not os.environ.get("OPENAI_API_KEY"):
            return "no OPENAI_API_KEY"
        if self.feature and self.feature not in features:
            return f"server built without {self.feature}"
        if self.needs and not self.needs[1]():
            return self.needs[0]
        return None


def a_pdf_exists():
    p = os.environ.get("DEMO_PDF_PATH")
    if p and pathlib.Path(p).is_file():
        return True
    return any((ROOT / "pdf").glob("*.pdf")) if (ROOT / "pdf").is_dir() else False


def a_dataset_exists():
    p = os.environ.get("DXMT_DATASET_DIR")
    return bool(p) and pathlib.Path(p).is_dir()


#: Every example, with what it needs. Kept in step with `examples/README.md` -- the test
#: `examples_are_all_listed` below fails if a file appears in one and not the other, which is
#: the only reason the two stay honest.
CATALOGUE = [
    Example("rag_minimal.py", imports=("sentence_transformers",)),
    Example("basic_crud.py"),
    Example("load_demo_corpus.py"),
    Example("rag_eval.py", imports=("sentence_transformers",), timeout=900),
    Example("full_text_search_example.py", feature="text-bm25"),
    Example("standard_dataset_recall.py", imports=("pyarrow",),
            needs=("no dataset; set DXMT_DATASET_DIR", a_dataset_exists), timeout=900),
    Example("pdf_rag_simple.py", imports=("sentence_transformers", "PyPDF2"),
            needs=("no PDF; set DEMO_PDF_PATH", a_pdf_exists)),
    Example("pdf_rag_demo.py", imports=("sentence_transformers", "PyPDF2"),
            needs=("no PDF; set DEMO_PDF_PATH", a_pdf_exists)),
    Example("query_pdf_docs.py", imports=("sentence_transformers",),
            needs=("no PDF; set DEMO_PDF_PATH", a_pdf_exists)),
    Example("langgraph_memory_agent.py", imports=("langgraph",)),
    Example("bisheng_integration.py", imports=("langchain_core", "langchain_openai"), key=True),
    Example("dxmt_haystack_rag.py", imports=("haystack", "sentence_transformers"), key=True),
    Example("crewai_dxmt_simple.py", imports=("crewai",), key=True),
    Example("crewai_dxmt_integration.py", imports=("crewai",), key=True),
    # mem0 runs every memory through an LLM to decide what is worth keeping, so these are
    # dominated by round trips rather than by anything this database does: 195 s against
    # OpenAI, more against Azure. The default 300 s cut the first one off mid-run.
    Example("mem0_integration.py", imports=("mem0",), key=True, timeout=1200),
    Example("dxmt_mem0_simple.py", imports=("mem0", "dotenv"), key=True, timeout=600),
    Example("metagpt_integration.py", imports=("llama_index",), key=True),
    # Needs the key in the *server's* environment as well as here: the table names a model
    # and the server does the embedding. The runner's server inherits this process's
    # environment, so one export covers both.
    Example("generative_benchmark.py", imports=("openai",), key=True, timeout=1800,
            args=("--limit", "40")),
]


def port_is_free(port):
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def check_sdk_is_ours(env):
    """Fail loudly if `import dxmt` finds a checkout that is not this one."""
    r = subprocess.run(
        [sys.executable, "-c", "import dxmt, sys; sys.stdout.write(dxmt.__file__)"],
        env=env, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if r.returncode != 0:
        sys.exit("`import dxmt` failed even with the client packages on the path:\n"
                 + r.stderr.decode("utf-8", "replace")[-800:])
    where = pathlib.Path(r.stdout.decode().strip()).resolve()
    if not str(where).startswith(str(ROOT.resolve())):
        sys.exit(f"`import dxmt` resolves to {where}, which is outside this checkout "
                 f"({ROOT}). Every example that uses the SDK would be validating that copy. "
                 f"Uninstall it (`pip uninstall dxmt`) or clear PYTHONPATH.")
    return where


def start_server(features, wal, port):
    env = dict(
        os.environ,
        DXMT_WAL_DIR=str(wal / "wal"),
        DXMT_BACKUP_ROOT=str(wal),
        DXMT_SNAPSHOT_INTERVAL_SECS="0",
        DXMT_EVICTION_INTERVAL_SECS="0",
        RUST_LOG="warn",
    )
    env.pop("DXMT_API_KEY", None)
    log = open(wal / "server.log", "w")
    proc = subprocess.Popen(
        [str(BIN), "serve", "--host", "127.0.0.1", "--port", str(port)],
        env=env, stdout=log, stderr=subprocess.STDOUT,
    )
    import urllib.request

    for _ in range(240):
        if proc.poll() is not None:
            sys.exit(f"the server exited on start; see {wal / 'server.log'}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/live", timeout=1)
            return proc, f"http://127.0.0.1:{port}"
        except Exception:
            time.sleep(0.25)
    proc.send_signal(signal.SIGKILL)
    sys.exit("the server never answered")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-server", action="store_true", help="use DXMT_URL, do not start one")
    ap.add_argument("--strict", action="store_true", help="a skip is a failure")
    ap.add_argument("--only", default=None, help="substring of the example name")
    ap.add_argument("--list", action="store_true", help="print the plan and exit")
    ap.add_argument("--features", default=os.environ.get("DXMT_FEATURES", ""),
                    help="cargo features the server was built with")
    args = ap.parse_args()

    load_dotenv()
    provider = resolve_provider()

    features = {f.strip() for f in args.features.replace(" ", ",").split(",") if f.strip()}
    plan = [e for e in CATALOGUE if not args.only or args.only in e.name]
    if not plan:
        sys.exit(f"no example matches {args.only!r}")

    missing = sorted({p.name for p in EXAMPLES.glob("*.py")} - {e.name for e in CATALOGUE})
    if missing:
        sys.exit("these examples are in examples/ and not in this script's catalogue, so "
                 "nothing runs them:\n  " + "\n  ".join(missing))

    if args.list:
        for e in plan:
            why = e.blocked_by(features)
            print(f"  {'SKIP' if why else 'RUN '}  {e.name:<34} {why or ''}")
        return 0

    server, url = None, os.environ.get("DXMT_URL", "http://127.0.0.1:9999")
    tmp = None
    if not args.no_server:
        if not BIN.exists():
            sys.exit(f"{BIN} does not exist; run `cargo build --bin vectordb` first")
        import tempfile

        if not port_is_free(PORT):
            sys.exit(f"port {PORT} is busy, and the examples hardcode it. Stop what is on it, "
                     f"or point this at a server you already have: "
                     f"DXMT_URL=http://127.0.0.1:{PORT} scripts/run_examples.py --no-server")
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="dxmt_examples_"))
        server, url = start_server(features, tmp, PORT)
        print(f"server at {url}, wal {tmp}")

    env = dict(os.environ, DXMT_URL=url,
               DXMT_HOST=url.split("//")[1].split(":")[0],
               DXMT_PORT=url.rsplit(":", 1)[1],
               PYTHONPATH=os.pathsep.join(
                   [str(p) for p in CLIENTS] + [os.environ.get("PYTHONPATH", "")]).rstrip(
                       os.pathsep))
    sdk = check_sdk_is_ours(env)
    print(f"dxmt SDK from {sdk.parent}")
    print(provider or "no LLM provider configured; the examples that need one will skip")
    passed, failed, skipped = [], [], []
    try:
        for e in plan:
            why = e.blocked_by(features)
            if why:
                skipped.append((e.name, why))
                print(f"SKIP  {e.name:<34} {why}")
                continue
            t0 = time.time()
            try:
                r = subprocess.run(
                    [sys.executable, str(EXAMPLES / e.name), *e.args],
                    cwd=ROOT, env=env, timeout=e.timeout,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                )
            except subprocess.TimeoutExpired as expired:
                # One example running long is a failure of that example, not of the run. The
                # exception used to propagate and take the remaining examples with it, so a
                # single slow provider hid everything after it in the catalogue -- which is
                # the opposite of what this script is for.
                dt = time.time() - t0
                tail = (expired.stdout or b"").decode("utf-8", "replace")
                failed.append((e.name, f"timed out after {e.timeout}s", tail))
                print(f"FAIL  {e.name:<34} {dt:5.1f}s  timed out")
                continue
            dt = time.time() - t0
            if r.returncode == 0:
                passed.append(e.name)
                print(f"ok    {e.name:<34} {dt:5.1f}s")
            else:
                failed.append((e.name, r.returncode, r.stdout.decode("utf-8", "replace")))
                print(f"FAIL  {e.name:<34} {dt:5.1f}s  exit {r.returncode}")
    finally:
        if server and server.poll() is None:
            server.send_signal(signal.SIGKILL)
            server.wait(timeout=30)

    for name, code, out in failed:
        print(f"\n--- {name}: {code} ---" if isinstance(code, str)
              else f"\n--- {name} exited {code} ---")
        print("\n".join(out.splitlines()[-25:]))

    print(f"\nexamples: {len(passed)} ran, {len(failed)} failed, {len(skipped)} skipped")
    if skipped:
        print("skipped, and why:")
        for name, why in skipped:
            print(f"  {name:<34} {why}")
    if args.strict and skipped:
        print("\n--strict: a skip is a failure here")
    return 1 if failed or (args.strict and skipped) else 0


if __name__ == "__main__":
    sys.exit(main())

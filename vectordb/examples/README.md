# Examples

Every file here uses the library and states what it needs. `scripts/run_examples.py` runs
them, starting a server itself, and **names every one it could not run and why**:

    cd vectordb && scripts/run_examples.py
    scripts/run_examples.py --list                 the plan, without running anything
    scripts/run_examples.py --only rag_minimal     one of them
    scripts/run_examples.py --strict               a skip is a failure

This paragraph used to say "nothing runs them automatically ... so this table is the contract:
if one stops working, it is a bug, not an expected casualty". The table was the contract and
nothing enforced it, which is a promise with no mechanism behind it. When the runner was
written, five of the seventeen were broken -- four framework integrations against APIs that
had moved on, and one that could not run unattended at all because it stopped to ask a
question. `import dxmt` was resolving to a different checkout entirely, so anything that
touched the SDK had been validated against code that is not in this repository.

The skips are the point as much as the runs. An example that needs `crewai`, a PDF or a key
this machine does not have is reported by name with the reason, and counted separately; a
runner that skips quietly is worse than none, because the line at the end says "ok" either
way. `--strict` makes a skip a failure, for a machine that was supposed to have everything.

All of them need a running server, and **they assume port 9999** -- eleven write it into a
constructor -- which is why the runner refuses to start when that port is busy rather than
picking a free one and letting half of them fail for a reason that is not theirs:

    cd vectordb && cargo run --release -- serve       # port 9999

Six read `DXMT_URL` or `DXMT_HOST`/`DXMT_PORT` and will follow a server elsewhere.

## What each one needs

| example | pip | OpenAI key | other |
| --- | --- | --- | --- |
| `rag_minimal.py` | `httpx`, optionally `sentence-transformers` | — | works offline; see below |
| `standard_dataset_recall.py` | `pyarrow httpx` | — | a VectorDBBench dataset on disk |
| `full_text_search_example.py` | — | — | server built `--features text-bm25` |
| `query_pdf_docs.py` | `sentence-transformers` | — | run `pdf_rag_simple.py` first |
| `pdf_rag_simple.py` | `sentence-transformers PyPDF2` | — | a PDF, see below |
| `pdf_rag_demo.py` | `sentence-transformers PyPDF2` | optional | a PDF; without a key it retrieves but does not generate |
| `langgraph_memory_agent.py` | `langgraph` | optional | the OpenAI section skips itself without one |
| `bisheng_integration.py` | `langchain-core langchain-openai` | yes | |
| `dxmt_haystack_rag.py` | `haystack-ai sentence-transformers` | yes | |
| `crewai_dxmt_simple.py` | `crewai crewai-tools` | yes | |
| `crewai_dxmt_integration.py` | `crewai crewai-tools` | yes | falls back to random vectors without `sentence-transformers` |
| `mem0_integration.py` | `mem0ai` | yes | mem0 configures an OpenAI embedder and LLM |
| `dxmt_mem0_simple.py` | `mem0ai python-dotenv` | yes | |
| `metagpt_integration.py` | `llama-index` | yes | |
| `generative_benchmark.py` | `openai` | yes | the key must also be in the **server's** environment |

`OPENAI_API_KEY` comes from the environment. Never put a key in a file here -- keys have had to
be scrubbed from this tree once already.

## RAG without a key

`rag_minimal.py` runs the retrieval half of RAG end to end -- chunk, embed, store, retrieve,
ground the answer in what came back -- with no API key and no network. It embeds locally with
`all-MiniLM-L6-v2` and answers by quoting the passage it retrieved.

    pip install sentence-transformers
    python3 examples/rag_minimal.py
    python3 examples/rag_minimal.py "how does the graph get built"

Without `sentence-transformers` it falls back to deterministic hash vectors and **says so
loudly**: the plumbing runs, the retrieval is meaningless, and the scores drop from about 0.5 to
about 0.05 so the difference is visible rather than merely stated. A RAG demo that silently
retrieves nonsense under a confident answer is worse than one that refuses.

Generation is deliberately absent. An LLM would need a key and would put the interesting
failure -- did retrieval find the right passage? -- behind a model's ability to paper over it.

Two things learned writing it, both in the file's comments. The first corpus had a passage
mentioning disk, memory *and* the WAL, and it won two questions it should have lost by touching
every topic a little; the retrieval was fine, the corpus was not. And a question phrased with
words no passage uses still retrieves the wrong thing -- which is why the example ends by asking
the reader to check the passage against the question rather than telling them it worked.

## Recall on a standard dataset

`standard_dataset_recall.py` loads a VectorDBBench dataset -- Cohere 100k x 768 by default --
and measures recall against the ground truth that ships with it. No key, no model, no network:
the embeddings are in the file.

    DXMT_DATASET_DIR=/path/to/vectordb_bench/dataset python3 examples/standard_dataset_recall.py
    DXMT_ROWS=20000 ... # a quicker look

Measured on 20,000 rows: loaded at 2,347 records/s, **recall@10 1.0000** over 90 queries,
serial p50 1.4 ms and p99 5.2 ms. Recall of 1.0 is expected at that size -- 65,536 rows is where
a table quantizes itself, so this is an exact graph. The point of this example is the *real
distribution and published ground truth*, which the generated corpus in `tests/api/property/`
cannot offer; the point of that one is that it runs in CI on every change, which this cannot.

If the dataset is not on disk it says so and names how to get it, rather than failing inside
parquet.

## The PDF ones

`pdf_rag_simple.py` and `pdf_rag_demo.py` read `DEMO_PDF_PATH`, defaulting to `pdf/sample.pdf`.
That directory is not in the repository, so out of the box they print "PDF 文件不存在" and exit
cleanly. Point them at any PDF:

    DEMO_PDF_PATH=~/some-paper.pdf python3 examples/pdf_rag_simple.py
    python3 examples/query_pdf_docs.py "what does it say about X"

They are the only examples that run with nothing but pip installs -- no key, no framework, no
network.

## Embeddings

None of these get embeddings from the database; the server stores vectors the caller supplies.
`clients/dxmt-python-sdk/examples/common_utils.py` has an `embed()` that uses `all-MiniLM-L6-v2` when
`sentence-transformers` is installed and falls back to a deterministic md5-derived vector when
it is not, which is what makes an example runnable offline. The fallback vectors are not
semantically meaningful -- they exercise the plumbing, not the retrieval quality.

## Rust examples

Declared in `Cargo.toml`, so `cargo clippy --all-targets` compiles them and a break shows up in
CI rather than the next time somebody opens the file:

| example | needs |
| --- | --- |
| `quickstart.rs` | nothing -- create, insert, search, filter, list, delete, drop, embedded |
| `mcp_demo.rs` | nothing |
| `dxql_demo.rs` | nothing |
| `hnsw_build_scaling.rs` | nothing |
| `chinese_tokenizer_demo.rs` | nothing |
| `adaptive_index_demo.rs` | `--features adaptive-index` |
| `full_text_search_example.rs` | `--features text-bm25` |

    cargo run --example quickstart

`quickstart.rs` is the shortest answer to "how do I put vectors in and get them out". It uses
the crate directly rather than HTTP, which is the embedded shape -- there is no Rust client SDK
in this repository; the four SDKs are Python, JavaScript, LangChain and LlamaIndex.

## Tutorials and other files

`01_getting_started.md`, `02_embeddings.md` and `filter_example.md` are prose walkthroughs and
cross-reference each other. `vectordbbench_example.sh` shows how to point VectorDBBench at a
running server. `nginx-jwt.conf` is a sample reverse-proxy configuration; nothing in the repo
refers to it.

## What is not here

Examples that did not run were removed rather than left to rot: four AutoGen examples importing
a module that exists in no package, and a multimodal example calling routes the server does not
serve. They were not abandoned by accident -- each one needs work this repository has not done,
and an example that cannot run is a promise the code does not keep.

## Two ways to score retrieval, and what each one answers

`rag_eval.py` and `generative_benchmark.py` look like the same example and are not.

`rag_eval.py` scores the two RAG metrics a database is responsible for -- context precision
and context recall -- against **hand-written questions with a known gold passage**, over this
repository's own documentation. No LLM, no key, no judge model, so the number is the same on
every run. It is the regression test of the two.

`generative_benchmark.py` works on **any corpus**, including yours, because it does not need
labels: an LLM writes the query each passage answers, and the pair is the label. That buys
generality and costs reproducibility -- the queries are generated, so read them before
trusting the score. `--show` prints them next to their passages.

The reason to have the second one is a question the first cannot ask. A perfect index still
retrieves badly if the embedding model puts the wrong passage first, and those are different
problems. Every query runs through the graph and through `search_mode: "exact"`, and each
miss is attributed to one or the other. On this repository's documentation, 165 generated
queries, `text-embedding-3-small`: recall@1 0.5333, recall@10 0.8848, and **all 77 misses
were the embedding -- the index placed none of them wrong**. That is the shape of answer this
example exists to produce, and no amount of index tuning would have moved it.

The method is Chroma's `sample_apps/generative_benchmarking` (Apache 2.0 -- the method, not
the code). The attribution half is ours, and is only possible because of `search_mode:
"exact"`.

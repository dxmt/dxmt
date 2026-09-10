# Examples

Every file here uses the library and states what it needs. Nothing runs them automatically --
they need frameworks, keys and PDFs that a CI job should not -- so this table is the contract:
if one stops working, it is a bug, not an expected casualty.

All of them need a running server:

    cd vectordb && cargo run --release -- serve       # standard mode, port 9999, NOT --simple

`DXMT_HOST` / `DXMT_PORT` are not read by these; they connect to localhost:9999.

## What each one needs

| example | pip | OpenAI key | other |
| --- | --- | --- | --- |
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

`OPENAI_API_KEY` comes from the environment. Never put a key in a file here -- keys have had to
be scrubbed from this tree once already.

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
`dxmt-python-sdk/examples/common_utils.py` has an `embed()` that uses `all-MiniLM-L6-v2` when
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

`planning/examples/` holds what was moved out and why: four AutoGen examples importing a module
that exists in no package, and a multimodal example calling routes the server does not serve.
Read the READMEs there before assuming either was abandoned by accident.

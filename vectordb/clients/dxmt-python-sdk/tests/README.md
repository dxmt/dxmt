# Tests

Two kinds, told apart by the `integration` marker (`pytest.ini` sets `--strict-markers`, so a
misspelled marker is an error rather than a silently unmarked test).

```bash
pip install -e ".[dev]"

pytest -m "not integration"      # mocked; no server
pytest                           # everything; needs a server on localhost:9999
DXMT_TEST_HOST=127.0.0.1 DXMT_TEST_PORT=19999 pytest tests/test_integration.py
```

A server that was started with `DXMT_API_KEY` needs the same value in the environment of the
test run; every live suite reads it from `DXMT_API_KEY` and passes it to the client. The export /
import round trip in `test_integration.py` runs only when `DXMT_BACKUP_ROOT` names the server's
backup root, and the hybrid and full-text tests skip on a server built without `text-bm25`
(the default build), because the server refuses those requests rather than degrading.

## What is where

| File | Needs | Covers |
| --- | --- | --- |
| `test_basic.py`, `test_client.py`, `test_client_complete.py` | nothing | construction, request bodies, filter conversion, sync/async parity |
| `test_filters.py`, `test_facets.py`, `test_dxql_client.py`, `test_scroll_upload.py`, `test_api_key.py`, `test_server_embedding.py` | nothing | one feature's request and response handling each |
| `test_integration.py` | server | every route the client wraps, judged by its effect on the server |
| `test_langchain_vectorstore.py`, `test_bisheng_integration.py` | `langchain-core` (mocked client); the second also `bisheng` | the `dxmt.langchain_vectorstore` adapter |
| `test_langgraph_store.py`, `test_langgraph_store_integration.py` | server + `langgraph` | the `BaseStore` implementation |
| `test_haystack_integration.py`, `test_crewai_integration.py`, `test_metagpt_vector_store.py`, `test_mem0_adapter.py` | the framework | each adapter, mocked where the framework is heavy |

CI (`.github/workflows/ci.yml`, job `sdk-live`) runs `test_integration.py` and the LangGraph
suites against a freshly built server, once keyless and once with `DXMT_API_KEY`.

## Writing a live test

Assert on what the server now holds -- a count, a record read back, a refused filter -- and
not on "the call returned". A `delete()` that raised on every call stayed green for months in
one adapter because its test only checked that something was returned. Make a fresh table per
module with a random suffix and delete it in a `finally`, so a failed run leaves nothing
behind on a shared server.

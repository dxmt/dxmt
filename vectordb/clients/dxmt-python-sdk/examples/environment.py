"""Where an example finds the server, and what it does when this machine cannot run it.

**The server.** Every example connects to `DXMT_HOST`:`DXMT_PORT` (default `localhost:9999`)
and sends `DXMT_API_KEY` when it is set, so the same file runs against a server started with a
key or on another port without editing it::

    DXMT_PORT=19999 python basic_usage.py

**Skipping.** An example that needs a library or a key this machine does not have calls
`skip()`, which says what is missing and exits 77. The examples used to print the reason and
`return` from `main()`, which exits 0 -- so a run with no OpenAI key, or with
`sentence-transformers` missing, looked exactly like a run that worked. 77 is the conventional
"skipped" status (automake, meson, ctest) and nothing else here uses it, so
`tests/test_examples.py` can tell "could not run here" from both "worked" and "failed".
"""

import os
import sys
from typing import Any

SKIPPED = 77


def client_kwargs() -> dict[str, Any]:
    """`host`, `port` and `api_key` for `Client(...)`, from the environment."""
    return {
        "host": os.environ.get("DXMT_HOST", "localhost"),
        "port": int(os.environ.get("DXMT_PORT", "9999")),
        "api_key": os.environ.get("DXMT_API_KEY") or None,
    }


def skip(reason: str) -> None:
    """Exit with `SKIPPED`, naming what is missing and how to get it."""
    print(f"SKIPPED: {reason}", file=sys.stderr)
    sys.exit(SKIPPED)

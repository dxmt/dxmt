#!/bin/bash
# Install the DXMT client into an installed VectorDBBench, and register it in its DB enum.
#
# Two things have to happen and only one of them is a file copy. VectorDBBench imports
# clients from its own package, so `dxmt/` has to be placed inside it -- and `DB.DXMT` has
# to exist in `backend/clients/__init__.py`, or `DB` has no member to dispatch on and every
# run fails with a KeyError that says nothing about the real cause.
#
# On the first benchmark box that registration was made by hand and lived only in a
# virtualenv. Setting up a second box is what found it: the client copied cleanly and the
# suite still could not see DXMT. Doing it as a script means the next box is one command.
#
# Idempotent: re-running against an already-patched tree changes nothing.
#
# Usage: install.sh /path/to/venv
set -u
VENV="${1:?usage: install.sh <venv path>}"
PY="$VENV/bin/python"
[ -x "$PY" ] || { echo "no python at $PY"; exit 1; }

SITE=$("$PY" -c "import vectordb_bench, os; print(os.path.dirname(vectordb_bench.__file__))") || {
  echo "vectordb_bench is not installed in $VENV"; exit 1; }
HERE=$(cd "$(dirname "$0")" && pwd)

mkdir -p "$SITE/backend/clients/dxmt"
cp "$HERE"/{dxmt.py,config.py,__init__.py} "$SITE/backend/clients/dxmt/" || exit 1
echo "client -> $SITE/backend/clients/dxmt"

"$PY" - "$SITE/backend/clients/__init__.py" <<'PATCH'
import io, re, sys

path = sys.argv[1]
src = io.open(path, encoding="utf-8").read()
before = src

# 1. The enum member. Anchored on Test rather than appended, so it lands inside the class.
if 'DXMT = "DXMT"' not in src:
    src = src.replace('    Test = "test"\n', '    Test = "test"\n    DXMT = "DXMT"\n', 1)


def insert_into_method(text, method, block, marker):
    """Put `block` before the first `if self == DB.…` arm inside `method`.

    Two earlier versions of this got the position wrong, both silently:

      * anchoring on a neighbouring arm's text (`if self == DB.Chroma:`) put all three
        blocks in the first method that happened to contain that string;
      * anchoring on "the first line after `def` ending in a colon" skipped the signature
        itself, because it ends in `# noqa: PLR0911, …` rather than the colon, and landed
        the block between `if self == DB.Milvus:` and its body.

    Anchoring on the first dispatch arm inside the method survives decorators, docstrings,
    multi-line signatures and trailing comments, none of which the other two did.
    """
    if marker in text:
        return text
    m = re.search(r"\n    def %s\b" % re.escape(method), text)
    if not m:
        raise SystemExit("method not found: %s" % method)
    arm = text.find("\n        if self == DB.", m.end())
    if arm == -1:
        raise SystemExit("no dispatch arm found in %s" % method)
    at = arm + 1  # just past the newline, at the arm's indentation
    return text[:at] + block + text[at:]


src = insert_into_method(
    src, "init_cls",
    "        if self == DB.DXMT:\n"
    "            from .dxmt.dxmt import DXMT\n\n"
    "            return DXMT\n\n",
    "from .dxmt.dxmt import DXMT",
)
src = insert_into_method(
    src, "config_cls",
    "        if self == DB.DXMT:\n"
    "            from .dxmt.config import DXMTConfig\n\n"
    "            return DXMTConfig\n\n",
    "from .dxmt.config import DXMTConfig",
)
src = insert_into_method(
    src, "case_config_cls",
    "        if self == DB.DXMT:\n"
    "            from .dxmt.config import DXMTAutoConfig, DXMTHNSWConfig, DXMTHNSWPQConfig\n\n"
    "            return {\n"
    "                IndexType.HNSW: DXMTHNSWConfig,\n"
    "                IndexType.AUTOINDEX: DXMTAutoConfig,\n"
    "            }.get(index_type, DXMTAutoConfig)\n\n",
    "IndexType.AUTOINDEX: DXMTAutoConfig",
)

if src != before:
    io.open(path, "w", encoding="utf-8").write(src)
    print("registered DB.DXMT in", path)
else:
    print("already registered, nothing to do")
PATCH

"$PY" -c "
from vectordb_bench.backend.clients import DB
assert 'DXMT' in DB.__members__, 'DB.DXMT missing after patch'
assert DB.DXMT.init_cls.__name__ == 'DXMT'
assert DB.DXMT.config_cls.__name__ == 'DXMTConfig'
print('verified: DB.DXMT resolves to', DB.DXMT.init_cls, DB.DXMT.config_cls)
"

"""Every example runs against a live server and exits the way it should.

The examples are the code most likely to be copied and the least likely to be run, and
nothing ran them: on 2026-10-06 a full pass found `memory_optimization_demo.py` calling
`batch_insert`, a method the client has never had in this repository's history, in all three
of its scenarios, each failure caught by a broad `except` and printed as a warning, the whole
run exiting 0. Four more examples exited 0 after printing that they could not run at all.

So each example is run as a user would run it, in its own process from `examples/`, and its
exit status is the verdict:

- **0** passes.
- **77** is `environment.skip()` -- a library or key this machine lacks -- and is a pytest
  skip carrying the example's own reason. An example that imports nothing outside the SDK has
  no reason to skip, and doing so fails (`test_an_example_with_no_extra_dependencies_never_skips`
  makes that a property of the source, not of a hand-kept list).
- **anything else** fails, with the tail of the output.

`OPENAI_API_KEY` is removed from the child's environment unless `DXMT_EXAMPLES_OPENAI=1`, so
the suite never spends money by accident. `PYTHONPATH` points at this checkout: `import dxmt`
otherwise finds whatever SDK is installed, which on a developer machine is often an older one.

Needs a server: `DXMT_TEST_HOST` / `DXMT_TEST_PORT`, as for `test_integration.py`.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

SDK_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = SDK_ROOT / "examples"
SKIPPED = 77

#: Helpers the examples import; not examples themselves.
NOT_EXAMPLES = {"environment.py", "common_utils.py"}

#: Libraries an example may need beyond the SDK. Importing any of these is a reason it may skip.
EXTRA_LIBRARIES = ("numpy", "sentence_transformers", "openai", "langgraph", "torch")

TIMEOUT = float(os.environ.get("DXMT_EXAMPLE_TIMEOUT", "300"))


def examples() -> list[Path]:
    return sorted(p for p in EXAMPLES.glob("*.py") if p.name not in NOT_EXAMPLES)


def needs_extra_library(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    pattern = r"^\s*(?:import|from)\s+(?:%s)\b" % "|".join(EXTRA_LIBRARIES)
    return re.search(pattern, source, re.M) is not None


def run(path: Path) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(SDK_ROOT), env.get("PYTHONPATH")]))
    env["DXMT_HOST"] = os.environ.get("DXMT_TEST_HOST", "localhost")
    env["DXMT_PORT"] = os.environ.get("DXMT_TEST_PORT", "9999")
    if os.environ.get("DXMT_EXAMPLES_OPENAI") != "1":
        env.pop("OPENAI_API_KEY", None)
    # The examples print emoji and Chinese; a C locale would fail them on the first print.
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return subprocess.run(
        [sys.executable, "-W", "ignore", path.name],
        cwd=EXAMPLES,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
    )


def tail(text: str, lines: int = 40) -> str:
    return "\n".join(text.strip().splitlines()[-lines:])


@pytest.mark.parametrize("path", examples(), ids=lambda p: p.stem)
def test_the_example_runs(path: Path) -> None:
    result = run(path)
    if result.returncode == SKIPPED:
        # The line `environment.skip()` wrote, not the stderr tail: torch's progress bars and
        # library deprecation warnings land there too.
        reasons = [l for l in result.stderr.splitlines() if l.startswith("SKIPPED:")]
        reason = reasons[-1][len("SKIPPED:"):].strip() if reasons else "no reason given"
        if not needs_extra_library(path):
            pytest.fail(
                f"{path.name} imports nothing outside the SDK and still skipped: {reason}"
            )
        pytest.skip(f"{path.name}: {reason}")
    assert result.returncode == 0, (
        f"{path.name} exited {result.returncode}\n"
        f"--- stdout (tail) ---\n{tail(result.stdout)}\n"
        f"--- stderr (tail) ---\n{tail(result.stderr)}"
    )


def test_there_are_examples_to_run() -> None:
    # A glob that matched nothing would make the parametrised test above vanish rather than
    # fail -- the same silence the suite is here to end.
    assert len(examples()) >= 20, [p.name for p in examples()]


def test_an_example_with_no_extra_dependencies_never_skips() -> None:
    # The classification `test_the_example_runs` relies on, checked on its own so a wrong
    # pattern shows up here and not as a mysterious pass: these need only the SDK.
    plain = {p.name for p in examples() if not needs_extra_library(p)}
    assert {"basic_usage.py", "async_usage.py", "memory_optimization_demo.py"} <= plain, plain
    assert "semantic_search_openai.py" not in plain

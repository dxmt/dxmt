"""The Bisheng adapter's surface.

Replaces `examples/verify_bisheng_integration.py`, which was a test wearing an example's
costume and had a flaw worth not repeating: when LangChain was absent it printed "Skipped" and
then `return False`, which its own summary counted as a failure. A clean machine running correct
code got a red report. `importorskip` is what that script was reaching for.
"""

import pytest

pytest.importorskip(
    "langchain_core", reason="optional LangChain dependency is not installed"
)

from dxmt import bisheng_integration  # noqa: E402
from dxmt.bisheng_integration import initialize_dxmt  # noqa: E402
from dxmt.langchain_vectorstore import DXMTVectorStore  # noqa: E402


def test_the_names_bisheng_imports_are_exported():
    """Bisheng imports these by name from its config. A rename that kept the class working
    would still break every deployment that names it in a YAML file."""
    assert callable(initialize_dxmt)
    assert hasattr(bisheng_integration, "initial_dxmt"), "the legacy alias is part of the API"
    assert bisheng_integration.initial_dxmt is initialize_dxmt


def test_initialize_dxmt_takes_the_signature_bisheng_calls():
    """Bisheng calls `initialize_dxmt(class_object, params, search_kwargs)` positionally."""
    import inspect

    params = list(inspect.signature(initialize_dxmt).parameters)
    assert params[:3] == ["class_object", "params", "search_kwargs"]


def test_the_vector_store_class_is_reachable_under_its_exported_name():
    """`DXMTVectorStore` is the name inside `dxmt.langchain_vectorstore`; the package exports
    it as `DXMTLangChainVectorStore`. Both are load-bearing -- the module-level name is what
    Bisheng's config points at -- so a rename of either breaks somebody."""
    import dxmt

    assert dxmt.DXMTLangChainVectorStore is DXMTVectorStore

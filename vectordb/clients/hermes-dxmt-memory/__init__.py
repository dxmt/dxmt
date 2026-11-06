"""hermes-dxmt-memory -- a DXMT-backed MemoryProvider plugin for Hermes Agent.

Hermes loads this directory as a plugin and calls ``register(ctx)``. The implementation
lives in the ``src`` subpackage; this module only re-exports the provider.
"""
from __future__ import annotations

if not __package__:
    # Loaded as a top-level module without a package (the tests do this from the repo
    # root). Manufacture a package rooted here so ``src`` and its relative imports resolve.
    import importlib
    import sys
    from pathlib import Path

    package_name = "hermes_dxmt_memory"
    package = sys.modules.setdefault(package_name, sys.modules[__name__])
    package.__path__ = [str(Path(__file__).parent)]
    DXMTMemoryProvider = importlib.import_module(f"{package_name}.src.provider").DXMTMemoryProvider
else:
    from .src.provider import DXMTMemoryProvider


EXTRACTION_TASK = "dxmt_extraction"


def register(ctx) -> None:
    """Register the DXMT memory provider with the Hermes plugin context.

    Fact extraction runs as the auxiliary task ``dxmt_extraction`` (its own ``auxiliary.<key>``
    config block), through the host-owned ``ctx.llm`` facade when the context offers one. The
    memory activation path hands providers a collector that forwards ``register_*`` calls but
    has no ``llm``; there the provider falls back to the auxiliary client directly.
    """
    provider = DXMTMemoryProvider()
    try:
        ctx.register_auxiliary_task(
            EXTRACTION_TASK,
            display_name="DXMT memory extraction",
            description="Extracts durable facts from a session for the DXMT memory provider.",
        )
    except Exception:  # an older Hermes, or a collector that cannot forward it
        pass
    try:
        provider.llm = ctx.llm
    except Exception:
        provider.llm = None
    ctx.register_memory_provider(provider)


__all__ = ["DXMTMemoryProvider", "register"]

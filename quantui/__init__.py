"""quantui — modular Textual TUI for multi-family model quantization.

IMPORTANT: this package is also imported (as a package) when the worker modules
are launched via ``python -m quantui.worker`` /
``python -m quantui.worker_ctq`` — including in the *separate* torch /
``convert_to_quant`` interpreter that may NOT have Textual installed. Therefore
this ``__init__`` MUST NOT import ``app`` (which pulls in Textual). Keep it
Textual-free. The TUI entry point is ``python -m quantui`` (see
``__main__``).
"""
from .quant_methods import (
    COMFY_FORMATS,
    METHODS,
    METHODS_BY_ID,
    Family,
)

# NTH-006: single source of truth for the semantic version. Bumped only via
# ``scripts/bump_version.py {major|minor|patch}`` (which also rotates CHANGELOG.md).
__version__ = "0.5.0"

__all__ = ["Family", "METHODS", "METHODS_BY_ID", "COMFY_FORMATS", "__version__"]

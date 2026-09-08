"""'List all methods' button removal (plan 2026-09-08-scifi-ui S1.2).

The button was redundant: the method picker modal (#pick_method / palette
`_palette_pick_method`) lists all 35 official methods interactively. The
button, its dispatch branch, and the orphaned `list_methods()` method are
removed together. These tests pin the absence.
"""

import os

import pytest
from textual.css.query import NoMatches

from quantui import app as appmod

_QUANTUI_DIR = os.path.join(os.path.dirname(__file__), "..", "quantui")


def _read(rel: str) -> str:
    with open(os.path.join(_QUANTUI_DIR, rel), encoding="utf-8") as fh:
        return fh.read()


def test_listm_source_tripwire():
    """Tripwire (mirrors the onthefly dead-id pattern): no trace of the
    'listm' feature may remain in panels.py or handlers.py."""
    for rel in ("panels.py", "handlers.py"):
        src = _read(rel)
        for needle in ("listm", "list_methods", '"List all methods"', "'List all methods'"):
            assert needle not in src, f"{rel} still contains {needle!r}"


async def test_listm_button_is_gone(tmp_path, monkeypatch):
    """Headless absence pin: #listm must not be mounted anywhere."""
    monkeypatch.setenv("UNSLOTH_CTQ_LOG_DIR", str(tmp_path))
    a = appmod.QuantApp()
    async with a.run_test():
        with pytest.raises(NoMatches):
            a.query_one("#listm")


async def test_run_button_still_composed(tmp_path, monkeypatch):
    """Guard against over-deletion: #run and #pick_method must survive."""
    monkeypatch.setenv("UNSLOTH_CTQ_LOG_DIR", str(tmp_path))
    a = appmod.QuantApp()
    async with a.run_test():
        run_btn = a.query_one("#run")
        assert run_btn.display
        pick = a.query_one("#pick_method")
        assert pick is not None

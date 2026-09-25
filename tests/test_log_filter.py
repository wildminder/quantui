"""Log-filter removal pins (user request 2026-09-09).

The S1.9 phase filter (click a progress row -> drawer filtered to that phase)
died with the ProgressRail: its only entry point was the rail click. Esc no
longer clears a filter (there is none); the drawer shows the plain capped
stream again. These tests pin the absence + the surviving drawer behavior.
"""

import json
import os

from conftest import wait_mounted
from textual.widgets import RichLog

from quantui import app as appmod

_QUANTUI_DIR = os.path.join(os.path.dirname(__file__), "..", "quantui")


def _read(rel: str) -> str:
    with open(os.path.join(_QUANTUI_DIR, rel), encoding="utf-8") as fh:
        return fh.read()


def _drawer_lines(a) -> list[str]:
    log = a.query_one("#log", RichLog)
    return ["".join(str(seg) for seg in line) for line in log.lines]


def test_filter_source_tripwire():
    """The filter state + Esc action are gone from app.py (docstring mentions
    of the removal are allowed)."""
    src = _read("app.py")
    for needle in ("_apply_log_filter", "_log_filter = ", "clear_log_filter"):
        assert needle not in src, f"app.py still contains {needle!r}"
    # The escape binding must be gone from the keymap.
    assert '"clear_log_filter"' not in _read("app.py")


async def test_esc_no_longer_clears_any_filter(tmp_path, monkeypatch):
    """Esc is unbound for filters; pressing it must not touch the drawer."""
    monkeypatch.setenv("UNSLOTH_CTQ_LOG_DIR", str(tmp_path))
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        await wait_mounted(a, pilot, "#log_drawer")
        a.log_msg("alpha")
        a.query_one("#log_drawer").display = True
        await pilot.pause()
        n_before = len(_drawer_lines(a))
        await pilot.press("escape")
        await pilot.pause()
        # No filter machinery: the stream is untouched.
        assert len(_drawer_lines(a)) >= n_before


async def test_drawer_shows_plain_stream_after_progress(tmp_path, monkeypatch):
    """Guard: the drawer still receives the normal stream — a progress frame
    collapses into the footer (no drawer noise), a real line is appended."""
    monkeypatch.setenv("UNSLOTH_CTQ_LOG_DIR", str(tmp_path))
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        a.log_msg("CTQ_PROGRESS " + json.dumps(
            {"phase": "quantize", "cur": 1, "total": 4, "pct": 25.0,
             "label": "Optimizing INT8"}
        ))
        a.log_msg("plain line after progress")
        a.query_one("#log_drawer").display = True
        await pilot.pause()
        lines = _drawer_lines(a)
        assert any("plain line after progress" in ln for ln in lines)
        # Progress frames never flood the drawer.
        assert not any("Optimizing INT8" in ln for ln in lines)


async def test_drawer_toggle_still_works(tmp_path, monkeypatch):
    """Regression guard: `l` still toggles the drawer."""
    monkeypatch.setenv("UNSLOTH_CTQ_LOG_DIR", str(tmp_path))
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        await wait_mounted(a, pilot, "#log_drawer")
        drawer = a.query_one("#log_drawer")
        assert drawer.display is False
        a.action_toggle_log()
        await pilot.pause()
        assert drawer.display is True
        a.action_toggle_log()
        await pilot.pause()
        assert drawer.display is False

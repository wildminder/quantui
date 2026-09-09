"""ProgressRail removal tripwire (user request 2026-09-09).

The per-phase chip rows were removed: they duplicated the footer stats line's
counts AND clicking them opened the log drawer — a behavior the user
explicitly rejected ("We do not need to open log window by clicking on the
footer"). The aggregate bar (#footer_bar) + boxed stats line (#footer_stats)
are the single progress surface. These tests pin the absence (mirrors the
listm / onthefly removal patterns).
"""

import os

import pytest
from textual.css.query import NoMatches

from quantui import app as appmod

_QUANTUI_DIR = os.path.join(os.path.dirname(__file__), "..", "quantui")


def _read(rel: str) -> str:
    with open(os.path.join(_QUANTUI_DIR, rel), encoding="utf-8") as fh:
        return fh.read()


def test_progress_rail_source_tripwire():
    """No trace of the rail feature may remain in the active source modules.

    Pinned strings: the widget classes, the click-to-filter handler chain and
    the phase-filter state. Docstring references to the removal are allowed.
    """
    for rel in ("panels.py", "app.py", "ids.py"):
        src = _read(rel)
        for needle in (
            "ProgressRailRow",
            "class ProgressRail",
            "BarClicked",
            ".rail_label",
            ".rail_row",
            ".rail_more",
            "set_states",
            "_apply_log_filter",
            "on_progress_rail_bar_clicked",
            "PROGRESS_RAIL",
            'id="progress_rail"',
            '"#progress_rail"',
        ):
            assert needle not in src, f"{rel} still contains {needle!r}"


async def test_progress_rail_is_gone(tmp_path, monkeypatch):
    """Headless absence pin: #progress_rail must not be mounted anywhere,
    even after progress frames stream in."""
    import json

    monkeypatch.setenv("UNSLOTH_CTQ_LOG_DIR", str(tmp_path))
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        a._show_run_footer()
        a.log_msg("CTQ_PROGRESS " + json.dumps(
            {"phase": "quantize", "cur": 1, "total": 4, "pct": 25.0,
             "label": "Optimizing INT8"}
        ))
        await pilot.pause()
        with pytest.raises(NoMatches):
            a.query_one("#progress_rail")
        with pytest.raises(NoMatches):
            a.query_one(".rail_row")


async def test_clicking_footer_does_not_open_log_drawer(tmp_path, monkeypatch):
    """The user-reported behavior is gone: clicking the run footer (bar, stats
    box, status strip) must NEVER open the log drawer."""
    monkeypatch.setenv("UNSLOTH_CTQ_LOG_DIR", str(tmp_path))
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        # Reveal the footer WITHOUT action_run: no run means no completion
        # toast intercepting pilot clicks.
        a._show_run_footer()
        await pilot.pause()
        assert a.query_one("#log_drawer").display is False

        # Click every footer surface: nothing may open the drawer.
        for selector in ("#footer_bar", "#footer_stats", "#status", "#footer_left"):
            await pilot.click(selector)
            await pilot.pause()
            assert a.query_one("#log_drawer").display is False, (
                f"clicking {selector} opened the log drawer"
            )


async def test_stats_line_still_shows_counts(tmp_path, monkeypatch):
    """Guard against over-deletion: the counts the rail used to show still
    appear in the boxed stats line (single surface, no duplication)."""
    import json

    from textual.widgets import Label

    monkeypatch.setenv("UNSLOTH_CTQ_LOG_DIR", str(tmp_path))
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        a._show_run_footer()
        a._run_start_ts = 1.0
        a.log_msg("CTQ_PROGRESS " + json.dumps(
            {"phase": "quantize", "cur": 1, "total": 4, "pct": 25.0,
             "label": "Optimizing INT8"}
        ))
        await pilot.pause()
        stats = str(a.query_one("#footer_stats", Label).content)
        assert "25" in stats
        assert "[1/4]" in stats

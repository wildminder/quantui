"""Headless tests for the log drawer (plan S1.4).

Covers: `l` toggle, `,`/`.` size cycling (S/M/L presets), and hidden-drawer
buffering (the RichLog stays mounted with display=False and keeps its
``max_lines`` cap while the authoritative temp file keeps every line).
"""

from conftest import wait_mounted
from textual.containers import Vertical
from textual.widgets import RichLog

from quantui import app as appmod
from quantui import panels


async def test_log_drawer_toggle():
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        await wait_mounted(a, pilot, "#log_drawer")
        drawer = a.query_one("#log_drawer")
        assert isinstance(drawer, Vertical)
        assert drawer.display is False  # hidden at start
        await pilot.press("l")
        assert drawer.display is True
        await pilot.press("l")
        assert drawer.display is False


async def test_log_drawer_size_cycle():
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        await wait_mounted(a, pilot, "#log_drawer")
        drawer = a.query_one("#log_drawer")
        # Start at M; two presses of "," -> S then L (wraps backwards).
        await pilot.press(",")
        assert a._log_drawer_size == "S"
        await pilot.press(",")
        assert a._log_drawer_size == "L"
        assert drawer.styles.height.value == panels.LOG_DRAWER_SIZES["L"]
        # "." wraps forward back to S.
        await pilot.press(".")
        assert a._log_drawer_size == "S"


async def test_drawer_hidden_keeps_log_buffering(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_CTQ_LOG_DIR", str(tmp_path))
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        await wait_mounted(a, pilot, "#log_drawer")
        assert a.query_one("#log_drawer").display is False
        for i in range(150):
            a.log_msg(f"hidden-line-{i}")
        await pilot.pause()
        log = a.query_one("#log", RichLog)
        # Capped display window survives hiding (plan Q2).
        assert len(log.lines) <= appmod.RUN_LOG_MAX_DISPLAY_LINES
        # Authoritative temp file has all 150 lines.
        with open(a._run_log_path, encoding="utf-8") as fh:
            assert len(fh.read().splitlines()) == 150


async def test_toggle_reopens_with_log_content(tmp_path, monkeypatch):
    """Lines written while hidden are visible after re-opening the drawer."""
    monkeypatch.setenv("UNSLOTH_CTQ_LOG_DIR", str(tmp_path))
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        a.log_msg("written-while-hidden")
        await pilot.press("l")  # open
        assert a.query_one("#log_drawer").display is True
        log = a.query_one(RichLog)
        texts = "".join(str(seg) for line in log.lines for seg in line)
        assert "written-while-hidden" in texts

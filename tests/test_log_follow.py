"""Headless tests for log tail-follow + scroll-pause (plan S1.5)."""

from textual.widgets import Label, RichLog

from quantui import app as appmod


async def _settle_scroll(log, pilot) -> None:
    """Wait until the RichLog has computed a real virtual size.

    A single ``pilot.pause()`` is not enough: the widget's ``max_scroll_y``
    stays 0 until the log has been laid out, so a test that reads ``scroll_y``
    at that point records a stale 0 and its later comparison is meaningless.
    """
    for _ in range(20):
        await pilot.pause()
        if log.max_scroll_y > 0:
            return


async def _settle_animated_scroll(log, pilot) -> None:
    """Wait until a scroll animation has come to rest.

    Scrolling a RichLog is animated: one pageup moves scroll_y 41 -> 22 and a
    further pause carries it on. A test that samples scroll_y after a single
    pause therefore reads a mid-animation value, and on a slow runner it lands
    somewhere different every time -- this is what made
    test_scroll_up_pauses_follow fail on CI (y_before read 0 while the log had
    already overflowed, so the assertion compared 22 against 1).

    Poll until the value stops changing between two consecutive pauses.
    """
    previous = None
    for _ in range(20):
        await pilot.pause()
        current = log.scroll_y
        if current == previous:
            return
        previous = current


async def test_follow_tail_by_default(tmp_path, monkeypatch):
    """New writes auto-scroll the RichLog to the tail while following."""
    monkeypatch.setenv("UNSLOTH_CTQ_LOG_DIR", str(tmp_path))
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        await pilot.press("l")  # open drawer
        assert a._log_follow is True
        for i in range(20):
            a.log_msg(f"follow-line-{i}")
        await pilot.pause()
        log = a.query_one("#log", RichLog)
        await _settle_scroll(log, pilot)
        await _settle_animated_scroll(log, pilot)
        assert len(log.lines) == 20
        # Auto-scrolled to (or near) the bottom: max_scroll_y == 0 when all
        # lines fit, otherwise scroll_y should be at max.
        assert log.scroll_y >= log.max_scroll_y - 1


async def test_scroll_up_pauses_follow(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_CTQ_LOG_DIR", str(tmp_path))
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        await pilot.press("l")
        for i in range(60):
            a.log_msg(f"pause-line-{i}")
        await pilot.pause()
        log = a.query_one("#log", RichLog)
        await _settle_scroll(log, pilot)
        # The log auto-scrolls to the tail on every write, and that scroll is
        # animated. Reading scroll_y while it is still travelling yields a
        # meaningless baseline (observed on CI: y_before came back 0.0 while
        # the log was in fact at 41, so the later comparison read 22 <= 1).
        await _settle_animated_scroll(log, pilot)
        y_before = log.scroll_y
        # The RichLog must be focused for pageup to reach it (keys go to the
        # focused widget); the drawer toggle leaves focus on the params column.
        log.focus()
        await pilot.pause()
        await pilot.press("pageup")
        await _settle_animated_scroll(log, pilot)
        assert a._log_follow is False
        assert a.query_one("#log_follow", Label).content == "paused"
        # More lines arrive -> line count grows, but the frozen view stays put.
        for i in range(10):
            a.log_msg(f"frozen-line-{i}")
        await pilot.pause()
        assert len(log.lines) > 0
        assert log.scroll_y <= y_before + 1


async def test_resume_on_bottom(tmp_path, monkeypatch):
    """Re-opening the drawer (`l`) resumes follow and jumps to the tail."""
    monkeypatch.setenv("UNSLOTH_CTQ_LOG_DIR", str(tmp_path))
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        await pilot.press("l")
        for i in range(30):
            a.log_msg(f"resume-line-{i}")
        await pilot.pause()
        log = a.query_one("#log", RichLog)
        log.focus()
        await pilot.pause()
        await pilot.press("pageup")
        await pilot.pause()
        assert a._log_follow is False
        a.log_msg("tail-line")
        await pilot.pause()
        await pilot.press("l")  # close
        await pilot.press("l")  # reopen -> resume follow
        assert a._log_follow is True
        assert a.query_one("#log_follow", Label).content == "follow"
        await pilot.pause()
        log = a.query_one("#log", RichLog)
        await _settle_scroll(log, pilot)
        await _settle_animated_scroll(log, pilot)
        assert log.scroll_y >= log.max_scroll_y - 1

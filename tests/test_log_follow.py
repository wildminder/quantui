"""Headless tests for log tail-follow + scroll-pause (plan S1.5)."""

from textual.widgets import Label, RichLog

from quantui import app as appmod


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
        y_before = log.scroll_y
        # The RichLog must be focused for pageup to reach it (keys go to the
        # focused widget); the drawer toggle leaves focus on the params column.
        log.focus()
        await pilot.pause()
        await pilot.press("pageup")
        await pilot.pause()
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
        assert log.scroll_y >= log.max_scroll_y - 1

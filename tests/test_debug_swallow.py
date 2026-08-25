"""Tests for QuantApp._debug_swallow (CRIT-002 STEP 2.2).

The helper records intentionally-swallowed errors into the authoritative
per-run log file so silent failures become diagnosable.
"""

from __future__ import annotations

import io

from quantui import app as appmod


def _make_app_with_log_fh():
    a = appmod.QuantApp()
    return a


async def test_debug_swallow_writes_to_run_log():
    a = _make_app_with_log_fh()
    async with a.run_test():
        buf = io.StringIO()
        a._run_log_fh = buf  # fake authoritative log handle
        try:
            raise ValueError("boom")
        except ValueError as exc:
            a._debug_swallow(exc, "test.context")
        assert "[swallowed] test.context: ValueError: boom" in buf.getvalue()


async def test_debug_swallow_no_fh_is_noop():
    a = _make_app_with_log_fh()
    async with a.run_test():
        # No _run_log_fh attribute set -> must not raise.
        if hasattr(a, "_run_log_fh"):
            del a._run_log_fh
        a._debug_swallow(RuntimeError("x"), "no.fh")


async def test_debug_swallow_closed_fh_is_noop():
    import os
    import tempfile

    a = _make_app_with_log_fh()
    async with a.run_test():
        path = os.path.join(tempfile.gettempdir(), "uqt-test-swallow.log")
        fh = open(path, "w", encoding="utf-8")
        fh.close()  # closed: write() would raise ValueError inside helper
        a._run_log_fh = fh
        a._debug_swallow(RuntimeError("y"), "closed.fh")  # must not raise


async def test_debug_swallow_does_not_mutate_widgets():
    a = _make_app_with_log_fh()
    async with a.run_test():
        from textual.widgets import Label

        status_before = str(a.query_one("#status", Label).content)
        buf = io.StringIO()
        a._run_log_fh = buf
        a._debug_swallow(Exception("z"), "widget.check")
        assert str(a.query_one("#status", Label).content) == status_before

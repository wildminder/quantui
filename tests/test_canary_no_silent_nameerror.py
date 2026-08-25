"""CRIT-002 STEP 2.7 historical-regression canary.

The original bug that motivated the silent-swallow overhaul (see
the issues tracker, CRIT-002): a ``NameError`` from a missing
import inside an ``except Exception: pass`` block was swallowed invisibly and
took a ``sys.settrace`` debugging session to locate.

This canary pins the NEW contract: any error raised inside a Class-L
(log-and-continue) guard MUST be recorded into the authoritative per-run log
via ``QuantApp._debug_swallow`` -- never lost. If a future refactor reintroduces
a silent swallow on these paths, this test fails.
"""

from __future__ import annotations

import io

from quantui import app as appmod
from quantui import profiles_store


def _make_app():
    return appmod.QuantApp()


async def test_class_l_nameerror_surfaces_in_run_log():
    """A NameError raised by profile persistence must land in the run log."""
    a = _make_app()
    async with a.run_test():
        buf = io.StringIO()
        a._run_log_fh = buf  # capture the authoritative log stream

        original = profiles_store.save_store

        def _boom(*args, **kwargs):
            raise NameError("missing_import_simulating_historical_bug")

        try:
            profiles_store.save_store = _boom
            # Drives the Class-L persistence guard -> _debug_swallow path.
            a._finish_run_record(rc=0)
        finally:
            profiles_store.save_store = original

        logged = buf.getvalue()
        assert "[swallowed] _finish_run_record.persist" in logged, logged
        assert "NameError" in logged, logged
        assert "missing_import_simulating_historical_bug" in logged, logged


async def test_class_l_error_does_not_break_run_completion():
    """The run-completion verdict still fires after the guarded failure."""
    a = _make_app()
    async with a.run_test():
        toasts: list[tuple[str, str]] = []
        a.emit_toast = lambda msg, severity="information": toasts.append(
            (msg, severity)
        )  # type: ignore[method-assign]
        a._run_log_fh = None  # helper must tolerate no-fh too

        original = profiles_store.save_store

        def _boom(*args, **kwargs):
            raise NameError("persist_failure_must_not_block_verdict")

        try:
            profiles_store.save_store = _boom
            a._finish_run_record(rc=0)
        finally:
            profiles_store.save_store = original

        assert any(
            "finished" in msg and sev == "information" for msg, sev in toasts
        ), toasts


async def test_widget_guard_no_longer_swallows_logic_errors():
    """Class-W sites catch only NoMatches -- a logic NameError must PROPAGATE,
    proving the historical hide-under-except-pass behavior is gone."""
    import pytest

    a = _make_app()
    async with a.run_test():
        calls: list[str] = []

        def _broken_query_one(*args, **kwargs):
            # Simulate the historical bug: a typo'd name inside a widget guard.
            calls.append("called")
            raise NameError("typo_in_handler_body")

        a.query_one = _broken_query_one  # type: ignore[method-assign]

        # set_status guards with ``except NoMatches`` only; the NameError must
        # escape instead of vanishing (the pre-CRIT-002 behavior).
        with pytest.raises(NameError, match="typo_in_handler_body"):
            a.set_status("hello")
        assert calls == ["called"]

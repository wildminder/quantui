"""Headless tests for the run footer (plan 2026-09-08-run-footer).

The footer replaces the right-hand rail (layout v2): #progress_rail + #status +
ResultsCard stack at the bottom of #body, full width, hidden until a run starts.
Composition + CSS pins live here; the visibility LIFECYCLE is S2.1 (same file);
the success-only results buttons are S2.2 (test_results_card*.py).
"""

from textual.widgets import Label

from quantui import app as appmod
from quantui import ids, panels
from quantui.app_css import MAIN_CSS
from quantui.widgets_results import ResultsCard


async def test_footer_holds_rail_status_results():
    """Composition: #run_footer wraps the historical rail ids + the results card.

    Left column #footer_left holds #progress_rail (ProgressRail) + #status (Label);
    ResultsCard is a direct child of the footer (right side)."""
    from textual.containers import Vertical

    a = appmod.QuantApp()
    async with a.run_test():
        footer = a.query_one("#run_footer", panels.RunFooter)
        rail = footer.query_one("#progress_rail", panels.ProgressRail)
        assert isinstance(rail, panels.ProgressRail)
        status = footer.query_one("#status", Label)
        assert isinstance(status, Label)
        left = footer.query_one("#footer_left", Vertical)
        assert left.query_one("#progress_rail") is not None
        assert left.query_one("#status") is not None
        card = footer.query_one(ResultsCard)
        assert isinstance(card, ResultsCard)


def test_params_full_width_css():
    """CSS contract: #params is 100% wide, #rail rule is gone, footer pinned hidden."""
    assert "width: 100%" in MAIN_CSS
    assert "#rail {" not in MAIN_CSS
    assert "#run_footer {" in MAIN_CSS
    # The footer must start hidden (S2.1 turns it on at run start).
    footer_block = MAIN_CSS.split("#run_footer {", 1)[1].split("}", 1)[0]
    assert "display: none" in footer_block
    assert "max-height" in footer_block
    # #params must no longer carry the old 30/70 split border.
    params_block = MAIN_CSS.split("#params {", 1)[1].split("}", 1)[0]
    assert "width: 100%" in params_block
    assert "border-right" not in params_block


def test_collapsible_has_bottom_margin():
    """F2-S1.1 (plan 2026-09-08-footer-v2): the Advanced Collapsible needs breathing
    room before the Run button — MAIN_CSS must give every Collapsible a bottom margin."""
    assert "Collapsible {" in MAIN_CSS
    block = MAIN_CSS.split("Collapsible {", 1)[1].split("}", 1)[0]
    assert "margin-bottom: 1" in block


# ---- S2.1: visibility lifecycle -----------------------------------------------


async def test_footer_hidden_until_run():
    """Direct seam unit: _show_run_footer() flips display on; idempotent."""
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        footer = a.query_one("#run_footer", panels.RunFooter)
        assert footer.display is False
        a._show_run_footer()
        await pilot.pause()
        assert footer.display is True
        # Idempotent: calling again stays visible, no error.
        a._show_run_footer()
        await pilot.pause()
        assert footer.display is True


async def test_footer_appears_when_run_pressed(tmp_path, monkeypatch):
    """Full lifecycle via action_run: footer shows at entry and SURVIVES completion."""
    from quantui import profiles_store as ps
    from tests.test_results_card_headless import FakeRunner, _gguf_cfg

    monkeypatch.setenv(ps.CONFIG_ENV_VAR, str(tmp_path / "cfg"))
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        cfg = _gguf_cfg(tmp_path)
        a._read_config = lambda: cfg
        a.runner = FakeRunner(rc=0)

        a.action_run()
        # Footer must be visible DURING the run (shown at action_run entry).
        await pilot.pause()
        assert a.query_one(ids.RUN_FOOTER, panels.RunFooter).display is True

        from tests.test_app_headless import _wait_until

        await a.workers.wait_for_complete()
        await _wait_until(lambda: "Done" in str(
            a.query_one(ResultsCard).query_one("#result_outcome").content), pilot)
        # ...and stays visible after completion.
        assert a.query_one(ids.RUN_FOOTER, panels.RunFooter).display is True


async def test_footer_shows_status_during_validation_failure(tmp_path, monkeypatch):
    """Early-return path (validation fails) still reveals the footer + status line."""
    from quantui import profiles_store as ps
    from quantui import run_config as rc_mod

    monkeypatch.setenv(ps.CONFIG_ENV_VAR, str(tmp_path / "cfg"))
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        # Empty model path -> run_config.validate fails -> action_run returns early.
        cfg = rc_mod.RunConfig(
            gguf=rc_mod.GgufConfig(model="", output=str(tmp_path / "o"),
                                   method="q4_k_m", pybin="python"),
        )
        a._read_config = lambda: cfg

        a.action_run()
        from tests.test_app_headless import _wait_until

        await _wait_until(
            lambda: "Validation failed" in str(a.query_one("#status", Label).content),
            pilot,
        )
        assert a.query_one(ids.RUN_FOOTER, panels.RunFooter).display is True


async def test_footer_survives_stop(tmp_path, monkeypatch):
    """User-stop outcome: footer stays up, card shows Stopped.

    Uses the blocking-runner pattern from test_app_headless (FakeRunner that
    waits on an Event until terminate()): FakeRunner(rc=0) would finish before
    the stop flag could ever matter."""
    import threading

    from quantui import profiles_store as ps
    from tests.test_app_headless import _wait_until
    from tests.test_results_card_headless import _gguf_cfg

    class BlockingRunner:
        def __init__(self) -> None:
            self._running = False
            self._ev = threading.Event()

        def run(self, cmd, cwd, observer, progress_debug_fh=None) -> int:
            self._running = True
            self._ev.wait()
            self._running = False
            return 1  # stopped, not a clean success

        def is_running(self) -> bool:
            return self._running

        def terminate(self) -> None:
            self._running = False
            self._ev.set()

    monkeypatch.setenv(ps.CONFIG_ENV_VAR, str(tmp_path / "cfg"))
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        cfg = _gguf_cfg(tmp_path)
        a._read_config = lambda: cfg
        a.runner = BlockingRunner()

        a.action_run()
        # Wait until the run is actually active, then stop via the real
        # _terminate_run path (sets _stop_requested + terminates the runner).
        await _wait_until(lambda: a._run_active is True, pilot)
        a._terminate_run()
        await pilot.pause()
        # The worker thread observes the flag and takes the stop branch.
        await _wait_until(lambda: "Stopped" in str(
            a.query_one(ResultsCard).query_one("#result_outcome").content), pilot)
        assert a.query_one(ids.RUN_FOOTER, panels.RunFooter).display is True


# ---- S3.1: e2e pins + regression guards ---------------------------------------


def _envelope(phase: str, cur: int, total: int, label: str) -> str:
    import json
    return "CTQ_PROGRESS " + json.dumps(
        {"phase": phase, "cur": cur, "total": total, "pct": round(100 * cur / total, 1),
         "label": label}
    )


async def test_footer_rail_receives_progress(tmp_path, monkeypatch):
    """Progress truly renders INSIDE the footer: an envelope produces a
    determinate .rail_row under #run_footer (not just a mounted rail)."""
    import textual.widgets as tw

    monkeypatch.setenv("UNSLOTH_CTQ_LOG_DIR", str(tmp_path))
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        a._show_run_footer()
        await pilot.pause()
        a.log_msg(_envelope("shard", 2, 3, "Quantizing shard"))
        await pilot.pause()
        footer = a.query_one("#run_footer", panels.RunFooter)
        rows = list(footer.query(".rail_row"))
        assert len(rows) == 1
        bar = rows[0].query_one(".rail_bar")
        assert isinstance(bar, tw.ProgressBar)


async def test_footer_updates_status_line():
    """set_status lands in the footer's #status; exactly ONE #status exists."""
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        assert len(list(a.query("#status"))) == 1
        a._show_run_footer()
        a.set_status("Running (q8_0)...")
        await pilot.pause()
        assert "Running (q8_0)" in str(a.query_one("#status", Label).content)


async def test_log_drawer_still_toggles():
    """Regression guard: the drawer binding still toggles the log above the footer."""
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        a._show_run_footer()
        await pilot.pause()
        drawer = a.query_one("#log_drawer")
        assert drawer.display is False
        a.action_toggle_log()
        await pilot.pause()
        assert drawer.display is True
        a.action_toggle_log()
        await pilot.pause()
        assert drawer.display is False

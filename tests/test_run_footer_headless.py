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
    """Composition (F2-S1.2): footer has BOTH panels but shows only one at a time.

    Progress mode (default): #footer_left visible (rail + bar + stats + status),
    ResultsCard hidden. Done mode: card visible (full width), left hidden."""
    from textual.containers import Vertical

    import quantui.profiles_store as ps

    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        footer = a.query_one("#run_footer", panels.RunFooter)
        rail = footer.query_one("#progress_rail", panels.ProgressRail)
        assert isinstance(rail, panels.ProgressRail)
        assert isinstance(footer.query_one("#status", Label), Label)
        # F2-S1.2: the new wide aggregate bar + stats line live in the left panel.
        left = footer.query_one("#footer_left", Vertical)
        assert left.query_one("#progress_rail") is not None
        assert left.query_one("#status") is not None
        from textual.widgets import ProgressBar

        bar = footer.query_one("#footer_bar", ProgressBar)
        assert bar.display is True
        assert footer.query_one("#footer_stats", Label) is not None
        card = footer.query_one(ResultsCard)
        assert isinstance(card, ResultsCard)

        # Default mode: progress only.
        assert footer._mode == "progress"
        assert left.display is True
        assert card.display is False

        # Done mode (record lands): card swaps in, progress panel out.
        rec = ps.RunRecord(
            ts="t", family="gguf", method="q4_k_m", output="/o/out.gguf",
            status="success", exit_code=0, duration_s=1.0,
        )
        card.show_record(rec)
        footer.set_mode("done")
        await pilot.pause()
        assert footer._mode == "done"
        assert left.display is False
        assert card.display is True
        assert footer.has_class("mode-done")
        # Buttons live on the card and are visible in done mode after success.
        assert card.query_one("#result_buttons").display is True

        # Back to progress for a new run.
        footer.set_mode("progress")
        await pilot.pause()
        assert left.display is True
        assert card.display is False
        assert not footer.has_class("mode-done")


def test_set_mode_rejects_unknown():
    """set_mode only accepts the two defined modes."""
    import pytest

    # No app boot needed: the API is a plain method; use an uncomposed instance.
    footer = panels.RunFooter(id="run_footer")
    with pytest.raises(ValueError):
        footer.set_mode("bogus")


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

    # F2-S1.2: done mode stretches the card; footer bar spans the left panel.
    assert ".mode-done > #results_card {" in MAIN_CSS
    assert "width: 100%" in MAIN_CSS.split(".mode-done > #results_card {", 1)[1].split("}", 1)[0]
    assert "#footer_bar {" in MAIN_CSS
    assert "#footer_stats {" in MAIN_CSS


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
    """Progress renders INSIDE the footer: an envelope produces a .rail_row
    under #run_footer (label-only — the wide bar is the ONE progress bar)."""
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
        label = rows[0].query_one(".rail_label")
        assert "Quantizing shard" in str(label.content)
        assert "[2/3]" in str(label.content)
        # Redundancy fix: rows must NOT contain a second progress bar.
        assert len(list(rows[0].query("ProgressBar"))) == 0


# ---- F2-S2.1: aggregate footer bar + stats line --------------------------------


async def test_footer_bar_tracks_progress(tmp_path, monkeypatch):
    """#footer_bar progress == aggregate pct; stats line has pct/Elapsed/ETA/counts."""
    monkeypatch.setenv("UNSLOTH_CTQ_LOG_DIR", str(tmp_path))
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        a._show_run_footer()
        a._run_start_ts = 1.0  # deterministic-ish elapsed > 0
        await pilot.pause()
        a.log_msg(_envelope("quantize", 2600, 4000, "Optimizing INT8"))
        await pilot.pause()
        from textual.widgets import ProgressBar

        footer = a.query_one("#run_footer", panels.RunFooter)
        bar = footer.query_one("#footer_bar", ProgressBar)
        stats = str(footer.query_one("#footer_stats", Label).content)
        assert bar.progress == 65  # 2600/4000
        assert "65" in stats
        assert "Elapsed" in stats
        assert "ETA" in stats
        assert "2600/4000" in stats


async def test_footer_bar_monotonic_like_header(tmp_path, monkeypatch):
    """A sparse line clearing the store must NOT reset the footer bar to 0."""
    monkeypatch.setenv("UNSLOTH_CTQ_LOG_DIR", str(tmp_path))
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        a._show_run_footer()
        await pilot.pause()
        a.log_msg(_envelope("quantize", 3000, 4000, "Optimizing INT8"))  # 75%
        await pilot.pause()
        from textual.widgets import ProgressBar

        footer = a.query_one("#run_footer", panels.RunFooter)
        bar = footer.query_one("#footer_bar", ProgressBar)
        assert bar.progress == 75
        # A plain log line clears the live store (header-hold semantics).
        a.log_msg("some ordinary log line without progress data")
        await pilot.pause()
        assert bar.progress == 75, "footer bar must hold, not reset"


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


# ---- F2-S3.1: mode-invariant pins -----------------------------------------------


async def test_mode_invariant_never_both(tmp_path, monkeypatch):
    """Walk the full lifecycle; after EVERY transition, progress panel and done
    card must never both be visible (the duplication the user reported)."""
    from quantui import profiles_store as ps
    from tests.test_app_headless import _wait_until
    from tests.test_results_card_headless import FakeRunner, _gguf_cfg

    def _invariant(a, step):
        left = a.query_one("#footer_left").display
        card = a.query_one(ResultsCard).display
        assert not (left and card), (
            f"both footer sides visible after: {step}"
        )

    monkeypatch.setenv(ps.CONFIG_ENV_VAR, str(tmp_path / "cfg"))
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        cfg = _gguf_cfg(tmp_path)
        a._read_config = lambda: cfg
        a.runner = FakeRunner(rc=0)

        # 1) run starts -> progress mode
        a.action_run()
        await pilot.pause()
        _invariant(a, "run start")
        # 2) completion -> done mode
        await a.workers.wait_for_complete()
        await _wait_until(lambda: "Done" in str(
            a.query_one(ResultsCard).query_one("#result_outcome").content), pilot)
        _invariant(a, "success done")
        # 3) new run -> back to progress
        a.action_run()
        await pilot.pause()
        _invariant(a, "second run start")
        # 4) second completion -> done mode again (outcome text is identical to
        # the first run's, so wait on the MODE, not the label content).
        await a.workers.wait_for_complete()
        await _wait_until(
            lambda: a.query_one("#footer_left").display is False, pilot)
        _invariant(a, "second run done")
        # In done mode, the left side must actually be hidden (not just XOR by luck).
        assert a.query_one("#footer_left").display is False
        assert a.query_one(ResultsCard).display is True


async def test_buttons_only_in_done_mode(tmp_path, monkeypatch):
    """#result_buttons visible implies #footer_left hidden (S2.2 gating x mode machine)."""
    from quantui import profiles_store as ps
    from tests.test_app_headless import _wait_until
    from tests.test_results_card_headless import FakeRunner, _gguf_cfg

    monkeypatch.setenv(ps.CONFIG_ENV_VAR, str(tmp_path / "cfg"))
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        cfg = _gguf_cfg(tmp_path)
        a._read_config = lambda: cfg
        a.runner = FakeRunner(rc=0)

        a.action_run()
        await a.workers.wait_for_complete()
        await _wait_until(lambda: "Done" in str(
            a.query_one(ResultsCard).query_one("#result_outcome").content), pilot)
        btns = a.query_one("#result_buttons")
        if btns.display:  # success record -> buttons visible in done mode
            assert a.query_one("#footer_left").display is False

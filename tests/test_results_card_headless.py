"""Headless tests for S2.2 results card + S2.5 recents recording/re-run."""

import sys

from quantui import app as appmod
from quantui import profiles_store as ps
from quantui import run_config as rc_mod
from quantui.widgets_results import ResultsCard
from tests.test_app_headless import _wait_until


class FakeRunner:
    def __init__(self, rc=0):
        self.rc = rc

    def run(self, cmd, cwd, observer, progress_debug_fh=None):
        return self.rc

    def is_running(self):
        return False

    def terminate(self):
        pass


def _gguf_cfg(tmp_path, out_name="out"):
    return rc_mod.RunConfig(
        family=Family_GGUF(),
        gguf=rc_mod.GgufConfig(
            model=str(tmp_path), output=str(tmp_path / out_name),
            method="q4_k_xl", pybin=sys.executable,
        ),
    )


def Family_GGUF():
    from quantui.quant_methods import Family

    return Family.GGUF


async def test_results_card_populated_after_run(tmp_path, monkeypatch):
    monkeypatch.setenv(ps.CONFIG_ENV_VAR, str(tmp_path / "cfg"))
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        cfg = _gguf_cfg(tmp_path)
        a._read_config = lambda: cfg
        a.runner = FakeRunner(rc=0)

        a.action_run()
        await a.workers.wait_for_complete()
        await pilot.pause()

        await _wait_until(lambda: "Done" in str(
            a.query_one(ResultsCard).query_one("#result_outcome").content), pilot)
        card = a.query_one(ResultsCard)
        assert "Done" in str(card.query_one("#result_outcome").content)
        assert card.output_path == cfg.gguf.output
        assert "exit 0" in str(card.query_one("#result_meta").content)


async def test_results_card_failed_shows_exit_code(tmp_path, monkeypatch):
    monkeypatch.setenv(ps.CONFIG_ENV_VAR, str(tmp_path / "cfg"))
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        cfg = _gguf_cfg(tmp_path)
        a._read_config = lambda: cfg
        a.runner = FakeRunner(rc=2)

        a.action_run()
        await a.workers.wait_for_complete()
        await pilot.pause()

        await _wait_until(lambda: "exit 2" in str(
            a.query_one(ResultsCard).query_one("#result_meta").content), pilot)
        outcome = str(a.query_one(ResultsCard).query_one("#result_outcome").content)
        assert "Failed" in outcome


async def test_recent_job_recorded_after_run(tmp_path, monkeypatch):
    """A finished run lands in store.json recents with correct fields."""
    cfg_dir = tmp_path / "cfg"
    monkeypatch.setenv(ps.CONFIG_ENV_VAR, str(cfg_dir))
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        cfg = _gguf_cfg(tmp_path, out_name="my-out")
        a._read_config = lambda: cfg
        a.runner = FakeRunner(rc=0)

        a.action_run()
        await a.workers.wait_for_complete()
        await pilot.pause()

        recents = ps.list_recents(config_dir=str(cfg_dir))
        assert len(recents) == 1
        rec = recents[0]
        assert rec["status"] == "success"
        assert rec["exit_code"] == 0
        assert rec["output"].endswith("my-out")
        assert rec["family"] == "gguf"


async def test_show_recents_no_recents_safe(tmp_path, monkeypatch):
    """Ctrl+R with an empty store -> toast only, no screen pushed."""
    monkeypatch.setenv(ps.CONFIG_ENV_VAR, str(tmp_path / "empty-cfg"))
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        toasts = []
        monkeypatch.setattr(
            a, "emit_toast", lambda msg, severity="information": toasts.append((msg, severity))
        )
        depth_before = len(a.screen_stack)
        a.action_show_recents()
        await pilot.pause()
        assert len(a.screen_stack) == depth_before
        assert any("No recent jobs" in m for m, _ in toasts)


async def test_rerun_reloads_widgets_from_recent(tmp_path, monkeypatch):
    """Activating a RecentJobsScreen row reloads the config into the widgets."""
    from quantui import screens

    cfg_dir = tmp_path / "cfg"
    monkeypatch.setenv(ps.CONFIG_ENV_VAR, str(cfg_dir))
    record = ps.RunRecord(
        ts="t0", family="gguf", method="q4_k_xl",
        output=str(tmp_path / "rerun-out"), status="success", exit_code=0,
        duration_s=1.0,
        config={"family": "gguf", "model": "/the/model", "output": "/the/out"},
    )
    ps.add_recent(record.to_dict(), config_dir=str(cfg_dir))

    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        from textual.widgets import Input

        a.action_show_recents()
        await pilot.pause()
        screen = a.screen_stack[-1]
        assert isinstance(screen, screens.RecentJobsScreen)

        # Dismiss with the first record (row activation).
        a.pop_screen_callback = None  # not a real attribute; just be explicit below
        # Simulate the callback path directly (deterministic vs DataTable events).
        screen.dismiss(record.to_dict())
        await pilot.pause()

        await _wait_until(
            lambda: a.query_one("#model", Input).value == "/the/model", pilot
        )
        from textual.widgets import Input as Inp

        assert a.query_one("#model", Inp).value == "/the/model"
        assert a.query_one("#output", Inp).value == "/the/out"

"""Headless tests for S2.1 (toast + bell on run end / validation error)."""

import sys
import threading

from quantui import app as appmod
from quantui import run_config as rc_mod
from quantui.quant_methods import Family
from tests.test_app_headless import _wait_until


class FakeRunner:
    def __init__(self, rc=0):
        self._running = False
        self._ev = threading.Event()
        self.rc = rc

    def run(self, cmd, cwd, observer, progress_debug_fh=None):
        self._running = True
        self._ev.set()
        self._running = False
        return self.rc

    def is_running(self):
        return False

    def terminate(self):
        pass


async def test_done_emits_toast_and_bell(tmp_path, monkeypatch):
    """Happy-path GGUF run -> information toast + terminal bell."""
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        toasts = []
        monkeypatch.setattr(
            a, "emit_toast", lambda msg, severity="information": toasts.append((msg, severity))
        )
        bells = []
        monkeypatch.setattr(a, "_ring_bell", lambda: bells.append(True))

        cfg = rc_mod.RunConfig(
            family=Family.GGUF,
            gguf=rc_mod.GgufConfig(
                model=str(tmp_path), output=str(tmp_path / "out"),
                method="q4_k_m", pybin=sys.executable,
            ),
        )
        a._read_config = lambda: cfg
        runner = FakeRunner(rc=0)
        a.runner = runner

        a.action_run()
        await a.workers.wait_for_complete()
        await pilot.pause()

        await _wait_until(lambda: any(s == "information" for _, s in toasts), pilot)
        assert any("finished" in m for m, _ in toasts)
        assert bells, "terminal bell must fire on success"


async def test_failed_run_emits_error_toast(tmp_path, monkeypatch):
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        toasts = []
        monkeypatch.setattr(
            a, "emit_toast", lambda msg, severity="information": toasts.append((msg, severity))
        )
        cfg = rc_mod.RunConfig(
            family=Family.GGUF,
            gguf=rc_mod.GgufConfig(
                model=str(tmp_path), output=str(tmp_path / "out"),
                method="q4_k_m", pybin=sys.executable,
            ),
        )
        a._read_config = lambda: cfg
        a.runner = FakeRunner(rc=2)

        a.action_run()
        await a.workers.wait_for_complete()
        await pilot.pause()

        await _wait_until(lambda: any(s == "error" for _, s in toasts), pilot)
        assert any("failed" in m.lower() and "exit 2" in m for m, _ in toasts)


async def test_validation_errors_toast_error_no_subprocess(tmp_path, monkeypatch):
    """action_run with empty model -> error toast; no subprocess launched."""
    launched = []
    from quantui import worker_runner as wr

    class BoomProc:
        def __init__(self, *args, **kwargs):
            launched.append(1)
            raise AssertionError("subprocess must not launch on validation error")

    monkeypatch.setattr(wr.subprocess, "Popen", BoomProc)

    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        toasts = []
        monkeypatch.setattr(
            a, "emit_toast", lambda msg, severity="information": toasts.append((msg, severity))
        )
        cfg = rc_mod.RunConfig(
            family=Family.GGUF,
            gguf=rc_mod.GgufConfig(model="", output="", method="", pybin=""),
        )
        a._read_config = lambda: cfg
        a.runner = FakeRunner()

        a.action_run()
        await a.workers.wait_for_complete()
        await pilot.pause()

        assert not launched
        assert any(s == "error" for _, s in toasts), toasts

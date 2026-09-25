"""Headless tests for threshold-gated stop-confirm (plan S1.10).

Below STOP_CONFIRM_AFTER_S: stop terminates immediately, no screen pushed
(the existing test_stop_button_terminates_comfy_run covers the lifecycle; this
adds the explicit screen-stack check). Above it: ConfirmModal is pushed and a
dismiss(True) triggers termination.
"""

import sys
import threading
import time

from conftest import wait_mounted

from quantui import app as appmod
from quantui import run_config as rc_mod
from quantui import screens
from quantui.quant_methods import Family


class FakeRunner:
    """Blocking fake WorkerRunner (same shape as in test_app_headless)."""

    def __init__(self) -> None:
        self._running = False
        self._ev = threading.Event()
        self.terminated = 0

    def run(self, cmd, cwd, observer, progress_debug_fh=None) -> int:
        self._running = True
        self._ev.wait()
        self._running = False
        return 1

    def is_running(self) -> bool:
        return self._running

    def terminate(self) -> None:
        self.terminated += 1
        self._running = False
        self._ev.set()


def _ctq_cfg(tmp_path):
    safetensors = tmp_path / "model.safetensors"
    safetensors.write_text("x")
    return rc_mod.RunConfig(
        family=Family.COMFY,
        ctq=rc_mod.CtqConfig(
            input=str(safetensors),
            output=str(tmp_path / "out-fp8_e4m3.safetensors"),
            pybin=sys.executable,
            format="fp8_e4m3",
            output_mode="sharded",
            option_values={},
            quant_tags=["fp8_e4m3"],
        ),
    )


async def _start_run(a, tmp_path):
    """Legacy helper kept for symmetry; tests inline their own setup."""
    raise NotImplementedError


async def test_stop_immediate_below_threshold(tmp_path):
    """Run younger than the threshold -> immediate terminate, no modal pushed."""
    from tests.test_app_headless import _wait_until

    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        await wait_mounted(a, pilot, "#gguf_panel")
        cfg = _ctq_cfg(tmp_path)
        a.family = Family.COMFY
        a.query_one("#gguf_panel").display = False
        a.query_one("#comfy_panel").display = True
        a.refresh_ctq_visibility()
        a._read_config = lambda: cfg
        runner = FakeRunner()
        a.runner = runner
        a.action_run()
        await _wait_until(lambda: runner.is_running() is True, pilot, timeout=5)

        assert len(a.screen_stack) == 1  # base screen only
        # Run just started -> elapsed ~0 < 15s.
        a.action_stop_run()
        await pilot.pause()
        assert a._stop_requested is True
        assert runner.terminated == 1
        assert len(a.screen_stack) == 1, "no confirm modal below threshold"


async def test_stop_confirm_modal_above_threshold(tmp_path):
    """Backdated start_ts (60s old) -> Stop pushes ConfirmModal; dismiss(True)
    terminates, dismiss(False) does not."""
    from tests.test_app_headless import _wait_until

    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        await wait_mounted(a, pilot, "#gguf_panel")
        cfg = _ctq_cfg(tmp_path)
        a.family = Family.COMFY
        a.query_one("#gguf_panel").display = False
        a.query_one("#comfy_panel").display = True
        a.refresh_ctq_visibility()
        a._read_config = lambda: cfg
        runner = FakeRunner()
        a.runner = runner
        a.action_run()
        await _wait_until(lambda: runner.is_running() is True, pilot, timeout=5)

        # Backdate the ETA clock: the run looks 60s old (> 15s threshold).
        a._run_start_ts = time.monotonic() - 60.0

        a.action_stop_run()
        await pilot.pause()
        # ConfirmModal on top of the base screen.
        assert len(a.screen_stack) == 2
        modal = a.screen_stack[-1]
        assert isinstance(modal, screens.ConfirmModal)

        # Cancel first: no termination.
        modal.dismiss(False)
        await pilot.pause()
        assert runner.terminated == 0
        assert a._stop_requested is False
        await _wait_until(lambda: len(a.screen_stack) == 1, pilot, timeout=5)

        # Now stop again and confirm: termination fires via the callback.
        a.action_stop_run()
        await pilot.pause()
        assert len(a.screen_stack) == 2
        a.screen_stack[-1].dismiss(True)
        await pilot.pause()
        assert runner.terminated == 1
        assert a._stop_requested is True


async def test_help_placeholder_toast():
    """`?` shows the keymap hint toast (Phase-3 placeholder)."""
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        toasts = []
        def monkey_toast(msg, severity="information"):
            toasts.append((msg, severity))
        a.emit_toast = monkey_toast  # type: ignore[method-assign]
        await pilot.press("?")
        await pilot.pause()
        assert toasts and "Keys:" in toasts[0][0]

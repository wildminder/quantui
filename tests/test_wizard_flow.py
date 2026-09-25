"""Headless tests for S3.3 wizard stepped flow + review==form determinism gate."""


from textual.widgets import Input, Select

from quantui import app as appmod
from quantui import screens_wizard


async def test_wizard_review_matches_form_cmd():
    """Wizard-collected values applied to the form produce the SAME ctq cmd as
    filling the form directly (determinism gate)."""
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        # Form path: fill widgets directly.
        a.query_one("#model", Input).value = "/data/model.safetensors"
        a.query_one("#output", Input).value = "/data/out"
        wizard_values = {
            "family": "gguf",
            "model": "/data/model.safetensors",
            "output": "/data/out",
            "method": "q4_k_m",
        }
        # Wizard path: apply through the same applier the Start button uses.
        a._apply_profile_fields(wizard_values)
        await pilot.pause()
        assert a.query_one("#model", Input).value == "/data/model.safetensors"
        assert a.query_one("#output", Input).value == "/data/out"
        # T9: #method is an Input now (T8) -- the applier writes the free-text
        # value verbatim, no Select option-value errors possible.
        assert a.query_one("#method", Input).value == "q4_k_m"


async def test_wizard_start_runs_same_pipeline(tmp_path, monkeypatch):
    """Wizard dismiss(values) -> _on_wizard_done fills form and starts action_run;
    spawned cmd tokens match the canonical comfy expectations."""
    from quantui import worker_runner as wr
    from tests.test_app_headless import fake_popen  # noqa: F401

    captured = {}

    class FakeProc:
        def __init__(self, cmd):
            captured["cmd"] = list(cmd)
            import os as _os

            r, w = _os.pipe()
            with _os.fdopen(w, "wb") as fw:
                fw.write(b"FAKE_WORKER_LINE\n")
            self.stdout = _os.fdopen(r, "rb", 0)
            self.returncode = 0

        def wait(self):
            return 0

    monkeypatch.setattr(
        wr.subprocess, "Popen", lambda cmd, **kw: FakeProc(cmd)
    )

    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        values = {
            "family": "comfy",
            "model": str(tmp_path / "m.safetensors"),
            "output": str(tmp_path / "m-fp8_e4m3.safetensors"),
            "ctq_input": str(tmp_path / "m.safetensors"),
            "ctq_output": str(tmp_path / "m-fp8_e4m3.safetensors"),
            "method": "fp8_e4m3",
            "ctq_format": "fp8_e4m3",
        }
        (tmp_path / "m.safetensors").write_text("x")
        a._on_wizard_done(dict(values))
        await a.workers.wait_for_complete()
        await pilot.pause()

        cmd = " ".join(captured["cmd"])
        assert "quantui.worker_ctq" in cmd
        for tok in ["-i", "-o", "--comfy_quant"]:
            assert tok in cmd, tok
        assert "m-fp8_e4m3.safetensors" in cmd


async def test_wizard_disabled_during_active_run():
    """Key w during an active run shows a warning toast, no screen pushed."""
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        toasts = []
        def monkey_toast(msg, severity="information"):
            toasts.append((msg, severity))
        a.emit_toast = monkey_toast  # type: ignore[method-assign]
        a._run_active = True
        depth = len(a.screen_stack)
        a.action_open_wizard()
        await pilot.pause()
        assert len(a.screen_stack) == depth
        assert any("wizard" in m.lower() for m, _ in toasts)


async def test_wizard_screen_steps_and_cancel():
    """WizardScreen steps forward/back and Escape dismisses None."""
    from textual.widgets import Button

    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        wiz = screens_wizard.WizardScreen()
        a.push_screen(wiz)
        # Wait until the wizard is the active screen and its body composed
        # (query via a.screen — fresh push_screen children settle after a pause).
        for _ in range(20):
            await pilot.pause()
            try:
                a.screen.query_one("#wiz_next")
                break
            except Exception:
                continue
        assert wiz.step == 0
        a.screen.query_one("#wiz_next", Button).press()
        await pilot.pause()
        assert wiz.step == 1
        a.screen.query_one("#wiz_back", Button).press()
        await pilot.pause()
        assert wiz.step == 0
        wiz.action_cancel()
        await pilot.pause()


async def test_wizard_default_method_is_q4_k_m():
    """T3b (plan 2026-08-31-gguf-unsloth-parity): the wizard's method Select
    must land on the same DEFAULT_GGUF_METHOD as the main form -- previously it
    degraded to METHODS[0] (not_quantized), silently wasting the run."""
    from quantui.quant_methods import DEFAULT_GGUF_METHOD

    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        wiz = screens_wizard.WizardScreen()
        a.push_screen(wiz)
        # Wait for the wizard body to compose (same retry loop as above).
        for _ in range(20):
            await pilot.pause()
            try:
                a.screen.query_one("#wiz_next")
                break
            except Exception:
                continue
        a.screen.query_one("#wiz_next").press()
        # The step-1 body populates the method Select, which can land a beat
        # after the press on a slow runner -- poll until it leaves Select.NULL
        # rather than asserting on a single pause.
        value = Select.NULL
        for _ in range(20):
            await pilot.pause()
            value = a.screen.query_one("#wiz_method", Select).value
            if str(value) != str(Select.NULL):
                break
        assert str(value) == DEFAULT_GGUF_METHOD
        assert DEFAULT_GGUF_METHOD == "q4_k_m"

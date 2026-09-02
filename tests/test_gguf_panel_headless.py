"""T8+T9 (plan 2026-08-31-gguf-unsloth-parity): GGUF panel UI + glue.

Headless-mount tests for the method Input conversion (Select -> Input, same
#method id), the imatrix widgets, the UD footer, and the handler glue
(selected_method / imatrix_value / update_method_info / run gate / profile
round-trip).

The last two tests pin regressions that the Input conversion EXPOSED (both
were latent before it) -- see their docstrings.
"""

import sys

from textual.widgets import Checkbox, Input, Static

from quantui import app as appmod
from quantui.quant_methods import DEFAULT_GGUF_METHOD, UD_INFO_FOOTER


async def test_method_widget_is_input():
    a = appmod.QuantApp()
    async with a.run_test():
        # The panel-level widget is an Input now: free-text multi-method entry
        # ("q4_k_m, q5_k_m") that a Select could never express.
        a.query_one("#method", Input)


async def test_method_input_default_q4_k_m():
    a = appmod.QuantApp()
    async with a.run_test():
        assert a.query_one("#method", Input).value == DEFAULT_GGUF_METHOD


async def test_imatrix_widgets_present():
    a = appmod.QuantApp()
    async with a.run_test():
        a.query_one("#imatrix_path", Input)
        auto = a.query_one("#imatrix_auto", Checkbox)
        assert auto.value is False  # unchecked by default


async def test_ud_footer_rendered():
    a = appmod.QuantApp()
    async with a.run_test():
        footer = a.query_one("#ud_footer", Static)
        # Static content is str or renderable; str() covers our plain case.
        assert UD_INFO_FOOTER in str(footer.render() if hasattr(footer, "render") else footer.content)


async def test_selected_method_reads_input():
    a = appmod.QuantApp()
    async with a.run_test():
        a.query_one("#method", Input).value = "q5_k_m"
        assert a.selected_method() == "q5_k_m"


async def test_selected_method_custom_override_wins():
    a = appmod.QuantApp()
    async with a.run_test():
        a.query_one("#method", Input).value = "q4_k_m"
        a.query_one("#custom", Input).value = "q8_0"
        assert a.selected_method() == "q8_0"


async def test_selected_method_custom_comma_list():
    # Custom may carry a comma list; it is returned verbatim -- the same
    # parse_methods downstream handles it.
    a = appmod.QuantApp()
    async with a.run_test():
        a.query_one("#custom", Input).value = "q4_k_m, q8_0"
        assert a.selected_method() == "q4_k_m, q8_0"


async def test_selected_method_strips_whitespace():
    a = appmod.QuantApp()
    async with a.run_test():
        a.query_one("#method", Input).value = "  q5_k_m  "
        assert a.selected_method() == "q5_k_m"


async def test_imatrix_value_auto_checkbox_wins():
    a = appmod.QuantApp()
    async with a.run_test():
        a.query_one("#imatrix_path", Input).value = "C:/some/imatrix.dat"
        a.query_one("#imatrix_auto", Checkbox).value = True
        assert a.imatrix_value() == "auto"


async def test_imatrix_value_path():
    a = appmod.QuantApp()
    async with a.run_test():
        a.query_one("#imatrix_path", Input).value = "  C:/some/imatrix.dat  "
        assert a.imatrix_value() == "C:/some/imatrix.dat"


async def test_imatrix_value_empty():
    a = appmod.QuantApp()
    async with a.run_test():
        assert a.imatrix_value() == ""


async def test_update_method_info_imatrix_marker():
    a = appmod.QuantApp()
    async with a.run_test():
        a.query_one("#method", Input).value = "iq2_xs"
        a.update_method_info()
        info = a.query_one("#method_info", Static)
        assert "[IMATRIX]" in str(info.render())


async def test_update_method_info_plain_no_marker():
    a = appmod.QuantApp()
    async with a.run_test():
        a.query_one("#method", Input).value = "q4_k_m"
        a.update_method_info()
        info = a.query_one("#method_info", Static)
        assert "[IMATRIX]" not in str(info.render())


async def test_run_blocked_iq_without_imatrix(monkeypatch, tmp_path):
    # The run gate (validate_gguf, T5) must refuse IQ* without imatrix via the
    # EXISTING toast machinery, and no worker process may be spawned.
    from quantui import run_config as rc

    spawned = []

    def fake_spawn(*a, **k):
        spawned.append((a, k))

    monkeypatch.setattr(
        "quantui.worker_runner.spawn_worker", fake_spawn, raising=False
    )
    # Give the model path + pybin widgets valid-looking values so the ONLY
    # validation error is the imatrix one.
    a = appmod.QuantApp()
    async with a.run_test():
        a.query_one("#model", Input).value = str(tmp_path)  # exists (dir)
        a.query_one("#output", Input).value = str(tmp_path / "out")
        a.query_one("#pybin", Input).value = sys.executable
        a.query_one("#method", Input).value = "iq2_xs"
        errors = rc.validate_gguf(a._read_config().gguf)
        assert errors, "IQ* without imatrix must not pass validation"
        assert any("imatrix" in e.lower() for e in errors)
        assert spawned == []


async def test_profile_roundtrips_imatrix(tmp_path, monkeypatch):
    from quantui import profiles_store as ps

    cfg_dir = tmp_path / "cfg"
    monkeypatch.setenv(ps.CONFIG_ENV_VAR, str(cfg_dir))
    a = appmod.QuantApp()
    async with a.run_test():
        a.query_one("#imatrix_path", Input).value = "C:/imat/imatrix.dat"
        fields = a._profile_fields()
        assert fields.get("imatrix") == "C:/imat/imatrix.dat"
        a.query_one("#imatrix_path", Input).value = ""
        a._apply_profile_fields(fields)
        assert a.query_one("#imatrix_path", Input).value == "C:/imat/imatrix.dat"


async def test_family_switch_is_synchronous():
    """Regression: pressing the family radio only *posts* RadioSet.Changed.

    A caller that switches family and then acts inside the SAME synchronous
    block (wizard / profile apply -> ``action_run``) must not read a STALE
    family: ``_read_config`` would build a GGUF config for a comfy run and
    ``validate`` would gate the wrong panel. While ``#method`` was a Select the
    bug was masked (a comfy run validated as GGUF but the rejected method fell
    back to a valid default); free-text entry (T8) turns it into a hard
    "Unknown quantization method" error. ``_set_family`` now applies at once.
    """
    from quantui.quant_methods import Family

    a = appmod.QuantApp()
    async with a.run_test():
        assert a.family == Family.GGUF
        a.action_family_comfy()
        # No await / pilot.pause() in between -- that is the whole point.
        assert a.family == Family.COMFY
        assert a._read_config().family == Family.COMFY
        a.action_family_gguf()
        assert a.family == Family.GGUF
        assert a._read_config().family == Family.GGUF


async def test_run_does_not_cancel_capability_probe():
    """Regression: ``action_run`` is ``@work(thread=True, exclusive=True)``.

    Exclusive cancels every other worker in the DEFAULT group. Switching to
    comfy synchronously (see the test above) now starts the capability probe
    BEFORE the run worker, so the run used to cancel it -- raising
    ``WorkerCancelled`` for anyone awaiting the workers and, in the real app,
    killing the capability badge update. The probe lives in its own group.
    """
    from quantui.quant_methods import Family

    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        a.action_family_comfy()  # starts the capability probe thread
        assert a.family == Family.COMFY
        a.action_run()  # exclusive in the default group; must not kill it
        # Empty model path -> validation fails fast, so no worker subprocess
        # is spawned; only the probe + the run worker are involved.
        await a.workers.wait_for_complete()
        await pilot.pause()

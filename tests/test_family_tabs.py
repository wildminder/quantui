"""Headless tests for family-tab keyboard shortcuts (plan S1.3).

Keys ``1`` / ``2`` programmatically press the #fam_gguf / #fam_comfy RadioButtons,
which fire ``RadioSet.Changed`` and drive the EXISTING ``on_radio_set_changed``
handler (no handler change). Regression: the output-mode radio must still NOT
switch families.
"""

from textual.widgets import RadioButton, RadioSet

from quantui import app as appmod
from quantui.quant_methods import Family


async def test_key_1_2_switch_family():
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        assert a.family == Family.GGUF
        await pilot.press("2")
        assert a.family == Family.COMFY
        assert a.query_one("#comfy_panel").display is True
        assert a.query_one("#gguf_panel").display is False
        await pilot.press("1")
        assert a.family == Family.GGUF
        assert a.query_one("#gguf_panel").display is True
        assert a.query_one("#comfy_panel").display is False


async def test_output_mode_radio_does_not_switch_family():
    # Ported regression guard: pressing "1"/"2" goes through the same radio-driven
    # path, so an unrelated RadioSet (#ctq_output_mode) still must not flip family.
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        await pilot.press("2")
        assert a.family == Family.COMFY
        rs = a.query_one("#ctq_output_mode", RadioSet)
        single = a.query_one("#om_single", RadioButton)
        a.on_radio_set_changed(RadioSet.Changed(rs, single))
        assert a.family == Family.COMFY

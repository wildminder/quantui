"""Method picker modal (plan: software-method-picker).

Headless tests for the ``[Pick from list]`` button (#pick_method) and the
``MethodPickerScreen`` modal it opens -- a ``SelectionList`` over all 35
official unsloth methods (``quant_methods.METHODS`` registry order), with
the current ``#method`` value preselected. Confirm dismisses with the
comma-joined selected ids in REGISTRY order (deterministic output, not
click order); Cancel/Escape dismiss ``None`` and leave ``#method`` alone.
"""

from textual.widgets import Button, Input, SelectionList, Static

from quantui import app as appmod
from quantui import screens
from quantui.quant_methods import METHODS


def _picker(app) -> SelectionList:
    """The modal's SelectionList (id=mp_list) on the pushed screen."""
    return app.screen.query_one("#mp_list", SelectionList)


async def _wait_mounted(app, pilot, selector: str) -> None:
    """Poll until ``selector`` exists on the current screen.

    Panel widgets are created during compose, and a single pilot.pause() is not
    reliably enough on a loaded or slow runner (observed failing the GitHub
    Actions gate). Wait for the widget instead of racing its mount.
    """
    for _ in range(20):
        await pilot.pause()
        try:
            app.screen.query_one(selector)
            return
        except Exception:
            continue


def _prompt_plain(option) -> str:
    """The option prompt as literal text (Content.plain strips markup)."""
    prompt = option.prompt
    return prompt.plain if hasattr(prompt, "plain") else str(prompt)


async def test_pick_method_button_present_and_opens_picker():
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        await _wait_mounted(a, pilot, "#pick_method")
        btn = a.query_one("#pick_method", Button)
        btn.press()
        await pilot.pause()
        assert isinstance(a.screen, screens.MethodPickerScreen)


async def test_picker_lists_all_35_methods_in_registry_order():
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        await _wait_mounted(a, pilot, "#pick_method")
        a.query_one("#pick_method", Button).press()
        for _ in range(20):  # wait for the modal body to compose
            await pilot.pause()
            try:
                _picker(a)
                break
            except Exception:
                continue
        sl = _picker(a)
        assert sl.option_count == len(METHODS)  # 35 official + 4 native (S4.1)
        # Values follow the METHODS registry order (import order = registry
        # order -- the picker must NOT re-sort).
        assert [sl.get_option_at_index(i).value for i in range(sl.option_count)] == [
            m.id for m in METHODS
        ]


async def test_current_method_preselected():
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        await _wait_mounted(a, pilot, "#pick_method")
        a.query_one("#method", Input).value = "q5_k_m"
        a.query_one("#pick_method", Button).press()
        for _ in range(20):
            await pilot.pause()
            try:
                _picker(a)
                break
            except Exception:
                continue
        sl = _picker(a)
        assert sl.selected == ["q5_k_m"]


async def test_multi_initial_value_preselected():
    # A comma list "q4_k_m, q5_k_m" preselects BOTH entries.
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        await _wait_mounted(a, pilot, "#pick_method")
        a.query_one("#method", Input).value = "q5_k_m, q4_k_m"
        a.query_one("#pick_method", Button).press()
        for _ in range(20):
            await pilot.pause()
            try:
                _picker(a)
                break
            except Exception:
                continue
        sl = _picker(a)
        assert set(sl.selected) == {"q4_k_m", "q5_k_m"}


async def test_unknown_initial_ids_ignored():
    # Free-text typos in the initial value must not crash or preselect junk.
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        await _wait_mounted(a, pilot, "#pick_method")
        a.query_one("#method", Input).value = "q9_typo, q5_k_m"
        a.query_one("#pick_method", Button).press()
        for _ in range(20):
            await pilot.pause()
            try:
                _picker(a)
                break
            except Exception:
                continue
        sl = _picker(a)
        assert sl.selected == ["q5_k_m"]  # typo ignored, valid id kept


async def test_confirm_single_selection_fills_method_input():
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        await _wait_mounted(a, pilot, "#pick_method")
        a.query_one("#method", Input).value = "q4_k_m"
        a.query_one("#pick_method", Button).press()
        for _ in range(20):
            await pilot.pause()
            try:
                _picker(a)
                break
            except Exception:
                continue
        sl = _picker(a)
        sl.select("q8_0")
        await pilot.pause()
        a.screen.query_one("#mp_confirm", Button).press()
        await pilot.pause()
        # Preselection = "the list being edited": picking q8_0 ADDS to the
        # existing q4_k_m (both are confirmed together, registry order).
        registry_ids = [m.id for m in METHODS]
        selected = {"q4_k_m", "q8_0"}
        expected = ", ".join(mid for mid in registry_ids if mid in selected)
        assert a.query_one("#method", Input).value == expected
        # A comma list is not a single registry id, so the info line shows
        # the honest multi-method pass-through message (single-id picks get
        # the per-method description; covered in test_gguf_panel_headless).
        # #custom lives in a Collapsible in the GGUF panel, so wait for the
        # panel to be fully composed before reading the info line.
        await _wait_mounted(a, pilot, "#custom")
        info = a.query_one("#method_info", Static)
        assert "Custom method" in str(info.render())


async def test_confirm_multiselect_registry_order():
    # Selecting in "reverse" order still dismisses in METHODS registry order.
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        await _wait_mounted(a, pilot, "#pick_method")
        a.query_one("#method", Input).value = "q4_k_m"
        a.query_one("#pick_method", Button).press()
        for _ in range(20):
            await pilot.pause()
            try:
                _picker(a)
                break
            except Exception:
                continue
        sl = _picker(a)
        # Click-order q8_0 -> q5_k_m -> q4_k_m (q4_k_m already preselected).
        sl.select("q8_0")
        sl.select("q5_k_m")
        await pilot.pause()
        a.screen.query_one("#mp_confirm", Button).press()
        await pilot.pause()
        # Registry order: q4_k_m(idx 7) < q5_k_m(idx 8) < q8_0(idx 6)?  No:
        # q8_0 comes BEFORE q5_k_m in the registry (idx 6 < 8).  Expected
        # output is joined by registry position.
        registry_ids = [m.id for m in METHODS]
        selected = {"q4_k_m", "q5_k_m", "q8_0"}
        expected = ", ".join(mid for mid in registry_ids if mid in selected)
        assert a.query_one("#method", Input).value == expected


async def test_cancel_leaves_method_untouched():
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        await _wait_mounted(a, pilot, "#pick_method")
        a.query_one("#method", Input).value = "q4_k_m"
        a.query_one("#pick_method", Button).press()
        for _ in range(20):
            await pilot.pause()
            try:
                _picker(a)
                break
            except Exception:
                continue
        a.screen.query_one("#mp_cancel", Button).press()
        await pilot.pause()
        assert a.query_one("#method", Input).value == "q4_k_m"


async def test_escape_leaves_method_untouched():
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        await _wait_mounted(a, pilot, "#pick_method")
        a.query_one("#method", Input).value = "iq2_xs"
        a.query_one("#pick_method", Button).press()
        for _ in range(20):
            await pilot.pause()
            try:
                _picker(a)
                break
            except Exception:
                continue
        await pilot.press("escape")
        await pilot.pause()
        assert a.query_one("#method", Input).value == "iq2_xs"


async def test_imatrix_marker_visible_in_picker():
    # The [IMATRIX] badge must render LITERALLY in iq* options -- Textual's
    # markup parser eats un-escaped uppercase tags, so the prompt is escaped.
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        await _wait_mounted(a, pilot, "#pick_method")
        a.query_one("#pick_method", Button).press()
        for _ in range(20):
            await pilot.pause()
            try:
                _picker(a)
                break
            except Exception:
                continue
        sl = _picker(a)
        iq_index = [m.id for m in METHODS].index("iq2_xs")
        prompt = _prompt_plain(sl.get_option_at_index(iq_index))
        assert "[IMATRIX]" in prompt
        # And a non-iq method shows no badge.
        q4_index = [m.id for m in METHODS].index("q4_k_m")
        assert "[IMATRIX]" not in _prompt_plain(sl.get_option_at_index(q4_index))

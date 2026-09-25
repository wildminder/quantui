"""QA verification for the method picker (commit 6bda155).

Independently authored edge cases -- complements
``tests/test_method_picker_headless.py`` (the engineer's tests) without
duplicating them: empty-field no-op, typo-in-field preselect, iq* round-trip,
full option-id surface, and the hand-typed-typo validation gate still being
enforced (the picker must not weaken it).
"""

from conftest import wait_mounted
from textual.widgets import Button, Input, SelectionList

from quantui import app as appmod
from quantui import screens
from quantui.quant_methods import (
    ALLOWED_QUANT_IDS,
    IMATRIX_QUANT_IDS,
    METHODS,
    NATIVE_QUANT_IDS,
)


def _picker(app) -> SelectionList:
    return app.screen.query_one("#mp_list", SelectionList)


async def _open_picker(a, pilot) -> SelectionList:
    """Press the button and wait for the modal body to compose."""
    a.query_one("#pick_method", Button).press()
    for _ in range(20):
        await pilot.pause()
        try:
            return _picker(a)
        except Exception:
            continue
    raise AssertionError("picker did not mount within 20 pauses")


async def test_empty_field_confirm_nothing_keeps_empty():
    # Empty #method: nothing preselected; confirming an EMPTY selection is a
    # no-op (the callback ignores an empty dismiss -- clearing the method
    # would break the run gate's expectations).
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        await wait_mounted(a, pilot, "#method")
        a.query_one("#method", Input).value = ""
        sl = await _open_picker(a, pilot)
        assert sl.selected == []
        a.screen.query_one("#mp_confirm", Button).press()
        await pilot.pause()
        await wait_mounted(a, pilot, "#method")
        assert a.query_one("#method", Input).value == ""


async def test_typo_in_field_preselects_nothing():
    # A hand-typed typo ("q4_km,") must not crash the picker and must not
    # preselect anything (unknown ids are silently ignored).
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        await wait_mounted(a, pilot, "#method")
        a.query_one("#method", Input).value = "q4_km,"
        sl = await _open_picker(a, pilot)
        assert sl.selected == []
        # Cancelling leaves the typo in place (field untouched by the modal).
        a.screen.query_one("#mp_cancel", Button).press()
        await pilot.pause()
        await wait_mounted(a, pilot, "#method")
        assert a.query_one("#method", Input).value == "q4_km,"


async def test_iq_star_preselected_roundtrips():
    # An imatrix-gated iq* id preselects and confirm round-trips it verbatim.
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        await wait_mounted(a, pilot, "#method")
        a.query_one("#method", Input).value = "iq2_xs"
        sl = await _open_picker(a, pilot)
        assert sl.selected == ["iq2_xs"]
        a.screen.query_one("#mp_confirm", Button).press()
        await pilot.pause()
        await wait_mounted(a, pilot, "#method")
        assert a.query_one("#method", Input).value == "iq2_xs"


async def test_all_official_ids_present_as_option_values():
    # The picker covers the 35 official unsloth ids PLUS the 4 native ids
    # (S4.1): official ids are a subset, natives complete the surface.
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        sl = await _open_picker(a, pilot)
        values = {sl.get_option_at_index(i).value for i in range(sl.option_count)}
        assert set(ALLOWED_QUANT_IDS) | set(IMATRIX_QUANT_IDS) <= values
        assert values == set(ALLOWED_QUANT_IDS) | set(IMATRIX_QUANT_IDS) | set(NATIVE_QUANT_IDS)
        assert sl.option_count == len(METHODS) == 40


async def test_hand_typed_typo_still_caught_by_validation():
    # Regression: the picker must not weaken the typo gate. A hand-typed
    # bogus id is refused by run_config.validate_gguf BEFORE any worker runs.
    from quantui import run_config as rc

    a = appmod.QuantApp()
    async with a.run_test():
        a.query_one("#method", Input).value = "q4_km"  # typo for q4_k_m
        cfg = a._read_config()
        assert cfg.gguf.method == "q4_km"
        errs = rc.validate_gguf(cfg.gguf)
        assert any("Unknown quantization method" in e for e in errs)
        assert any("q4_km" in e for e in errs)


async def test_picker_is_modal_and_modal_only():
    # The picker is a modal screen pushed ABOVE the main screen; the main
    # screen's widgets stay queryable but the active screen is the picker
    # (so Escape routes to the picker's cancel binding).
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        sl = await _open_picker(a, pilot)
        assert isinstance(a.screen, screens.MethodPickerScreen)
        assert sl.id == "mp_list"
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(a.screen, screens.MethodPickerScreen)

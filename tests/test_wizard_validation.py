"""Headless tests for S3.2 inline blur validation."""

from textual.widgets import Input, Label

from quantui import app as appmod


async def test_blur_empty_model_shows_hint():
    """Focus + blur an empty #model -> hint label visible + -invalid class."""
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        model = a.query_one("#model", Input)
        model.focus()
        await pilot.pause()
        # Blur while empty.
        a.set_focus(None)
        await pilot.pause()

        assert model.has_class("-invalid")
        hint = a.query_one("#field_hint_model", Label)
        assert hint.display is True
        assert "required" in str(hint.content).lower()


async def test_valid_input_clears_hint():
    """Blur with content -> no -invalid class; re-blur after filling clears hint."""
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        model = a.query_one("#model", Input)
        # First blur empty -> invalid state on.
        model.focus()
        await pilot.pause()
        a.set_focus(None)
        await pilot.pause()
        assert model.has_class("-invalid")

        # Fill and blur again -> cleared.
        model.value = "/some/model"
        model.focus()
        await pilot.pause()
        a.set_focus(None)
        await pilot.pause()
        assert not model.has_class("-invalid")
        hint = a.query_one("#field_hint_model", Label)
        assert hint.display is False


async def test_non_required_field_ignored():
    """Blurring #custom (not in REQUIRED_FIELDS) never sets -invalid."""
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        custom = a.query_one("#custom", Input)
        custom.focus()
        await pilot.pause()
        a.set_focus(None)
        await pilot.pause()
        assert not custom.has_class("-invalid")

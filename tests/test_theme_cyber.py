"""Tests for the quantui-cyber theme + app identity (plan 2026-09-08-scifi-ui).

S1.1: title pins. S2.1: theme registration + variable-value pins.
"""

from quantui import app as appmod
from quantui.theme import build_cyber_theme


async def test_app_title_is_quantui(tmp_path, monkeypatch):
    """The app (Header + terminal window) must be titled 'QuantUI', not 'QuantApp'."""
    monkeypatch.setenv("UNSLOTH_CTQ_LOG_DIR", str(tmp_path))
    a = appmod.QuantApp()
    async with a.run_test():
        assert a.title == "QuantUI"


def test_title_is_class_constant():
    """The title must come from the QuantApp.TITLE constant (deterministic, no drift)."""
    assert appmod.QuantApp.TITLE == "QuantUI"


# ---- S2.1: theme registration ----------------------------------------------------


def test_theme_builder_values_pinned():
    """The pure builder returns the exact sci-fi palette (catches accidental retunes)."""
    t = build_cyber_theme()
    assert t.name == "quantui-cyber"
    assert t.primary == "#00e5ff"
    assert t.accent == "#ff2d95"
    assert t.background == "#050b14"
    assert t.dark is True


async def test_theme_registered_on_app(tmp_path, monkeypatch):
    """'quantui-cyber' is registered and retrievable from a booted app."""
    monkeypatch.setenv("UNSLOTH_CTQ_LOG_DIR", str(tmp_path))
    a = appmod.QuantApp()
    async with a.run_test():
        theme = a.get_theme("quantui-cyber")
        assert theme is not None
        assert theme.primary == "#00e5ff"


async def test_app_defaults_to_cyber(tmp_path, monkeypatch):
    """A booted app uses the cyber theme by default."""
    monkeypatch.setenv("UNSLOTH_CTQ_LOG_DIR", str(tmp_path))
    a = appmod.QuantApp()
    async with a.run_test():
        assert a.theme == "quantui-cyber"


# ---- S2.2: sci-fi input styling ---------------------------------------------------

_MAIN_CSS = appmod.QuantApp.CSS  # aliases MAIN_CSS (pinned contract)


def test_input_css_sci_fi_pins():
    """MAIN_CSS drives the sci-fi input look: cyan round frame, focus power-up."""
    assert "Input {" in _MAIN_CSS and "border: round $primary" in _MAIN_CSS
    assert "Input:focus" in _MAIN_CSS and "border: round $accent" in _MAIN_CSS


def test_invalid_input_still_red():
    """The -invalid state keeps a red border (was pinned pre-theme)."""
    assert "Input.-invalid" in _MAIN_CSS and "$error" in _MAIN_CSS


async def test_input_focus_border_changes(tmp_path, monkeypatch):
    """Focusing a real Input changes its rendered border (cyan -> accent hue)."""
    monkeypatch.setenv("UNSLOTH_CTQ_LOG_DIR", str(tmp_path))
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        inp = a.query_one("#model")
        a.set_focus(inp)
        await pilot.pause()
        border_focused = inp.styles.border
        a.set_focus(None)
        await pilot.pause()
        border_blurred = inp.styles.border
        assert border_focused != border_blurred, (
            "focus must change the input border (sci-fi power-up)"
        )


# ---- S2.3: sci-fi button styling ---------------------------------------------------


def test_button_css_sci_fi_pins():
    """MAIN_CSS gives buttons the sci-fi treatment: bold labels + accent focus."""
    assert "Button {" in _MAIN_CSS
    # Bold labels + a focus/hover rule are the two required pins.
    assert "text-style: bold" in _MAIN_CSS
    assert "Button:focus" in _MAIN_CSS
    assert "$accent" in _MAIN_CSS.split("Button:focus")[1].split("}")[0]


async def test_run_button_variant_survives(tmp_path, monkeypatch):
    """The Run button keeps its success variant (the theme only recolors)."""
    monkeypatch.setenv("UNSLOTH_CTQ_LOG_DIR", str(tmp_path))
    a = appmod.QuantApp()
    async with a.run_test():
        btn = a.query_one("#run")
        assert btn.variant == "success"
        assert btn.display


# ---- S3.1: boot stability ----------------------------------------------------------


async def test_theme_and_title_survive_repeated_boot(tmp_path, monkeypatch):
    """Two fresh app instances in one process: theme + title stable both times
    (guards against theme registration leaking across instances)."""
    monkeypatch.setenv("UNSLOTH_CTQ_LOG_DIR", str(tmp_path))
    for _ in range(2):
        a = appmod.QuantApp()
        async with a.run_test():
            assert a.theme == "quantui-cyber"
            assert a.title == "QuantUI"
            assert a.get_theme("quantui-cyber").primary == "#00e5ff"

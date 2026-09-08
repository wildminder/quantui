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

"""Tests for the quantui-cyber theme + app identity (plan 2026-09-08-scifi-ui).

S1.1 lands the title pins; S2.1 joins the theme registration tests here.
"""

from quantui import app as appmod


async def test_app_title_is_quantui(tmp_path, monkeypatch):
    """The app (Header + terminal window) must be titled 'QuantUI', not 'QuantApp'."""
    monkeypatch.setenv("UNSLOTH_CTQ_LOG_DIR", str(tmp_path))
    a = appmod.QuantApp()
    async with a.run_test():
        assert a.title == "QuantUI"


def test_title_is_class_constant():
    """The title must come from the QuantApp.TITLE constant (deterministic, no drift)."""
    assert appmod.QuantApp.TITLE == "QuantUI"

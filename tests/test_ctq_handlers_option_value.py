"""Regression test for handlers.ctq_option_value.

Every registry OptionField now has a matching widget in the ComfyUI panel
(#scaling_mode, #block_size, #convrot, #convrot_group_size, ...), but
ctq_option_value must still fall back to the option's declared default when a
widget is not mounted instead of crashing _read_config with NoMatches.
"""


from textual.css.query import NoMatches

from quantui.handlers import HandlersMixin
from quantui.quant_methods import comfy_format


class _FakeHandlers(HandlersMixin):
    """Stand-in whose query_one always raises (simulating an unrendered widget).

    Raises the real ``NoMatches`` type: handlers.ctq_option_value deliberately
    narrows to ``except NoMatches`` (CRIT-002 S2.3). A generic Exception must no
    longer be swallowed — that was the silent-swallow bug class this plan
    eliminates.
    """

    def query_one(self, *args, **kwargs):
        raise NoMatches("no node matches")


def _first_int8_option():
    fmt = comfy_format("int8")
    assert fmt.extra_options, "unified int8 must declare options"
    return fmt.extra_options[0]  # scaling_mode


def test_ctq_option_value_falls_back_when_widget_missing():
    opt = _first_int8_option()
    fake = _FakeHandlers()
    val = fake.ctq_option_value(opt)
    assert val == opt.default, f"expected default {opt.default!r}, got {val!r}"


def test_ctq_option_value_returns_widget_value_when_present():
    # When the widget exists, the value is returned unchanged (no fallback).
    sentinel = object()

    class _FakeWithWidget(HandlersMixin):
        def query_one(self, *args, **kwargs):
            return type("W", (), {"value": sentinel})()

    fake = _FakeWithWidget()
    assert fake.ctq_option_value(_first_int8_option()) is sentinel

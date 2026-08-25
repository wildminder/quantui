"""Regression test for handlers.ctq_option_value.

The ComfyUI panel renders only a fixed set of option widgets (#ctq_scaling_mode,
#convrot_group_size). A format may declare an extra option (e.g. #block_size for
int8_block) that has NO corresponding widget. ctq_option_value must fall back to the
option's declared default instead of crashing _read_config with NoMatches.
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


def test_ctq_option_value_falls_back_when_widget_missing():
    fmt = comfy_format("int8_block")
    assert fmt.extra_options, "int8_block must declare at least block_size"
    opt = fmt.extra_options[0]  # block_size
    fake = _FakeHandlers()
    val = fake.ctq_option_value(opt)
    assert val == opt.default, f"expected default {opt.default!r}, got {val!r}"


def test_ctq_option_value_returns_widget_value_when_present():
    # When the widget exists, the value is returned unchanged (no fallback).
    sentinel = object()

    class _FakeWithWidget(HandlersMixin):
        def query_one(self, *args, **kwargs):
            return type("W", (), {"value": sentinel})()

    opt = comfy_format("int8_block").extra_options[0]
    fake = _FakeWithWidget()
    assert fake.ctq_option_value(opt) is sentinel

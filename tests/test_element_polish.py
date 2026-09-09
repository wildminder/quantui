"""Control-panel element-polish tests (plan 2026-09-09-control-panel S2.2).

CSS-only step: the stats line gets a boxed readout treatment and radio
buttons keep a readable $text color. Updated 2026-09-09 (user request): the
rail phase chips were REMOVED (duplicate of the stats line + click opened
the log drawer), so the pill pins are replaced by absence pins.
"""

from quantui import app as appmod
from quantui.app_css import MAIN_CSS


def _css_block(selector: str) -> str:
    """Extract the FIRST `selector { ... }` block from MAIN_CSS."""
    key = selector + " {"
    assert key in MAIN_CSS, f"{selector} rule missing from MAIN_CSS"
    return MAIN_CSS.split(key, 1)[1].split("}", 1)[0]


def test_rail_css_rules_are_gone():
    """The chip pill rules died with the rail (user request 2026-09-09)."""
    for selector in (".rail_label {", ".rail_row {", ".rail_more {"):
        assert selector not in MAIN_CSS, f"{selector!r} still in MAIN_CSS"


def test_stats_boxed_css_pins():
    """#footer_stats gets a boxed readout: round panel border + surface
    background, and the explicit height: 3 (a 2-row box clips its own text:
    both rows are consumed by top/bottom borders)."""
    block = _css_block("#footer_stats")
    assert "border: round $panel" in block
    assert "background: $surface" in block
    assert "height: 3" in block


def test_close_row_css_pins():
    """The Close ✕ row is right-aligned like the other button rows."""
    block = _css_block("#footer_close_row")
    assert "align: right middle" in block


def test_radio_accent_pinned():
    """RadioSet > RadioButton keeps a readable theme-driven $text color."""
    block = _css_block("RadioSet > RadioButton")
    assert "color: $text" in block


def _boot_and_query(tmp_path, monkeypatch, coro_fn):
    """Boot a QuantApp, run coro_fn(a, pilot) inside the app context, return its
    result (widgets must be reduced to plain values INSIDE — regions reset to
    zero once the app exits)."""
    import asyncio

    async def _run():
        monkeypatch.setenv("UNSLOTH_CTQ_LOG_DIR", str(tmp_path))
        a = appmod.QuantApp()
        async with a.run_test() as pilot:
            return await coro_fn(a, pilot)

    return asyncio.run(_run())


def test_stats_box_computed(tmp_path, monkeypatch):
    """#footer_stats computes a 3-row round box on a booted app."""

    async def scenario(a, pilot):
        a._show_run_footer()
        await pilot.pause()
        stats = a.query_one("#footer_stats")
        return {"border_top": stats.styles.border.top[0], "height": stats.region.height}

    got = _boot_and_query(tmp_path, monkeypatch, scenario)
    assert got["border_top"] == "round"
    assert got["height"] == 3

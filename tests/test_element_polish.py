"""Control-panel element-polish tests (plan 2026-09-09-control-panel S2.2).

CSS-only step: rail phase chips become pill boxes, the stats line gets a
boxed readout treatment, and radio buttons keep a readable $text color.
Pins are source-level (MAIN_CSS blocks) + computed-style assertions on a
booted app (the classes already exist; no widget-tree changes).
"""

from quantui import app as appmod
from quantui.app_css import MAIN_CSS
from quantui.panels import ProgressRail


def _css_block(selector: str) -> str:
    """Extract the FIRST `selector { ... }` block from MAIN_CSS."""
    key = selector + " {"
    assert key in MAIN_CSS, f"{selector} rule missing from MAIN_CSS"
    return MAIN_CSS.split(key, 1)[1].split("}", 1)[0]


def test_rail_label_pill_css_pins():
    """MAIN_CSS gives .rail_label a pill treatment: round panel border +
    surface background + padding, auto height with a max-width cap."""
    block = _css_block(".rail_label")
    assert "border: round $panel" in block
    assert "background: $surface" in block
    assert "padding: 0 1" in block
    assert "max-width: 100%" in block
    # The rail row must grow to fit the 3-row pill (the old fixed height: 1
    # clipped the box).
    row_block = _css_block(".rail_row")
    assert "height: auto" in row_block
    # The widget-internal .rail_label rule must not force height: 1 anymore.
    assert ".rail_label { height: 1" not in ProgressRail.DEFAULT_CSS


def test_stats_boxed_css_pins():
    """#footer_stats gets a boxed readout: round panel border + surface
    background, and the explicit height: 3 (a 2-row box clips its own text:
    both rows are consumed by top/bottom borders)."""
    block = _css_block("#footer_stats")
    assert "border: round $panel" in block
    assert "background: $surface" in block
    assert "height: 3" in block


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


def test_rail_label_pill_computed_styles(tmp_path, monkeypatch):
    """A booted app's rail chip actually computes the pill styles (the MAIN_CSS
    rule wins over the old height: 1 label default)."""
    import json

    async def scenario(a, pilot):
        a._show_run_footer()
        await pilot.pause()
        env = "CTQ_PROGRESS " + json.dumps(
            {"phase": "quantize", "cur": 1, "total": 4, "pct": 25.0,
             "label": "Optimizing INT8"}
        )
        a.log_msg(env)
        await pilot.pause()
        await pilot.pause()
        lbl = a.query_one(".rail_label")
        return {"border_top": lbl.styles.border.top[0], "height": lbl.region.height}

    got = _boot_and_query(tmp_path, monkeypatch, scenario)
    assert got["border_top"] == "round"
    # The pill box is 3 rows tall (border + text + border).
    assert got["height"] == 3


def test_footer_stats_box_computed(tmp_path, monkeypatch):
    """#footer_stats computes a 3-row round box on a booted app."""

    async def scenario(a, pilot):
        a._show_run_footer()
        await pilot.pause()
        stats = a.query_one("#footer_stats")
        return {"border_top": stats.styles.border.top[0], "height": stats.region.height}

    got = _boot_and_query(tmp_path, monkeypatch, scenario)
    assert got["border_top"] == "round"
    assert got["height"] == 3

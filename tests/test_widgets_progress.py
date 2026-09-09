"""BlockBar render-math tests (plan 2026-09-09-control-panel S1.2).

Pure tests: no app boot — the row math lives in the module-level
``build_bar_text`` (Widget.size is only meaningful after layout), and the
widget API (update/progress alias/reactive) is driven on unmounted instances,
which Textual allows.
"""

import pytest

from quantui.widgets_progress import BlockBar, build_bar_text

# ---- pure render math (build_bar_text) -----------------------------------------


def test_rows_filled_proportionally():
    """pct 0.6 at width 10 -> 6 filled + 4 track glyphs per row."""
    out = build_bar_text(0.6, 10, 3).plain
    for line in out.splitlines():
        assert line.count("█") == 6
        assert line.count("░") == 4
        assert len(line) == 10


def test_rows_clamp_overflow():
    """pct 1.5 -> fully filled; pct -1 -> empty track."""
    full = build_bar_text(1.5, 10, 1).plain
    assert "█" in full and "░" not in full
    empty = build_bar_text(-1, 10, 1).plain
    assert "█" not in empty and "░" in empty


def test_complete_state_uses_success_color():
    """pct >= 1.0 flips the fill style to the success green (styles live in
    the joined Text's spans, not the base style)."""
    done_styles = {str(s.style) for s in build_bar_text(1.0, 10, 1).spans}
    mid_styles = {str(s.style) for s in build_bar_text(0.6, 10, 1).spans}
    assert any("#5fd7a0" in s for s in done_styles)
    assert not any("#5fd7a0" in s for s in mid_styles)
    assert any("#4dc3ff" in s for s in mid_styles)


def test_three_rows_by_default():
    """Default rows=3: render output is exactly 3 identical lines."""
    out = build_bar_text(0.4, 12, 3).plain
    lines = out.splitlines()
    assert len(lines) == 3
    assert len(set(lines)) == 1


def test_zero_width_guard():
    """width 0 (pre-layout) clamps to a single cell, never a zero-division."""
    assert len(build_bar_text(0.5, 0, 1).plain) == 1
    assert build_bar_text(None, 0, 1).plain == "░"


@pytest.mark.parametrize("pct,expect_filled", [(0.0, 0), (0.25, 3), (0.5, 5), (1.0, 10)])
def test_fill_boundaries(pct, expect_filled):
    """Boundary mapping pct->filled-glyph count at width 10."""
    out = build_bar_text(pct, 10, 1).plain
    assert out.count("█") == expect_filled


# ---- widget API (unmounted instances) -------------------------------------------


def test_update_api_matches_progressbar():
    """update(total=, progress=) — ProgressBar-compatible: progress on the
    0-total scale (the #footer_bar caller's contract)."""
    bar = BlockBar(rows=2)
    bar.update(total=100, progress=65)
    assert bar.percentage == 0.65
    assert bar.progress == 65  # 0-total alias for legacy callers/tests


def test_progress_property_alias_roundtrip():
    """progress alias reads percentage * 100 without float noise."""
    bar = BlockBar(total=100, rows=1)
    bar.update(progress=40)
    assert bar.progress == 40
    assert bar.percentage == 0.4
    bar.update(progress=65)
    assert bar.progress == 65  # 0.65 * 100 must not be 65.00000000000001


def test_update_clamps():
    """update() clamps out-of-range progress instead of corrupting the math."""
    bar = BlockBar(total=100, rows=1)
    bar.update(progress=150)
    assert bar.percentage == 1.0
    bar.update(progress=-5)
    assert bar.percentage == 0.0


def test_update_total_change_rescales():
    """A new total rescales subsequent progress values (native semantics)."""
    bar = BlockBar(total=100, rows=1)
    bar.update(total=200, progress=100)
    assert bar.percentage == 0.5


def test_unmounted_render_defaults_to_empty_track():
    """An unmounted bar's size is 0x0 — render clamps to one cell, no crash."""
    bar = BlockBar(total=100, rows=1)
    bar.update(progress=50)
    assert len(bar.render().plain) == 1

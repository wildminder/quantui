"""Headless tests for the run footer (plan 2026-09-08-run-footer).

The footer replaces the right-hand rail (layout v2): #progress_rail + #status +
ResultsCard stack at the bottom of #body, full width, hidden until a run starts.
Composition + CSS pins live here; the visibility LIFECYCLE is S2.1 (same file);
the success-only results buttons are S2.2 (test_results_card*.py).
"""

from textual.widgets import Label

from quantui import app as appmod
from quantui import panels
from quantui.app_css import MAIN_CSS
from quantui.widgets_results import ResultsCard


async def test_footer_holds_rail_status_results():
    """Composition: #run_footer wraps the historical rail ids + the results card.

    Left column #footer_left holds #progress_rail (ProgressRail) + #status (Label);
    ResultsCard is a direct child of the footer (right side)."""
    from textual.containers import Vertical

    a = appmod.QuantApp()
    async with a.run_test():
        footer = a.query_one("#run_footer", panels.RunFooter)
        rail = footer.query_one("#progress_rail", panels.ProgressRail)
        assert isinstance(rail, panels.ProgressRail)
        status = footer.query_one("#status", Label)
        assert isinstance(status, Label)
        left = footer.query_one("#footer_left", Vertical)
        assert left.query_one("#progress_rail") is not None
        assert left.query_one("#status") is not None
        card = footer.query_one(ResultsCard)
        assert isinstance(card, ResultsCard)


def test_params_full_width_css():
    """CSS contract: #params is 100% wide, #rail rule is gone, footer pinned hidden."""
    assert "width: 100%" in MAIN_CSS
    assert "#rail {" not in MAIN_CSS
    assert "#run_footer {" in MAIN_CSS
    # The footer must start hidden (S2.1 turns it on at run start).
    footer_block = MAIN_CSS.split("#run_footer {", 1)[1].split("}", 1)[0]
    assert "display: none" in footer_block
    assert "max-height" in footer_block
    # #params must no longer carry the old 30/70 split border.
    params_block = MAIN_CSS.split("#params {", 1)[1].split("}", 1)[0]
    assert "width: 100%" in params_block
    assert "border-right" not in params_block
